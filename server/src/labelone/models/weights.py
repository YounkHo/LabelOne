from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from threading import Lock, RLock
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from labelone.errors import LabelOneError

from .catalog import ModelRecord
from .types import Availability, AvailabilityState


DEFAULT_ALLOWED_DOMAINS = frozenset({
    "github.com",
    "githubusercontent.com",
    "huggingface.co",
    "modelscope.cn",
    "modelscope.ai",
    "modelscope.oss-cn-beijing.aliyuncs.com",
})
DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_PARALLEL_DOWNLOAD_THRESHOLD = 32 * 1024 * 1024
DEFAULT_PARALLEL_DOWNLOAD_WORKERS = 4
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MODEL_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MANIFEST_VERSION = 1


class ModelWeightError(LabelOneError):
    code = "model_weight_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ModelWeightCancelled(ModelWeightError):
    code = "model_weight_cancelled"


class WeightResponse(Protocol):
    headers: Any
    status: int

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def geturl(self) -> str: ...


WeightOpener = Callable[[Request, float], WeightResponse]


@dataclass(frozen=True, slots=True)
class RemoteWeight:
    url_index: int
    url: str
    filename: str
    downloaded: bool
    local_path: Path | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    source_id: str = "other"
    preferred: bool = False


@dataclass(frozen=True, slots=True)
class DownloadedWeight:
    model_id: str
    url_index: int
    source_url: str
    final_url: str
    local_path: Path
    size_bytes: int
    sha256: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    model_id: str
    url_index: int
    source_url: str
    final_url: str
    received_bytes: int
    total_bytes: int | None
    cache_hit: bool = False

    @property
    def progress(self) -> float | None:
        if self.total_bytes is None or self.total_bytes <= 0:
            return None
        return min(1.0, self.received_bytes / self.total_bytes)


ProgressCallback = Callable[[DownloadProgress], None]
CancelCallback = Callable[[], bool]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):  # noqa: ANN001
        del request, file_pointer, code, message, headers, new_url
        return None


def model_weight_source(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").casefold().strip(".")
    if (
        hostname == "github.com"
        or hostname.endswith(".github.com")
        or hostname.endswith(".githubusercontent.com")
    ):
        return "github"
    if (
        hostname == "modelscope.cn"
        or hostname.endswith(".modelscope.cn")
        or hostname == "modelscope.ai"
        or hostname.endswith(".modelscope.ai")
        or hostname == "modelscope.oss-cn-beijing.aliyuncs.com"
    ):
        return "modelscope"
    if hostname == "huggingface.co" or hostname.endswith(".huggingface.co"):
        return "huggingface"
    return "other"


def _default_open(request: Request, timeout: float) -> WeightResponse:
    opener = build_opener(_NoRedirect())
    try:
        return opener.open(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code in _REDIRECT_STATUSES or exc.code == 416:
            return exc
        raise


def _header(headers: Any, name: str) -> str | None:
    value = headers.get(name) if headers is not None else None
    return str(value).strip() if value is not None else None


def _status(response: WeightResponse) -> int:
    raw = getattr(response, "status", None)
    if raw is None and hasattr(response, "getcode"):
        raw = response.getcode()
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ModelWeightError("Weight server response did not provide a valid HTTP status") from exc


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_range(value: str | None) -> tuple[int | None, int | None, int] | None:
    if not value:
        return None
    complete = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
    if complete:
        start, end, total = (int(item) for item in complete.groups())
        return start, end, total
    unsatisfied = re.fullmatch(r"bytes \*/(\d+)", value)
    if unsatisfied:
        return None, None, int(unsatisfied.group(1))
    return None


class ModelWeightStore:
    """Explicit, allowlisted and crash-safe model weight downloads."""

    def __init__(
        self,
        data_dir: Path,
        *,
        root_dir: Path | None = None,
        allowed_domains: set[str] | frozenset[str] | tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_seconds: float = 30.0,
        max_redirects: int = 5,
        parallel_workers: int = DEFAULT_PARALLEL_DOWNLOAD_WORKERS,
        parallel_threshold_bytes: int = DEFAULT_PARALLEL_DOWNLOAD_THRESHOLD,
        opener: WeightOpener | None = None,
    ) -> None:
        self.root = root_dir.expanduser().resolve() if root_dir is not None else (data_dir.expanduser().resolve() / "model-weights")
        normalized_domains = {domain.strip(".").casefold() for domain in allowed_domains if domain.strip(".")}
        if not normalized_domains:
            raise ValueError("allowed_domains must not be empty")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_redirects < 0:
            raise ValueError("max_redirects must not be negative")
        if not 1 <= parallel_workers <= 8:
            raise ValueError("parallel_workers must be between 1 and 8")
        if parallel_threshold_bytes <= 0:
            raise ValueError("parallel_threshold_bytes must be positive")
        self.allowed_domains = frozenset(normalized_domains)
        self.max_bytes = int(max_bytes)
        self.timeout_seconds = float(timeout_seconds)
        self.max_redirects = int(max_redirects)
        self.parallel_workers = int(parallel_workers)
        self.parallel_threshold_bytes = int(parallel_threshold_bytes)
        self.opener = opener or _default_open
        self._lock = RLock()
        self._download_locks: dict[tuple[str, str], Lock] = {}
        self._active_partials: set[Path] = set()

    def _download_lock(self, model_id: str, target_name: str) -> Lock:
        with self._lock:
            return self._download_locks.setdefault((model_id, target_name), Lock())

    def _model_dir(self, model_id: str, *, create: bool = False) -> Path:
        if not _MODEL_ID.fullmatch(model_id):
            raise ModelWeightError(
                "Model ID is not safe for a weight directory",
                details={"model_id": model_id},
            )
        directory = (self.root / model_id).resolve()
        if directory.parent != self.root:
            raise ModelWeightError("Model weight directory escapes the configured data directory")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _validate_url(self, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme.casefold() != "https":
            raise ModelWeightError("Model weights may only be downloaded over HTTPS", details={"url": url})
        if parsed.username is not None or parsed.password is not None:
            raise ModelWeightError("Model weight URLs must not contain credentials", details={"url": url})
        if parsed.port not in {None, 443}:
            raise ModelWeightError("Model weight URLs may only use the default HTTPS port", details={"url": url})
        hostname = (parsed.hostname or "").casefold().strip(".")
        if not hostname or not any(hostname == domain or hostname.endswith(f".{domain}") for domain in self.allowed_domains):
            raise ModelWeightError(
                "Model weight URL domain is not allowlisted",
                details={"url": url, "hostname": hostname, "allowed_domains": sorted(self.allowed_domains)},
            )
        if parsed.fragment:
            raise ModelWeightError("Model weight URLs must not contain fragments", details={"url": url})
        return url

    @staticmethod
    def _remote_location(record: ModelRecord, url_index: int) -> str:
        locations = record.descriptor.weight_locations
        if isinstance(url_index, bool) or not isinstance(url_index, int) or not 0 <= url_index < len(locations):
            raise ModelWeightError(
                "Weight URL index is out of range",
                details={"url_index": url_index, "location_count": len(locations)},
            )
        return str(locations[url_index])

    @staticmethod
    def _filename(url: str) -> str:
        parsed_path = urlsplit(url).path
        if parsed_path.endswith("/"):
            raise ModelWeightError("Model weight URL does not contain a safe filename", details={"url": url})
        name = unquote(Path(parsed_path).name)
        if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
            raise ModelWeightError("Model weight URL does not contain a safe filename", details={"url": url})
        if len(name.encode("utf-8")) > 240:
            raise ModelWeightError("Model weight filename is too long", details={"url": url})
        return name

    def _manifest_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "manifest.json"

    def _read_manifest(self, model_id: str) -> dict[str, Any]:
        path = self._manifest_path(model_id)
        if not path.is_file():
            return {"version": _MANIFEST_VERSION, "model_id": model_id, "files": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelWeightError(
                "Model weight manifest is unreadable",
                details={"model_id": model_id, "path": str(path), "error": str(exc)},
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _MANIFEST_VERSION
            or payload.get("model_id") != model_id
            or not isinstance(payload.get("files"), dict)
        ):
            raise ModelWeightError("Model weight manifest has an invalid structure", details={"path": str(path)})
        return payload

    def _manifest_entry_path(self, model_id: str, entry: object) -> Path | None:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None
        directory = self._model_dir(model_id)
        candidate = (directory / entry["path"]).resolve()
        if candidate.parent != directory or not candidate.is_file():
            return None
        expected_size = entry.get("size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0 or candidate.stat().st_size != expected_size:
            return None
        return candidate

    def _write_manifest(self, model_id: str, payload: dict[str, Any]) -> None:
        directory = self._model_dir(model_id, create=True)
        path = directory / "manifest.json"
        partial = directory / ".manifest.json.part"
        partial.unlink(missing_ok=True)
        try:
            with partial.open("xb") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            partial.unlink(missing_ok=True)

    @staticmethod
    def _partial_metadata_path(partial_path: Path) -> Path:
        return partial_path.with_name(f"{partial_path.name}.json")

    def _read_partial_metadata(self, partial_path: Path, source_url: str) -> dict[str, Any] | None:
        metadata_path = self._partial_metadata_path(partial_path)
        if not partial_path.is_file() or not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("source_url") != source_url
            or not isinstance(payload.get("final_url"), str)
        ):
            return None
        size = partial_path.stat().st_size
        total = payload.get("total_bytes")
        if size <= 0 or size > self.max_bytes or (total is not None and (not isinstance(total, int) or total <= 0 or size > total)):
            return None
        return payload

    def _write_partial_metadata(self, partial_path: Path, payload: dict[str, Any]) -> None:
        metadata_path = self._partial_metadata_path(partial_path)
        temporary = metadata_path.with_name(f".{metadata_path.name}.part")
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("xb") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _discard_partial(self, partial_path: Path) -> None:
        partial_path.unlink(missing_ok=True)
        self._partial_metadata_path(partial_path).unlink(missing_ok=True)

    def cleanup_stale_parts(self, model_id: str) -> int:
        directory = self._model_dir(model_id)
        if not directory.is_dir():
            return 0
        removed = 0
        with self._lock:
            active_partials = set(self._active_partials)
        for candidate in directory.iterdir():
            if not candidate.is_file() or not candidate.name.endswith(".part"):
                continue
            if candidate in active_partials:
                continue
            metadata_path = self._partial_metadata_path(candidate)
            keep = False
            if metadata_path.is_file():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    source_url = str(payload.get("source_url") or "") if isinstance(payload, dict) else ""
                    keep = self._read_partial_metadata(candidate, self._validate_url(source_url)) is not None
                except (OSError, UnicodeError, json.JSONDecodeError, ModelWeightError):
                    keep = False
            if keep:
                continue
            self._discard_partial(candidate)
            removed += 1
        return removed

    def cleanup_all_stale_parts(self) -> int:
        if not self.root.is_dir():
            return 0
        removed = 0
        for candidate in self.root.iterdir():
            if not candidate.is_dir() or not _MODEL_ID.fullmatch(candidate.name):
                continue
            try:
                removed += self.cleanup_stale_parts(candidate.name)
            except ModelWeightError:
                continue
        return removed

    def local_overrides(self, model_id: str) -> dict[str, Path]:
        with self._lock:
            manifest = self._read_manifest(model_id)
            overrides: dict[str, Path] = {}
            for source_url, entry in manifest["files"].items():
                if not isinstance(source_url, str):
                    continue
                try:
                    self._validate_url(source_url)
                except ModelWeightError:
                    continue
                path = self._manifest_entry_path(model_id, entry)
                if path is not None:
                    overrides[source_url] = path
            return overrides

    def effective_record(self, record: ModelRecord) -> ModelRecord:
        overrides = self.local_overrides(record.descriptor.id)
        if not overrides:
            return record

        def replace(value: object) -> object:
            if isinstance(value, str):
                return str(overrides.get(value, value))
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        config = replace(record.config)
        assert isinstance(config, dict)
        locations = [str(overrides.get(location, location)) for location in record.descriptor.weight_locations]
        adapter = record.descriptor.adapter

        def local_file(location: object) -> bool:
            if not isinstance(location, str) or urlsplit(location).scheme:
                return False
            path = Path(location).expanduser()
            resolved = path if path.is_absolute() else record.descriptor.config_path.parent / path
            return resolved.resolve().is_file()

        if adapter == "ppocr_onnx":
            keys = ["det_model_path", "rec_model_path"]
            if bool(config.get("use_angle_cls", True)):
                keys.append("cls_model_path")
            available = all(local_file(config.get(key)) for key in keys)
        elif adapter == "segment_anything_onnx":
            available = all(local_file(config.get(key)) for key in ("encoder_model_path", "decoder_model_path"))
        else:
            available = any(local_file(location) for location in locations)
        descriptor = record.descriptor.model_copy(update={
            "weight_locations": locations,
            "availability": Availability(
                state=AvailabilityState.AVAILABLE if available else record.descriptor.availability.state,
                reason=None if available else record.descriptor.availability.reason,
            ),
        })
        return ModelRecord(descriptor=descriptor, config=config)

    def list_remote(self, model_id: str, record: ModelRecord, preferred_source: str = "auto") -> list[RemoteWeight]:
        manifest = self._read_manifest(model_id)
        results: list[RemoteWeight] = []
        for index, raw in enumerate(record.descriptor.weight_locations):
            url = str(raw)
            if urlsplit(url).scheme.casefold() != "https":
                continue
            self._validate_url(url)
            entry = manifest["files"].get(url)
            local_path = self._manifest_entry_path(model_id, entry)
            source_id = model_weight_source(url)
            results.append(RemoteWeight(
                url_index=index,
                url=url,
                filename=self._filename(url),
                downloaded=local_path is not None,
                local_path=local_path,
                size_bytes=entry.get("size_bytes") if local_path is not None and isinstance(entry, dict) else None,
                sha256=entry.get("sha256") if local_path is not None and isinstance(entry, dict) else None,
                source_id=source_id,
                preferred=preferred_source == "auto" or source_id == preferred_source,
            ))
        return sorted(results, key=lambda item: (not item.preferred, item.url_index))

    def _cached(self, model_id: str, source_url: str, url_index: int, manifest: dict[str, Any]) -> DownloadedWeight | None:
        entry = manifest["files"].get(source_url)
        path = self._manifest_entry_path(model_id, entry)
        if path is None or not isinstance(entry, dict):
            return None
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or _hash_file(path) != expected_hash:
            return None
        return DownloadedWeight(
            model_id=model_id,
            url_index=url_index,
            source_url=source_url,
            final_url=str(entry.get("final_url") or source_url),
            local_path=path,
            size_bytes=int(entry["size_bytes"]),
            sha256=expected_hash,
            cache_hit=True,
        )

    def _open_final(
        self,
        source_url: str,
        *,
        headers: dict[str, str] | None = None,
        accepted_statuses: frozenset[int] = frozenset({200}),
    ) -> tuple[WeightResponse, str]:
        current_url = source_url
        for redirect_count in range(self.max_redirects + 1):
            self._validate_url(current_url)
            request = Request(
                current_url,
                headers={"User-Agent": "LabelOne/0.1 model-weight-downloader", "Accept-Encoding": "identity", **(headers or {})},
                method="GET",
            )
            try:
                response = self.opener(request, self.timeout_seconds)
            except Exception as exc:
                raise ModelWeightError(
                    "Could not open model weight URL",
                    details={"url": current_url, "error": str(exc)},
                ) from exc
            status = _status(response)
            response_url = response.geturl() or current_url
            try:
                self._validate_url(response_url)
            except Exception:
                response.close()
                raise
            if status in _REDIRECT_STATUSES:
                location = _header(response.headers, "Location")
                response.close()
                if not location:
                    raise ModelWeightError("Weight download redirect did not provide a Location header")
                if redirect_count >= self.max_redirects:
                    raise ModelWeightError("Weight download exceeded the redirect limit")
                current_url = self._validate_url(urljoin(response_url, location))
                continue
            if status not in accepted_statuses:
                response.close()
                raise ModelWeightError(
                    "Weight server returned an unsuccessful status",
                    details={"url": response_url, "status": status},
                )
            return response, self._validate_url(response_url)
        raise ModelWeightError("Weight download exceeded the redirect limit")

    def _download_parallel_segments(
        self,
        source_url: str,
        partial_path: Path,
        *,
        start_offset: int,
        total_bytes: int,
        cancel: CancelCallback | None,
        progress: Callable[[int], None] | None,
    ) -> list[Path] | None:
        """Fetch disjoint ranges concurrently without exposing a sparse partial file.

        Segment files are merged only after every range succeeds. A transport or
        range failure returns ``None`` so the caller can consume its already-open
        sequential response instead. Cancellation remains observable immediately.
        """
        remaining = total_bytes - start_offset
        worker_count = min(self.parallel_workers, remaining)
        if worker_count <= 1 or remaining < self.parallel_threshold_bytes:
            return None
        segment_size = (remaining + worker_count - 1) // worker_count
        ranges: list[tuple[int, int, Path]] = []
        progress_lock = Lock()
        downloaded_bytes = 0
        for worker_index in range(worker_count):
            start = start_offset + worker_index * segment_size
            if start >= total_bytes:
                break
            end = min(total_bytes - 1, start + segment_size - 1)
            segment_path = partial_path.with_name(
                f"{partial_path.name}.{start}-{end}.segment.part"
            )
            segment_path.unlink(missing_ok=True)
            ranges.append((start, end, segment_path))

        with self._lock:
            self._active_partials.update(segment_path for _, _, segment_path in ranges)

        def fetch(start: int, end: int, segment_path: Path) -> Path:
            nonlocal downloaded_bytes
            response: WeightResponse | None = None
            try:
                if cancel is not None and cancel():
                    raise ModelWeightCancelled("Model weight download was canceled")
                response, _ = self._open_final(
                    source_url,
                    headers={"Range": f"bytes={start}-{end}"},
                    accepted_statuses=frozenset({206}),
                )
                content_range = _content_range(_header(response.headers, "Content-Range"))
                if content_range != (start, end, total_bytes):
                    raise ModelWeightError("Weight server returned an invalid parallel Content-Range")
                expected_size = end - start + 1
                raw_length = _header(response.headers, "Content-Length")
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError as exc:
                        raise ModelWeightError("Weight server returned an invalid Content-Length") from exc
                    if content_length != expected_size:
                        raise ModelWeightError("Parallel weight segment length did not match its range")
                received = 0
                with segment_path.open("xb") as handle:
                    while True:
                        if cancel is not None and cancel():
                            raise ModelWeightCancelled("Model weight download was canceled")
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > expected_size:
                            raise ModelWeightError("Parallel weight segment exceeded its requested range")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if received != expected_size:
                    raise ModelWeightError("Parallel weight segment was incomplete")
                if progress is not None:
                    with progress_lock:
                        downloaded_bytes += received
                        progress(start_offset + downloaded_bytes)
                return segment_path
            except Exception:
                segment_path.unlink(missing_ok=True)
                raise
            finally:
                if response is not None:
                    response.close()

        try:
            with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix="labelone-weight") as executor:
                futures = {
                    executor.submit(fetch, start, end, segment_path): segment_path
                    for start, end, segment_path in ranges
                }
                for future in as_completed(futures):
                    future.result()
        except ModelWeightCancelled:
            for _, _, segment_path in ranges:
                segment_path.unlink(missing_ok=True)
            with self._lock:
                self._active_partials.difference_update(segment_path for _, _, segment_path in ranges)
            raise
        except Exception:
            for _, _, segment_path in ranges:
                segment_path.unlink(missing_ok=True)
            with self._lock:
                self._active_partials.difference_update(segment_path for _, _, segment_path in ranges)
            return None
        return [segment_path for _, _, segment_path in ranges]

    def download(
        self,
        model_id: str,
        record: ModelRecord,
        url_index: int,
        *,
        expected_sha256: str | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> DownloadedWeight:
        source_url = self._validate_url(self._remote_location(record, url_index))
        target_name = self._filename(source_url)
        with self._download_lock(model_id, target_name):
            if expected_sha256 is not None:
                expected_sha256 = expected_sha256.casefold()
                if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                    raise ModelWeightError("Expected model weight SHA-256 is invalid")

            def check_canceled() -> None:
                if cancel is not None and cancel():
                    raise ModelWeightCancelled(
                        "Model weight download was canceled",
                        details={"model_id": model_id, "url_index": url_index},
                    )

            check_canceled()
            directory = self._model_dir(model_id, create=True)
            with self._lock:
                manifest = self._read_manifest(model_id)
                cached = self._cached(model_id, source_url, url_index, manifest)
            if cached is not None:
                if expected_sha256 is not None and cached.sha256 != expected_sha256:
                    raise ModelWeightError(
                        "Cached model weight SHA-256 does not match the expected value",
                        details={"expected": expected_sha256, "actual": cached.sha256},
                    )
                if progress is not None:
                    progress(DownloadProgress(
                        model_id=model_id,
                        url_index=url_index,
                        source_url=source_url,
                        final_url=cached.final_url,
                        received_bytes=cached.size_bytes,
                        total_bytes=cached.size_bytes,
                        cache_hit=True,
                    ))
                return cached

            filename = target_name
            conflicting = {
                entry.get("path")
                for url, entry in manifest["files"].items()
                if url != source_url and isinstance(entry, dict)
            }
            if filename in conflicting:
                parsed = Path(filename)
                filename = f"{parsed.stem}-{sha256(source_url.encode()).hexdigest()[:8]}{parsed.suffix}"
            final_path = directory / filename
            partial_path = directory / f".{filename}.part"
            with self._lock:
                self._active_partials.add(partial_path)
            try:
                self.cleanup_stale_parts(model_id)
            except Exception:
                with self._lock:
                    self._active_partials.discard(partial_path)
                raise
            metadata = self._read_partial_metadata(partial_path, source_url)
            if partial_path.exists() and metadata is None:
                self._discard_partial(partial_path)
            resume_size = partial_path.stat().st_size if metadata is not None else 0
            request_headers: dict[str, str] = {}
            if resume_size > 0:
                request_headers["Range"] = f"bytes={resume_size}-"
                validator = str(metadata.get("etag") or metadata.get("last_modified") or "")
                if validator:
                    request_headers["If-Range"] = validator

            response: WeightResponse | None = None
            final_url = str(metadata.get("final_url") or source_url) if metadata else source_url
            expected_length: int | None = None
            append_from = 0
            already_complete = False
            try:
                check_canceled()
                response, final_url = self._open_final(
                    source_url,
                    headers=request_headers,
                    accepted_statuses=frozenset({200, 206, 416}),
                )
                status = _status(response)
                raw_length = _header(response.headers, "Content-Length")
                content_length: int | None = None
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError as exc:
                        self._discard_partial(partial_path)
                        raise ModelWeightError(
                            "Weight server returned an invalid Content-Length",
                            details={"content_length": raw_length},
                        ) from exc
                    if content_length < 0:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight server returned a negative Content-Length")

                if status == 416:
                    content_range = _content_range(_header(response.headers, "Content-Range"))
                    if resume_size <= 0 or content_range is None or content_range[:2] != (None, None) or content_range[2] != resume_size:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight server rejected the resume range")
                    expected_length = resume_size
                    append_from = resume_size
                    already_complete = True
                elif status == 206:
                    content_range = _content_range(_header(response.headers, "Content-Range"))
                    if resume_size <= 0 or content_range is None or content_range[0] != resume_size:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight server returned an invalid Content-Range")
                    start, end, expected_length = content_range
                    assert start is not None and end is not None
                    if end < start or expected_length <= end:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight server returned an invalid Content-Range")
                    if content_length is not None and content_length != end - start + 1:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight server Content-Length does not match Content-Range")
                    previous_validator = str(metadata.get("etag") or metadata.get("last_modified") or "") if metadata else ""
                    current_validator = _header(response.headers, "ETag") or _header(response.headers, "Last-Modified") or ""
                    if previous_validator and current_validator and previous_validator != current_validator:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight resource changed while resuming")
                    append_from = resume_size
                else:
                    # A 200 response to a Range request means the server cannot
                    # resume this resource. Reuse the full response safely.
                    if resume_size > 0:
                        self._discard_partial(partial_path)
                    append_from = 0
                    expected_length = content_length

                if expected_length is not None:
                    if expected_length <= 0:
                        self._discard_partial(partial_path)
                        raise ModelWeightError("Weight server returned an empty model weight")
                    if expected_length > self.max_bytes:
                        self._discard_partial(partial_path)
                        raise ModelWeightError(
                            "Model weight exceeds the configured size limit",
                            details={"content_length": expected_length, "maximum": self.max_bytes},
                        )

                etag = _header(response.headers, "ETag") or (str(metadata.get("etag")) if metadata and metadata.get("etag") else None)
                last_modified = _header(response.headers, "Last-Modified") or (str(metadata.get("last_modified")) if metadata and metadata.get("last_modified") else None)
                if append_from == 0 and not partial_path.exists():
                    partial_path.touch()
                self._write_partial_metadata(partial_path, {
                    "version": 1,
                    "source_url": source_url,
                    "final_url": final_url,
                    "etag": etag,
                    "last_modified": last_modified,
                    "total_bytes": expected_length,
                })

                digest = sha256()
                size = 0
                if append_from > 0:
                    with partial_path.open("rb") as existing:
                        while chunk := existing.read(1024 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                if progress is not None:
                    progress(DownloadProgress(
                        model_id=model_id,
                        url_index=url_index,
                        source_url=source_url,
                        final_url=final_url,
                        received_bytes=size,
                        total_bytes=expected_length,
                    ))

                reported_bytes = size
                report_lock = Lock()

                def report_received(received_bytes: int) -> None:
                    nonlocal reported_bytes
                    if progress is None:
                        return
                    with report_lock:
                        if received_bytes <= reported_bytes:
                            return
                        reported_bytes = received_bytes
                        progress(DownloadProgress(
                            model_id=model_id,
                            url_index=url_index,
                            source_url=source_url,
                            final_url=final_url,
                            received_bytes=received_bytes,
                            total_bytes=expected_length,
                        ))

                parallel_segments: list[Path] | None = None
                accepts_ranges = (_header(response.headers, "Accept-Ranges") or "").casefold() == "bytes"
                if (
                    not already_complete
                    and expected_length is not None
                    and (status == 206 or accepts_ranges)
                ):
                    parallel_segments = self._download_parallel_segments(
                        source_url,
                        partial_path,
                        start_offset=append_from,
                        total_bytes=expected_length,
                        cancel=cancel,
                        progress=report_received,
                    )

                if parallel_segments is not None:
                    try:
                        with partial_path.open("ab") as handle:
                            for segment_path in parallel_segments:
                                check_canceled()
                                with segment_path.open("rb") as segment:
                                    while chunk := segment.read(1024 * 1024):
                                        handle.write(chunk)
                                        digest.update(chunk)
                                        size += len(chunk)
                                        report_received(size)
                                segment_path.unlink(missing_ok=True)
                            handle.flush()
                            os.fsync(handle.fileno())
                    finally:
                        for segment_path in parallel_segments:
                            segment_path.unlink(missing_ok=True)
                        with self._lock:
                            self._active_partials.difference_update(parallel_segments)
                    response.close()
                    response = None
                elif not already_complete:
                    mode = "ab" if append_from > 0 else "wb"
                    with partial_path.open(mode) as handle:
                        while True:
                            check_canceled()
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > self.max_bytes:
                                self._discard_partial(partial_path)
                                raise ModelWeightError(
                                    "Model weight exceeded the configured size limit while streaming",
                                    details={"received_bytes": size, "maximum": self.max_bytes},
                                )
                            handle.write(chunk)
                            digest.update(chunk)
                            report_received(size)
                        handle.flush()
                        os.fsync(handle.fileno())

                if expected_length is not None and size != expected_length:
                    raise ModelWeightError(
                        "Model weight size did not match Content-Length",
                        details={"expected_bytes": expected_length, "received_bytes": size, "resumable": True},
                    )
                if size == 0:
                    self._discard_partial(partial_path)
                    raise ModelWeightError("Weight server returned an empty model weight")
                checksum = digest.hexdigest()
                if expected_sha256 is not None and checksum != expected_sha256:
                    self._discard_partial(partial_path)
                    raise ModelWeightError(
                        "Model weight SHA-256 does not match the expected value",
                        details={"expected": expected_sha256, "actual": checksum},
                    )
                check_canceled()
                os.replace(partial_path, final_path)
                self._partial_metadata_path(partial_path).unlink(missing_ok=True)
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

                with self._lock:
                    manifest = self._read_manifest(model_id)
                    manifest["files"][source_url] = {
                        "path": final_path.name,
                        "final_url": final_url,
                        "size_bytes": size,
                        "sha256": checksum,
                    }
                    self._write_manifest(model_id, manifest)
                return DownloadedWeight(
                    model_id=model_id,
                    url_index=url_index,
                    source_url=source_url,
                    final_url=final_url,
                    local_path=final_path,
                    size_bytes=size,
                    sha256=checksum,
                    cache_hit=False,
                )
            except ModelWeightError:
                if partial_path.exists() and partial_path.stat().st_size == 0:
                    self._discard_partial(partial_path)
                raise
            except Exception as exc:
                if partial_path.exists() and partial_path.stat().st_size == 0:
                    self._discard_partial(partial_path)
                raise ModelWeightError(
                    "Model weight download failed",
                    details={"url": source_url, "error": str(exc), "resumable": partial_path.exists()},
                ) from exc
            finally:
                if response is not None:
                    response.close()
                with self._lock:
                    self._active_partials.discard(partial_path)
