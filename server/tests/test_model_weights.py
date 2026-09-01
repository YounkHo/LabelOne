from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from threading import Barrier
from typing import Any
from urllib.request import Request

import pytest

from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor
from labelone.models.weights import ModelWeightError, ModelWeightStore, model_weight_source


class FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        fail_after_reads: int | None = None,
    ) -> None:
        self.url = url
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.fail_after_reads = fail_after_reads
        self.offset = 0
        self.read_count = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        if self.fail_after_reads is not None and self.read_count > self.fail_after_reads:
            raise OSError("simulated connection loss")
        if self.offset >= len(self.body):
            return b""
        end = len(self.body) if size < 0 else min(len(self.body), self.offset + size)
        chunk = self.body[self.offset:end]
        self.offset = end
        return chunk

    def close(self) -> None:
        self.closed = True

    def geturl(self) -> str:
        return self.url


class FakeOpener:
    def __init__(self, responses: dict[str, list[FakeResponse] | FakeResponse]) -> None:
        self.responses = {
            url: list(value) if isinstance(value, list) else [value]
            for url, value in responses.items()
        }
        self.calls: list[tuple[str, float, dict[str, str]]] = []

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        url = request.full_url
        self.calls.append((url, timeout, dict(request.header_items())))
        queue = self.responses.get(url)
        if not queue:
            raise AssertionError(f"Unexpected network request: {url}")
        return queue.pop(0)


class SynchronizedRangeResponse(FakeResponse):
    def __init__(self, *, barrier: Barrier, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.barrier = barrier

    def read(self, size: int = -1) -> bytes:
        if self.read_count == 0:
            self.barrier.wait(5)
        return super().read(size)


class RangeOpener:
    def __init__(self, url: str, payload: bytes, workers: int, *, reject_ranges: bool = False) -> None:
        self.url = url
        self.payload = payload
        self.barrier = Barrier(workers)
        self.reject_ranges = reject_ranges
        self.calls: list[dict[str, str]] = []
        self.initial_response: FakeResponse | None = None

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        del timeout
        assert request.full_url == self.url
        headers = {key.casefold(): value for key, value in request.header_items()}
        self.calls.append(headers)
        requested_range = headers.get("range")
        if requested_range is None:
            self.initial_response = FakeResponse(
                url=self.url,
                body=self.payload,
                headers={
                    "Content-Length": str(len(self.payload)),
                    "Accept-Ranges": "bytes",
                },
            )
            return self.initial_response
        start_text, end_text = requested_range.removeprefix("bytes=").split("-", 1)
        start, end = int(start_text), int(end_text)
        body = self.payload[start:end + 1]
        if self.reject_ranges:
            return FakeResponse(
                url=self.url,
                status=200,
                body=self.payload,
                headers={"Content-Length": str(len(self.payload))},
            )
        return SynchronizedRangeResponse(
            barrier=self.barrier,
            url=self.url,
            status=206,
            body=body,
            headers={
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
            },
        )


def _record(tmp_path: Path, locations: list[str]) -> ModelRecord:
    descriptor = ModelDescriptor(
        id="fixture",
        name="fixture",
        display_name="Fixture",
        model_type="fixture",
        task="detection",
        family="fixture",
        adapter="fixture",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "fixture.yaml",
        weight_locations=locations,
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(),
    )
    return ModelRecord(descriptor=descriptor, config={})


def _store(tmp_path: Path, opener: FakeOpener, *, max_bytes: int = 1024, **kwargs: Any) -> ModelWeightStore:
    return ModelWeightStore(tmp_path / "data", opener=opener, max_bytes=max_bytes, **kwargs)


def test_lists_only_https_locations_with_original_indices(tmp_path: Path) -> None:
    record = _record(tmp_path, [
        "weights/local.onnx",
        "http://github.com/org/insecure.onnx",
        "https://github.com/org/model.onnx",
        "https://huggingface.co/org/repo/resolve/main/model.bin?download=1",
    ])
    store = _store(tmp_path, FakeOpener({}))

    remote = store.list_remote("fixture", record)

    assert [item.url_index for item in remote] == [2, 3]
    assert [item.filename for item in remote] == ["model.onnx", "model.bin"]
    assert all(not item.downloaded for item in remote)


def test_explicit_download_streams_hashes_and_persists_manifest(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"model-weight-content"
    response = FakeResponse(url=url, body=payload, headers={"Content-Length": str(len(payload))})
    opener = FakeOpener({url: response})
    store = _store(tmp_path, opener)
    record = _record(tmp_path, [url])
    stale_directory = tmp_path / "data" / "model-weights" / "fixture"
    stale_directory.mkdir(parents=True)
    (stale_directory / ".crashed.onnx.part").write_bytes(b"partial")

    result = store.download("fixture", record, 0)

    expected_path = tmp_path / "data" / "model-weights" / "fixture" / "model.onnx"
    assert result.local_path == expected_path
    assert result.local_path.read_bytes() == payload
    assert result.sha256 == sha256(payload).hexdigest()
    assert result.size_bytes == len(payload)
    assert result.cache_hit is False
    assert response.closed is True
    assert not list(expected_path.parent.glob("*.part"))
    assert not list(expected_path.parent.glob(".*.part"))
    assert store.local_overrides("fixture") == {url: expected_path}
    reopened = _store(tmp_path, FakeOpener({}))
    assert reopened.local_overrides("fixture") == {url: expected_path}

    manifest = json.loads((expected_path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][url]["sha256"] == sha256(payload).hexdigest()
    assert manifest["files"][url]["path"] == "model.onnx"


def test_single_large_weight_uses_parallel_range_segments(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = bytes(range(64))
    opener = RangeOpener(url, payload, workers=4)
    store = ModelWeightStore(
        tmp_path / "data",
        opener=opener,
        max_bytes=1024,
        parallel_workers=4,
        parallel_threshold_bytes=1,
    )
    received: list[int] = []

    result = store.download(
        "fixture",
        _record(tmp_path, [url]),
        0,
        progress=lambda update: received.append(update.received_bytes),
    )

    assert result.local_path.read_bytes() == payload
    ranges = sorted(call["range"] for call in opener.calls if "range" in call)
    assert ranges == ["bytes=0-15", "bytes=16-31", "bytes=32-47", "bytes=48-63"]
    assert opener.initial_response is not None and opener.initial_response.closed
    assert not list(result.local_path.parent.glob("*.segment.part"))
    assert received == sorted(received)
    assert received[-1] == len(payload)


def test_parallel_range_rejection_falls_back_to_open_sequential_response(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = bytes(range(64))
    opener = RangeOpener(url, payload, workers=4, reject_ranges=True)
    store = ModelWeightStore(
        tmp_path / "data",
        opener=opener,
        max_bytes=1024,
        parallel_workers=4,
        parallel_threshold_bytes=1,
    )

    result = store.download("fixture", _record(tmp_path, [url]), 0)

    assert result.local_path.read_bytes() == payload
    assert len([call for call in opener.calls if "range" in call]) == 4
    assert opener.initial_response is not None and opener.initial_response.closed


def test_duplicate_download_is_a_verified_cache_hit_without_network(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"same-content"
    opener = FakeOpener({url: FakeResponse(url=url, body=payload, headers={"Content-Length": str(len(payload))})})
    store = _store(tmp_path, opener)
    record = _record(tmp_path, [url])

    first = store.download("fixture", record, 0)
    second = store.download("fixture", record, 0)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(opener.calls) == 1
    assert store.list_remote("fixture", record)[0].downloaded is True


def test_tampered_cached_file_is_redownloaded(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    original = b"original"
    replacement = b"replaced"
    opener = FakeOpener({
        url: [
            FakeResponse(url=url, body=original, headers={"Content-Length": str(len(original))}),
            FakeResponse(url=url, body=replacement, headers={"Content-Length": str(len(replacement))}),
        ]
    })
    store = _store(tmp_path, opener)
    record = _record(tmp_path, [url])
    first = store.download("fixture", record, 0)
    first.local_path.write_bytes(b"tampered")

    second = store.download("fixture", record, 0)

    assert second.cache_hit is False
    assert second.local_path.read_bytes() == replacement
    assert len(opener.calls) == 2


@pytest.mark.parametrize(
    "location, message",
    [
        ("http://github.com/org/model.onnx", "HTTPS"),
        ("https://evil.example/model.onnx", "allowlisted"),
        ("https://user:secret@github.com/org/model.onnx", "credentials"),
        ("https://github.com:444/org/model.onnx", "default HTTPS port"),
        ("https://github.com/org/", "safe filename"),
    ],
)
def test_unsafe_urls_are_rejected_before_network(tmp_path: Path, location: str, message: str) -> None:
    opener = FakeOpener({})
    store = _store(tmp_path, opener)
    record = _record(tmp_path, [location])

    with pytest.raises(ModelWeightError, match=message):
        store.download("fixture", record, 0)
    assert opener.calls == []


def test_invalid_model_id_and_url_index_are_rejected(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    store = _store(tmp_path, FakeOpener({}))
    record = _record(tmp_path, [url])

    with pytest.raises(ModelWeightError, match="Model ID"):
        store.download("../escape", record, 0)
    with pytest.raises(ModelWeightError, match="index"):
        store.download("fixture", record, 1)


def test_allowed_redirect_is_followed_and_final_url_is_recorded(tmp_path: Path) -> None:
    source = "https://github.com/org/releases/model.onnx"
    final = "https://objects.githubusercontent.com/release-assets/model.onnx"
    payload = b"redirected"
    first = FakeResponse(url=source, status=302, headers={"Location": final})
    second = FakeResponse(url=final, body=payload, headers={"Content-Length": str(len(payload))})
    opener = FakeOpener({source: first, final: second})
    store = _store(tmp_path, opener)

    result = store.download("fixture", _record(tmp_path, [source]), 0)

    assert result.final_url == final
    assert [call[0] for call in opener.calls] == [source, final]
    assert first.closed and second.closed


def test_redirect_target_and_implicit_final_url_are_revalidated(tmp_path: Path) -> None:
    source = "https://github.com/org/model.onnx"
    evil = "https://evil.example/model.onnx"
    redirect = FakeResponse(url=source, status=302, headers={"Location": evil})
    store = _store(tmp_path, FakeOpener({source: redirect}))

    with pytest.raises(ModelWeightError, match="allowlisted"):
        store.download("fixture", _record(tmp_path, [source]), 0)
    assert redirect.closed

    followed = FakeResponse(url=evil, body=b"bad")
    store = _store(tmp_path, FakeOpener({source: followed}))
    with pytest.raises(ModelWeightError, match="allowlisted"):
        store.download("fixture", _record(tmp_path, [source]), 0)
    assert followed.closed


def test_redirect_limit_is_enforced(tmp_path: Path) -> None:
    first = "https://github.com/org/first.onnx"
    second = "https://github.com/org/second.onnx"
    opener = FakeOpener({first: FakeResponse(url=first, status=302, headers={"Location": second})})
    store = _store(tmp_path, opener, max_redirects=0)

    with pytest.raises(ModelWeightError, match="redirect limit"):
        store.download("fixture", _record(tmp_path, [first]), 0)


@pytest.mark.parametrize(
    "headers, body, max_bytes, message",
    [
        ({"Content-Length": "100"}, b"", 10, "size limit"),
        ({"Content-Length": "0"}, b"", 100, "empty"),
        ({"Content-Length": "not-a-number"}, b"", 100, "invalid Content-Length"),
        ({"Content-Length": "10"}, b"short", 100, "did not match"),
        ({}, b"streamed-past-limit", 5, "while streaming"),
    ],
)
def test_content_length_and_streaming_limits_keep_only_resumable_partials(
    tmp_path: Path,
    headers: dict[str, str],
    body: bytes,
    max_bytes: int,
    message: str,
) -> None:
    url = "https://github.com/org/model.onnx"
    response = FakeResponse(url=url, body=body, headers=headers)
    store = _store(tmp_path, FakeOpener({url: response}), max_bytes=max_bytes)

    with pytest.raises(ModelWeightError, match=message):
        store.download("fixture", _record(tmp_path, [url]), 0)

    directory = tmp_path / "data" / "model-weights" / "fixture"
    if message == "did not match":
        assert (directory / ".model.onnx.part").read_bytes() == body
        assert (directory / ".model.onnx.part.json").is_file()
    else:
        assert not list(directory.glob("*.part"))
        assert not list(directory.glob(".*.part"))
    assert response.closed


def test_connection_failure_cleans_partial_and_does_not_write_manifest(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    response = FakeResponse(url=url, body=b"partial", fail_after_reads=0)
    store = _store(tmp_path, FakeOpener({url: response}))

    with pytest.raises(ModelWeightError, match="download failed"):
        store.download("fixture", _record(tmp_path, [url]), 0)

    directory = tmp_path / "data" / "model-weights" / "fixture"
    assert not list(directory.glob("*.part"))
    assert not list(directory.glob(".*.part"))
    assert not (directory / "manifest.json").exists()


def test_connection_failure_resumes_with_range_and_if_range(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"a" * (1024 * 1024) + b"resumed-tail"
    first = FakeResponse(
        url=url,
        body=payload,
        headers={"Content-Length": str(len(payload)), "ETag": '"fixture-v1"'},
        fail_after_reads=1,
    )
    remaining = payload[1024 * 1024:]
    second = FakeResponse(
        url=url,
        status=206,
        body=remaining,
        headers={
            "Content-Length": str(len(remaining)),
            "Content-Range": f"bytes {1024 * 1024}-{len(payload) - 1}/{len(payload)}",
            "ETag": '"fixture-v1"',
        },
    )
    opener = FakeOpener({url: [first, second]})
    store = _store(tmp_path, opener, max_bytes=2 * 1024 * 1024)
    record = _record(tmp_path, [url])

    with pytest.raises(ModelWeightError, match="download failed"):
        store.download("fixture", record, 0)
    partial = tmp_path / "data" / "model-weights" / "fixture" / ".model.onnx.part"
    assert partial.stat().st_size == 1024 * 1024

    result = store.download("fixture", record, 0)

    assert result.local_path.read_bytes() == payload
    resumed_headers = {key.casefold(): value for key, value in opener.calls[1][2].items()}
    assert resumed_headers["range"] == f"bytes={1024 * 1024}-"
    assert resumed_headers["if-range"] == '"fixture-v1"'
    assert not partial.exists()
    assert not partial.with_name(f"{partial.name}.json").exists()


def test_server_ignoring_range_restarts_from_full_response(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"a" * (1024 * 1024) + b"tail"
    first = FakeResponse(
        url=url,
        body=payload,
        headers={"Content-Length": str(len(payload)), "ETag": '"fixture-v1"'},
        fail_after_reads=1,
    )
    full_retry = FakeResponse(
        url=url,
        status=200,
        body=payload,
        headers={"Content-Length": str(len(payload)), "ETag": '"fixture-v2"'},
    )
    opener = FakeOpener({url: [first, full_retry]})
    store = _store(tmp_path, opener, max_bytes=2 * 1024 * 1024)
    record = _record(tmp_path, [url])

    with pytest.raises(ModelWeightError):
        store.download("fixture", record, 0)
    result = store.download("fixture", record, 0)

    assert result.local_path.read_bytes() == payload
    assert len(opener.calls) == 2


def test_stale_part_cleanup_is_explicit_and_runs_before_download(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "model-weights" / "fixture"
    directory.mkdir(parents=True)
    (directory / ".old.onnx.part").write_bytes(b"stale")
    (directory / "keep.onnx").write_bytes(b"keep")
    store = _store(tmp_path, FakeOpener({}))

    assert store.cleanup_stale_parts("fixture") == 1
    assert not (directory / ".old.onnx.part").exists()
    assert (directory / "keep.onnx").exists()


def test_manifest_path_escape_is_ignored_by_local_overrides(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "model-weights" / "fixture"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    url = "https://github.com/org/model.onnx"
    (directory / "manifest.json").write_text(json.dumps({
        "version": 1,
        "model_id": "fixture",
        "files": {url: {"path": "../../../outside.onnx", "size_bytes": len(b"outside"), "sha256": sha256(b"outside").hexdigest()}},
    }), encoding="utf-8")

    assert _store(tmp_path, FakeOpener({})).local_overrides("fixture") == {}


def test_custom_allowlist_can_enable_an_explicit_domain(tmp_path: Path) -> None:
    url = "https://models.example.com/model.onnx"
    payload = b"custom"
    opener = FakeOpener({url: FakeResponse(url=url, body=payload)})
    store = _store(tmp_path, opener, allowed_domains=("example.com",))

    result = store.download("fixture", _record(tmp_path, [url]), 0)

    assert result.local_path.read_bytes() == payload


def test_modelscope_urls_are_allowlisted_and_preferred_source_sorts_real_locations(tmp_path: Path) -> None:
    github = "https://github.com/org/model.onnx"
    modelscope = "https://modelscope.cn/models/org/model/resolve/master/model.onnx"
    huggingface = "https://huggingface.co/org/model/resolve/main/model.onnx"
    store = _store(tmp_path, FakeOpener({}))

    weights = store.list_remote("fixture", _record(tmp_path, [github, modelscope, huggingface]), "modelscope")

    assert [item.source_id for item in weights] == ["modelscope", "github", "huggingface"]
    assert [item.preferred for item in weights] == [True, False, False]
    assert model_weight_source("https://www.modelscope.ai/models/org/model") == "modelscope"
    assert model_weight_source("https://raw.githubusercontent.com/org/repo/model.onnx") == "github"


def test_effective_record_rewrites_downloaded_urls_without_mutating_catalog_record(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"model"
    opener = FakeOpener({url: FakeResponse(url=url, body=payload, headers={"Content-Length": str(len(payload))})})
    store = _store(tmp_path, opener)
    record = _record(tmp_path, [url])
    record.config["model_path"] = url
    store.download("fixture", record, 0)

    effective = store.effective_record(record)

    assert record.descriptor.weight_locations == [url]
    assert record.config["model_path"] == url
    assert Path(effective.descriptor.weight_locations[0]).is_file()
    assert effective.config["model_path"] == effective.descriptor.weight_locations[0]
    assert effective.descriptor.availability.state is AvailabilityState.AVAILABLE
