from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from threading import RLock
from typing import Iterable
from uuid import uuid4

from PIL import Image

from labelone.errors import InvalidPathError

from .models import (
    DerivedDatasetPublishResult,
    DerivedOutput,
    PipelineDerivedItemResult,
    PipelineOutputPolicy,
)


@dataclass(frozen=True, slots=True)
class PreparedDerivedOutput:
    image_relative_path: str
    annotation_relative_path: str
    image: Image.Image
    document: dict
    tile: dict[str, int] | None = None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.parent / f".{path.name}.{uuid4().hex}.part"
    try:
        with partial.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        partial.unlink(missing_ok=True)


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise InvalidPathError("Derived dataset relative path is unsafe", details={"path": value})
    return Path(*pure.parts)


def _safe_join(root: Path, relative: str) -> Path:
    path = (root / _safe_relative(relative)).resolve(strict=False)
    if root != path and root not in path.parents:
        raise InvalidPathError("Derived output path escapes its staging root", details={"path": relative})
    current = root
    for part in path.relative_to(root).parts[:-1]:
        current /= part
        if current.exists() and current.is_symlink():
            raise InvalidPathError("Derived output refuses to follow a symlink", details={"path": str(current)})
    return path


def _reject_existing_symlinks(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise InvalidPathError("Derived dataset output path contains a symlink", details={"path": str(current)})


def validate_output_root(output_root: Path, source_root: Path) -> Path:
    if not output_root.is_absolute():
        raise InvalidPathError("Derived dataset output_root must be absolute")
    _reject_existing_symlinks(output_root)
    resolved = output_root.resolve(strict=False)
    source = source_root.resolve()
    if resolved == source or resolved in source.parents or source in resolved.parents:
        raise InvalidPathError(
            "Derived dataset output_root must not overlap the source dataset",
            details={"output_root": str(resolved), "source_root": str(source)},
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlinks(resolved.parent)
    return resolved


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[index][1] * points[(index + 1) % len(points)][0]
        for index in range(len(points))
    )) * 0.5


def _clip_polygon(points: list[list[float]], left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    output = [[float(point[0]), float(point[1])] for point in points]
    boundaries = (
        (lambda point: point[0] >= left, lambda a, b: [left, a[1] + (b[1] - a[1]) * (left - a[0]) / (b[0] - a[0])]),
        (lambda point: point[0] <= right, lambda a, b: [right, a[1] + (b[1] - a[1]) * (right - a[0]) / (b[0] - a[0])]),
        (lambda point: point[1] >= top, lambda a, b: [a[0] + (b[0] - a[0]) * (top - a[1]) / (b[1] - a[1]), top]),
        (lambda point: point[1] <= bottom, lambda a, b: [a[0] + (b[0] - a[0]) * (bottom - a[1]) / (b[1] - a[1]), bottom]),
    )
    for inside, intersection in boundaries:
        source = output
        output = []
        if not source:
            break
        previous = source[-1]
        previous_inside = inside(previous)
        for current in source:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous, previous_inside = current, current_inside
    return output


def _clip_segment(
    first: list[float], second: list[float], left: float, top: float, right: float, bottom: float
) -> list[list[float]]:
    x1, y1 = map(float, first)
    x2, y2 = map(float, second)
    dx, dy = x2 - x1, y2 - y1
    low, high = 0.0, 1.0
    for p, q in ((-dx, x1 - left), (dx, right - x1), (-dy, y1 - top), (dy, bottom - y1)):
        if p == 0:
            if q < 0:
                return []
            continue
        ratio = q / p
        if p < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return []
    return [[x1 + low * dx, y1 + low * dy], [x1 + high * dx, y1 + high * dy]]


def _same_point(first: list[float], second: list[float]) -> bool:
    return abs(first[0] - second[0]) <= 1e-9 and abs(first[1] - second[1]) <= 1e-9


def _clip_linestrip(
    points: list[list[float]], left: float, top: float, right: float, bottom: float
) -> list[list[list[float]]]:
    runs: list[list[list[float]]] = []
    current: list[list[float]] = []
    for first, second in zip(points, points[1:], strict=False):
        segment = _clip_segment(first, second, left, top, right, bottom)
        if len(segment) != 2 or _same_point(segment[0], segment[1]):
            if len(current) >= 2:
                runs.append(current)
            current = []
            continue
        start, end = segment
        if current and _same_point(current[-1], start):
            if not _same_point(current[-1], end):
                current.append(end)
            continue
        if len(current) >= 2:
            runs.append(current)
        current = [start, end]
    if len(current) >= 2:
        runs.append(current)
    return runs


def clip_annotation_document(document: dict, *, x: int, y: int, width: int, height: int, image_name: str) -> dict:
    result = deepcopy(document)
    clipped_shapes: list[dict] = []
    right, bottom = x + width, y + height
    for raw_shape in result.get("shapes", []):
        if not isinstance(raw_shape, dict) or not isinstance(raw_shape.get("points"), list):
            continue
        shape = deepcopy(raw_shape)
        points = shape["points"]
        if not points or not all(isinstance(point, list) and len(point) == 2 for point in points):
            continue
        shape_type = str(shape.get("shape_type") or "polygon")
        all_inside = all(x <= float(point[0]) <= right and y <= float(point[1]) <= bottom for point in points)
        if shape_type == "linestrip" and len(points) >= 2:
            for run in _clip_linestrip(points, x, y, right, bottom):
                clipped_shape = deepcopy(shape)
                clipped_shape["points"] = [[point[0] - x, point[1] - y] for point in run]
                clipped_shapes.append(clipped_shape)
            continue
        if shape_type == "point" or len(points) == 1:
            clipped = [[float(points[0][0]), float(points[0][1])]] if all_inside else []
        elif shape_type == "line" and len(points) == 2:
            clipped = _clip_segment(points[0], points[1], x, y, right, bottom)
        elif len(points) >= 3:
            clipped = _clip_polygon(points, x, y, right, bottom)
            if len(clipped) >= 3 and _polygon_area(clipped) <= 1e-6:
                clipped = []
            if clipped and not all_inside and shape_type in {"rectangle", "rotation"}:
                shape["shape_type"] = "polygon"
                shape.pop("direction", None)
        else:
            clipped = [[float(point[0]), float(point[1])] for point in points] if all_inside else []
        if not clipped:
            continue
        translated = [[point[0] - x, point[1] - y] for point in clipped]
        if len(translated) == 2 and translated[0] == translated[1]:
            continue
        shape["points"] = translated
        clipped_shapes.append(shape)
    result["shapes"] = clipped_shapes
    result["imagePath"] = image_name
    result["imageWidth"] = width
    result["imageHeight"] = height
    result["imageData"] = None
    return result


def tile_windows(
    width: int,
    height: int,
    *,
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
    include_partial: bool,
) -> list[tuple[int, int, int, int, int, int]]:
    def starts(length: int, size: int, overlap: int) -> list[int]:
        if length < size:
            return [0] if include_partial else []
        stride = size - overlap
        values: list[int] = []
        position = 0
        while True:
            if position + size > length and not include_partial:
                break
            values.append(position)
            if position + size >= length:
                break
            position += stride
        return values

    xs = starts(width, tile_width, overlap_x)
    ys = starts(height, tile_height, overlap_y)
    return [
        (column, row, x, y, min(tile_width, width - x), min(tile_height, height - y))
        for row, y in enumerate(ys)
        for column, x in enumerate(xs)
    ]


class DerivedDatasetWriter:
    def __init__(self) -> None:
        self._lock = RLock()

    @staticmethod
    def staging_root(output_root: Path, job_id: str) -> Path:
        suffix = sha256(job_id.encode("utf-8")).hexdigest()[:16]
        return output_root.parent / f".{output_root.name}.labelone-{suffix}.part"

    @staticmethod
    def _manifest_relative(asset_id: str) -> str:
        return f".labelone-items/{sha256(asset_id.encode('utf-8')).hexdigest()}.json"

    def write_item(
        self,
        *,
        job_id: str,
        dataset_id: str,
        asset_id: str,
        source_root: Path,
        policy: PipelineOutputPolicy,
        item_fingerprint: str,
        outputs: list[PreparedDerivedOutput],
    ) -> PipelineDerivedItemResult:
        if policy.mode != "derived_dataset" or policy.output_root is None:
            raise InvalidPathError("Derived writer requires a derived_dataset output policy")
        output_root = validate_output_root(policy.output_root, source_root)
        staging = self.staging_root(output_root, job_id)
        if staging.exists() and staging.is_symlink():
            raise InvalidPathError("Derived staging path must not be a symlink", details={"path": str(staging)})
        manifest_relative = self._manifest_relative(asset_id)
        with self._lock:
            staging.mkdir(parents=True, exist_ok=True)
            manifest_path = _safe_join(staging, manifest_relative)
            if manifest_path.is_file():
                try:
                    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    existing = None
                if isinstance(existing, dict) and existing.get("item_fingerprint") == item_fingerprint:
                    relative_outputs = existing.get("outputs")
                    if isinstance(relative_outputs, list) and all(
                        isinstance(item, dict)
                        and _safe_join(staging, str(item.get("image_relative_path"))).is_file()
                        and _safe_join(staging, str(item.get("annotation_relative_path"))).is_file()
                        for item in relative_outputs
                    ):
                        return PipelineDerivedItemResult(
                            dataset_id=dataset_id,
                            asset_id=asset_id,
                            output_root=output_root,
                            item_fingerprint=item_fingerprint,
                            outputs=[DerivedOutput.model_validate(item) for item in relative_outputs],
                            cache_hit=True,
                        )
                if isinstance(existing, dict):
                    for item in existing.get("outputs", []):
                        if isinstance(item, dict):
                            for key in ("image_relative_path", "annotation_relative_path"):
                                value = item.get(key)
                                if isinstance(value, str):
                                    _safe_join(staging, value).unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)

            written: list[Path] = []
            rendered: list[DerivedOutput] = []
            try:
                for prepared in outputs:
                    image_path = _safe_join(staging, prepared.image_relative_path)
                    annotation_path = _safe_join(staging, prepared.annotation_relative_path)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    annotation_path.parent.mkdir(parents=True, exist_ok=True)
                    image_buffer = BytesIO()
                    encoder = "JPEG" if policy.image_format == "jpeg" else policy.image_format.upper()
                    image = prepared.image
                    if encoder == "JPEG" and image.mode not in {"RGB", "L"}:
                        image = image.convert("RGB")
                    options = {"quality": 90} if encoder == "JPEG" else {"quality": 88} if encoder == "WEBP" else {}
                    image.save(image_buffer, encoder, **options)
                    _atomic_bytes(image_path, image_buffer.getvalue())
                    written.append(image_path)
                    encoded_document = json.dumps(prepared.document, ensure_ascii=False, indent=2).encode("utf-8")
                    _atomic_bytes(annotation_path, encoded_document)
                    written.append(annotation_path)
                    rendered.append(DerivedOutput(
                        image_relative_path=prepared.image_relative_path,
                        annotation_relative_path=prepared.annotation_relative_path,
                        width=prepared.image.width,
                        height=prepared.image.height,
                        tile=prepared.tile,
                        annotation_count=len(prepared.document.get("shapes", [])),
                    ))
                payload = {
                    "asset_id": asset_id,
                    "item_fingerprint": item_fingerprint,
                    "outputs": [item.model_dump(mode="json") for item in rendered],
                }
                _atomic_bytes(manifest_path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            except Exception:
                for path in written:
                    path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                for partial in staging.rglob("*.part"):
                    partial.unlink(missing_ok=True)
                if not any(path.is_file() for path in staging.rglob("*")):
                    shutil.rmtree(staging)
                raise
        return PipelineDerivedItemResult(
            dataset_id=dataset_id,
            asset_id=asset_id,
            output_root=output_root,
            item_fingerprint=item_fingerprint,
            outputs=rendered,
        )

    def _validated_paths(self, staging: Path, item: PipelineDerivedItemResult) -> list[Path]:
        item_manifest = _safe_join(staging, self._manifest_relative(item.asset_id))
        if not item_manifest.is_file():
            raise InvalidPathError("Derived item manifest is missing", details={"asset_id": item.asset_id})
        try:
            persisted_item = json.loads(item_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidPathError(
                "Derived item manifest is unreadable", details={"asset_id": item.asset_id}
            ) from exc
        if not isinstance(persisted_item, dict) or persisted_item.get("item_fingerprint") != item.item_fingerprint:
            raise InvalidPathError("Derived item fingerprint does not match staging", details={"asset_id": item.asset_id})
        paths = [item_manifest]
        for output in item.outputs:
            image_path = _safe_join(staging, output.image_relative_path)
            annotation_path = _safe_join(staging, output.annotation_relative_path)
            if not image_path.is_file() or not annotation_path.is_file():
                raise InvalidPathError("Derived item output is incomplete", details={"asset_id": item.asset_id})
            paths.extend((image_path, annotation_path))
        return paths

    def finalize(
        self,
        *,
        job_id: str,
        dataset_id: str,
        source_root: Path,
        policy: PipelineOutputPolicy,
        items: Iterable[PipelineDerivedItemResult],
        expected_item_count: int,
    ) -> DerivedDatasetPublishResult:
        if policy.output_root is None:
            raise InvalidPathError("Derived dataset output_root is missing")
        output_root = validate_output_root(policy.output_root, source_root)
        staging = self.staging_root(output_root, job_id)
        with self._lock:
            if not staging.is_dir() or staging.is_symlink():
                raise InvalidPathError("Derived dataset staging directory is missing", details={"path": str(staging)})
            final_manifest = staging / ".labelone-derived.json"
            manifest_partial = staging / ".labelone-derived.json.part"
            references_path = staging / ".labelone-finalize-refs.sqlite3"
            manifest_partial.unlink(missing_ok=True)
            references_path.unlink(missing_ok=True)
            references = sqlite3.connect(references_path)
            references.execute("PRAGMA journal_mode=OFF")
            references.execute("PRAGMA synchronous=OFF")
            references.execute("CREATE TABLE refs(path TEXT PRIMARY KEY)")
            fingerprint = sha256()
            fingerprint.update(b'{"dataset_id":')
            fingerprint.update(json.dumps(dataset_id, separators=(",", ":")).encode("utf-8"))
            fingerprint.update(b',"items":[')
            item_count = 0
            output_count = 0
            previous_asset_id: str | None = None
            try:
                with manifest_partial.open("xb") as handle:
                    handle.write(b'{"schema_version":1,"dataset_id":')
                    handle.write(json.dumps(dataset_id, ensure_ascii=False).encode("utf-8"))
                    handle.write(b',"items":[')
                    first = True
                    for item in items:
                        if item.dataset_id != dataset_id or item.output_root.resolve() != output_root:
                            raise InvalidPathError(
                                "Derived item result belongs to a different output",
                                details={"asset_id": item.asset_id},
                            )
                        if previous_asset_id is not None and item.asset_id <= previous_asset_id:
                            raise InvalidPathError("Derived item results must be streamed in asset_id order")
                        previous_asset_id = item.asset_id
                        validated_paths = self._validated_paths(staging, item)
                        references.executemany(
                            "INSERT INTO refs(path) VALUES(?)",
                            [(path.relative_to(staging).as_posix(),) for path in validated_paths],
                        )
                        output_count += len(item.outputs)
                        pair = json.dumps([item.asset_id, item.item_fingerprint], separators=(",", ":"))
                        if item_count:
                            fingerprint.update(b",")
                        fingerprint.update(pair.encode("utf-8"))
                        encoded_item = json.dumps(item.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
                        if not first:
                            handle.write(b",")
                        handle.write(encoded_item.encode("utf-8"))
                        first = False
                        item_count += 1
                        if item_count % 1000 == 0:
                            references.commit()
                    if item_count != expected_item_count:
                        raise InvalidPathError(
                            "Derived dataset item count does not match the job",
                            details={"expected": expected_item_count, "actual": item_count},
                        )
                    fingerprint.update(b"]}")
                    dataset_fingerprint = fingerprint.hexdigest()
                    handle.write(b'],"dataset_fingerprint":')
                    handle.write(json.dumps(dataset_fingerprint).encode("ascii"))
                    handle.write(b',"item_count":')
                    handle.write(str(item_count).encode("ascii"))
                    handle.write(b',"output_count":')
                    handle.write(str(output_count).encode("ascii"))
                    handle.write(b"}")
                    handle.flush()
                    os.fsync(handle.fileno())
                references.commit()
                os.replace(manifest_partial, final_manifest)
                _fsync_directory(staging)
                for path in staging.rglob("*"):
                    if path.is_symlink():
                        raise InvalidPathError("Derived staging contains a symlink", details={"path": str(path)})
                    if not path.is_file() or path in {final_manifest, references_path}:
                        continue
                    relative_path = path.relative_to(staging).as_posix()
                    if references.execute("SELECT 1 FROM refs WHERE path=?", (relative_path,)).fetchone() is None:
                        path.unlink()
                references.close()
                references_path.unlink(missing_ok=True)
                for path in sorted((item for item in staging.rglob("*") if item.is_dir()), reverse=True):
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            except Exception:
                references.close()
                references_path.unlink(missing_ok=True)
                manifest_partial.unlink(missing_ok=True)
                raise
            _fsync_directory(staging)
            if output_root.exists():
                if output_root.is_symlink() or not output_root.is_dir():
                    raise InvalidPathError("Derived output path already exists and is unsafe", details={"path": str(output_root)})
                existing_manifest = output_root / ".labelone-derived.json"
                try:
                    existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    existing = None
                if policy.conflict == "reuse" and isinstance(existing, dict) and existing.get("dataset_fingerprint") == dataset_fingerprint:
                    shutil.rmtree(staging)
                    return DerivedDatasetPublishResult(
                        output_root=output_root,
                        dataset_fingerprint=dataset_fingerprint,
                        item_count=item_count,
                        output_count=output_count,
                        reused=True,
                    )
                raise InvalidPathError(
                    "Derived dataset output already exists with different content",
                    details={"output_root": str(output_root), "dataset_fingerprint": dataset_fingerprint},
                )
            os.replace(staging, output_root)
            _fsync_directory(output_root.parent)
            return DerivedDatasetPublishResult(
                output_root=output_root,
                dataset_fingerprint=dataset_fingerprint,
                item_count=item_count,
                output_count=output_count,
            )

    def abort(self, *, job_id: str, source_root: Path, policy: PipelineOutputPolicy) -> None:
        if policy.output_root is None:
            return
        output_root = validate_output_root(policy.output_root, source_root)
        staging = self.staging_root(output_root, job_id)
        with self._lock:
            if staging.is_symlink():
                raise InvalidPathError("Derived staging path is a symlink", details={"path": str(staging)})
            if staging.is_dir():
                shutil.rmtree(staging)
