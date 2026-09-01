from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, UnidentifiedImageError

from labelone.errors import InvalidPathError

from .models import AssetStatus, DatasetAsset, DatasetScanRequest, DatasetScanResult, DatasetScanSummary


class DatasetScanInterrupted(RuntimeError):
    """Raised when a cooperative scan interrupt is observed."""


class _CancellationProbe:
    def __init__(self, callback: Callable[[], bool] | None, *, interval: int = 128) -> None:
        self.callback = callback
        self.interval = interval
        self.operations = 0

    def check(self, *, force: bool = False) -> None:
        if self.callback is None:
            return
        self.operations += 1
        if (force or self.operations % self.interval == 0) and self.callback():
            raise DatasetScanInterrupted("Dataset scan was interrupted")


@dataclass(frozen=True, slots=True)
class _Roots:
    root: Path
    image_root: Path
    annotation_roots: tuple[Path, ...]
    annotation_output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetScanMetadata:
    dataset_id: str
    root_dir: Path
    image_root: Path
    annotation_roots: tuple[Path, ...]


def _resolve_under(root: Path, candidate: Path | None) -> Path | None:
    if candidate is None:
        return None
    return (candidate if candidate.is_absolute() else root / candidate).expanduser().resolve()


def _discover_roots(request: DatasetScanRequest) -> _Roots:
    root = request.root_dir.expanduser().resolve()
    if not root.is_dir():
        raise InvalidPathError("Dataset root is not a readable directory", details={"root_dir": str(root)})

    explicit_image = _resolve_under(root, request.image_dir)
    explicit_annotation = _resolve_under(root, request.annotation_dir)
    if request.annotation_storage_root is not None:
        storage_root = request.annotation_storage_root.expanduser().resolve()
        if not storage_root.is_dir():
            raise InvalidPathError(
                "Annotation storage root is not a readable directory",
                details={"annotation_storage_root": str(storage_root)},
            )
        image_root = explicit_image or root
        annotation_output_root = storage_root
        annotation_roots = (annotation_output_root,)
    elif request.layout == "custom":
        if explicit_image is None or explicit_annotation is None:
            raise InvalidPathError("Custom layout requires image_dir and annotation_dir")
        image_root = explicit_image
        annotation_roots = (explicit_annotation,)
        annotation_output_root = None
    elif request.layout == "same_directory":
        image_root = explicit_image or root
        annotation_roots = (image_root,)
        annotation_output_root = None
    else:
        image_root = explicit_image or (root / "images" if (root / "images").is_dir() else root)
        candidates = [explicit_annotation] if explicit_annotation else [root / "annotations", root / "labels"]
        existing = [item for item in candidates if item is not None and item.is_dir()]
        if request.layout == "parallel" and not existing:
            raise InvalidPathError("Parallel layout requires an annotations/ or labels/ directory")
        if request.layout == "parallel":
            annotation_roots = tuple(dict.fromkeys(existing))
        else:
            # Auto mode accepts mixed datasets: parallel annotations/labels
            # plus JSON sidecars next to images in the same recursive scan.
            annotation_roots = tuple(dict.fromkeys([*existing, image_root]))
        annotation_output_root = None

    if not image_root.is_dir():
        raise InvalidPathError("Image directory is not readable", details={"image_dir": str(image_root)})
    for annotation_root in annotation_roots:
        if not annotation_root.is_dir():
            raise InvalidPathError("Annotation directory is not readable", details={"annotation_dir": str(annotation_root)})
    return _Roots(
        root=root,
        image_root=image_root,
        annotation_roots=annotation_roots,
        annotation_output_root=annotation_output_root,
    )


def resolve_scan_metadata(request: DatasetScanRequest) -> DatasetScanMetadata:
    """Resolve cheap dataset identity and roots before the expensive recursive scan."""
    roots = _discover_roots(request)
    dataset_id = request.dataset_id or sha1(
        roots.root.as_posix().encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]
    return DatasetScanMetadata(
        dataset_id=dataset_id,
        root_dir=roots.root,
        image_root=roots.image_root,
        annotation_roots=roots.annotation_roots,
    )


def _iter_files(root: Path, recursive: bool, probe: _CancellationProbe | None = None) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    for item in iterator:
        if probe is not None:
            probe.check()
        if item.is_file() and ".labelone" not in item.parts:
            yield item


def _match_key(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return relative.with_suffix("").as_posix()


def _asset_id(dataset_id: str, match_key: str) -> str:
    return sha1(f"{dataset_id}|{match_key}".encode("utf-8"), usedforsecurity=False).hexdigest()[:20]


def _relative_display(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_annotation(path: Path) -> tuple[int | None, str | None, str | None, list[str], list[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, None, f"JSON decode failed: {exc}", [], []
    if not isinstance(payload, dict):
        return None, None, "Annotation root must be an object", [], []
    shapes = payload.get("shapes", [])
    if not isinstance(shapes, list):
        return None, None, "Annotation field 'shapes' must be a list", [], []
    image_path = payload.get("imagePath")
    labels = sorted({str(shape.get("label")) for shape in shapes if isinstance(shape, dict) and shape.get("label")})
    shape_types = sorted({str(shape.get("shape_type")) for shape in shapes if isinstance(shape, dict) and shape.get("shape_type")})
    return len(shapes), image_path if isinstance(image_path, str) else None, None, labels, shape_types


def _validate_image(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return width, height, None
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        return None, None, f"Image decode failed: {exc}"


def _relative_candidate(image: Path, image_root: Path, annotation_root: Path, suffix: str) -> Path:
    try:
        relative = image.relative_to(image_root)
    except ValueError:
        relative = Path(image.name)
    return (annotation_root / relative).with_suffix(suffix)


def _build_candidates(
    image: Path,
    roots: _Roots,
    annotation_set: set[Path],
    basename_index: dict[str, list[Path]],
    image_path_index: dict[str, list[Path]],
    request: DatasetScanRequest,
) -> list[Path]:
    if roots.annotation_output_root is not None:
        # Image and annotation trees are independent. Managed annotation roots
        # contain JSON only, so pair recursively discovered files by stem.
        return list(basename_index.get(image.stem.casefold(), []))
    if request.match_strategy == "same_directory":
        candidate = image.with_suffix(request.annotation_extension)
        return [candidate] if candidate in annotation_set else []
    if request.match_strategy == "basename":
        return list(basename_index.get(image.stem.casefold(), []))
    if request.match_strategy == "image_path":
        keys = {image.name.casefold(), _relative_display(roots.image_root, image).casefold()}
        return sorted({path for key in keys for path in image_path_index.get(key, [])})

    candidates = {image.with_suffix(request.annotation_extension)}
    candidates.update(
        _relative_candidate(image, roots.image_root, annotation_root, request.annotation_extension)
        for annotation_root in roots.annotation_roots
    )
    return sorted(candidate for candidate in candidates if candidate in annotation_set)


def scan_dataset(
    request: DatasetScanRequest,
    *,
    item_sink: Callable[[DatasetAsset], None] | None = None,
    collect_items: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> DatasetScanResult:
    probe = _CancellationProbe(cancel_check)
    probe.check(force=True)
    roots = _discover_roots(request)
    probe.check(force=True)
    dataset_id = request.dataset_id or sha1(roots.root.as_posix().encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    image_paths = sorted(
        path.resolve()
        for path in _iter_files(roots.image_root, request.recursive, probe)
        if path.suffix.lower() in request.image_extensions
    )
    probe.check(force=True)
    annotation_paths = sorted({
        path.resolve()
        for annotation_root in roots.annotation_roots
        for path in _iter_files(annotation_root, request.recursive, probe)
        if path.suffix.lower() == request.annotation_extension
    })
    probe.check(force=True)
    annotation_set = set(annotation_paths)
    basename_index: dict[str, list[Path]] = defaultdict(list)
    image_path_index: dict[str, list[Path]] = defaultdict(list)
    annotation_metadata: dict[Path, tuple[int | None, str | None, str | None, list[str], list[str]]] = {}
    for annotation in annotation_paths:
        probe.check()
        basename_index[annotation.stem.casefold()].append(annotation)
        metadata = _validate_annotation(annotation) if (request.validate_annotations or request.match_strategy == "image_path") else (None, None, None, [], [])
        annotation_metadata[annotation] = metadata
        _, image_path, _, _, _ = metadata
        if image_path:
            image_path_index[Path(image_path).name.casefold()].append(annotation)
            image_path_index[Path(image_path).as_posix().casefold()].append(annotation)

    candidates_by_image: dict[Path, list[Path]] = {}
    image_stem_counts = Counter(image.stem.casefold() for image in image_paths)
    for image in image_paths:
        probe.check()
        candidates_by_image[image] = _build_candidates(
            image, roots, annotation_set, basename_index, image_path_index, request
        )
    probe.check(force=True)
    images_by_annotation: dict[Path, list[Path]] = defaultdict(list)
    for image, candidates in candidates_by_image.items():
        probe.check()
        match_key = _match_key(roots.image_root, image)
        for annotation in candidates:
            images_by_annotation[annotation].append(image)

    items: list[DatasetAsset] = []

    def emit(item: DatasetAsset) -> None:
        if collect_items:
            items.append(item)
        if item_sink is not None:
            item_sink(item)

    consumed_annotations: set[Path] = set()
    summary = DatasetScanSummary()
    for image, candidates in candidates_by_image.items():
        probe.check()
        match_key = _match_key(roots.image_root, image)
        if not candidates:
            if roots.annotation_output_root is None:
                summary.hidden_image_only += 1
                continue
            if image_stem_counts[image.stem.casefold()] > 1:
                summary.duplicate_match += 1
                emit(DatasetAsset(
                    asset_id=_asset_id(dataset_id, match_key), match_key=match_key,
                    display_path=_relative_display(roots.root, image), image_path=image,
                    status=AssetStatus.DUPLICATE_MATCH, selectable=False,
                    reason="Multiple images have the same filename stem",
                    issues=["ambiguous_image_stem"],
                ))
                continue
            annotation = _relative_candidate(
                image,
                roots.image_root,
                roots.annotation_output_root,
                request.annotation_extension,
            )
            width = height = None
            if request.validate_images:
                width, height, image_error = _validate_image(image)
                if image_error:
                    summary.corrupt_image += 1
                    emit(DatasetAsset(
                        asset_id=_asset_id(dataset_id, match_key), match_key=match_key,
                        display_path=_relative_display(roots.root, image), image_path=image,
                        annotation_paths=[annotation], status=AssetStatus.CORRUPT_IMAGE,
                        selectable=False, reason=image_error, issues=["image_corrupt"],
                    ))
                    continue
            summary.valid += 1
            emit(DatasetAsset(
                asset_id=_asset_id(dataset_id, match_key), match_key=match_key,
                display_path=_relative_display(roots.root, image), image_path=image,
                annotation_paths=[annotation], status=AssetStatus.VALID, selectable=True,
                width=width, height=height, annotation_count=0,
            ))
            continue
        consumed_annotations.update(candidates)
        duplicated = len(candidates) > 1 or any(len(images_by_annotation[path]) > 1 for path in candidates)
        if duplicated:
            summary.duplicate_match += 1
            emit(DatasetAsset(
                asset_id=_asset_id(dataset_id, match_key),
                match_key=match_key,
                display_path=_relative_display(roots.root, image),
                image_path=image,
                annotation_paths=candidates,
                annotation_file_exists=any(path.is_file() for path in candidates),
                status=AssetStatus.DUPLICATE_MATCH,
                selectable=False,
                reason="Image and annotation matching is ambiguous",
                issues=["ambiguous_match"],
            ))
            continue

        annotation = candidates[0]
        annotation_count, _, annotation_error, labels, shape_types = annotation_metadata.get(annotation, (None, None, None, [], []))
        if request.validate_annotations and annotation_error:
            summary.corrupt_annotation += 1
            emit(DatasetAsset(
                asset_id=_asset_id(dataset_id, match_key), match_key=match_key, display_path=_relative_display(roots.root, image),
                image_path=image, annotation_paths=[annotation], status=AssetStatus.CORRUPT_ANNOTATION,
                annotation_file_exists=annotation.is_file(),
                selectable=False, reason=annotation_error, issues=["annotation_invalid_json"],
            ))
            continue

        width = height = None
        if request.validate_images:
            width, height, image_error = _validate_image(image)
            if image_error:
                summary.corrupt_image += 1
                emit(DatasetAsset(
                    asset_id=_asset_id(dataset_id, match_key), match_key=match_key, display_path=_relative_display(roots.root, image),
                    image_path=image, annotation_paths=[annotation], status=AssetStatus.CORRUPT_IMAGE,
                    annotation_file_exists=annotation.is_file(),
                    selectable=False, reason=image_error, issues=["image_corrupt"],
                ))
                continue

        summary.valid += 1
        emit(DatasetAsset(
            asset_id=_asset_id(dataset_id, match_key), match_key=match_key, display_path=_relative_display(roots.root, image),
            image_path=image, annotation_paths=[annotation], status=AssetStatus.VALID, selectable=True,
            width=width, height=height, annotation_count=annotation_count, annotation_file_exists=annotation.is_file(),
            labels=labels, shape_types=shape_types,
        ))

    for annotation in sorted(annotation_set - consumed_annotations):
        probe.check()
        annotation_root = next((root for root in roots.annotation_roots if root == annotation.parent or root in annotation.parents), roots.root)
        match_key = _match_key(annotation_root, annotation)
        _, _, annotation_error, labels, shape_types = annotation_metadata.get(annotation, (None, None, None, [], []))
        status = AssetStatus.CORRUPT_ANNOTATION if annotation_error else AssetStatus.ORPHAN_ANNOTATION
        if annotation_error:
            summary.corrupt_annotation += 1
        else:
            summary.orphan_annotation += 1
        emit(DatasetAsset(
            asset_id=_asset_id(dataset_id, match_key), match_key=match_key, display_path=_relative_display(roots.root, annotation),
            annotation_paths=[annotation], status=status, selectable=False,
            annotation_file_exists=annotation.is_file(),
            reason=annotation_error or "Annotation has no matching image",
            issues=["annotation_invalid_json" if annotation_error else "orphan_annotation"],
            labels=labels, shape_types=shape_types,
        ))

    probe.check(force=True)
    items.sort(key=lambda item: (not item.selectable, item.display_path.casefold()))
    return DatasetScanResult(
        dataset_id=dataset_id,
        root_dir=roots.root,
        image_root=roots.image_root,
        annotation_roots=list(roots.annotation_roots),
        items=items,
        summary=summary,
    )
