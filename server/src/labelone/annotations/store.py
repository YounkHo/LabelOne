from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from labelone.datasets.models import DatasetAsset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import AnnotationValidationError, RevisionConflictError
from labelone.keyed_lock import KeyedLockPool

from .models import AnnotationEnvelope, AnnotationSaveResponse
from .codec import normalize_annotation_document
from .validation import validate_annotation_document


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


class AnnotationStore:
    def __init__(self, repository: DatasetRepository, backup_root: Path) -> None:
        self.repository = repository
        self.backup_root = backup_root.expanduser().resolve()
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self._asset_locks = KeyedLockPool()

    def _asset_path(self, dataset_id: str, asset_id: str) -> tuple[DatasetAsset, Path]:
        asset = self.repository.get_asset(dataset_id, asset_id, require_selectable=True)
        if len(asset.annotation_paths) != 1:
            raise AnnotationValidationError(
                "A selectable asset must have exactly one annotation path",
                details={"dataset_id": dataset_id, "asset_id": asset_id},
            )
        return asset, asset.annotation_paths[0].resolve()

    @staticmethod
    def _empty(asset: DatasetAsset) -> tuple[bytes, dict[str, Any]]:
        document: dict[str, Any] = {
            "version": "LabelOne",
            "flags": {},
            "shapes": [],
            "imagePath": asset.image_path.name if asset.image_path is not None else None,
            "imageData": None,
            "imageWidth": asset.width,
            "imageHeight": asset.height,
        }
        content = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        return content, document

    @staticmethod
    def _read(path: Path) -> tuple[bytes, dict[str, Any]]:
        try:
            content = path.read_bytes()
            document = json.loads(content.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AnnotationValidationError("Could not read annotation JSON", details={"path": str(path), "error": str(exc)}) from exc
        if not isinstance(document, dict):
            raise AnnotationValidationError("Annotation root must be an object", details={"path": str(path)})
        normalized = normalize_annotation_document(document)
        validate_annotation_document(normalized)
        return content, normalized

    def load(self, dataset_id: str, asset_id: str) -> AnnotationEnvelope:
        asset, path = self._asset_path(dataset_id, asset_id)
        content, document = self._read(path) if path.is_file() else self._empty(asset)
        return AnnotationEnvelope(
            dataset_id=dataset_id,
            asset_id=asset_id,
            path=path,
            revision=_revision(content),
            document=document,
        )

    def save(self, dataset_id: str, asset_id: str, document: dict[str, Any], *, if_match: str) -> AnnotationSaveResponse:
        document = normalize_annotation_document(document)
        validate_annotation_document(document)
        asset, path = self._asset_path(dataset_id, asset_id)
        with self._asset_locks.hold((dataset_id, asset_id)):
            target_existed = path.is_file()
            current_content, _ = self._read(path) if target_existed else self._empty(asset)
            current_revision = _revision(current_content)
            expected = if_match.strip().strip('"')
            if expected != current_revision:
                raise RevisionConflictError(
                    "Annotation has changed since it was loaded",
                    details={"expected": expected, "current": current_revision},
                )
            backup_dir = self.backup_root / dataset_id / asset_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{current_revision}.json"
            if not backup_path.exists():
                backup_partial = backup_dir / f".{current_revision}.{uuid4().hex}.part"
                with backup_partial.open("xb") as handle:
                    handle.write(current_content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(backup_partial, backup_path)
                backup_directory_fd = os.open(backup_dir, os.O_RDONLY)
                try:
                    os.fsync(backup_directory_fd)
                finally:
                    os.close(backup_directory_fd)

            encoded = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.parent / f".{path.name}.{uuid4().hex}.tmp"
            try:
                with partial.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                if target_existed:
                    os.chmod(partial, path.stat().st_mode & 0o777)
                latest_content = path.read_bytes() if path.is_file() else None
                if (target_existed and latest_content is None) or (latest_content is not None and _revision(latest_content) != current_revision):
                    raise RevisionConflictError(
                        "Annotation changed during save",
                        details={
                            "expected": current_revision,
                            "current": _revision(latest_content) if latest_content is not None else None,
                        },
                    )
                os.replace(partial, path)
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if partial.exists():
                    partial.unlink()
            new_revision = _revision(encoded)
            self.repository.update_annotation_metadata(
                dataset_id,
                asset_id,
                annotation_count=len(document.get("shapes", [])),
                revision=new_revision,
                labels=[str(shape.get("label")) for shape in document.get("shapes", []) if isinstance(shape, dict) and shape.get("label")],
                shape_types=[str(shape.get("shape_type")) for shape in document.get("shapes", []) if isinstance(shape, dict) and shape.get("shape_type")],
            )
            return AnnotationSaveResponse(
                dataset_id=dataset_id,
                asset_id=asset_id,
                path=path,
                previous_revision=current_revision,
                revision=new_revision,
                backup_path=backup_path,
            )
