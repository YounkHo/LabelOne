from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from labelone.errors import InvalidPathError

from .models import AssetStatus, DatasetAsset
from .repository import DatasetRepository


def revalidate_asset(repository: DatasetRepository, dataset_id: str, asset_id: str) -> DatasetAsset:
    asset = repository.get_asset(dataset_id, asset_id)
    if asset.image_path is None or len(asset.annotation_paths) != 1:
        raise InvalidPathError(
            "This matching error requires a full dataset rescan",
            details={"dataset_id": dataset_id, "asset_id": asset_id, "status": asset.status.value},
        )
    image_path = asset.image_path.resolve()
    annotation_path = asset.annotation_paths[0].resolve()
    if not image_path.is_file() or not annotation_path.is_file():
        raise InvalidPathError(
            "Dataset asset files are still missing",
            details={"image_path": str(image_path), "annotation_path": str(annotation_path)},
        )
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        repository.update_asset_validation(
            dataset_id,
            asset_id,
            status=AssetStatus.CORRUPT_IMAGE,
            selectable=False,
            reason=f"Image decode failed: {exc}",
        )
        return repository.get_asset(dataset_id, asset_id)
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        shapes = payload.get("shapes", []) if isinstance(payload, dict) else None
        if not isinstance(shapes, list):
            raise ValueError("Annotation field 'shapes' must be a list")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        repository.update_asset_validation(
            dataset_id,
            asset_id,
            status=AssetStatus.CORRUPT_ANNOTATION,
            selectable=False,
            reason=f"Annotation decode failed: {exc}",
        )
        return repository.get_asset(dataset_id, asset_id)
    labels = [str(shape.get("label")) for shape in shapes if isinstance(shape, dict) and shape.get("label")]
    shape_types = [str(shape.get("shape_type")) for shape in shapes if isinstance(shape, dict) and shape.get("shape_type")]
    repository.update_asset_validation(
        dataset_id,
        asset_id,
        status=AssetStatus.VALID,
        selectable=True,
        reason=None,
        width=width,
        height=height,
        annotation_count=len(shapes),
        annotation_file_exists=True,
        labels=labels,
        shape_types=shape_types,
    )
    return repository.get_asset(dataset_id, asset_id)
