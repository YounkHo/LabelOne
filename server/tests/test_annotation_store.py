from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from labelone.annotations import AnnotationStore
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import AnnotationValidationError, RevisionConflictError


def _store(tmp_path: Path) -> tuple[AnnotationStore, str, str, Path]:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (20, 10), "white").save(root / "image.png")
    annotation_path = root / "image.json"
    annotation_path.write_text(json.dumps({
        "version": "x",
        "customTopLevel": {"keep": True},
        "shapes": [{
            "label": "object",
            "shape_type": "rotation",
            "points": [[1, 1], [10, 1], [10, 5], [1, 5]],
            "direction": 0.0,
            "customShapeField": "keep",
        }],
    }), encoding="utf-8")
    result = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(result)
    return AnnotationStore(repository, tmp_path / "backups"), "dataset", result.items[0].asset_id, annotation_path


def test_load_save_preserves_unknown_fields_and_creates_backup(tmp_path: Path) -> None:
    store, dataset_id, asset_id, annotation_path = _store(tmp_path)
    loaded = store.load(dataset_id, asset_id)
    loaded.document["shapes"][0]["label"] = "changed"

    saved = store.save(dataset_id, asset_id, loaded.document, if_match=loaded.revision)

    assert saved.backup_path.is_file()
    assert saved.previous_revision == loaded.revision
    assert saved.revision != loaded.revision
    persisted = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert persisted["customTopLevel"] == {"keep": True}
    assert persisted["shapes"][0]["customShapeField"] == "keep"
    assert persisted["shapes"][0]["label"] == "changed"


def test_first_save_creates_missing_annotation_in_managed_storage(tmp_path: Path) -> None:
    images = tmp_path / "images" / "fresh-set"
    storage = tmp_path / "labels"
    images.mkdir(parents=True)
    storage.mkdir()
    Image.new("RGB", (20, 10), "white").save(images / "nested.png")
    result = scan_dataset(DatasetScanRequest(
        root_dir=images,
        annotation_storage_root=storage,
        dataset_id="fresh",
    ))
    repository = DatasetRepository(tmp_path / "fresh-index.sqlite3")
    repository.register(result)
    store = AnnotationStore(repository, tmp_path / "fresh-backups")
    asset = result.items[0]

    loaded = store.load("fresh", asset.asset_id)
    assert repository.get_asset("fresh", asset.asset_id).annotation_file_exists is False
    assert loaded.document["shapes"] == []
    assert loaded.document["imagePath"] == "nested.png"
    assert loaded.document["imageWidth"] == 20
    assert loaded.document["imageHeight"] == 10
    assert not loaded.path.exists()

    loaded.document["shapes"] = [{
        "label": "new-object",
        "shape_type": "rectangle",
        "points": [[1, 1], [10, 1], [10, 5], [1, 5]],
    }]
    saved = store.save("fresh", asset.asset_id, loaded.document, if_match=loaded.revision)

    assert saved.path == (storage / "nested.json").resolve()
    assert saved.path.is_file()
    assert saved.backup_path.is_file()
    assert repository.get_asset("fresh", asset.asset_id).annotation_file_exists is True
    assert store.load("fresh", asset.asset_id).document["shapes"][0]["label"] == "new-object"


def test_stale_revision_is_rejected_without_overwrite(tmp_path: Path) -> None:
    store, dataset_id, asset_id, annotation_path = _store(tmp_path)
    loaded = store.load(dataset_id, asset_id)
    changed = dict(loaded.document)
    changed["version"] = "new"
    store.save(dataset_id, asset_id, changed, if_match=loaded.revision)
    current = annotation_path.read_bytes()

    with pytest.raises(RevisionConflictError):
        store.save(dataset_id, asset_id, loaded.document, if_match=loaded.revision)

    assert annotation_path.read_bytes() == current


def test_missing_rotation_direction_is_derived_and_degenerate_edge_rejected(tmp_path: Path) -> None:
    store, dataset_id, asset_id, _ = _store(tmp_path)
    loaded = store.load(dataset_id, asset_id)
    del loaded.document["shapes"][0]["direction"]
    saved = store.save(dataset_id, asset_id, loaded.document, if_match=loaded.revision)
    reloaded = store.load(dataset_id, asset_id)
    assert reloaded.document["shapes"][0]["direction"] == 0.0

    reloaded.document["shapes"][0]["points"][1] = reloaded.document["shapes"][0]["points"][0]

    with pytest.raises(AnnotationValidationError):
        store.save(dataset_id, asset_id, reloaded.document, if_match=saved.revision)


def test_legacy_rectangle_is_upgraded_to_four_points(tmp_path: Path) -> None:
    store, dataset_id, asset_id, _ = _store(tmp_path)
    loaded = store.load(dataset_id, asset_id)
    loaded.document["shapes"] = [{
        "label": "rectangle",
        "shape_type": "rectangle",
        "points": [[9, 7], [2, 3]],
    }]

    saved = store.save(dataset_id, asset_id, loaded.document, if_match=loaded.revision)
    reloaded = store.load(dataset_id, asset_id)

    assert saved.revision == reloaded.revision
    assert reloaded.document["shapes"][0]["points"] == [[2, 3], [9, 3], [9, 7], [2, 7]]
