from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.models import AssetStatus


def _image(path: Path, *, color: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (12, 8), color=color).save(path)


def _annotation(path: Path, *, image_path: str | None = None, count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"shapes": [{"label": "object", "points": [[0, 0], [1, 1]]}] * count}
    if image_path:
        payload["imagePath"] = image_path
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parallel_layout_hides_image_only_and_keeps_errors(tmp_path: Path) -> None:
    _image(tmp_path / "images" / "train" / "valid.png")
    _annotation(tmp_path / "annotations" / "train" / "valid.json", count=2)
    _image(tmp_path / "images" / "train" / "hidden.png")
    _annotation(tmp_path / "annotations" / "orphan.json")

    (tmp_path / "images" / "broken.png").write_bytes(b"not an image")
    _annotation(tmp_path / "annotations" / "broken.json")
    _image(tmp_path / "images" / "bad-json.png")
    (tmp_path / "annotations" / "bad-json.json").write_text("{", encoding="utf-8")

    result = scan_dataset(DatasetScanRequest(root_dir=tmp_path, layout="auto"))

    assert result.summary.valid == 1
    assert result.summary.hidden_image_only == 1
    assert result.summary.orphan_annotation == 1
    assert result.summary.corrupt_image == 1
    assert result.summary.corrupt_annotation == 1
    assert all("hidden.png" not in item.display_path for item in result.items)
    assert next(item for item in result.items if item.status is AssetStatus.VALID).annotation_count == 2
    assert all(not item.selectable for item in result.items if item.status is not AssetStatus.VALID)


def test_same_directory_sidecars(tmp_path: Path) -> None:
    _image(tmp_path / "nested" / "a.png")
    _annotation(tmp_path / "nested" / "a.json")
    _image(tmp_path / "nested" / "image-only.png")
    _annotation(tmp_path / "nested" / "json-only.json")

    result = scan_dataset(DatasetScanRequest(root_dir=tmp_path, layout="same_directory"))

    assert result.summary.valid == 1
    assert result.summary.hidden_image_only == 1
    assert result.summary.orphan_annotation == 1
    assert [item.display_path for item in result.items if item.selectable] == ["nested/a.png"]


def test_auto_layout_reads_parallel_and_sidecar_annotations_together(tmp_path: Path) -> None:
    _image(tmp_path / "images" / "parallel.png")
    _annotation(tmp_path / "annotations" / "parallel.json")
    _image(tmp_path / "images" / "sidecar.png")
    _annotation(tmp_path / "images" / "sidecar.json")
    _image(tmp_path / "images" / "duplicate.png")
    _annotation(tmp_path / "annotations" / "duplicate.json")
    _annotation(tmp_path / "images" / "duplicate.json")

    result = scan_dataset(DatasetScanRequest(root_dir=tmp_path, layout="auto"))

    assert result.summary.valid == 2
    assert result.summary.duplicate_match == 1
    assert {item.display_path for item in result.items if item.selectable} == {"images/parallel.png", "images/sidecar.png"}
    duplicate = next(item for item in result.items if item.status is AssetStatus.DUPLICATE_MATCH)
    assert duplicate.display_path == "images/duplicate.png"
    assert len(duplicate.annotation_paths) == 2


def test_annotation_storage_matches_json_by_stem_and_keeps_unlabeled_images(tmp_path: Path) -> None:
    images = tmp_path / "image-datasets" / "wafer-set"
    storage = tmp_path / "annotation-datasets"
    storage.mkdir()
    _image(images / "nested" / "existing.png")
    _image(images / "nested" / "new.png")
    _annotation(storage / "another-layout" / "existing.json", count=2)

    result = scan_dataset(DatasetScanRequest(
        root_dir=images,
        annotation_storage_root=storage,
    ))

    assert result.annotation_roots == [storage.resolve()]
    assert result.summary.valid == 2
    assert result.summary.hidden_image_only == 0
    by_name = {item.image_path.name: item for item in result.items if item.image_path is not None}
    assert by_name["existing.png"].annotation_count == 2
    assert by_name["existing.png"].annotation_file_exists is True
    assert by_name["new.png"].annotation_count == 0
    assert by_name["new.png"].annotation_file_exists is False
    assert by_name["new.png"].annotation_paths == [(storage / "nested" / "new.json").resolve()]
    assert not by_name["new.png"].annotation_paths[0].exists()


def test_annotation_storage_scans_json_only_and_does_not_create_dataset_directory(tmp_path: Path) -> None:
    images = tmp_path / "new-dataset"
    storage = tmp_path / "annotation-datasets"
    storage.mkdir()
    _image(images / "image.png")
    _image(storage / "must-not-be-scanned.png")

    result = scan_dataset(DatasetScanRequest(root_dir=images, annotation_storage_root=storage))

    assert not (storage / "new-dataset").exists()
    assert result.items[0].selectable is True
    assert all(item.image_path == (images / "image.png").resolve() for item in result.items)


def test_annotation_storage_marks_duplicate_json_stems_ambiguous(tmp_path: Path) -> None:
    images = tmp_path / "images"
    storage = tmp_path / "labels"
    _image(images / "same.jpg")
    _annotation(storage / "a" / "same.json")
    _annotation(storage / "b" / "same.json")

    result = scan_dataset(DatasetScanRequest(root_dir=images, annotation_storage_root=storage))

    assert result.summary.duplicate_match == 1
    assert result.items[0].status is AssetStatus.DUPLICATE_MATCH
    assert result.items[0].selectable is False


def test_annotation_storage_marks_unlabeled_duplicate_image_stems_ambiguous(tmp_path: Path) -> None:
    images = tmp_path / "images"
    storage = tmp_path / "labels"
    storage.mkdir()
    _image(images / "a" / "same.jpg")
    _image(images / "b" / "same.png")

    result = scan_dataset(DatasetScanRequest(root_dir=images, annotation_storage_root=storage))

    assert result.summary.duplicate_match == 2
    assert len(result.items) == 2
    assert all(item.status is AssetStatus.DUPLICATE_MATCH for item in result.items)
    assert all(not item.selectable for item in result.items)


def test_basename_collision_disables_every_ambiguous_image(tmp_path: Path) -> None:
    _image(tmp_path / "images" / "a" / "same.png")
    _image(tmp_path / "images" / "b" / "same.png")
    _annotation(tmp_path / "annotations" / "same.json")

    result = scan_dataset(DatasetScanRequest(
        root_dir=tmp_path,
        layout="parallel",
        match_strategy="basename",
    ))

    duplicates = [item for item in result.items if item.status is AssetStatus.DUPLICATE_MATCH]
    assert len(duplicates) == 2
    assert result.summary.duplicate_match == 2
    assert all(not item.selectable for item in duplicates)


def test_image_path_strategy(tmp_path: Path) -> None:
    _image(tmp_path / "images" / "nested" / "target.png")
    _annotation(tmp_path / "annotations" / "custom-name.json", image_path="nested/target.png")

    result = scan_dataset(DatasetScanRequest(
        root_dir=tmp_path,
        layout="parallel",
        match_strategy="image_path",
    ))

    assert result.summary.valid == 1
    assert result.summary.orphan_annotation == 0


def test_asset_id_survives_orphan_repair(tmp_path: Path) -> None:
    _annotation(tmp_path / "nested" / "stable.json")
    orphan = scan_dataset(DatasetScanRequest(root_dir=tmp_path, layout="same_directory", dataset_id="dataset"))
    orphan_id = orphan.items[0].asset_id

    _image(tmp_path / "nested" / "stable.png")
    repaired = scan_dataset(DatasetScanRequest(root_dir=tmp_path, layout="same_directory", dataset_id="dataset"))

    assert repaired.items[0].status is AssetStatus.VALID
    assert repaired.items[0].asset_id == orphan_id
