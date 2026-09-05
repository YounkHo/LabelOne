from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError


def _dataset(root: Path) -> None:
    root.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (12 + index, 8), (index * 40, 20, 30)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")


def test_register_persist_list_and_page_assets(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    result = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="stable"))
    database = tmp_path / "index.sqlite3"
    repository = DatasetRepository(database)
    registered = repository.register(result, name="Test Dataset")

    assert registered.dataset_id == "stable"
    assert repository.list_datasets().datasets[0].name == "Test Dataset"
    first_page = repository.list_assets("stable", limit=2)
    assert first_page.total == 3
    assert len(first_page.items) == 2
    assert first_page.next_offset == 2
    first_asset = repository.get_asset("stable", first_page.items[0].asset_id, require_selectable=True)
    assert first_asset.image_path is not None
    repository.close()

    reopened = DatasetRepository(database)
    assert reopened.get_dataset("stable").summary.valid == 3
    assert reopened.list_assets("stable", offset=2, limit=2).next_offset is None
    reopened.close()


def test_missing_dataset_root_is_reported_before_stale_assets_are_returned(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    repository = DatasetRepository(tmp_path / "missing-root.sqlite3")
    repository.register(scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="missing-root")))

    root.rename(tmp_path / "dataset-moved")

    listed = repository.list_datasets().datasets[0]
    assert listed.source_available is False
    assert listed.source_error == "root_missing"
    with pytest.raises(InvalidPathError, match="数据集源目录不存在或不可访问") as error:
        repository.list_assets_cursor("missing-root")
    assert error.value.details["reason"] == "root_missing"
    with pytest.raises(InvalidPathError):
        repository.search_assets("missing-root", query="", mode="smart")
    repository.close()


def test_json_file_filter_distinguishes_empty_json_from_missing_json(tmp_path: Path) -> None:
    images = tmp_path / "images"
    storage = tmp_path / "labels"
    images.mkdir()
    storage.mkdir()
    Image.new("RGB", (12, 8), "white").save(images / "existing.png")
    Image.new("RGB", (12, 8), "white").save(images / "missing.png")
    (storage / "existing.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    result = scan_dataset(DatasetScanRequest(
        root_dir=images,
        annotation_storage_root=storage,
        dataset_id="json-filter",
    ))
    database = tmp_path / "json-filter.sqlite3"
    repository = DatasetRepository(database)
    repository.register(result)

    with_json = repository.search_assets(
        "json-filter", query="", mode="smart", has_annotation_file=True,
    )
    without_json = repository.search_assets(
        "json-filter", query="", mode="smart", has_annotation_file=False,
    )

    assert [item.display_path for item in with_json.items] == ["existing.png"]
    assert with_json.items[0].annotation_count == 0
    assert with_json.items[0].annotation_file_exists is True
    assert [item.display_path for item in without_json.items] == ["missing.png"]
    assert without_json.items[0].annotation_file_exists is False
    repository.close()


def test_legacy_index_backfills_json_file_existence_from_disk(tmp_path: Path) -> None:
    root = tmp_path / "legacy-dataset"
    _dataset(root)
    database = tmp_path / "legacy.sqlite3"
    repository = DatasetRepository(database)
    repository.register(scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="legacy")))
    repository.close()

    connection = sqlite3.connect(database)
    connection.execute("DROP INDEX assets_dataset_annotation_file_path")
    connection.execute("ALTER TABLE assets DROP COLUMN annotation_file_exists")
    connection.commit()
    connection.close()

    migrated = DatasetRepository(database)
    assert migrated.get_dataset("legacy").index_revision == 2
    assert all(item.annotation_file_exists for item in migrated.list_assets("legacy").items)
    migrated.close()
