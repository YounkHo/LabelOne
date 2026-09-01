from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.cursor import InvalidDatasetCursorError, StaleDatasetCursorError
from labelone.datasets.repository import DatasetRepository


def _register(repository: DatasetRepository, root: Path, dataset_id: str, count: int = 9):
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (20 + index, 12), (index, 30, 40)).save(root / f"asset-{index:03d}.png")
        shapes = [] if index % 3 else [{"label": "scratch", "shape_type": "polygon", "points": [[1, 1], [2, 1], [2, 2]]}]
        (root / f"asset-{index:03d}.json").write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
    result = scan_dataset(DatasetScanRequest(dataset_id=dataset_id, root_dir=root, layout="same_directory"))
    return result, repository.register(result)


def _walk(repository: DatasetRepository, dataset_id: str, *, limit: int = 2) -> list[str]:
    cursor = None
    assets: list[str] = []
    while True:
        page = repository.list_assets_cursor(dataset_id, cursor=cursor, limit=limit)
        assets.extend(item.asset_id for item in page.items)
        cursor = page.next_cursor
        if cursor is None:
            return assets


def test_keyset_cursor_walk_matches_compatible_offset_order_and_revision_persists(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    repository = DatasetRepository(database)
    _, registered = _register(repository, tmp_path / "dataset", "cursor")
    expected = [item.asset_id for item in repository.list_assets("cursor", limit=100).items]

    walked = _walk(repository, "cursor", limit=2)

    assert walked == expected
    assert len(walked) == len(set(walked)) == 9
    first = repository.list_assets_cursor("cursor", limit=3)
    assert first.total == 9
    assert first.index_revision == registered.index_revision == 1
    repository.close()
    reopened = DatasetRepository(database)
    assert reopened.get_dataset("cursor").index_revision == 1
    assert _walk(reopened, "cursor", limit=4) == expected
    reopened.close()


def test_invalid_dataset_query_and_tampered_cursors_are_rejected_as_400(tmp_path: Path) -> None:
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    _register(repository, tmp_path / "one", "one", count=4)
    _register(repository, tmp_path / "two", "two", count=4)
    cursor = repository.list_assets_cursor("one", limit=2).next_cursor
    assert cursor is not None

    with pytest.raises(InvalidDatasetCursorError) as tampered:
        repository.list_assets_cursor("one", cursor=cursor[:-1] + ("A" if cursor[-1] != "A" else "B"), limit=2)
    assert tampered.value.status_code == 400
    with pytest.raises(InvalidDatasetCursorError, match="different dataset") as wrong_dataset:
        repository.list_assets_cursor("two", cursor=cursor, limit=2)
    assert wrong_dataset.value.status_code == 400
    with pytest.raises(InvalidDatasetCursorError, match="fingerprint"):
        repository.search_assets_cursor("one", query="asset", mode="text", cursor=cursor, limit=2)
    json_cursor = repository.search_assets_cursor(
        "one", query="", mode="smart", has_annotation_file=True, limit=2,
    ).next_cursor
    assert json_cursor is not None
    with pytest.raises(InvalidDatasetCursorError, match="fingerprint"):
        repository.search_assets_cursor(
            "one", query="", mode="smart", has_annotation_file=False, cursor=json_cursor, limit=2,
        )
    repository.close()


def test_cursor_becomes_409_stale_when_index_or_search_metadata_changes(tmp_path: Path) -> None:
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    scan, _ = _register(repository, tmp_path / "dataset", "stale", count=5)
    first = repository.list_assets_cursor("stale", limit=2)
    assert first.next_cursor is not None
    asset = scan.items[0]
    repository.update_annotation_metadata(
        "stale",
        asset.asset_id,
        annotation_count=1,
        revision="revision",
        labels=["changed"],
        shape_types=["point"],
    )

    with pytest.raises(StaleDatasetCursorError) as stale:
        repository.list_assets_cursor("stale", cursor=first.next_cursor, limit=2)

    assert stale.value.status_code == 409
    assert stale.value.details == {"cursor_revision": 1, "index_revision": 2}
    assert repository.get_dataset("stale").index_revision == 2
    repository.close()


def test_search_cursor_supports_condition_and_regex_without_duplicates(tmp_path: Path) -> None:
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    _register(repository, tmp_path / "dataset", "search", count=12)
    cursor = None
    paths: list[str] = []
    total = None
    while True:
        page = repository.search_assets_cursor(
            "search",
            query="class:scratch OR annotations=0",
            mode="condition",
            cursor=cursor,
            limit=3,
        )
        total = page.total
        paths.extend(item.display_path for item in page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    regex = repository.search_assets_cursor(
        "search", query=r"asset-00[89]\.png$", mode="regex", limit=10
    )

    assert len(paths) == total == 12
    assert len(paths) == len(set(paths))
    assert [item.display_path for item in regex.items] == ["asset-008.png", "asset-009.png"]
    repository.close()
