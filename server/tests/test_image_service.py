from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.images import ImageService
from labelone.errors import InvalidPathError
import pytest


def _service(tmp_path: Path) -> tuple[ImageService, str, str]:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (400, 200), (100, 120, 140)).save(root / "image.jpg")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    result = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(result)
    return ImageService(repository, tmp_path / "cache"), "dataset", result.items[0].asset_id


def test_thumbnail_and_region_are_cached_with_expected_dimensions(tmp_path: Path) -> None:
    service, dataset_id, asset_id = _service(tmp_path)
    first = service.thumbnail(dataset_id, asset_id, max_size=100)
    second = service.thumbnail(dataset_id, asset_id, max_size=100)
    region = service.region(dataset_id, asset_id, x=50, y=20, width=100, height=60, scale=2)

    assert (first.width, first.height) == (100, 50)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.etag == first.etag
    assert (region.width, region.height) == (200, 120)


def test_same_thumbnail_key_is_single_flight_and_region_budget_is_enforced(tmp_path: Path) -> None:
    service, dataset_id, asset_id = _service(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: service.thumbnail(dataset_id, asset_id, max_size=96), range(8)))

    assert sum(not result.cache_hit for result in results) == 1
    assert len({result.etag for result in results}) == 1
    with pytest.raises(InvalidPathError):
        service.region(dataset_id, asset_id, x=0, y=0, width=10000, height=10000, scale=1)
