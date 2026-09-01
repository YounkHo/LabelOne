from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
from threading import Lock
from time import sleep

import numpy as np
from PIL import Image
import pytest

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError
from labelone.images.tiles import DeepZoomTileService


def _service(
    tmp_path: Path,
    *,
    tile_size: int = 256,
    service_type: type[DeepZoomTileService] = DeepZoomTileService,
) -> tuple[DeepZoomTileService, DatasetRepository, Path, str, str]:
    root = tmp_path / "dataset"
    root.mkdir()
    y, x = np.indices((350, 600))
    pixels = np.stack(((x % 256), (y % 256), ((x + y) % 256)), axis=2).astype(np.uint8)
    image_path = root / "large.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)
    (root / "large.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="tiles"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    service = service_type(repository, tmp_path / "tile-cache", tile_size=tile_size)
    return service, repository, image_path, "tiles", scan.items[0].asset_id


def test_metadata_and_edge_tiles_follow_deep_zoom_geometry(tmp_path: Path) -> None:
    service, repository, _, dataset_id, asset_id = _service(tmp_path)

    metadata = service.metadata(dataset_id, asset_id)
    full_edge = service.tile(dataset_id, asset_id, level=metadata.max_level, x=2, y=1)
    half_edge = service.tile(dataset_id, asset_id, level=metadata.max_level - 1, x=1, y=0, format_name="png")
    smallest = service.tile(dataset_id, asset_id, level=0, x=0, y=0)

    assert (metadata.width, metadata.height) == (600, 350)
    assert metadata.tile_size == 256
    assert metadata.max_level == 10
    assert metadata.format == "webp"
    assert metadata.backend == "pillow"
    assert len(metadata.source_etag) == 64
    assert (full_edge.width, full_edge.height) == (88, 94)
    assert Image.open(BytesIO(full_edge.content)).size == (88, 94)
    assert half_edge.media_type == "image/png"
    assert (half_edge.width, half_edge.height) == (44, 175)
    assert Image.open(BytesIO(half_edge.content)).size == (44, 175)
    assert (smallest.width, smallest.height) == (1, 1)
    repository.close()


def test_etag_is_stable_for_304_and_source_mtime_invalidates_cache(tmp_path: Path) -> None:
    service, repository, image_path, dataset_id, asset_id = _service(tmp_path, tile_size=128)
    metadata = service.metadata(dataset_id, asset_id)

    first = service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0)
    second = service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0)
    previous_stat = image_path.stat()
    with Image.open(image_path) as source:
        changed = source.copy()
    changed.putpixel((0, 0), (255, 255, 255))
    changed.save(image_path)
    os.utime(image_path, ns=(previous_stat.st_atime_ns, max(image_path.stat().st_mtime_ns, previous_stat.st_mtime_ns + 1_000_000)))
    third = service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.etag == first.etag
    assert third.cache_hit is False
    assert third.etag != first.etag
    assert service.metadata(dataset_id, asset_id).source_etag != metadata.source_etag
    repository.close()


class _CountingTileService(DeepZoomTileService):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.render_count = 0
        self.render_lock = Lock()

    def _render_pillow(self, *args, **kwargs) -> bytes:
        with self.render_lock:
            self.render_count += 1
        sleep(0.04)
        return super()._render_pillow(*args, **kwargs)


def test_same_tile_key_is_single_flight_and_lock_entries_are_released(tmp_path: Path) -> None:
    service, repository, _, dataset_id, asset_id = _service(
        tmp_path,
        tile_size=128,
        service_type=_CountingTileService,
    )
    metadata = service.metadata(dataset_id, asset_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: service.tile(dataset_id, asset_id, level=metadata.max_level, x=1, y=1),
                range(8),
            )
        )

    assert service.render_count == 1
    assert sum(not result.cache_hit for result in results) == 1
    assert len({result.etag for result in results}) == 1
    assert service._single_flights == {}
    repository.close()


def test_corrupt_cache_is_discarded_and_atomically_rebuilt(tmp_path: Path) -> None:
    service, repository, _, dataset_id, asset_id = _service(tmp_path, tile_size=128)
    metadata = service.metadata(dataset_id, asset_id)
    first = service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0)
    cache_files = [path for path in service.cache_root.rglob("*") if path.is_file()]
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"not an image")

    rebuilt = service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0)
    cached = service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0)

    assert rebuilt.cache_hit is False
    assert rebuilt.etag == first.etag
    assert Image.open(BytesIO(rebuilt.content)).size == (128, 128)
    assert cached.cache_hit is True
    assert not list(service.cache_root.rglob("*.part"))
    repository.close()


@pytest.mark.parametrize(
    ("level_offset", "x", "y"),
    [(-100, 0, 0), (1, 0, 0), (0, 3, 0), (0, 0, 2)],
)
def test_illegal_levels_and_coordinates_are_rejected(
    tmp_path: Path,
    level_offset: int,
    x: int,
    y: int,
) -> None:
    service, repository, _, dataset_id, asset_id = _service(tmp_path)
    metadata = service.metadata(dataset_id, asset_id)
    level = metadata.max_level + level_offset

    with pytest.raises(InvalidPathError):
        service.tile(dataset_id, asset_id, level=level, x=x, y=y)

    repository.close()


def test_format_coordinate_and_source_budgets_are_validated(tmp_path: Path) -> None:
    service, repository, _, dataset_id, asset_id = _service(tmp_path)
    metadata = service.metadata(dataset_id, asset_id)

    with pytest.raises(InvalidPathError, match="Unsupported tile format"):
        service.tile(dataset_id, asset_id, level=metadata.max_level, x=0, y=0, format_name="gif")
    with pytest.raises(InvalidPathError, match="non-negative integers"):
        service.tile(dataset_id, asset_id, level=metadata.max_level, x=0.5, y=0)  # type: ignore[arg-type]

    limited = DeepZoomTileService(repository, tmp_path / "limited-cache", max_source_pixels=1_000)
    with pytest.raises(InvalidPathError, match="decode budget"):
        limited.metadata(dataset_id, asset_id)
    repository.close()
