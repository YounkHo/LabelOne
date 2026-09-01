from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.datasets.revalidate import revalidate_asset
from labelone.errors import InvalidPathError


def test_revalidate_repairs_a_corrupt_annotation_without_enabling_matching_errors(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (20, 10), "white").save(root / "image.png")
    (root / "image.json").write_text("{broken", encoding="utf-8")
    (root / "orphan.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(dataset_id="dataset", root_dir=root, layout="same_directory"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    corrupt = next(item for item in scan.items if item.status.value == "corrupt_annotation")
    orphan = next(item for item in scan.items if item.status.value == "orphan_annotation")

    (root / "image.json").write_text(json.dumps({"shapes": [{"label": "fixed", "shape_type": "point", "points": [[1, 2]]}]}), encoding="utf-8")
    repaired = revalidate_asset(repository, "dataset", corrupt.asset_id)

    assert repaired.selectable is True
    assert repaired.labels == ["fixed"]
    assert repaired.shape_types == ["point"]
    with pytest.raises(InvalidPathError, match="full dataset rescan"):
        revalidate_asset(repository, "dataset", orphan.asset_id)
    repository.close()
