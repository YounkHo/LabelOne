from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from labelone.annotations import AnnotationStore
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError


def _repository(tmp_path: Path) -> tuple[DatasetRepository, AnnotationStore, str]:
    root = tmp_path / "dataset"
    root.mkdir()
    fixtures = [
        ("large-scratch.png", (9001, 10), [{"label": "scratch", "shape_type": "rotation", "points": [[1, 1], [2, 2]]}]),
        ("particle.png", (20, 20), [{"label": "particle", "shape_type": "polygon", "points": [[1, 1], [2, 1], [2, 2]]}]),
        ("empty.png", (20, 10), []),
    ]
    for name, size, shapes in fixtures:
        Image.new("RGB", size, "white").save(root / name)
        (root / Path(name).with_suffix(".json")).write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(dataset_id="dataset", root_dir=root, layout="same_directory"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    particle = next(item for item in scan.items if item.labels == ["particle"])
    return repository, AnnotationStore(repository, tmp_path / "backups"), particle.asset_id


def test_text_regex_and_condition_search_are_server_paginated(tmp_path: Path) -> None:
    repository, _, _ = _repository(tmp_path)

    assert repository.search_assets("dataset", query="particle", mode="text").total == 1
    assert repository.search_assets("dataset", query=r"large-.*\.png", mode="regex").total == 1
    result = repository.search_assets(
        "dataset",
        query="class:scratch AND type:rotation OR annotations=0",
        mode="condition",
        limit=1,
    )

    assert result.total == 2
    assert len(result.items) == 1
    assert result.next_offset == 1
    repository.close()


@pytest.mark.parametrize("query", [r"(a+)+$", r"(?=a)", r"(a|aa)+", ".*a.*b"])
def test_unsafe_regex_is_rejected(tmp_path: Path, query: str) -> None:
    repository, _, _ = _repository(tmp_path)
    with pytest.raises(InvalidPathError, match="Unsafe regular expression"):
        repository.search_assets("dataset", query=query, mode="regex")
    repository.close()


def test_annotation_save_refreshes_search_labels_and_shape_types(tmp_path: Path) -> None:
    repository, annotations, asset_id = _repository(tmp_path)
    envelope = annotations.load("dataset", asset_id)
    document = envelope.document
    document["shapes"] = [{"label": "updated", "shape_type": "point", "points": [[3, 4]]}]

    annotations.save("dataset", asset_id, document, if_match=envelope.revision)

    assert repository.search_assets("dataset", query="class:updated type:point", mode="condition").total == 1
    assert repository.search_assets("dataset", query="class:particle", mode="condition").total == 0
    repository.close()
