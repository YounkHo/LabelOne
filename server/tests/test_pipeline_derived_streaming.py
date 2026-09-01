from __future__ import annotations

import gc
from hashlib import sha256
import json
from pathlib import Path
import weakref

import pytest

from labelone.pipelines.derived import DerivedDatasetWriter
from labelone.pipelines.models import PipelineDerivedItemResult, PipelineOutputPolicy


class _TrackedItems:
    def __init__(self, count: int, output_root: Path) -> None:
        self.count = count
        self.output_root = output_root
        self.produced = 0
        self.alive = 0
        self.peak_alive = 0

    def _released(self) -> None:
        self.alive -= 1

    def __iter__(self):
        for index in range(self.count):
            item = PipelineDerivedItemResult(
                dataset_id="dataset",
                asset_id=f"asset-{index:06d}",
                output_root=self.output_root,
                item_fingerprint=sha256(f"item-{index}".encode()).hexdigest(),
                outputs=[],
            )
            self.produced += 1
            self.alive += 1
            self.peak_alive = max(self.peak_alive, self.alive)
            weakref.finalize(item, self._released)
            yield item


@pytest.mark.parametrize("count", [10_000, 100_000])
def test_finalize_streams_large_item_sets_with_constant_live_model_count(
    tmp_path: Path,
    monkeypatch,
    count: int,
) -> None:
    writer = DerivedDatasetWriter()
    source_root = tmp_path / f"source-{count}"
    source_root.mkdir()
    output_root = (tmp_path / f"derived-{count}").resolve()
    policy = PipelineOutputPolicy(mode="derived_dataset", output_root=output_root)
    staging = writer.staging_root(output_root, f"job-{count}")
    staging.mkdir()
    tracked = _TrackedItems(count, output_root)
    monkeypatch.setattr(writer, "_validated_paths", lambda staging, item: [])

    published = writer.finalize(
        job_id=f"job-{count}",
        dataset_id="dataset",
        source_root=source_root,
        policy=policy,
        items=iter(tracked),
        expected_item_count=count,
    )
    gc.collect()

    assert published.item_count == count
    assert published.output_count == 0
    assert tracked.produced == count
    assert tracked.peak_alive <= 3
    assert tracked.alive == 0
    manifest = output_root / ".labelone-derived.json"
    assert manifest.is_file()
    with manifest.open("rb") as handle:
        prefix = handle.read(80)
        handle.seek(-120, 2)
        suffix = handle.read()
    assert b'"schema_version":1' in prefix
    assert f'"item_count":{count}'.encode() in suffix
    assert not list(output_root.parent.glob("*.part"))


def test_streaming_fingerprint_matches_previous_sorted_definition(tmp_path: Path, monkeypatch) -> None:
    writer = DerivedDatasetWriter()
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_root = (tmp_path / "derived").resolve()
    policy = PipelineOutputPolicy(mode="derived_dataset", output_root=output_root)
    staging = writer.staging_root(output_root, "fingerprint-job")
    staging.mkdir()
    items = [
        PipelineDerivedItemResult(
            dataset_id="dataset",
            asset_id=asset_id,
            output_root=output_root,
            item_fingerprint=fingerprint,
            outputs=[],
        )
        for asset_id, fingerprint in (("a", "1" * 64), ("b", "2" * 64), ("c", "3" * 64))
    ]
    monkeypatch.setattr(writer, "_validated_paths", lambda staging, item: [])

    published = writer.finalize(
        job_id="fingerprint-job",
        dataset_id="dataset",
        source_root=source_root,
        policy=policy,
        items=iter(items),
        expected_item_count=3,
    )
    expected = sha256(json.dumps({
        "dataset_id": "dataset",
        "items": sorted((item.asset_id, item.item_fingerprint) for item in items),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert published.dataset_fingerprint == expected
