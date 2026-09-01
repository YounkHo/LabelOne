from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from labelone.datasets.models import (
    AssetStatus,
    DatasetAsset,
    DatasetScanRequest,
    DatasetScanResult,
    DatasetScanSummary,
)
from labelone.datasets.scan_sessions import DatasetScanSessionStore
from labelone.datasets.scanner import DatasetScanInterrupted, scan_dataset
from labelone.errors import InvalidPathError


def _image(path: Path, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        Image.new("RGB", (12, 8), "white").save(path)
    else:
        path.write_bytes(b"broken image")


def _annotation(path: Path, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"shapes": []}) if valid else "{broken", encoding="utf-8")


def test_scan_session_persists_incremental_results_without_image_only_entries(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _image(root / "valid.png")
    _annotation(root / "valid.json")
    _image(root / "hidden-image-only.png")
    _annotation(root / "orphan.json")
    _image(root / "corrupt.png", valid=False)
    _annotation(root / "corrupt.json")
    database = tmp_path / "index.sqlite3"
    store = DatasetScanSessionStore(database, flush_size=1)
    created = store.create(DatasetScanRequest(dataset_id="scan", root_dir=root, layout="same_directory"))

    finished = store.run(created.session_id)
    first = store.list_items(created.session_id, limit=2)
    second = store.list_items(created.session_id, after_sequence=first.next_after or -1, limit=10)
    items = [*first.items, *second.items]

    assert created.state == "queued"
    assert finished.state == "succeeded"
    assert finished.persisted_items == 3
    assert finished.summary is not None
    assert finished.summary.valid == 1
    assert finished.summary.hidden_image_only == 1
    assert finished.summary.orphan_annotation == 1
    assert finished.summary.corrupt_image == 1
    assert first.total == 3
    assert first.next_after is not None
    assert len(items) == 3
    assert not any("hidden-image-only" in item.display_path for item in items)
    valid = next(item for item in items if item.status is AssetStatus.VALID)
    assert valid.selectable is True
    assert all(not item.selectable for item in items if item is not valid)
    store.close()

    reopened = DatasetScanSessionStore(database)
    restored = reopened.get(created.session_id)
    assert restored.state == "succeeded"
    assert restored.persisted_items == 3
    assert reopened.list_items(created.session_id, limit=10).items == items
    reopened.close()


def test_scanner_sink_mode_preserves_rules_without_collecting_response_items(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _image(root / "valid.png")
    _annotation(root / "valid.json")
    _image(root / "hidden.png")
    streamed = []

    result = scan_dataset(
        DatasetScanRequest(dataset_id="stream", root_dir=root, layout="same_directory"),
        item_sink=streamed.append,
        collect_items=False,
    )

    assert result.items == []
    assert len(streamed) == 1
    assert streamed[0].status is AssetStatus.VALID
    assert result.summary.valid == 1
    assert result.summary.hidden_image_only == 1


def test_running_session_becomes_interrupted_and_can_be_retried_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _image(root / "valid.png")
    _annotation(root / "valid.json")
    database = tmp_path / "index.sqlite3"
    store = DatasetScanSessionStore(database)
    created = store.create(DatasetScanRequest(dataset_id="retry", root_dir=root, layout="same_directory"))
    with store._lock, store._connection:
        store._connection.execute(
            "UPDATE dataset_scan_sessions SET state='running' WHERE session_id=?", (created.session_id,)
        )
    store.close()

    reopened = DatasetScanSessionStore(database)
    assert reopened.get(created.session_id).state == "interrupted"
    assert reopened.run(created.session_id).state == "succeeded"
    reopened.close()


def test_failed_scan_session_keeps_queryable_state_and_error(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    store = DatasetScanSessionStore(database)
    created = store.create(DatasetScanRequest(root_dir=tmp_path / "missing"))

    failed = store.run(created.session_id)

    assert failed.state == "failed"
    assert failed.persisted_items == 0
    assert failed.error is not None and "readable directory" in failed.error
    assert store.list_items(created.session_id, limit=10).items == []
    store.close()


def test_queued_interrupt_is_idempotent_and_interrupted_session_can_rerun(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _image(root / "valid.png")
    _annotation(root / "valid.json")
    store = DatasetScanSessionStore(tmp_path / "index.sqlite3")
    created = store.create(DatasetScanRequest(dataset_id="queued", root_dir=root, layout="same_directory"))

    interrupted = store.interrupt(created.session_id)
    repeated = store.interrupt(created.session_id)

    assert interrupted.state == repeated.state == "interrupted"
    assert interrupted.interrupted_at == repeated.interrupted_at
    assert interrupted.interruption_reason == repeated.interruption_reason == "Interrupted before scan started"
    completed = store.run(created.session_id)
    assert completed.state == "succeeded"
    assert completed.interrupted_at is None
    assert completed.interruption_reason is None
    assert completed.persisted_items == 1
    with pytest.raises(InvalidPathError, match="queued or running"):
        store.interrupt(created.session_id)
    store.close()


def test_running_interrupt_cooperatively_stops_and_preserves_buffered_items_for_query(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from threading import Event, Thread
    from labelone.datasets import scan_sessions as scan_session_module

    root = tmp_path / "dataset"
    root.mkdir()
    store = DatasetScanSessionStore(tmp_path / "index.sqlite3", flush_size=3)
    session = store.create(DatasetScanRequest(dataset_id="running", root_dir=root, layout="same_directory"))
    checkpoint = Event()
    proceed = Event()
    call_count = 0

    def asset(index: int) -> DatasetAsset:
        return DatasetAsset(
            asset_id=f"asset-{index}",
            match_key=f"asset-{index}",
            display_path=f"asset-{index}.png",
            image_path=root / f"asset-{index}.png",
            annotation_paths=[root / f"asset-{index}.json"],
            status=AssetStatus.VALID,
            selectable=True,
        )

    def controlled_scan(request, *, item_sink, collect_items, cancel_check):
        nonlocal call_count
        call_count += 1
        count = 10 if call_count == 1 else 2
        for index in range(count):
            item_sink(asset(index + call_count * 100))
            if call_count == 1 and index == 3:
                checkpoint.set()
                assert proceed.wait(timeout=2)
            if cancel_check():
                raise DatasetScanInterrupted("fixture interruption")
        return DatasetScanResult(
            dataset_id="running",
            root_dir=root,
            image_root=root,
            annotation_roots=[root],
            items=[] if not collect_items else [asset(index) for index in range(count)],
            summary=DatasetScanSummary(valid=count),
        )

    monkeypatch.setattr(scan_session_module, "scan_dataset", controlled_scan)
    results: list = []
    thread = Thread(target=lambda: results.append(store.run(session.session_id)))
    thread.start()
    assert checkpoint.wait(timeout=2)
    requested = store.interrupt(session.session_id)
    assert requested.state == "interrupted"
    proceed.set()
    thread.join(timeout=3)

    assert len(results) == 1
    stopped = results[0]
    assert stopped.state == "interrupted"
    assert stopped.error is None
    assert stopped.persisted_items == 4
    assert stopped.interruption_reason == "Interrupted cooperatively after persisting 4 items"
    assert store.list_items(session.session_id, limit=10).total == 4

    rerun = store.run(session.session_id)
    assert rerun.state == "succeeded"
    assert rerun.persisted_items == 2
    assert store.list_items(session.session_id, limit=10).total == 2
    assert rerun.interruption_reason is None
    store.close()


def test_scanner_cooperative_callback_interrupts_between_discovery_stages(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _image(root / "valid.png")
    _annotation(root / "valid.json")
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(DatasetScanInterrupted):
        scan_dataset(
            DatasetScanRequest(dataset_id="cancel", root_dir=root, layout="same_directory"),
            cancel_check=cancel,
        )

    assert checks == 3


def test_immediate_rerun_generation_rejects_late_items_from_interrupted_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from threading import Event, Thread
    from labelone.datasets import scan_sessions as scan_session_module

    root = tmp_path / "dataset"
    root.mkdir()
    store = DatasetScanSessionStore(tmp_path / "index.sqlite3", flush_size=10)
    session = store.create(DatasetScanRequest(dataset_id="generation", root_dir=root, layout="same_directory"))
    old_waiting = Event()
    release_old = Event()
    calls = 0

    def make_asset(asset_id: str) -> DatasetAsset:
        return DatasetAsset(
            asset_id=asset_id,
            match_key=asset_id,
            display_path=f"{asset_id}.png",
            status=AssetStatus.VALID,
            selectable=True,
        )

    def generation_scan(request, *, item_sink, collect_items, cancel_check):
        nonlocal calls
        calls += 1
        current_call = calls
        item_sink(make_asset("old" if current_call == 1 else "new"))
        if current_call == 1:
            old_waiting.set()
            assert release_old.wait(timeout=2)
            if cancel_check():
                raise DatasetScanInterrupted("old generation stopped")
        return DatasetScanResult(
            dataset_id="generation",
            root_dir=root,
            image_root=root,
            annotation_roots=[root],
            items=[],
            summary=DatasetScanSummary(valid=1),
        )

    monkeypatch.setattr(scan_session_module, "scan_dataset", generation_scan)
    old_results: list = []
    old_thread = Thread(target=lambda: old_results.append(store.run(session.session_id)))
    old_thread.start()
    assert old_waiting.wait(timeout=2)
    interrupted = store.interrupt(session.session_id)
    old_generation = interrupted.run_generation

    rerun = store.run(session.session_id)
    release_old.set()
    old_thread.join(timeout=3)

    assert rerun.state == "succeeded"
    assert rerun.run_generation == old_generation + 1
    assert len(old_results) == 1 and old_results[0].run_generation == rerun.run_generation
    items = store.list_items(session.session_id, limit=10).items
    assert [item.asset_id for item in items] == ["new"]
    store.close()
