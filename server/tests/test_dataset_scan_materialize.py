from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from threading import Event, Thread

from PIL import Image
import pytest

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.datasets.scan_sessions import DatasetScanSessionStore
from labelone.errors import InvalidPathError


def _dataset(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True)
    for index, name in enumerate(names):
        Image.new("RGB", (12, 8), (index, 30, 40)).save(root / f"{name}.png")
        (root / f"{name}.json").write_text(
            json.dumps({"shapes": [{"label": name, "shape_type": "rectangle", "points": [[1, 1], [2, 2]]}]}),
            encoding="utf-8",
        )


def test_successful_session_materializes_with_insert_select_and_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    repository = DatasetRepository(database)
    old_root = tmp_path / "old"
    _dataset(old_root, ["old"])
    repository.register(scan_dataset(DatasetScanRequest(dataset_id="stable", root_dir=old_root, layout="same_directory")))
    new_root = tmp_path / "new"
    _dataset(new_root, ["new-a", "new-b", "new-c"])
    sessions = DatasetScanSessionStore(database, flush_size=1)
    session = sessions.create(DatasetScanRequest(dataset_id="stable", root_dir=new_root, layout="same_directory"))
    assert sessions.run(session.session_id).state == "succeeded"
    statements: list[str] = []
    repository._connection.set_trace_callback(statements.append)

    registered = sessions.register(session.session_id, repository, name="Materialized")
    repository._connection.set_trace_callback(None)
    retried = sessions.register(session.session_id, repository, name="Materialized")
    restored_session = sessions.get(session.session_id)

    assert registered.name == retried.name == "Materialized"
    assert registered.index_revision == retried.index_revision == 2
    assert repository.list_assets("stable", limit=100).total == 3
    assert [item.display_path for item in repository.list_assets("stable", limit=100).items] == [
        "new-a.png",
        "new-b.png",
        "new-c.png",
    ]
    assert restored_session.registration_name == "Materialized"
    assert restored_session.registered_dataset_id == "stable"
    assert restored_session.registered_index_revision == 2
    assert restored_session.registered_at is not None
    normalized_sql = [" ".join(statement.split()).upper() for statement in statements]
    assert any(statement.startswith("INSERT INTO ASSETS") and "SELECT" in statement for statement in normalized_sql)
    sessions.close()
    repository.close()


def test_materialize_failure_rolls_back_deleted_assets_revision_and_registration_state(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    repository = DatasetRepository(database)
    old_root = tmp_path / "old"
    _dataset(old_root, ["old"])
    repository.register(scan_dataset(DatasetScanRequest(dataset_id="stable", root_dir=old_root, layout="same_directory")))
    new_root = tmp_path / "new"
    _dataset(new_root, ["new"])
    sessions = DatasetScanSessionStore(database)
    session = sessions.create(DatasetScanRequest(dataset_id="stable", root_dir=new_root, layout="same_directory"))
    assert sessions.run(session.session_id).state == "succeeded"
    with repository._lock, repository._connection:
        repository._connection.executescript(
            """
            CREATE TRIGGER reject_new_scan_asset
            BEFORE INSERT ON assets
            WHEN NEW.dataset_id='stable' AND NEW.display_path LIKE 'new%'
            BEGIN
                SELECT RAISE(ABORT, 'fixture rejects materialized asset');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="fixture rejects materialized asset"):
        sessions.register(session.session_id, repository)

    current = repository.get_dataset("stable")
    current_assets = repository.list_assets("stable", limit=100)
    scan_state = sessions.get(session.session_id)
    assert current.index_revision == 1
    assert current_assets.total == 1
    assert current_assets.items[0].display_path == "old.png"
    assert scan_state.registered_at is None
    assert scan_state.registered_dataset_id is None
    sessions.close()
    repository.close()


def test_concurrent_run_uses_database_cas_and_only_one_scanner_writes(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "index.sqlite3"
    first_store = DatasetScanSessionStore(database, flush_size=1)
    second_store = DatasetScanSessionStore(database, flush_size=1)
    root = tmp_path / "dataset"
    _dataset(root, ["only"])
    session = first_store.create(DatasetScanRequest(dataset_id="cas", root_dir=root, layout="same_directory"))
    started = Event()
    proceed = Event()
    from labelone.datasets import scan_sessions as scan_session_module

    original_scan = scan_session_module.scan_dataset

    def slow_scan(*args, **kwargs):
        started.set()
        assert proceed.wait(timeout=2)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(scan_session_module, "scan_dataset", slow_scan)
    completed: list = []
    thread = Thread(target=lambda: completed.append(first_store.run(session.session_id)))
    thread.start()
    assert started.wait(timeout=2)

    with pytest.raises(InvalidPathError, match="current state") as conflict:
        second_store.run(session.session_id)

    assert conflict.value.details["state"] == "running"
    proceed.set()
    thread.join(timeout=3)
    assert len(completed) == 1 and completed[0].state == "succeeded"
    assert first_store.get(session.session_id).persisted_items == 1
    assert first_store.list_items(session.session_id, limit=10).total == 1
    first_store.close()
    second_store.close()


def test_running_scan_materializes_first_batch_then_appends_final_batch(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "index.sqlite3"
    repository = DatasetRepository(database)
    sessions = DatasetScanSessionStore(database, flush_size=1)
    root = tmp_path / "dataset"
    root.mkdir()
    created = sessions.create(DatasetScanRequest(dataset_id="progressive", root_dir=root, layout="same_directory"))
    first_batch_ready = Event()
    continue_scan = Event()
    from labelone.datasets import scan_sessions as scan_session_module

    def item(index: int):
        from labelone.datasets.models import AssetStatus, DatasetAsset

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
        del request, cancel_check
        for index in range(2):
            item_sink(item(index))
        first_batch_ready.set()
        assert continue_scan.wait(timeout=3)
        for index in range(2, 4):
            item_sink(item(index))
        from labelone.datasets.models import DatasetScanResult, DatasetScanSummary

        return DatasetScanResult(
            dataset_id="progressive",
            root_dir=root,
            image_root=root,
            annotation_roots=[root],
            items=[] if not collect_items else [item(index) for index in range(4)],
            summary=DatasetScanSummary(valid=4),
        )

    monkeypatch.setattr(scan_session_module, "scan_dataset", controlled_scan)
    thread = Thread(target=lambda: sessions.run(created.session_id))
    thread.start()
    assert first_batch_ready.wait(timeout=3)

    first = sessions.register(created.session_id, repository, name="Progressive")
    running = sessions.get(created.session_id)

    assert first.index_revision == 1
    assert repository.list_assets("progressive", limit=10).total == 2
    assert running.state == "running"
    assert running.registered_items == 2

    continue_scan.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    final = sessions.register(created.session_id, repository, name="Progressive")
    finished = sessions.get(created.session_id)

    assert final.index_revision == 2
    assert repository.list_assets("progressive", limit=10).total == 4
    assert finished.state == "succeeded"
    assert finished.registered_items == 4
    sessions.close()
    repository.close()


def test_atomic_registration_rejects_different_sqlite_databases(tmp_path: Path) -> None:
    first_repository = DatasetRepository(tmp_path / "one.sqlite3")
    sessions = DatasetScanSessionStore(tmp_path / "two.sqlite3")
    root = tmp_path / "dataset"
    _dataset(root, ["one"])
    session = sessions.create(DatasetScanRequest(dataset_id="different", root_dir=root, layout="same_directory"))
    assert sessions.run(session.session_id).state == "succeeded"

    with pytest.raises(InvalidPathError, match="share one SQLite database"):
        sessions.register(session.session_id, first_repository)

    sessions.close()
    first_repository.close()
