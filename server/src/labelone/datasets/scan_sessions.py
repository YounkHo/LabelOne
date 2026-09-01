from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4

from labelone.errors import InvalidPathError

from .models import (
    AssetStatus,
    DatasetAsset,
    DatasetScanItemPage,
    DatasetScanRequest,
    DatasetScanSession,
    DatasetScanSessionList,
    DatasetScanSummary,
)
from .scanner import DatasetScanInterrupted, resolve_scan_metadata, scan_dataset

if TYPE_CHECKING:
    from .repository import DatasetRepository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetScanSessionStore:
    """Persistent, incrementally queryable scan results without a jobs dependency."""

    def __init__(self, database_path: Path, *, flush_size: int = 32) -> None:
        if isinstance(flush_size, bool) or not isinstance(flush_size, int) or flush_size <= 0:
            raise ValueError("flush_size must be a positive integer")
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_size = flush_size
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_scan_sessions (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    dataset_id TEXT,
                    root_dir TEXT,
                    image_root TEXT,
                    annotation_roots_json TEXT NOT NULL DEFAULT '[]',
                    summary_json TEXT,
                    persisted_items INTEGER NOT NULL DEFAULT 0,
                    run_generation INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    registration_name TEXT,
                    registered_dataset_id TEXT,
                    registered_index_revision INTEGER,
                    registered_sequence INTEGER NOT NULL DEFAULT 0,
                    registered_at TEXT,
                    interrupted_at TEXT,
                    interruption_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_scan_session_items (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    asset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    selectable INTEGER NOT NULL,
                    display_path TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, sequence),
                    FOREIGN KEY(session_id) REFERENCES dataset_scan_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS dataset_scan_items_status_sequence
                    ON dataset_scan_session_items(session_id, status, sequence);
                CREATE INDEX IF NOT EXISTS dataset_scan_items_display
                    ON dataset_scan_session_items(session_id, selectable DESC, display_path, asset_id);
                """
            )
            columns = {
                row[1] for row in self._connection.execute("PRAGMA table_info(dataset_scan_sessions)").fetchall()
            }
            if "registration_name" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN registration_name TEXT")
            if "registered_dataset_id" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN registered_dataset_id TEXT")
            if "registered_index_revision" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN registered_index_revision INTEGER")
            if "registered_at" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN registered_at TEXT")
            if "registered_sequence" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN registered_sequence INTEGER NOT NULL DEFAULT 0")
            if "interrupted_at" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN interrupted_at TEXT")
            if "interruption_reason" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN interruption_reason TEXT")
            if "run_generation" not in columns:
                self._connection.execute("ALTER TABLE dataset_scan_sessions ADD COLUMN run_generation INTEGER NOT NULL DEFAULT 0")
            now = _now()
            self._connection.execute(
                """
                UPDATE dataset_scan_sessions
                SET state='interrupted', interrupted_at=?,
                    interruption_reason='Service restarted while scan was running', updated_at=?
                WHERE state='running'
                """,
                (now, now),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _session(row: sqlite3.Row) -> DatasetScanSession:
        return DatasetScanSession(
            session_id=str(row["session_id"]),
            state=str(row["state"]),
            request=DatasetScanRequest.model_validate_json(row["request_json"]),
            dataset_id=str(row["dataset_id"]) if row["dataset_id"] else None,
            root_dir=Path(row["root_dir"]) if row["root_dir"] else None,
            image_root=Path(row["image_root"]) if row["image_root"] else None,
            annotation_roots=[Path(path) for path in json.loads(row["annotation_roots_json"])],
            summary=DatasetScanSummary.model_validate_json(row["summary_json"]) if row["summary_json"] else None,
            persisted_items=int(row["persisted_items"]),
            run_generation=int(row["run_generation"]),
            error=str(row["error"]) if row["error"] else None,
            registration_name=str(row["registration_name"]) if row["registration_name"] else None,
            registered_dataset_id=str(row["registered_dataset_id"]) if row["registered_dataset_id"] else None,
            registered_index_revision=int(row["registered_index_revision"]) if row["registered_index_revision"] is not None else None,
            registered_items=int(row["registered_sequence"]),
            registered_at=str(row["registered_at"]) if row["registered_at"] else None,
            interrupted_at=str(row["interrupted_at"]) if row["interrupted_at"] else None,
            interruption_reason=str(row["interruption_reason"]) if row["interruption_reason"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def create(self, request: DatasetScanRequest) -> DatasetScanSession:
        session_id = uuid4().hex
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO dataset_scan_sessions(
                    session_id, state, request_json, dataset_id, created_at, updated_at
                ) VALUES(?, 'queued', ?, ?, ?, ?)
                """,
                (session_id, request.model_dump_json(), request.dataset_id, now, now),
            )
        return self.get(session_id)

    def get(self, session_id: str) -> DatasetScanSession:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise InvalidPathError("Unknown dataset scan session", details={"session_id": session_id})
        return self._session(row)

    def list(self, *, limit: int = 100) -> DatasetScanSessionList:
        safe_limit = max(1, min(limit, 500))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM dataset_scan_sessions ORDER BY created_at DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return DatasetScanSessionList(sessions=[self._session(row) for row in rows])

    def _flush(
        self,
        session_id: str,
        start_sequence: int,
        items: list[DatasetAsset],
        *,
        run_generation: int,
        allowed_states: frozenset[str] = frozenset({"running"}),
    ) -> int:
        if not items:
            return start_sequence
        now = _now()
        with self._lock, self._connection:
            state = self._connection.execute(
                "SELECT state, run_generation FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if (
                state is None
                or state["state"] not in allowed_states
                or int(state["run_generation"]) != run_generation
            ):
                raise InvalidPathError(
                    "Dataset scan session run is no longer current",
                    details={
                        "session_id": session_id,
                        "state": state["state"] if state else None,
                        "run_generation": int(state["run_generation"]) if state else None,
                        "expected_generation": run_generation,
                    },
                )
            self._connection.executemany(
                """
                INSERT INTO dataset_scan_session_items(
                    session_id, sequence, asset_id, status, selectable, display_path, payload_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        start_sequence + offset,
                        item.asset_id,
                        item.status.value,
                        int(item.selectable),
                        item.display_path,
                        item.model_dump_json(),
                    )
                    for offset, item in enumerate(items)
                ],
            )
            next_sequence = start_sequence + len(items)
            self._connection.execute(
                """
                UPDATE dataset_scan_sessions SET persisted_items=?, updated_at=?
                WHERE session_id=? AND run_generation=?
                """,
                (next_sequence, now, session_id, run_generation),
            )
        return next_sequence

    def run(self, session_id: str) -> DatasetScanSession:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE dataset_scan_sessions
                SET state='running', persisted_items=0, summary_json=NULL, error=NULL,
                    run_generation=run_generation+1,
                    registration_name=NULL, registered_dataset_id=NULL,
                    registered_index_revision=NULL, registered_sequence=0, registered_at=NULL,
                    interrupted_at=NULL, interruption_reason=NULL, updated_at=?
                WHERE session_id=? AND state IN ('queued','interrupted','failed')
                """,
                (_now(), session_id),
            )
            if cursor.rowcount != 1:
                row = self._connection.execute(
                    "SELECT state FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if row is None:
                    raise InvalidPathError("Unknown dataset scan session", details={"session_id": session_id})
                raise InvalidPathError(
                    "Dataset scan session cannot be run from its current state",
                    details={"session_id": session_id, "state": row["state"]},
                )
            self._connection.execute(
                "DELETE FROM dataset_scan_session_items WHERE session_id=?", (session_id,)
            )
            row = self._connection.execute(
                "SELECT request_json, run_generation FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            request = DatasetScanRequest.model_validate_json(row["request_json"])
            run_generation = int(row["run_generation"])

        buffer: list[DatasetAsset] = []
        sequence = 0

        def persist(item: DatasetAsset) -> None:
            nonlocal sequence
            buffer.append(item)
            if len(buffer) >= self.flush_size:
                sequence = self._flush(
                    session_id, sequence, buffer, run_generation=run_generation
                )
                buffer.clear()

        def interrupted() -> bool:
            with self._lock:
                row = self._connection.execute(
                    "SELECT state, run_generation FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
            return (
                row is None
                or row["state"] == "interrupted"
                or int(row["run_generation"]) != run_generation
            )

        def settle_interrupted() -> None:
            nonlocal sequence
            with self._lock:
                current = self._connection.execute(
                    "SELECT run_generation FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
            if current is None or int(current["run_generation"]) != run_generation:
                buffer.clear()
                return
            if buffer:
                sequence = self._flush(
                    session_id,
                    sequence,
                    buffer,
                    run_generation=run_generation,
                    allowed_states=frozenset({"running", "interrupted"}),
                )
                buffer.clear()
            now = _now()
            reason = f"Interrupted cooperatively after persisting {sequence} items"
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    UPDATE dataset_scan_sessions
                    SET state='interrupted', error=NULL, persisted_items=?,
                        interrupted_at=COALESCE(interrupted_at, ?), interruption_reason=?, updated_at=?
                    WHERE session_id=? AND run_generation=? AND state IN ('running','interrupted')
                    """,
                    (sequence, now, reason, now, session_id, run_generation),
                )

        try:
            metadata = resolve_scan_metadata(request)
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    UPDATE dataset_scan_sessions
                    SET dataset_id=?, root_dir=?, image_root=?, annotation_roots_json=?, updated_at=?
                    WHERE session_id=? AND state='running' AND run_generation=?
                    """,
                    (
                        metadata.dataset_id,
                        str(metadata.root_dir),
                        str(metadata.image_root),
                        json.dumps([str(path) for path in metadata.annotation_roots]),
                        _now(),
                        session_id,
                        run_generation,
                    ),
                )
            result = scan_dataset(
                request,
                item_sink=persist,
                collect_items=False,
                cancel_check=interrupted,
            )
            sequence = self._flush(
                session_id, sequence, buffer, run_generation=run_generation
            )
            buffer.clear()
            now = _now()
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    UPDATE dataset_scan_sessions
                    SET state='succeeded', dataset_id=?, root_dir=?, image_root=?,
                        annotation_roots_json=?, summary_json=?, persisted_items=?, updated_at=?
                    WHERE session_id=? AND state='running' AND run_generation=?
                    """,
                    (
                        result.dataset_id,
                        str(result.root_dir),
                        str(result.image_root),
                        json.dumps([str(path) for path in result.annotation_roots]),
                        result.summary.model_dump_json(),
                        sequence,
                        now,
                        session_id,
                        run_generation,
                    ),
                )
        except DatasetScanInterrupted:
            settle_interrupted()
        except Exception as exc:
            with self._lock:
                current = self._connection.execute(
                    "SELECT state, run_generation FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
            if (
                current is not None
                and current["state"] == "interrupted"
                and int(current["run_generation"]) == run_generation
            ):
                settle_interrupted()
            elif current is not None and int(current["run_generation"]) != run_generation:
                buffer.clear()
            else:
                if buffer:
                    try:
                        sequence = self._flush(
                            session_id, sequence, buffer, run_generation=run_generation
                        )
                    except Exception:
                        pass
                    buffer.clear()
                with self._lock, self._connection:
                    self._connection.execute(
                        """
                        UPDATE dataset_scan_sessions
                        SET state='failed', error=?, persisted_items=?, updated_at=?
                        WHERE session_id=? AND state='running' AND run_generation=?
                        """,
                        (str(exc), sequence, _now(), session_id, run_generation),
                    )
        return self.get(session_id)

    def interrupt(self, session_id: str) -> DatasetScanSession:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE dataset_scan_sessions
                SET state='interrupted', interrupted_at=?,
                    interruption_reason=CASE
                        WHEN state='queued' THEN 'Interrupted before scan started'
                        ELSE 'Interrupt requested while scan was running'
                    END,
                    error=NULL, updated_at=?
                WHERE session_id=? AND state IN ('queued','running')
                """,
                (now, now, session_id),
            )
            if cursor.rowcount != 1:
                row = self._connection.execute(
                    "SELECT state FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if row is None:
                    raise InvalidPathError("Unknown dataset scan session", details={"session_id": session_id})
                if row["state"] != "interrupted":
                    raise InvalidPathError(
                        "Only queued or running scan sessions can be interrupted",
                        details={"session_id": session_id, "state": row["state"]},
                    )
        return self.get(session_id)

    def register(
        self,
        session_id: str,
        repository: "DatasetRepository",
        *,
        name: str | None = None,
    ):
        if self.database_path != repository.database_path:
            raise InvalidPathError(
                "Atomic scan registration requires the scan session and dataset index to share one SQLite database",
                details={"scan_database": str(self.database_path), "dataset_database": str(repository.database_path)},
            )
        return repository.register_scan_session(session_id, name=name)

    def list_items(
        self,
        session_id: str,
        *,
        after_sequence: int = -1,
        limit: int = 200,
        status: AssetStatus | None = None,
    ) -> DatasetScanItemPage:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < -1:
            raise ValueError("after_sequence must be an integer greater than or equal to -1")
        safe_limit = max(1, min(limit, 1000))
        session = self.get(session_id)
        where = ["session_id=?", "sequence>?"]
        parameters: list[object] = [session_id, after_sequence]
        count_where = ["session_id=?"]
        count_parameters: list[object] = [session_id]
        if status is not None:
            where.append("status=?")
            parameters.append(status.value)
            count_where.append("status=?")
            count_parameters.append(status.value)
        with self._lock:
            total = int(self._connection.execute(
                f"SELECT COUNT(*) FROM dataset_scan_session_items WHERE {' AND '.join(count_where)}",
                count_parameters,
            ).fetchone()[0])
            rows = self._connection.execute(
                f"""
                SELECT sequence, payload_json
                FROM dataset_scan_session_items
                WHERE {' AND '.join(where)}
                ORDER BY sequence
                LIMIT ?
                """,
                [*parameters, safe_limit + 1],
            ).fetchall()
        has_more = len(rows) > safe_limit
        visible = rows[:safe_limit]
        next_after = int(visible[-1]["sequence"]) if has_more and visible else None
        return DatasetScanItemPage(
            items=[DatasetAsset.model_validate_json(row["payload_json"]) for row in visible],
            total=total,
            next_after=next_after,
            state=session.state,
        )
