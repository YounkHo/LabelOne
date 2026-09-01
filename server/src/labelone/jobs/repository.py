from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4

from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError, RevisionConflictError

from .models import (
    BatchJobRequest,
    JobEvent,
    JobEventType,
    JobItem,
    JobItemListResponse,
    JobListResponse,
    JobRecord,
    JobState,
    is_terminal_job_state,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRepository:
    def __init__(self, database_path: Path, datasets: DatasetRepository) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.datasets = datasets
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    total INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    desired_state TEXT NOT NULL DEFAULT 'run',
                    generation INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS job_items (
                    job_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    claim_token TEXT,
                    run_generation INTEGER,
                    progress_json TEXT,
                    PRIMARY KEY(job_id, asset_id),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS job_items_state_position
                    ON job_items(job_id, state, position);
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS job_events_job_event_id
                    ON job_events(job_id, event_id);
                """
            )
            job_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(jobs)").fetchall()}
            if "desired_state" not in job_columns:
                self._connection.execute("ALTER TABLE jobs ADD COLUMN desired_state TEXT NOT NULL DEFAULT 'run'")
            if "generation" not in job_columns:
                self._connection.execute("ALTER TABLE jobs ADD COLUMN generation INTEGER NOT NULL DEFAULT 0")
            item_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(job_items)").fetchall()}
            if "claim_token" not in item_columns:
                self._connection.execute("ALTER TABLE job_items ADD COLUMN claim_token TEXT")
            if "run_generation" not in item_columns:
                self._connection.execute("ALTER TABLE job_items ADD COLUMN run_generation INTEGER")
            if "progress_json" not in item_columns:
                self._connection.execute("ALTER TABLE job_items ADD COLUMN progress_json TEXT")
            self._connection.execute(
                "UPDATE jobs SET desired_state='cancel' WHERE state='canceling'"
            )
            self._connection.execute(
                "UPDATE jobs SET desired_state='pause' WHERE state IN ('pausing','paused')"
            )
            self._recover_startup(_now())

    def _insert_event(
        self,
        job_id: str,
        event_type: JobEventType,
        payload: dict[str, object],
        *,
        created_at: str | None = None,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO job_events(job_id, event_type, created_at, payload_json) VALUES(?, ?, ?, ?)",
            (
                job_id,
                event_type,
                created_at or _now(),
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ),
        )
        return int(cursor.lastrowid)

    def _progress_payload(self, job_id: str) -> dict[str, object]:
        job = self._connection.execute(
            "SELECT total FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if job is None:
            raise InvalidPathError("Unknown job", details={"job_id": job_id})
        rows = self._connection.execute(
            "SELECT state, COUNT(*) AS count FROM job_items WHERE job_id=? GROUP BY state",
            (job_id,),
        ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        total = int(job["total"])
        finished = counts.get("succeeded", 0) + counts.get("failed", 0) + counts.get("canceled", 0)
        return {
            "total": total,
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "succeeded": counts.get("succeeded", 0),
            "failed": counts.get("failed", 0),
            "canceled": counts.get("canceled", 0),
            "finished": finished,
            "progress": finished / total if total else 1.0,
        }

    def _emit_progress(self, job_id: str, *, created_at: str | None = None) -> int:
        return self._insert_event(
            job_id,
            "job.progress",
            self._progress_payload(job_id),
            created_at=created_at,
        )

    def _emit_job_state(
        self,
        job_id: str,
        previous_state: str,
        state: str,
        *,
        reason: str,
        created_at: str | None = None,
        error: str | None = None,
    ) -> int:
        row = self._connection.execute(
            "SELECT desired_state, generation FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        payload: dict[str, object] = {
            "previous_state": previous_state,
            "state": state,
            "reason": reason,
            "desired_state": str(row["desired_state"]) if row else None,
            "generation": int(row["generation"]) if row else None,
        }
        if error is not None:
            payload["error"] = error
        return self._insert_event(job_id, "job.state", payload, created_at=created_at)

    def _emit_item_state(
        self,
        job_id: str,
        asset_id: str,
        previous_state: str,
        state: str,
        *,
        reason: str,
        created_at: str | None = None,
        error: str | None = None,
        has_result: bool = False,
    ) -> int:
        row = self._connection.execute(
            "SELECT attempts, run_generation FROM job_items WHERE job_id=? AND asset_id=?",
            (job_id, asset_id),
        ).fetchone()
        payload: dict[str, object] = {
            "asset_id": asset_id,
            "previous_state": previous_state,
            "state": state,
            "reason": reason,
            "attempts": int(row["attempts"]) if row else None,
            "run_generation": int(row["run_generation"]) if row and row["run_generation"] is not None else None,
            "has_result": has_result,
        }
        if error is not None:
            payload["error"] = error
        return self._insert_event(job_id, "item.state", payload, created_at=created_at)

    def _emit_terminal(self, job_id: str, state: str, *, created_at: str | None = None) -> int | None:
        if not is_terminal_job_state(state):
            return None
        return self._insert_event(
            job_id,
            "job.terminal",
            {"state": state, **self._progress_payload(job_id)},
            created_at=created_at,
        )

    def _recover_startup(self, now: str) -> None:
        rows = self._connection.execute(
            """
            SELECT job_id, state, desired_state
            FROM jobs
            WHERE (desired_state='cancel' AND state NOT IN ('succeeded','succeeded_with_errors','failed','canceled'))
               OR (desired_state='pause' AND (
                    state='pausing'
                    OR (state='paused' AND EXISTS (
                        SELECT 1 FROM job_items WHERE job_items.job_id=jobs.job_id AND job_items.state='running'
                    ))
               ))
               OR (desired_state='run' AND state='running')
            ORDER BY created_at
            """
        ).fetchall()
        for row in rows:
            job_id = str(row["job_id"])
            previous_state = str(row["state"])
            desired_state = str(row["desired_state"])
            running_items = self._connection.execute(
                "SELECT asset_id FROM job_items WHERE job_id=? AND state='running' ORDER BY position",
                (job_id,),
            ).fetchall()
            if desired_state == "cancel":
                changed_items = self._connection.execute(
                    "SELECT asset_id, state FROM job_items WHERE job_id=? AND state IN ('queued','running') ORDER BY position",
                    (job_id,),
                ).fetchall()
                self._connection.execute(
                    """
                    UPDATE job_items
                    SET state='canceled', finished_at=?, claim_token=NULL, run_generation=NULL
                    WHERE job_id=? AND state IN ('queued','running')
                    """,
                    (now, job_id),
                )
                self._connection.execute(
                    "UPDATE jobs SET state='canceled', updated_at=? WHERE job_id=?",
                    (now, job_id),
                )
                for item in changed_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        str(item["state"]),
                        "canceled",
                        reason="startup_recovery",
                        created_at=now,
                    )
                self._emit_job_state(job_id, previous_state, "canceled", reason="startup_recovery", created_at=now)
                recovered_state = "canceled"
            elif desired_state == "pause":
                self._connection.execute(
                    """
                    UPDATE job_items
                    SET state='queued', started_at=NULL, claim_token=NULL, run_generation=NULL
                    WHERE job_id=? AND state='running'
                    """,
                    (job_id,),
                )
                if previous_state == "pausing":
                    self._connection.execute(
                        "UPDATE jobs SET state='paused', updated_at=?, generation=generation+1 WHERE job_id=?",
                        (now, job_id),
                    )
                    self._emit_job_state(job_id, previous_state, "paused", reason="startup_recovery", created_at=now)
                for item in running_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        "running",
                        "queued",
                        reason="startup_recovery",
                        created_at=now,
                    )
                recovered_state = "paused"
            else:
                self._connection.execute(
                    """
                    UPDATE job_items
                    SET state='queued', started_at=NULL, claim_token=NULL, run_generation=NULL
                    WHERE job_id=? AND state='running'
                    """,
                    (job_id,),
                )
                self._connection.execute(
                    "UPDATE jobs SET state='interrupted', updated_at=?, generation=generation+1 WHERE job_id=?",
                    (now, job_id),
                )
                for item in running_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        "running",
                        "queued",
                        reason="startup_recovery",
                        created_at=now,
                    )
                self._emit_job_state(job_id, previous_state, "interrupted", reason="startup_recovery", created_at=now)
                recovered_state = "interrupted"
            self._insert_event(
                job_id,
                "job.recovered",
                {
                    "previous_state": previous_state,
                    "state": recovered_state,
                    "desired_state": desired_state,
                    "requeued_items": len(running_items),
                },
                created_at=now,
            )
            self._emit_progress(job_id, created_at=now)
            self._emit_terminal(job_id, recovered_state, created_at=now)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create(self, request: BatchJobRequest, *, idempotency_key: str | None = None) -> JobRecord:
        request_payload = request.model_dump(mode="json")
        if not request.preferred_asset_ids:
            request_payload.pop("preferred_asset_ids", None)
        if request.source_category is None and request.target_category is None:
            request_payload.pop("source_category", None)
            request_payload.pop("target_category", None)
        if request.pipeline_context is None:
            request_payload.pop("pipeline_context", None)
        request_json = json.dumps(request_payload, sort_keys=True, separators=(",", ":"))
        request_hash = sha256(request_json.encode()).hexdigest()
        compatible_request_hashes = {request_hash}
        if request.pipeline_context is not None:
            legacy_payload = dict(request_payload)
            legacy_payload.pop("pipeline_context", None)
            compatible_request_hashes.add(sha256(
                json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest())
        if idempotency_key:
            with self._lock:
                existing = self._connection.execute(
                    "SELECT job_id, request_hash FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing:
                    if existing["request_hash"] not in compatible_request_hashes:
                        raise RevisionConflictError("Idempotency key was reused with a different job request")
                    return self.get(str(existing["job_id"]), include_items=False)
        if request.kind == "model_download":
            work_item_ids = [f"weight:{index}" for index in request.weight_url_indices]
            normalized = request
        elif request.kind == "category_rename":
            assert request.source_category is not None
            work_item_ids = self.datasets.selectable_asset_ids_for_category(request.dataset_id, request.source_category)
            if not work_item_ids:
                raise InvalidPathError(
                    "Dataset category rename has no matching assets",
                    details={"dataset_id": request.dataset_id, "source_category": request.source_category},
                )
            normalized = request
        else:
            work_item_ids = list(request.asset_ids) or self.datasets.selectable_asset_ids(request.dataset_id)
            if not work_item_ids:
                raise InvalidPathError("Batch job has no selectable assets")
            work_item_set = set(work_item_ids)
            unknown_preferred = [
                asset_id for asset_id in request.preferred_asset_ids
                if asset_id not in work_item_set
            ]
            if unknown_preferred:
                raise InvalidPathError(
                    "Preferred assets must belong to the batch job scope",
                    details={"asset_ids": unknown_preferred},
                )
            for asset_id in work_item_ids:
                self.datasets.get_asset(request.dataset_id, asset_id, require_selectable=True)
            if request.preferred_asset_ids:
                preferred = set(request.preferred_asset_ids)
                work_item_ids = [
                    *request.preferred_asset_ids,
                    *(asset_id for asset_id in work_item_ids if asset_id not in preferred),
                ]
            # The job_items table is the immutable execution snapshot. Keeping
            # an implicit "all assets" request compact avoids serializing the
            # complete dataset into jobs.request_json and every API response.
            normalized = request
        with self._lock, self._connection:
            if idempotency_key:
                existing = self._connection.execute(
                    "SELECT job_id, request_hash FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing:
                    if existing["request_hash"] not in compatible_request_hashes:
                        raise RevisionConflictError("Idempotency key was reused with a different job request")
                    return self.get(str(existing["job_id"]), include_items=False)
            job_id = uuid4().hex
            now = _now()
            self._connection.execute(
                """
                INSERT INTO jobs(
                    job_id, kind, dataset_id, state, request_json, request_hash,
                    idempotency_key, total, created_at, updated_at, error,
                    desired_state, generation
                ) VALUES(?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, NULL, 'run', 0)
                """,
                (job_id, normalized.kind, normalized.dataset_id, request_json, request_hash, idempotency_key, len(work_item_ids), now, now),
            )
            self._connection.executemany(
                "INSERT INTO job_items(job_id, asset_id, position, state, attempts) VALUES(?, ?, ?, 'queued', 0)",
                [(job_id, asset_id, position) for position, asset_id in enumerate(work_item_ids)],
            )
            self._insert_event(
                job_id,
                "job.created",
                {
                    "kind": normalized.kind,
                    "dataset_id": normalized.dataset_id,
                    "total": len(work_item_ids),
                    "priority": normalized.priority,
                },
                created_at=now,
            )
            self._emit_job_state(job_id, "none", "queued", reason="created", created_at=now)
            self._emit_progress(job_id, created_at=now)
        return self.get(job_id, include_items=False)

    def _record(self, row: sqlite3.Row, *, include_items: bool) -> JobRecord:
        counts = self._connection.execute(
            "SELECT state, COUNT(*) AS count FROM job_items WHERE job_id = ? GROUP BY state", (row["job_id"],)
        ).fetchall()
        totals = {item["state"]: int(item["count"]) for item in counts}
        items: list[JobItem] = []
        if include_items:
            item_rows = self._connection.execute(
                "SELECT * FROM job_items WHERE job_id = ? ORDER BY position", (row["job_id"],)
            ).fetchall()
            items = [JobItem(
                asset_id=item["asset_id"], position=item["position"], state=item["state"], attempts=item["attempts"],
                result=json.loads(item["result_json"]) if item["result_json"] else None,
                error=item["error"], started_at=item["started_at"], finished_at=item["finished_at"],
                progress=json.loads(item["progress_json"]) if item["progress_json"] else None,
            ) for item in item_rows]
        return JobRecord(
            job_id=row["job_id"], kind=row["kind"], dataset_id=row["dataset_id"], state=row["state"],
            desired_state=row["desired_state"], generation=row["generation"],
            request=BatchJobRequest.model_validate_json(row["request_json"]), total=row["total"],
            completed=totals.get("succeeded", 0), failed=totals.get("failed", 0), canceled=totals.get("canceled", 0),
            created_at=row["created_at"], updated_at=row["updated_at"], error=row["error"], items=items,
        )

    def get(self, job_id: str, *, include_items: bool = True) -> JobRecord:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise InvalidPathError("Unknown job", details={"job_id": job_id})
            return self._record(row, include_items=include_items)

    def list(self, limit: int = 100) -> JobListResponse:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
            return JobListResponse(jobs=[self._record(row, include_items=False) for row in rows])

    def find_pipeline_precompute(
        self,
        dataset_id: str,
        signature: str,
        *,
        full_dataset_only: bool = False,
    ) -> JobRecord | None:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM jobs
                WHERE dataset_id=? AND kind='pipeline'
                  AND state NOT IN ('failed','canceled','canceling','pausing')
                ORDER BY created_at DESC
                """,
                (dataset_id,),
            ).fetchall()
            for row in rows:
                request = BatchJobRequest.model_validate_json(row["request_json"])
                context = request.pipeline_context
                if (
                    context is not None
                    and context.signature == signature
                    and (not full_dataset_only or not request.asset_ids)
                ):
                    return self._record(row, include_items=False)
        return None

    def superseded_background_pipeline_job_ids(
        self,
        dataset_id: str,
        keep_signature: str,
    ) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT job_id, request_json FROM jobs
                WHERE dataset_id=? AND kind='pipeline'
                  AND state IN ('queued','running','pausing','paused','interrupted')
                ORDER BY created_at
                """,
                (dataset_id,),
            ).fetchall()
        superseded: list[str] = []
        for row in rows:
            request = BatchJobRequest.model_validate_json(row["request_json"])
            context = request.pipeline_context
            if (
                request.priority == "background"
                and request.output_policy.mode == "preview"
                and not request.asset_ids
                and context is not None
                and context.signature != keep_signature
            ):
                superseded.append(str(row["job_id"]))
        return superseded

    def has_active_dataset_jobs(self, dataset_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE dataset_id=?
                  AND state NOT IN ('succeeded','succeeded_with_errors','failed','canceled')
                LIMIT 1
                """,
                (dataset_id,),
            ).fetchone()
        return row is not None

    def list_events(self, job_id: str, *, after: int = 0, limit: int = 200) -> list[JobEvent]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a non-negative integer event id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        with self._lock:
            exists = self._connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if exists is None:
                raise InvalidPathError("Unknown job", details={"job_id": job_id})
            rows = self._connection.execute(
                """
                SELECT event_id, job_id, event_type, created_at, payload_json
                FROM job_events
                WHERE job_id=? AND event_id>?
                ORDER BY event_id
                LIMIT ?
                """,
                (job_id, after, limit),
            ).fetchall()
        return [
            JobEvent(
                event_id=int(row["event_id"]),
                job_id=str(row["job_id"]),
                event_type=str(row["event_type"]),
                created_at=str(row["created_at"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    def report_job_phase(self, job_id: str, phase: str) -> bool:
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a non-empty string")
        now = _now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT state, desired_state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise InvalidPathError("Unknown job", details={"job_id": job_id})
            if str(row["state"]) != "running" or str(row["desired_state"]) != "run":
                return False
            self._insert_event(
                job_id,
                "job.progress",
                {**self._progress_payload(job_id), "phase": phase.strip()},
                created_at=now,
            )
            return True

    def latest_event_id(self, job_id: str | None = None) -> int:
        with self._lock:
            if job_id is not None:
                exists = self._connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if exists is None:
                    raise InvalidPathError("Unknown job", details={"job_id": job_id})
                row = self._connection.execute(
                    "SELECT MAX(event_id) FROM job_events WHERE job_id=?", (job_id,)
                ).fetchone()
            else:
                row = self._connection.execute("SELECT MAX(event_id) FROM job_events").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def is_terminal(self, job_id: str) -> bool:
        with self._lock:
            row = self._connection.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise InvalidPathError("Unknown job", details={"job_id": job_id})
        return is_terminal_job_state(str(row["state"]))

    def list_items(self, job_id: str, *, offset: int = 0, limit: int = 200, state: str | None = None) -> JobItemListResponse:
        self.get(job_id, include_items=False)
        where = "job_id = ?" + (" AND state = ?" if state else "")
        parameters: list[object] = [job_id]
        if state:
            parameters.append(state)
        with self._lock:
            total = int(self._connection.execute(f"SELECT COUNT(*) FROM job_items WHERE {where}", parameters).fetchone()[0])
            rows = self._connection.execute(
                f"SELECT * FROM job_items WHERE {where} ORDER BY position LIMIT ? OFFSET ?",
                [*parameters, max(1, min(limit, 1000)), max(0, offset)],
            ).fetchall()
        items = [JobItem(
            asset_id=row["asset_id"], position=row["position"], state=row["state"], attempts=row["attempts"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"], started_at=row["started_at"], finished_at=row["finished_at"],
            progress=json.loads(row["progress_json"]) if row["progress_json"] else None,
        ) for row in rows]
        next_offset = offset + len(items) if offset + len(items) < total else None
        return JobItemListResponse(items=items, total=total, next_offset=next_offset)

    def lookup_items(self, job_id: str, asset_ids: list[str]) -> JobItemListResponse:
        self.get(job_id, include_items=False)
        if not 1 <= len(asset_ids) <= 200 or len(set(asset_ids)) != len(asset_ids) or any(not asset_id for asset_id in asset_ids):
            raise ValueError("asset_ids must contain 1 to 200 unique non-empty values")
        placeholders = ",".join("?" for _ in asset_ids)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM job_items WHERE job_id=? AND asset_id IN ({placeholders}) ORDER BY position",
                [job_id, *asset_ids],
            ).fetchall()
        items = [JobItem(
            asset_id=row["asset_id"], position=row["position"], state=row["state"], attempts=row["attempts"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"], started_at=row["started_at"], finished_at=row["finished_at"],
            progress=json.loads(row["progress_json"]) if row["progress_json"] else None,
        ) for row in rows]
        return JobItemListResponse(items=items, total=len(items), next_offset=None)

    def iter_item_results(
        self,
        job_id: str,
        *,
        state: str = "succeeded",
        page_size: int = 1000,
    ):
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 5000:
            raise ValueError("page_size must be an integer between 1 and 5000")
        self.get(job_id, include_items=False)
        last_asset_id = ""
        while True:
            with self._lock:
                rows = self._connection.execute(
                    """
                    SELECT asset_id, result_json
                    FROM job_items
                    WHERE job_id=? AND state=? AND asset_id>?
                    ORDER BY asset_id
                    LIMIT ?
                    """,
                    (job_id, state, last_asset_id, page_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                asset_id = str(row["asset_id"])
                if not row["result_json"]:
                    raise InvalidPathError(
                        "Completed job item has no result payload",
                        details={"job_id": job_id, "asset_id": asset_id},
                    )
                payload = json.loads(row["result_json"])
                if not isinstance(payload, dict):
                    raise InvalidPathError(
                        "Completed job item result must be an object",
                        details={"job_id": job_id, "asset_id": asset_id},
                    )
                yield payload
                last_asset_id = asset_id

    def set_state(self, job_id: str, state: JobState, *, error: str | None = None) -> None:
        now = _now()
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if previous is None:
                raise InvalidPathError("Unknown job", details={"job_id": job_id})
            cursor = self._connection.execute(
                "UPDATE jobs SET state = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (state, error, now, job_id),
            )
            if cursor.rowcount != 1:
                raise InvalidPathError("Unknown job", details={"job_id": job_id})
            self._emit_job_state(
                job_id,
                str(previous["state"]),
                state,
                reason="set_state",
                created_at=now,
                error=error,
            )
            if is_terminal_job_state(state):
                self._emit_progress(job_id, created_at=now)
                self._emit_terminal(job_id, state, created_at=now)

    def resumable_job_ids(self) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT job_id FROM jobs WHERE desired_state='run' AND state IN ('queued','interrupted') ORDER BY created_at"
            ).fetchall()
        return [str(row["job_id"]) for row in rows]

    def transition_to_running(self, job_id: str) -> bool:
        now = _now()
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            cursor = self._connection.execute(
                """
                UPDATE jobs SET state='running', error=NULL, updated_at=?
                WHERE job_id=? AND desired_state='run' AND state IN ('queued','interrupted')
                """,
                (now, job_id),
            )
            if cursor.rowcount == 1 and previous is not None:
                self._emit_job_state(
                    job_id, str(previous["state"]), "running", reason="transition_to_running", created_at=now
                )
        return cursor.rowcount == 1

    def request_pause(self, job_id: str) -> None:
        now = _now()
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET desired_state='pause',
                    state=CASE WHEN state='queued' THEN 'paused' ELSE 'pausing' END,
                    updated_at=?
                WHERE job_id=? AND state IN ('queued','running')
                """,
                (now, job_id),
            )
            if cursor.rowcount != 1 or previous is None:
                raise InvalidPathError("Only queued/running jobs can be paused")
            current = self._connection.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            self._emit_job_state(
                job_id,
                str(previous["state"]),
                str(current["state"]),
                reason="request_pause",
                created_at=now,
            )

    def request_resume(self, job_id: str) -> None:
        now = _now()
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET desired_state='run', state='queued', error=NULL,
                    generation=generation+1, updated_at=?
                WHERE job_id=? AND state IN ('paused','interrupted','failed','succeeded_with_errors')
                """,
                (now, job_id),
            )
            if cursor.rowcount != 1 or previous is None:
                raise InvalidPathError("Job cannot be resumed")
            self._emit_job_state(
                job_id, str(previous["state"]), "queued", reason="request_resume", created_at=now
            )

    def request_cancel(self, job_id: str) -> None:
        now = _now()
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET desired_state='cancel', state='canceling', generation=generation+1, updated_at=?
                WHERE job_id=? AND state NOT IN ('succeeded','succeeded_with_errors','failed','canceled')
                """,
                (now, job_id),
            )
            if cursor.rowcount == 1:
                queued_items = self._connection.execute(
                    "SELECT asset_id FROM job_items WHERE job_id=? AND state='queued' ORDER BY position",
                    (job_id,),
                ).fetchall()
                self._connection.execute(
                    "UPDATE job_items SET state='canceled', finished_at=? WHERE job_id=? AND state='queued'",
                    (now, job_id),
                )
                if previous is not None:
                    self._emit_job_state(
                        job_id,
                        str(previous["state"]),
                        "canceling",
                        reason="request_cancel",
                        created_at=now,
                    )
                for item in queued_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        "queued",
                        "canceled",
                        reason="request_cancel",
                        created_at=now,
                    )
                self._emit_progress(job_id, created_at=now)
                running = int(self._connection.execute(
                    "SELECT COUNT(*) FROM job_items WHERE job_id=? AND state='running'", (job_id,)
                ).fetchone()[0])
                if running == 0:
                    self._connection.execute(
                        "UPDATE jobs SET state='canceled', updated_at=? WHERE job_id=?", (now, job_id)
                    )
                    self._emit_job_state(
                        job_id, "canceling", "canceled", reason="request_cancel_settled", created_at=now
                    )
                    self._emit_terminal(job_id, "canceled", created_at=now)

    def settle_pause(self, job_id: str) -> bool:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE jobs SET state='paused', updated_at=?
                WHERE job_id=? AND state='pausing' AND desired_state='pause'
                  AND NOT EXISTS (
                    SELECT 1 FROM job_items WHERE job_id=? AND state='running'
                  )
                """,
                (now, job_id, job_id),
            )
            if cursor.rowcount == 1:
                self._emit_job_state(job_id, "pausing", "paused", reason="settle_pause", created_at=now)
        return cursor.rowcount == 1

    def settle_cancel(self, job_id: str) -> bool:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE jobs SET state='canceled', updated_at=?
                WHERE job_id=? AND state='canceling' AND desired_state='cancel'
                  AND NOT EXISTS (
                    SELECT 1 FROM job_items WHERE job_id=? AND state='running'
                  )
                """,
                (now, job_id, job_id),
            )
            if cursor.rowcount == 1:
                self._emit_job_state(job_id, "canceling", "canceled", reason="settle_cancel", created_at=now)
                self._emit_progress(job_id, created_at=now)
                self._emit_terminal(job_id, "canceled", created_at=now)
        return cursor.rowcount == 1

    def settle_completed(self, job_id: str) -> bool:
        now = _now()
        with self._lock, self._connection:
            failed = int(self._connection.execute(
                "SELECT COUNT(*) FROM job_items WHERE job_id=? AND state='failed'", (job_id,)
            ).fetchone()[0])
            cursor = self._connection.execute(
                """
                UPDATE jobs SET state=?, updated_at=?
                WHERE job_id=? AND state='running' AND desired_state='run'
                  AND NOT EXISTS (
                    SELECT 1 FROM job_items WHERE job_id=? AND state IN ('queued','running')
                  )
                """,
                ("succeeded_with_errors" if failed else "succeeded", now, job_id, job_id),
            )
            if cursor.rowcount == 1:
                state = "succeeded_with_errors" if failed else "succeeded"
                self._emit_job_state(job_id, "running", state, reason="settle_completed", created_at=now)
                self._emit_progress(job_id, created_at=now)
                self._emit_terminal(job_id, state, created_at=now)
        return cursor.rowcount == 1

    def queued_items(self, job_id: str, limit: int) -> list[JobItem]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM job_items WHERE job_id = ? AND state = 'queued' ORDER BY position LIMIT ?",
                (job_id, limit),
            ).fetchall()
        return [JobItem(
            asset_id=row["asset_id"], position=row["position"], state=row["state"], attempts=row["attempts"],
            result=None, error=row["error"], started_at=row["started_at"], finished_at=row["finished_at"],
            progress=json.loads(row["progress_json"]) if row["progress_json"] else None,
        ) for row in rows]

    def prioritize_queued_items(self, job_id: str, asset_ids: list[str]) -> JobRecord:
        if not asset_ids or len(asset_ids) > 64 or len(set(asset_ids)) != len(asset_ids) or any(not item for item in asset_ids):
            raise ValueError("Priority asset IDs must contain 1 to 64 unique non-empty values")
        with self._lock, self._connection:
            job = self._connection.execute(
                "SELECT kind, state, desired_state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise InvalidPathError("Unknown job", details={"job_id": job_id})
            if job["kind"] not in {"pipeline", "inference"}:
                raise InvalidPathError("Only image processing jobs support nearby-item priority")
            if job["state"] not in {"queued", "running"} or job["desired_state"] != "run":
                raise InvalidPathError("Only active jobs can reprioritize queued items")
            placeholders = ",".join("?" for _ in asset_ids)
            rows = self._connection.execute(
                f"SELECT asset_id, state FROM job_items WHERE job_id=? AND asset_id IN ({placeholders})",
                [job_id, *asset_ids],
            ).fetchall()
            known = {str(row["asset_id"]): str(row["state"]) for row in rows}
            unknown = [asset_id for asset_id in asset_ids if asset_id not in known]
            if unknown:
                raise InvalidPathError(
                    "Priority assets do not belong to this job",
                    details={"job_id": job_id, "asset_ids": unknown},
                )
            queued = [asset_id for asset_id in asset_ids if known[asset_id] == "queued"]
            if queued:
                minimum = int(self._connection.execute(
                    "SELECT COALESCE(MIN(position), 0) FROM job_items WHERE job_id=? AND state='queued'",
                    (job_id,),
                ).fetchone()[0])
                for offset, asset_id in enumerate(queued):
                    self._connection.execute(
                        "UPDATE job_items SET position=? WHERE job_id=? AND asset_id=? AND state='queued'",
                        (minimum - len(queued) + offset, job_id, asset_id),
                    )
                self._connection.execute(
                    "UPDATE jobs SET updated_at=? WHERE job_id=?", (_now(), job_id)
                )
        return self.get(job_id, include_items=False)

    def mark_item_running(self, job_id: str, asset_id: str) -> str | None:
        claim_token = uuid4().hex
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE job_items
                SET state='running', attempts=attempts+1, started_at=?, finished_at=NULL,
                    error=NULL, progress_json=NULL, claim_token=?,
                    run_generation=(SELECT generation FROM jobs WHERE job_id=?)
                WHERE job_id=? AND asset_id=? AND state='queued'
                  AND EXISTS (
                    SELECT 1 FROM jobs
                    WHERE job_id=? AND state='running' AND desired_state='run'
                  )
                """,
                (now, claim_token, job_id, job_id, asset_id, job_id),
            )
            if cursor.rowcount == 1:
                self._emit_item_state(
                    job_id, asset_id, "queued", "running", reason="claimed", created_at=now
                )
                self._emit_progress(job_id, created_at=now)
        return claim_token if cursor.rowcount == 1 else None

    def update_item_progress(
        self,
        job_id: str,
        asset_id: str,
        *,
        claim_token: str,
        payload: dict[str, object],
    ) -> bool:
        now = _now()
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT item.run_generation, job.generation, job.desired_state
                FROM job_items AS item
                JOIN jobs AS job ON job.job_id=item.job_id
                WHERE item.job_id=? AND item.asset_id=?
                  AND item.state='running' AND item.claim_token=?
                """,
                (job_id, asset_id, claim_token),
            ).fetchone()
            if (
                row is None
                or str(row["desired_state"]) != "run"
                or int(row["run_generation"]) != int(row["generation"])
            ):
                return False
            self._insert_event(
                job_id,
                "item.progress",
                {"asset_id": asset_id, **payload},
                created_at=now,
            )
            self._connection.execute(
                "UPDATE job_items SET progress_json=? WHERE job_id=? AND asset_id=?",
                (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), job_id, asset_id),
            )
            return True

    def finish_item(
        self,
        job_id: str,
        asset_id: str,
        *,
        claim_token: str,
        state: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> bool:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE job_items
                SET state=?, result_json=?, error=?, finished_at=?, claim_token=NULL
                WHERE job_id=? AND asset_id=? AND state='running' AND claim_token=?
                  AND EXISTS (
                    SELECT 1 FROM jobs
                    WHERE job_id=? AND desired_state!='cancel'
                      AND generation=job_items.run_generation
                  )
                """,
                (state, json.dumps(result) if result is not None else None, error, now, job_id, asset_id, claim_token, job_id),
            )
            if cursor.rowcount == 1:
                self._emit_item_state(
                    job_id,
                    asset_id,
                    "running",
                    state,
                    reason="finished",
                    created_at=now,
                    error=error,
                    has_result=result is not None,
                )
                self._emit_progress(job_id, created_at=now)
        return cursor.rowcount == 1

    def finish_committed_item(
        self,
        job_id: str,
        asset_id: str,
        *,
        claim_token: str,
        result: dict,
    ) -> bool:
        """Record an irreversible mutation even if pause/cancel arrived after its claim."""
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE job_items
                SET state='succeeded', result_json=?, error=NULL, finished_at=?, claim_token=NULL
                WHERE job_id=? AND asset_id=? AND state='running' AND claim_token=?
                """,
                (json.dumps(result), now, job_id, asset_id, claim_token),
            )
            if cursor.rowcount == 1:
                self._emit_item_state(
                    job_id,
                    asset_id,
                    "running",
                    "succeeded",
                    reason="committed_mutation_finished",
                    created_at=now,
                    has_result=True,
                )
                self._emit_progress(job_id, created_at=now)
        return cursor.rowcount == 1

    def cancel_running_item(self, job_id: str, asset_id: str, *, claim_token: str) -> bool:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE job_items
                SET state='canceled', result_json=NULL, error=NULL, finished_at=?, claim_token=NULL
                WHERE job_id=? AND asset_id=? AND state='running' AND claim_token=?
                """,
                (now, job_id, asset_id, claim_token),
            )
            if cursor.rowcount == 1:
                self._emit_item_state(
                    job_id, asset_id, "running", "canceled", reason="cancel_running", created_at=now
                )
                self._emit_progress(job_id, created_at=now)
        return cursor.rowcount == 1

    def requeue_running_item(self, job_id: str, asset_id: str, *, claim_token: str) -> bool:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE job_items
                SET state='queued', started_at=NULL, finished_at=NULL,
                    claim_token=NULL, run_generation=NULL, error=NULL
                WHERE job_id=? AND asset_id=? AND state='running' AND claim_token=?
                """,
                (job_id, asset_id, claim_token),
            )
            if cursor.rowcount == 1:
                self._emit_item_state(
                    job_id, asset_id, "running", "queued", reason="requeue_running", created_at=now
                )
                self._emit_progress(job_id, created_at=now)
        return cursor.rowcount == 1

    def interrupt_running(self, job_id: str) -> None:
        now = _now()
        with self._lock, self._connection:
            running_items = self._connection.execute(
                "SELECT asset_id FROM job_items WHERE job_id=? AND state='running' ORDER BY position",
                (job_id,),
            ).fetchall()
            cursor = self._connection.execute(
                """
                UPDATE jobs SET state='interrupted', generation=generation+1, updated_at=?
                WHERE job_id=? AND state='running' AND desired_state='run'
                """,
                (now, job_id),
            )
            if cursor.rowcount == 1:
                self._connection.execute(
                    """
                    UPDATE job_items SET state='queued', started_at=NULL, claim_token=NULL, run_generation=NULL
                    WHERE job_id=? AND state='running'
                    """,
                    (job_id,),
                )
                for item in running_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        "running",
                        "queued",
                        reason="interrupt_running",
                        created_at=now,
                    )
                self._emit_job_state(
                    job_id, "running", "interrupted", reason="interrupt_running", created_at=now
                )
                self._emit_progress(job_id, created_at=now)

    def cancel_queued(self, job_id: str) -> None:
        now = _now()
        with self._lock, self._connection:
            queued_items = self._connection.execute(
                "SELECT asset_id FROM job_items WHERE job_id=? AND state='queued' ORDER BY position",
                (job_id,),
            ).fetchall()
            cursor = self._connection.execute(
                "UPDATE job_items SET state='canceled', finished_at=? WHERE job_id=? AND state='queued'",
                (now, job_id),
            )
            if cursor.rowcount:
                for item in queued_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        "queued",
                        "canceled",
                        reason="cancel_queued",
                        created_at=now,
                    )
                self._emit_progress(job_id, created_at=now)

    def retry_failed(self, job_id: str) -> int:
        now = _now()
        with self._lock, self._connection:
            retry_items = self._connection.execute(
                "SELECT asset_id, state FROM job_items WHERE job_id=? AND state IN ('failed','canceled') ORDER BY position",
                (job_id,),
            ).fetchall()
            cursor = self._connection.execute(
                "UPDATE job_items SET state='queued', result_json=NULL, error=NULL, started_at=NULL, finished_at=NULL WHERE job_id=? AND state IN ('failed','canceled')",
                (job_id,),
            )
            if cursor.rowcount:
                for item in retry_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        str(item["state"]),
                        "queued",
                        reason="retry_failed",
                        created_at=now,
                    )
                self._emit_progress(job_id, created_at=now)
        return cursor.rowcount

    def retry_all_items(self, job_id: str) -> int:
        now = _now()
        with self._lock, self._connection:
            retry_items = self._connection.execute(
                """
                SELECT asset_id, state FROM job_items
                WHERE job_id=? AND state IN ('succeeded','failed','canceled')
                ORDER BY position
                """,
                (job_id,),
            ).fetchall()
            cursor = self._connection.execute(
                """
                UPDATE job_items
                SET state='queued', result_json=NULL, error=NULL,
                    started_at=NULL, finished_at=NULL, claim_token=NULL, run_generation=NULL
                WHERE job_id=? AND state IN ('succeeded','failed','canceled')
                """,
                (job_id,),
            )
            if cursor.rowcount:
                for item in retry_items:
                    self._emit_item_state(
                        job_id,
                        str(item["asset_id"]),
                        str(item["state"]),
                        "queued",
                        reason="retry_all_items",
                        created_at=now,
                    )
                self._emit_progress(job_id, created_at=now)
        return cursor.rowcount
