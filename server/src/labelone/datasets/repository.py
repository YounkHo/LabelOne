from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from unicodedata import normalize

from labelone.errors import InvalidPathError, RevisionConflictError
from labelone.workspace_settings import DatasetWorkspaceSettings, DatasetWorkspaceSettingsResponse

from .models import (
    AssetCursorPage,
    AssetListResponse,
    AssetStatus,
    DatasetAsset,
    DatasetListResponse,
    DatasetScanResult,
    DatasetScanSummary,
    RegisteredDataset,
)
from .cursor import (
    DatasetCursor,
    InvalidDatasetCursorError,
    StaleDatasetCursorError,
    decode_cursor,
    encode_cursor,
    query_fingerprint,
)
from .search import SearchMode, compile_asset_sql


class DatasetRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_dir TEXT NOT NULL,
                    image_root TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    index_revision INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS assets (
                    dataset_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    match_key TEXT NOT NULL,
                    display_path TEXT NOT NULL,
                    image_path TEXT,
                    annotation_paths_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    selectable INTEGER NOT NULL,
                    reason TEXT,
                    issues_json TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    annotation_count INTEGER,
                    annotation_file_exists INTEGER NOT NULL DEFAULT 0,
                    annotation_revision TEXT,
                    labels_json TEXT NOT NULL DEFAULT '[]',
                    shape_types_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (dataset_id, asset_id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS assets_dataset_status_path
                    ON assets(dataset_id, status, display_path);
                CREATE INDEX IF NOT EXISTS assets_dataset_selectable_path
                    ON assets(dataset_id, selectable DESC, display_path, asset_id);
                CREATE INDEX IF NOT EXISTS assets_dataset_annotation_path
                    ON assets(dataset_id, annotation_count, display_path, asset_id);
                CREATE TABLE IF NOT EXISTS dataset_workspace_settings (
                    dataset_id TEXT PRIMARY KEY,
                    settings_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
                );
                """
            )
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(assets)").fetchall()}
            if "annotation_revision" not in columns:
                self._connection.execute("ALTER TABLE assets ADD COLUMN annotation_revision TEXT")
            if "labels_json" not in columns:
                self._connection.execute("ALTER TABLE assets ADD COLUMN labels_json TEXT NOT NULL DEFAULT '[]'")
            if "shape_types_json" not in columns:
                self._connection.execute("ALTER TABLE assets ADD COLUMN shape_types_json TEXT NOT NULL DEFAULT '[]'")
            annotation_file_exists_added = "annotation_file_exists" not in columns
            if annotation_file_exists_added:
                self._connection.execute("ALTER TABLE assets ADD COLUMN annotation_file_exists INTEGER NOT NULL DEFAULT 0")
            dataset_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(datasets)").fetchall()}
            if "index_revision" not in dataset_columns:
                self._connection.execute("ALTER TABLE datasets ADD COLUMN index_revision INTEGER NOT NULL DEFAULT 1")
            if annotation_file_exists_added:
                legacy_rows = self._connection.execute(
                    "SELECT dataset_id, asset_id, annotation_paths_json FROM assets"
                ).fetchall()
                self._connection.executemany(
                    "UPDATE assets SET annotation_file_exists=? WHERE dataset_id=? AND asset_id=?",
                    [
                        (
                            int(any(Path(path).is_file() for path in json.loads(row["annotation_paths_json"]))),
                            row["dataset_id"],
                            row["asset_id"],
                        )
                        for row in legacy_rows
                    ],
                )
                self._connection.execute(
                    "UPDATE datasets SET index_revision=index_revision+1, updated_at=?",
                    (datetime.now(timezone.utc).isoformat(),),
                )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS assets_dataset_annotation_file_path
                ON assets(dataset_id, annotation_file_exists, display_path, asset_id)
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def delete_dataset(self, dataset_id: str) -> None:
        """Remove only the local index entry; source images/annotations are untouched."""
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM datasets WHERE dataset_id=?", (dataset_id,))
            if cursor.rowcount != 1:
                raise InvalidPathError("Unknown dataset", details={"dataset_id": dataset_id})

    def register(self, result: DatasetScanResult, *, name: str | None = None) -> RegisteredDataset:
        now = datetime.now(timezone.utc).isoformat()
        display_name = name or result.root_dir.name or result.dataset_id
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at, index_revision FROM datasets WHERE dataset_id = ?", (result.dataset_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            index_revision = int(existing["index_revision"]) + 1 if existing else 1
            self._connection.execute(
                """
                INSERT INTO datasets(dataset_id, name, root_dir, image_root, summary_json, created_at, updated_at, index_revision)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name=excluded.name,
                    root_dir=excluded.root_dir,
                    image_root=excluded.image_root,
                    summary_json=excluded.summary_json,
                    updated_at=excluded.updated_at,
                    index_revision=excluded.index_revision
                """,
                (
                    result.dataset_id,
                    display_name,
                    str(result.root_dir),
                    str(result.image_root),
                    result.summary.model_dump_json(),
                    created_at,
                    now,
                    index_revision,
                ),
            )
            self._connection.execute("DELETE FROM assets WHERE dataset_id = ?", (result.dataset_id,))
            self._connection.executemany(
                """
                INSERT INTO assets(
                    dataset_id, asset_id, match_key, display_path, image_path,
                    annotation_paths_json, status, selectable, reason, issues_json,
                    width, height, annotation_count, annotation_file_exists, labels_json, shape_types_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.dataset_id,
                        item.asset_id,
                        item.match_key,
                        item.display_path,
                        str(item.image_path) if item.image_path else None,
                        json.dumps([str(path) for path in item.annotation_paths]),
                        item.status.value,
                        int(item.selectable),
                        item.reason,
                        json.dumps(item.issues),
                        item.width,
                        item.height,
                        item.annotation_count,
                        int(item.annotation_file_exists),
                        json.dumps(item.labels, ensure_ascii=False),
                        json.dumps(item.shape_types, ensure_ascii=False),
                    )
                    for item in result.items
                ],
            )
        return RegisteredDataset(
            dataset_id=result.dataset_id,
            name=display_name,
            root_dir=result.root_dir,
            image_root=result.image_root,
            summary=result.summary,
            created_at=created_at,
            updated_at=now,
            index_revision=index_revision,
        )

    def register_scan_session(self, session_id: str, *, name: str | None = None) -> RegisteredDataset:
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be a non-empty string when provided")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                table = self._connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dataset_scan_sessions'"
                ).fetchone()
                if table is None:
                    raise InvalidPathError("Dataset scan session storage is not initialized")
                session = self._connection.execute(
                    "SELECT * FROM dataset_scan_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if session is None:
                    raise InvalidPathError("Unknown dataset scan session", details={"session_id": session_id})
                if session["state"] not in {"running", "succeeded", "interrupted", "failed"}:
                    raise InvalidPathError(
                        "Dataset scan session has not produced registerable data",
                        details={"session_id": session_id, "state": session["state"]},
                    )
                dataset_id = str(session["dataset_id"] or "")
                if not dataset_id or not session["root_dir"] or not session["image_root"]:
                    raise InvalidPathError(
                        "Dataset scan session is missing resolved metadata",
                        details={"session_id": session_id},
                    )

                item_count = int(self._connection.execute(
                    "SELECT COUNT(*) FROM dataset_scan_session_items WHERE session_id=?", (session_id,)
                ).fetchone()[0])
                if item_count != int(session["persisted_items"]):
                    raise InvalidPathError(
                        "Dataset scan session item count is inconsistent",
                        details={
                            "session_id": session_id,
                            "persisted_items": int(session["persisted_items"]),
                            "stored_items": item_count,
                        },
                    )
                if item_count == 0:
                    raise InvalidPathError(
                        "Dataset scan session has not persisted its first batch yet",
                        details={"session_id": session_id, "state": session["state"]},
                    )

                previous_sequence = int(session["registered_sequence"] or 0)
                registered_revision = session["registered_index_revision"]
                existing_registered = self._connection.execute(
                    "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)
                ).fetchone()
                if session["registered_at"] is not None:
                    if (
                        session["registered_dataset_id"] != dataset_id
                        or existing_registered is None
                        or registered_revision is None
                        or int(existing_registered["index_revision"]) != int(registered_revision)
                    ):
                        raise RevisionConflictError(
                            "Progressive scan index changed outside its scan session",
                            details={
                                "session_id": session_id,
                                "registered_revision": registered_revision,
                                "current_revision": int(existing_registered["index_revision"]) if existing_registered else None,
                            },
                        )
                    final_summary_pending = (
                        session["state"] == "succeeded"
                        and str(session["updated_at"]) > str(session["registered_at"])
                    )
                    if previous_sequence == item_count and not final_summary_pending:
                        self._connection.commit()
                        return self._dataset(existing_registered)

                now = datetime.now(timezone.utc).isoformat()
                created_at = str(existing_registered["created_at"]) if existing_registered else now
                revision = int(existing_registered["index_revision"]) + 1 if existing_registered else 1
                root_dir = Path(str(session["root_dir"]))
                display_name = name.strip() if name is not None else str(session["registration_name"] or root_dir.name or dataset_id)
                if session["summary_json"]:
                    summary_json = str(session["summary_json"])
                else:
                    counts = {
                        str(row["status"]): int(row["count"])
                        for row in self._connection.execute(
                            """
                            SELECT status, COUNT(*) AS count
                            FROM dataset_scan_session_items
                            WHERE session_id=?
                            GROUP BY status
                            """,
                            (session_id,),
                        ).fetchall()
                    }
                    summary_json = DatasetScanSummary(
                        valid=counts.get(AssetStatus.VALID.value, 0),
                        duplicate_match=counts.get(AssetStatus.DUPLICATE_MATCH.value, 0),
                        orphan_annotation=counts.get(AssetStatus.ORPHAN_ANNOTATION.value, 0),
                        corrupt_image=counts.get(AssetStatus.CORRUPT_IMAGE.value, 0),
                        corrupt_annotation=counts.get(AssetStatus.CORRUPT_ANNOTATION.value, 0),
                    ).model_dump_json()
                self._connection.execute(
                    """
                    INSERT INTO datasets(
                        dataset_id, name, root_dir, image_root, summary_json,
                        created_at, updated_at, index_revision
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_id) DO UPDATE SET
                        name=excluded.name,
                        root_dir=excluded.root_dir,
                        image_root=excluded.image_root,
                        summary_json=excluded.summary_json,
                        updated_at=excluded.updated_at,
                        index_revision=excluded.index_revision
                    """,
                    (
                        dataset_id,
                        display_name,
                        str(session["root_dir"]),
                        str(session["image_root"]),
                        summary_json,
                        created_at,
                        now,
                        revision,
                    ),
                )
                if session["registered_at"] is None:
                    self._connection.execute("DELETE FROM assets WHERE dataset_id=?", (dataset_id,))
                if previous_sequence < item_count:
                    self._connection.execute(
                        """
                    INSERT INTO assets(
                        dataset_id, asset_id, match_key, display_path, image_path,
                        annotation_paths_json, status, selectable, reason, issues_json,
                        width, height, annotation_count, annotation_file_exists, annotation_revision,
                        labels_json, shape_types_json
                    )
                    SELECT
                        ?,
                        json_extract(payload_json, '$.asset_id'),
                        json_extract(payload_json, '$.match_key'),
                        json_extract(payload_json, '$.display_path'),
                        json_extract(payload_json, '$.image_path'),
                        COALESCE(json_extract(payload_json, '$.annotation_paths'), '[]'),
                        json_extract(payload_json, '$.status'),
                        CAST(json_extract(payload_json, '$.selectable') AS INTEGER),
                        json_extract(payload_json, '$.reason'),
                        COALESCE(json_extract(payload_json, '$.issues'), '[]'),
                        json_extract(payload_json, '$.width'),
                        json_extract(payload_json, '$.height'),
                        json_extract(payload_json, '$.annotation_count'),
                        COALESCE(CAST(json_extract(payload_json, '$.annotation_file_exists') AS INTEGER), 0),
                        NULL,
                        COALESCE(json_extract(payload_json, '$.labels'), '[]'),
                        COALESCE(json_extract(payload_json, '$.shape_types'), '[]')
                    FROM dataset_scan_session_items
                    WHERE session_id=? AND sequence>=? AND sequence<?
                    ORDER BY sequence
                    """,
                        (dataset_id, session_id, previous_sequence, item_count),
                    )
                registered_count = int(self._connection.execute(
                    "SELECT COUNT(*) FROM assets WHERE dataset_id=?", (dataset_id,)
                ).fetchone()[0])
                if registered_count != item_count:
                    raise InvalidPathError(
                        "Dataset scan materialization did not preserve every item",
                        details={"expected": item_count, "registered": registered_count},
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE dataset_scan_sessions
                    SET registration_name=?, registered_dataset_id=?,
                        registered_index_revision=?, registered_sequence=?, registered_at=?
                    WHERE session_id=? AND registered_sequence=?
                    """,
                    (display_name, dataset_id, revision, item_count, now, session_id, previous_sequence),
                )
                if cursor.rowcount != 1:
                    raise RevisionConflictError(
                        "Dataset scan session was registered concurrently",
                        details={"session_id": session_id},
                    )
                self._connection.commit()
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
        return self.get_dataset(dataset_id)

    @staticmethod
    def _dataset(row: sqlite3.Row) -> RegisteredDataset:
        root_dir = Path(row["root_dir"])
        image_root = Path(row["image_root"])
        source_error = "root_missing" if not root_dir.is_dir() else "image_root_missing" if not image_root.is_dir() else None
        return RegisteredDataset(
            dataset_id=row["dataset_id"],
            name=row["name"],
            root_dir=root_dir,
            image_root=image_root,
            summary=DatasetScanSummary.model_validate_json(row["summary_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            index_revision=int(row["index_revision"]),
            source_available=source_error is None,
            source_error=source_error,
        )

    @staticmethod
    def _asset(row: sqlite3.Row) -> DatasetAsset:
        return DatasetAsset(
            asset_id=row["asset_id"],
            match_key=row["match_key"],
            display_path=row["display_path"],
            image_path=Path(row["image_path"]) if row["image_path"] else None,
            annotation_paths=[Path(path) for path in json.loads(row["annotation_paths_json"])],
            status=AssetStatus(row["status"]),
            selectable=bool(row["selectable"]),
            reason=row["reason"],
            issues=json.loads(row["issues_json"]),
            width=row["width"],
            height=row["height"],
            annotation_count=row["annotation_count"],
            annotation_file_exists=bool(row["annotation_file_exists"]),
            labels=json.loads(row["labels_json"]),
            shape_types=json.loads(row["shape_types_json"]),
        )

    def list_datasets(self) -> DatasetListResponse:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM datasets ORDER BY updated_at DESC").fetchall()
        return DatasetListResponse(datasets=[self._dataset(row) for row in rows])

    def get_dataset(self, dataset_id: str) -> RegisteredDataset:
        with self._lock:
            row = self._connection.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise InvalidPathError("Unknown dataset", details={"dataset_id": dataset_id})
        return self._dataset(row)

    def require_dataset_source(self, dataset_id: str) -> RegisteredDataset:
        dataset = self.get_dataset(dataset_id)
        if dataset.source_available:
            return dataset
        raise InvalidPathError(
            "数据集源目录不存在或不可访问，请重新选择当前项目文件夹",
            details={
                "dataset_id": dataset_id,
                "reason": dataset.source_error or "source_unavailable",
                "root_dir": str(dataset.root_dir),
            },
        )

    def list_assets(self, dataset_id: str, *, offset: int = 0, limit: int = 200) -> AssetListResponse:
        self.require_dataset_source(dataset_id)
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        with self._lock:
            total = int(self._connection.execute(
                "SELECT COUNT(*) FROM assets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()[0])
            rows = self._connection.execute(
                "SELECT * FROM assets WHERE dataset_id = ? ORDER BY selectable DESC, display_path LIMIT ? OFFSET ?",
                (dataset_id, limit, offset),
            ).fetchall()
        next_offset = offset + len(rows) if offset + len(rows) < total else None
        return AssetListResponse(items=[self._asset(row) for row in rows], total=total, next_offset=next_offset)

    def _cursor_page(
        self,
        dataset_id: str,
        *,
        where_sql: str,
        parameters: list[object],
        fingerprint: str,
        cursor: str | None,
        limit: int,
        regex=None,
    ) -> AssetCursorPage:
        self.require_dataset_source(dataset_id)
        safe_limit = max(1, min(limit, 1000))
        with self._lock:
            dataset = self._connection.execute(
                "SELECT index_revision FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            if dataset is None:
                raise InvalidPathError("Unknown dataset", details={"dataset_id": dataset_id})
            revision = int(dataset["index_revision"])
            decoded: DatasetCursor | None = decode_cursor(cursor) if cursor else None
            if decoded is not None:
                if decoded.dataset_id != dataset_id:
                    raise InvalidDatasetCursorError(
                        "Dataset cursor belongs to a different dataset",
                        details={"cursor_dataset_id": decoded.dataset_id, "dataset_id": dataset_id},
                    )
                if decoded.query_fingerprint != fingerprint:
                    raise InvalidDatasetCursorError("Dataset cursor query fingerprint does not match")
                if decoded.revision != revision:
                    raise StaleDatasetCursorError(
                        "Dataset cursor is stale because the index changed",
                        details={"cursor_revision": decoded.revision, "index_revision": revision},
                    )
            if regex is not None:
                self._connection.create_function(
                    "LABELONE_REGEX",
                    1,
                    lambda value: int(regex.search(str(value)[:4_096]) is not None),
                    deterministic=True,
                )
            try:
                if decoded is None:
                    total = int(self._connection.execute(
                        f"SELECT COUNT(*) FROM assets WHERE {where_sql}", parameters
                    ).fetchone()[0])
                    page_where = where_sql
                    page_parameters = list(parameters)
                else:
                    total = decoded.total
                    page_where = (
                        f"{where_sql} AND ("
                        "selectable < ? OR "
                        "(selectable = ? AND display_path > ?) OR "
                        "(selectable = ? AND display_path = ? AND asset_id > ?)"
                        ")"
                    )
                    page_parameters = [
                        *parameters,
                        decoded.selectable,
                        decoded.selectable,
                        decoded.display_path,
                        decoded.selectable,
                        decoded.display_path,
                        decoded.asset_id,
                    ]
                rows = self._connection.execute(
                    f"""
                    SELECT * FROM assets
                    WHERE {page_where}
                    ORDER BY selectable DESC, display_path, asset_id
                    LIMIT ?
                    """,
                    [*page_parameters, safe_limit + 1],
                ).fetchall()
            finally:
                if regex is not None:
                    self._connection.create_function("LABELONE_REGEX", 1, None)
        has_more = len(rows) > safe_limit
        visible_rows = rows[:safe_limit]
        next_cursor = None
        if has_more and visible_rows:
            last = visible_rows[-1]
            next_cursor = encode_cursor(DatasetCursor(
                dataset_id=dataset_id,
                revision=revision,
                query_fingerprint=fingerprint,
                selectable=int(last["selectable"]),
                display_path=str(last["display_path"]),
                asset_id=str(last["asset_id"]),
                total=total,
            ))
        return AssetCursorPage(
            items=[self._asset(row) for row in visible_rows],
            total=total,
            next_cursor=next_cursor,
            index_revision=revision,
        )

    def list_assets_cursor(
        self,
        dataset_id: str,
        *,
        cursor: str | None = None,
        limit: int = 200,
    ) -> AssetCursorPage:
        return self._cursor_page(
            dataset_id,
            where_sql="dataset_id=?",
            parameters=[dataset_id],
            fingerprint=query_fingerprint({"kind": "list"}),
            cursor=cursor,
            limit=limit,
        )

    def search_assets(
        self,
        dataset_id: str,
        *,
        query: str,
        mode: SearchMode,
        offset: int = 0,
        limit: int = 200,
        status: str | None = None,
        annotated: bool | None = None,
        has_annotation_file: bool | None = None,
    ) -> AssetListResponse:
        self.require_dataset_source(dataset_id)
        expression, query_parameters, regex = compile_asset_sql(query, mode)
        where = ["dataset_id=?", f"({expression})"]
        parameters: list[object] = [dataset_id, *query_parameters]
        if status:
            if status in {"error", "exception"}:
                where.append("status != 'valid'")
            else:
                where.append("status=?")
                parameters.append(status)
        if annotated is not None:
            where.append("COALESCE(annotation_count, 0) > 0" if annotated else "COALESCE(annotation_count, 0) = 0")
        if has_annotation_file is not None:
            where.append("annotation_file_exists = ?")
            parameters.append(int(has_annotation_file))
        safe_offset = max(0, offset)
        safe_limit = max(1, min(limit, 1000))
        where_sql = " AND ".join(where)
        with self._lock:
            if regex is not None:
                self._connection.create_function(
                    "LABELONE_REGEX",
                    1,
                    lambda value: int(regex.search(str(value)[:4_096]) is not None),
                    deterministic=True,
                )
            try:
                total = int(self._connection.execute(
                    f"SELECT COUNT(*) FROM assets WHERE {where_sql}", parameters
                ).fetchone()[0])
                rows = self._connection.execute(
                    f"SELECT * FROM assets WHERE {where_sql} ORDER BY selectable DESC, display_path, asset_id LIMIT ? OFFSET ?",
                    [*parameters, safe_limit, safe_offset],
                ).fetchall()
            finally:
                if regex is not None:
                    self._connection.create_function("LABELONE_REGEX", 1, None)
        page = [self._asset(row) for row in rows]
        next_offset = safe_offset + len(page) if safe_offset + len(page) < total else None
        return AssetListResponse(items=page, total=total, next_offset=next_offset)

    def search_assets_cursor(
        self,
        dataset_id: str,
        *,
        query: str,
        mode: SearchMode,
        cursor: str | None = None,
        limit: int = 200,
        status: str | None = None,
        annotated: bool | None = None,
        has_annotation_file: bool | None = None,
    ) -> AssetCursorPage:
        expression, query_parameters, regex = compile_asset_sql(query, mode)
        where = ["dataset_id=?", f"({expression})"]
        parameters: list[object] = [dataset_id, *query_parameters]
        if status:
            if status in {"error", "exception"}:
                where.append("status != 'valid'")
            else:
                where.append("status=?")
                parameters.append(status)
        if annotated is not None:
            where.append("COALESCE(annotation_count, 0) > 0" if annotated else "COALESCE(annotation_count, 0) = 0")
        if has_annotation_file is not None:
            where.append("annotation_file_exists = ?")
            parameters.append(int(has_annotation_file))
        fingerprint = query_fingerprint({
            "kind": "search",
            "query": query.strip(),
            "mode": mode,
            "status": status,
            "annotated": annotated,
            "has_annotation_file": has_annotation_file,
        })
        return self._cursor_page(
            dataset_id,
            where_sql=" AND ".join(where),
            parameters=parameters,
            fingerprint=fingerprint,
            cursor=cursor,
            limit=limit,
            regex=regex,
        )

    def get_asset(self, dataset_id: str, asset_id: str, *, require_selectable: bool = False) -> DatasetAsset:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM assets WHERE dataset_id = ? AND asset_id = ?", (dataset_id, asset_id)
            ).fetchone()
        if row is None:
            raise InvalidPathError("Unknown dataset asset", details={"dataset_id": dataset_id, "asset_id": asset_id})
        asset = self._asset(row)
        if require_selectable and not asset.selectable:
            raise InvalidPathError(
                "Dataset asset is disabled",
                details={"dataset_id": dataset_id, "asset_id": asset_id, "issues": asset.issues},
            )
        return asset

    def get_workspace_settings(self, dataset_id: str) -> DatasetWorkspaceSettingsResponse:
        self.get_dataset(dataset_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT settings_json, revision, updated_at FROM dataset_workspace_settings WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return DatasetWorkspaceSettingsResponse()
        try:
            settings = DatasetWorkspaceSettings.model_validate_json(row["settings_json"])
        except ValueError:
            settings = DatasetWorkspaceSettings()
        return DatasetWorkspaceSettingsResponse(
            **settings.model_dump(mode="python"),
            revision=int(row["revision"]),
            updated_at=str(row["updated_at"]),
        )

    def set_workspace_settings(
        self,
        dataset_id: str,
        settings: DatasetWorkspaceSettings,
        *,
        expected_revision: int,
    ) -> DatasetWorkspaceSettingsResponse:
        now = datetime.now(timezone.utc).isoformat()
        encoded = settings.model_dump_json()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                dataset = self._connection.execute(
                    "SELECT 1 FROM datasets WHERE dataset_id=?",
                    (dataset_id,),
                ).fetchone()
                if dataset is None:
                    raise InvalidPathError("Unknown dataset", details={"dataset_id": dataset_id})
                if settings.last_asset_id is not None:
                    asset = self._connection.execute(
                        "SELECT selectable FROM assets WHERE dataset_id=? AND asset_id=?",
                        (dataset_id, settings.last_asset_id),
                    ).fetchone()
                    if asset is None or not bool(asset["selectable"]):
                        raise InvalidPathError(
                            "Dataset workspace asset is missing or disabled",
                            details={"dataset_id": dataset_id, "asset_id": settings.last_asset_id},
                        )
                existing = self._connection.execute(
                    "SELECT revision FROM dataset_workspace_settings WHERE dataset_id=?",
                    (dataset_id,),
                ).fetchone()
                current_revision = int(existing["revision"]) if existing else 0
                if current_revision != expected_revision:
                    raise RevisionConflictError(
                        "Dataset workspace settings changed concurrently",
                        details={"dataset_id": dataset_id, "expected_revision": expected_revision, "current_revision": current_revision},
                    )
                next_revision = current_revision + 1
                self._connection.execute(
                    """
                    INSERT INTO dataset_workspace_settings(dataset_id, settings_json, revision, updated_at)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(dataset_id) DO UPDATE SET
                        settings_json=excluded.settings_json,
                        revision=excluded.revision,
                        updated_at=excluded.updated_at
                    """,
                    (dataset_id, encoded, next_revision, now),
                )
                self._connection.commit()
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
        return DatasetWorkspaceSettingsResponse(
            **settings.model_dump(mode="python"),
            revision=next_revision,
            updated_at=now,
        )

    def update_annotation_metadata(
        self,
        dataset_id: str,
        asset_id: str,
        *,
        annotation_count: int,
        revision: str,
        labels: list[str],
        shape_types: list[str],
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE assets SET annotation_count=?, annotation_file_exists=1, annotation_revision=?, labels_json=?, shape_types_json=?
                WHERE dataset_id=? AND asset_id=?
                """,
                (
                    annotation_count,
                    revision,
                    json.dumps(sorted(set(labels)), ensure_ascii=False),
                    json.dumps(sorted(set(shape_types)), ensure_ascii=False),
                    dataset_id,
                    asset_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidPathError("Unknown dataset asset", details={"dataset_id": dataset_id, "asset_id": asset_id})
            self._connection.execute(
                "UPDATE datasets SET index_revision=index_revision+1, updated_at=? WHERE dataset_id=?",
                (datetime.now(timezone.utc).isoformat(), dataset_id),
            )

    def update_asset_validation(
        self,
        dataset_id: str,
        asset_id: str,
        *,
        status: AssetStatus,
        selectable: bool,
        reason: str | None,
        width: int | None = None,
        height: int | None = None,
        annotation_count: int | None = None,
        annotation_file_exists: bool | None = None,
        labels: list[str] | None = None,
        shape_types: list[str] | None = None,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE assets SET status=?, selectable=?, reason=?,
                    width=COALESCE(?, width), height=COALESCE(?, height),
                    annotation_count=COALESCE(?, annotation_count),
                    annotation_file_exists=COALESCE(?, annotation_file_exists),
                    labels_json=COALESCE(?, labels_json),
                    shape_types_json=COALESCE(?, shape_types_json)
                WHERE dataset_id=? AND asset_id=?
                """,
                (
                    status.value,
                    int(selectable),
                    reason,
                    width,
                    height,
                    annotation_count,
                    int(annotation_file_exists) if annotation_file_exists is not None else None,
                    json.dumps(sorted(set(labels)), ensure_ascii=False) if labels is not None else None,
                    json.dumps(sorted(set(shape_types)), ensure_ascii=False) if shape_types is not None else None,
                    dataset_id,
                    asset_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidPathError("Unknown dataset asset", details={"dataset_id": dataset_id, "asset_id": asset_id})
            self._connection.execute(
                "UPDATE datasets SET index_revision=index_revision+1, updated_at=? WHERE dataset_id=?",
                (datetime.now(timezone.utc).isoformat(), dataset_id),
            )

    def selectable_asset_ids(self, dataset_id: str) -> list[str]:
        self.get_dataset(dataset_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT asset_id FROM assets WHERE dataset_id = ? AND selectable = 1 ORDER BY display_path",
                (dataset_id,),
            ).fetchall()
        return [str(row["asset_id"]) for row in rows]

    def selectable_asset_ids_for_category(self, dataset_id: str, category: str) -> list[str]:
        normalized_category = normalize("NFC", category.strip())
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT asset_id, labels_json
                FROM assets
                WHERE dataset_id=? AND selectable=1
                ORDER BY display_path, asset_id
                """,
                (dataset_id,),
            ).fetchall()
        matches: list[str] = []
        for row in rows:
            try:
                labels = json.loads(row["labels_json"])
            except (TypeError, json.JSONDecodeError):
                labels = []
            if any(normalize("NFC", str(label).strip()) == normalized_category for label in labels):
                matches.append(str(row["asset_id"]))
        return matches
