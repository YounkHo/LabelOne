from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4

from labelone.errors import InvalidPathError

from .models import AgentAuditRecord, AgentProposal, AgentRun, AgentToolResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    asset_id TEXT,
                    message TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    state TEXT NOT NULL,
                    tool_results_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_proposals (
                    run_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    requires_confirmation INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    PRIMARY KEY(run_id, proposal_id),
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_tool_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    status TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS agent_tool_audit_run_id
                    ON agent_tool_audit(run_id, audit_id);
                """
            )
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(agent_runs)").fetchall()}
            if "tool_results_json" not in columns:
                self._connection.execute("ALTER TABLE agent_runs ADD COLUMN tool_results_json TEXT NOT NULL DEFAULT '[]'")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create(
        self,
        *,
        dataset_id: str,
        asset_id: str | None,
        message: str,
        reply: str,
        state: str,
        proposals: list[tuple[AgentProposal, dict[str, object]]],
        tool_results: list[AgentToolResult] | None = None,
        audits: list[tuple[str, str, str, dict[str, object], dict[str, object] | None]] | None = None,
    ) -> AgentRun:
        run_id = uuid4().hex
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO agent_runs(
                    run_id, dataset_id, asset_id, message, reply, state,
                    tool_results_json, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset_id,
                    asset_id,
                    message,
                    reply,
                    state,
                    json.dumps([item.model_dump(mode="json") for item in tool_results or []], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO agent_proposals(
                    run_id, proposal_id, tool, title, description, risk,
                    requires_confirmation, payload_json, executed, result_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                """,
                [
                    (
                        run_id,
                        proposal.id,
                        proposal.tool,
                        proposal.title,
                        proposal.description,
                        proposal.risk,
                        int(proposal.requires_confirmation),
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    )
                    for proposal, payload in proposals
                ],
            )
            audit_rows = list(audits or [])
            audit_rows.extend(
                (proposal.tool, proposal.risk, "proposed", payload, None)
                for proposal, payload in proposals
            )
            self._connection.executemany(
                """
                INSERT INTO agent_tool_audit(
                    run_id, tool, risk, status, arguments_json, result_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        tool,
                        risk,
                        status,
                        json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                        json.dumps(result, ensure_ascii=False, sort_keys=True) if result is not None else None,
                        now,
                    )
                    for tool, risk, status, arguments, result in audit_rows
                ],
            )
        return self.get(run_id)

    def get(self, run_id: str) -> AgentRun:
        with self._lock:
            row = self._connection.execute("SELECT * FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise InvalidPathError("Unknown agent run", details={"run_id": run_id})
            proposal_rows = self._connection.execute(
                "SELECT * FROM agent_proposals WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
        return AgentRun(
            run_id=row["run_id"],
            dataset_id=row["dataset_id"],
            asset_id=row["asset_id"],
            message=row["message"],
            reply=row["reply"],
            state=row["state"],
            tool_results=[AgentToolResult.model_validate(item) for item in json.loads(row["tool_results_json"])],
            proposals=[
                AgentProposal(
                    id=item["proposal_id"],
                    tool=item["tool"],
                    title=item["title"],
                    description=item["description"],
                    risk=item["risk"],
                    requires_confirmation=bool(item["requires_confirmation"]),
                    executed=bool(item["executed"]),
                    result=json.loads(item["result_json"]) if item["result_json"] else None,
                )
                for item in proposal_rows
            ],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def conversation_history(self, dataset_id: str, *, limit: int = 12) -> list[tuple[str, str]]:
        safe_limit = max(0, min(limit, 24))
        if safe_limit == 0:
            return []
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT message, reply FROM agent_runs
                WHERE dataset_id=?
                ORDER BY created_at DESC LIMIT ?
                """,
                (dataset_id, safe_limit),
            ).fetchall()
        history: list[tuple[str, str]] = []
        for row in reversed(rows):
            history.append(("user", str(row["message"])[:4_000]))
            history.append(("assistant", str(row["reply"])[:4_000]))
        return history

    def record_audit(
        self,
        run_id: str,
        *,
        tool: str,
        risk: str,
        status: str,
        arguments: dict[str, object],
        result: dict[str, object] | None = None,
    ) -> AgentAuditRecord:
        now = _now()
        with self._lock, self._connection:
            exists = self._connection.execute("SELECT 1 FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
            if exists is None:
                raise InvalidPathError("Unknown agent run", details={"run_id": run_id})
            cursor = self._connection.execute(
                """
                INSERT INTO agent_tool_audit(
                    run_id, tool, risk, status, arguments_json, result_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    tool,
                    risk,
                    status,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    json.dumps(result, ensure_ascii=False, sort_keys=True) if result is not None else None,
                    now,
                ),
            )
            audit_id = int(cursor.lastrowid)
        return self.list_audit(run_id, after=audit_id - 1, limit=1)[0]

    def list_audit(self, run_id: str, *, after: int = 0, limit: int = 200) -> list[AgentAuditRecord]:
        safe_limit = max(1, min(limit, 1000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_tool_audit
                WHERE run_id=? AND audit_id>?
                ORDER BY audit_id LIMIT ?
                """,
                (run_id, max(0, after), safe_limit),
            ).fetchall()
        return [
            AgentAuditRecord(
                audit_id=int(row["audit_id"]),
                run_id=str(row["run_id"]),
                tool=str(row["tool"]),
                risk=str(row["risk"]),
                status=str(row["status"]),
                arguments=json.loads(row["arguments_json"]),
                result=json.loads(row["result_json"]) if row["result_json"] else None,
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def proposal_payload(self, run_id: str, proposal_id: str) -> tuple[str, dict[str, object], bool]:
        with self._lock:
            row = self._connection.execute(
                "SELECT tool, payload_json, executed FROM agent_proposals WHERE run_id=? AND proposal_id=?",
                (run_id, proposal_id),
            ).fetchone()
        if row is None:
            raise InvalidPathError(
                "Unknown agent proposal",
                details={"run_id": run_id, "proposal_id": proposal_id},
            )
        return str(row["tool"]), json.loads(row["payload_json"]), bool(row["executed"])

    def complete_proposal(self, run_id: str, proposal_id: str, result: dict[str, object], reply: str) -> AgentRun:
        now = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE agent_proposals SET executed=1, result_json=?
                WHERE run_id=? AND proposal_id=? AND executed=0
                """,
                (json.dumps(result, ensure_ascii=False, sort_keys=True), run_id, proposal_id),
            )
            if cursor.rowcount == 1:
                remaining = int(self._connection.execute(
                    "SELECT COUNT(*) FROM agent_proposals WHERE run_id=? AND executed=0", (run_id,)
                ).fetchone()[0])
                self._connection.execute(
                    "UPDATE agent_runs SET state=?, reply=?, updated_at=? WHERE run_id=?",
                    ("proposed" if remaining else "completed", reply, now, run_id),
                )
        return self.get(run_id)
