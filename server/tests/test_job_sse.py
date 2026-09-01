from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic, sleep

from fastapi.testclient import TestClient
from PIL import Image

from labelone.config import Settings
from labelone.jobs.repository import JobRepository
from labelone.main import create_app


@dataclass(frozen=True)
class _Event:
    event_id: int
    event_type: str
    payload: dict[str, object]


EVENTS = [
    _Event(1, "job.created", {"state": "queued"}),
    _Event(2, "item.state", {"asset_id": "a", "state": "succeeded"}),
    _Event(3, "job.terminal", {"state": "succeeded", "terminal": True}),
]


def _client(tmp_path: Path, monkeypatch, calls: list[int]) -> TestClient:
    def get_job(self, job_id: str, *, include_items: bool = True):
        del self, include_items
        if job_id != "job-1":
            raise AssertionError(job_id)
        return object()

    def list_events(self, job_id: str, after: int, limit: int):
        del self
        assert job_id == "job-1"
        calls.append(after)
        return [event for event in EVENTS if event.event_id > after][:limit]

    monkeypatch.setattr(JobRepository, "get", get_job)
    monkeypatch.setattr(JobRepository, "list_events", list_events, raising=False)
    return TestClient(create_app(Settings(data_dir=tmp_path / "data")))


def _parse_sse(body: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for block in body.strip().split("\n\n"):
        values: dict[str, object] = {}
        for line in block.splitlines():
            if line.startswith("id: "):
                values["id"] = int(line[4:])
            elif line.startswith("event: "):
                values["event"] = line[7:]
            elif line.startswith("data: "):
                values["data"] = json.loads(line[6:])
        if values:
            parsed.append(values)
    return parsed


def test_sse_initial_replay_uses_sqlite_page_and_closes_after_terminal(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    with _client(tmp_path, monkeypatch, calls) as client:
        response = client.get("/api/v1/jobs/job-1/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    events = _parse_sse(response.text)
    assert [event["id"] for event in events] == [1, 2, 3]
    assert [event["event"] for event in events] == ["job.created", "item.state", "job.terminal"]
    assert events[-1]["data"] == {"state": "succeeded", "terminal": True}
    assert calls == [0]


def test_query_cursor_wins_over_last_event_id_and_does_not_repeat(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    with _client(tmp_path, monkeypatch, calls) as client:
        response = client.get(
            "/api/v1/jobs/job-1/events?after=1",
            headers={"Last-Event-ID": "not-an-integer"},
        )

    assert response.status_code == 200
    assert [event["id"] for event in _parse_sse(response.text)] == [2, 3]
    assert calls == [1]


def test_sse_drains_multiple_sqlite_pages_without_duplicate_events(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    with _client(tmp_path, monkeypatch, calls) as client:
        response = client.get("/api/v1/jobs/job-1/events?limit=1")

    assert [event["id"] for event in _parse_sse(response.text)] == [1, 2, 3]
    assert calls == [0, 1, 2]


def test_json_event_page_is_diagnostic_and_uses_same_cursor_rules(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    with _client(tmp_path, monkeypatch, calls) as client:
        response = client.get(
            "/api/v1/jobs/job-1/events?format=json",
            headers={"Last-Event-ID": "1"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "after": 1,
        "next_after": 3,
        "events": [
            {"id": 2, "event": "item.state", "data": {"asset_id": "a", "state": "succeeded"}},
            {"id": 3, "event": "job.terminal", "data": {"state": "succeeded", "terminal": True}},
        ],
    }
    assert calls == [1]


def test_invalid_query_or_header_cursor_and_format_return_400(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    with _client(tmp_path, monkeypatch, calls) as client:
        invalid_query = client.get("/api/v1/jobs/job-1/events?after=-1")
        invalid_header = client.get("/api/v1/jobs/job-1/events?format=json", headers={"Last-Event-ID": "abc"})
        invalid_format = client.get("/api/v1/jobs/job-1/events?format=xml&after=0")

    assert invalid_query.status_code == 400
    assert invalid_query.json()["detail"]["code"] == "invalid_event_cursor"
    assert invalid_header.status_code == 400
    assert invalid_format.status_code == 400
    assert calls == []


def test_completed_real_job_replays_persisted_sqlite_events(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (24, 16), "white").save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        client.post("/api/v1/datasets/register", json={
            "dataset_id": "dataset",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        created = client.post(
            "/api/v1/jobs",
            headers={"Idempotency-Key": "sse-real-job"},
            json={
                "kind": "pipeline",
                "dataset_id": "dataset",
                "pipeline_nodes": [{"id": "color", "kind": "color"}],
            },
        ).json()
        deadline = monotonic() + 5
        while monotonic() < deadline:
            state = client.get(f"/api/v1/jobs/{created['job_id']}").json()["state"]
            if state in {"succeeded", "succeeded_with_errors", "failed", "canceled"}:
                break
            sleep(0.02)
        response = client.get(f"/api/v1/jobs/{created['job_id']}/events")

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0]["event"] == "job.created"
    assert events[-1]["event"] == "job.terminal"
    assert [event["id"] for event in events] == sorted(event["id"] for event in events)
