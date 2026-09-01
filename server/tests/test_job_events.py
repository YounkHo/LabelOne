from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.jobs.models import BatchJobRequest
from labelone.jobs.repository import JobRepository
from labelone.pipelines import PipelineNode


def _repositories(tmp_path: Path, *, count: int = 3) -> tuple[DatasetRepository, JobRepository]:
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(count):
        Image.new("RGB", (16, 12), (index, 20, 30)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="events"))
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    datasets.register(scan)
    return datasets, JobRepository(tmp_path / "index.sqlite3", datasets)


def _request() -> BatchJobRequest:
    return BatchJobRequest(
        kind="pipeline",
        dataset_id="events",
        pipeline_nodes=[PipelineNode(id="color", kind="color")],
    )


def test_events_are_strictly_increasing_paginated_and_progress_is_aggregated(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=2)
    job = jobs.create(_request())
    assert jobs.is_terminal(job.job_id) is False
    assert jobs.transition_to_running(job.job_id)
    for index, item in enumerate(jobs.queued_items(job.job_id, 10)):
        claim = jobs.mark_item_running(job.job_id, item.asset_id)
        assert claim is not None
        assert jobs.finish_item(
            job.job_id,
            item.asset_id,
            claim_token=claim,
            state="succeeded" if index == 0 else "failed",
            result={"index": index} if index == 0 else None,
            error="fixture failure" if index else None,
        )
    assert jobs.settle_completed(job.job_id)

    events = jobs.list_events(job.job_id, limit=1000)
    event_ids = [event.event_id for event in events]
    assert event_ids == sorted(event_ids)
    assert len(event_ids) == len(set(event_ids))
    assert event_ids[0] > 0
    assert jobs.latest_event_id(job.job_id) == event_ids[-1]
    assert jobs.latest_event_id() >= event_ids[-1]
    assert [event.event_type for event in events[:3]] == ["job.created", "job.state", "job.progress"]
    assert events[-1].event_type == "job.terminal"
    assert events[-1].payload["state"] == "succeeded_with_errors"
    assert events[-1].payload["finished"] == 2
    assert events[-1].payload["succeeded"] == 1
    assert events[-1].payload["failed"] == 1
    assert jobs.is_terminal(job.job_id) is True

    first_page = jobs.list_events(job.job_id, limit=4)
    second_page = jobs.list_events(job.job_id, after=first_page[-1].event_id, limit=1000)
    assert [*first_page, *second_page] == events
    for index, event in enumerate(events[:-1]):
        if event.event_type == "item.state" and event.payload["state"] in {"succeeded", "failed"}:
            assert events[index + 1].event_type == "job.progress"
    jobs.close()
    datasets.close()


def test_event_insert_failure_rolls_back_item_cas_and_attempt_counter(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=1)
    job = jobs.create(_request())
    assert jobs.transition_to_running(job.job_id)
    before_event = jobs.latest_event_id(job.job_id)
    item = jobs.queued_items(job.job_id, 1)[0]
    with jobs._lock, jobs._connection:
        jobs._connection.executescript(
            """
            CREATE TRIGGER reject_item_event
            BEFORE INSERT ON job_events
            WHEN NEW.event_type='item.state'
            BEGIN
                SELECT RAISE(ABORT, 'fixture rejects item event');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="fixture rejects item event"):
        jobs.mark_item_running(job.job_id, item.asset_id)

    restored = jobs.get(job.job_id)
    assert restored.items[0].state == "queued"
    assert restored.items[0].attempts == 0
    assert jobs.latest_event_id(job.job_id) == before_event
    jobs.close()
    datasets.close()


def test_item_progress_snapshot_is_persisted_and_visible_lookup_is_bounded(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=3)
    job = jobs.create(_request())
    assert jobs.transition_to_running(job.job_id)
    items = jobs.queued_items(job.job_id, 3)
    claim = jobs.mark_item_running(job.job_id, items[1].asset_id)
    assert claim is not None
    payload = {
        "kind": "pipeline",
        "progress": 0.5,
        "phase": "operator",
        "completed_steps": 2,
        "total_steps": 4,
    }
    assert jobs.update_item_progress(job.job_id, items[1].asset_id, claim_token=claim, payload=payload)

    visible = jobs.lookup_items(job.job_id, [items[0].asset_id, items[1].asset_id])

    assert visible.total == 2
    running = next(item for item in visible.items if item.asset_id == items[1].asset_id)
    assert running.state == "running"
    assert running.progress == payload
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    persisted = reopened.lookup_items(job.job_id, [items[1].asset_id]).items[0]
    assert persisted.progress == payload
    with pytest.raises(ValueError, match="1 to 200"):
        reopened.lookup_items(job.job_id, [])
    reopened.close()
    datasets.close()


def test_startup_recovery_events_persist_continue_ids_and_do_not_repeat(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=2)
    job = jobs.create(_request())
    assert jobs.transition_to_running(job.job_id)
    item = jobs.queued_items(job.job_id, 1)[0]
    assert jobs.mark_item_running(job.job_id, item.asset_id)
    before_restart = jobs.latest_event_id(job.job_id)
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    recovered = reopened.get(job.job_id)
    recovery_events = reopened.list_events(job.job_id, after=before_restart, limit=100)
    assert recovered.state == "interrupted"
    assert recovered.items[0].state == "queued"
    assert recovery_events
    assert all(event.event_id > before_restart for event in recovery_events)
    assert [event.event_id for event in recovery_events] == sorted(event.event_id for event in recovery_events)
    assert "job.recovered" in [event.event_type for event in recovery_events]
    assert any(
        event.event_type == "item.state"
        and event.payload["previous_state"] == "running"
        and event.payload["state"] == "queued"
        for event in recovery_events
    )
    latest_after_recovery = reopened.latest_event_id(job.job_id)
    reopened.close()

    reopened_again = JobRepository(tmp_path / "index.sqlite3", datasets)
    assert reopened_again.list_events(job.job_id, after=latest_after_recovery, limit=100) == []
    assert reopened_again.latest_event_id(job.job_id) == latest_after_recovery
    reopened_again.close()
    datasets.close()


def test_startup_recovery_preserves_pause_and_cancel_intent_in_events(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=2)
    paused = jobs.create(_request())
    assert jobs.transition_to_running(paused.job_id)
    paused_item = jobs.queued_items(paused.job_id, 1)[0]
    assert jobs.mark_item_running(paused.job_id, paused_item.asset_id)
    jobs.request_pause(paused.job_id)
    pause_before = jobs.latest_event_id(paused.job_id)

    canceled = jobs.create(_request())
    assert jobs.transition_to_running(canceled.job_id)
    canceled_item = jobs.queued_items(canceled.job_id, 1)[0]
    assert jobs.mark_item_running(canceled.job_id, canceled_item.asset_id)
    jobs.request_cancel(canceled.job_id)
    cancel_before = jobs.latest_event_id(canceled.job_id)
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    pause_events = reopened.list_events(paused.job_id, after=pause_before, limit=100)
    cancel_events = reopened.list_events(canceled.job_id, after=cancel_before, limit=100)

    assert reopened.get(paused.job_id).state == "paused"
    assert reopened.is_terminal(paused.job_id) is False
    assert any(
        event.event_type == "job.recovered"
        and event.payload["state"] == "paused"
        and event.payload["desired_state"] == "pause"
        for event in pause_events
    )
    assert reopened.get(canceled.job_id).state == "canceled"
    assert reopened.is_terminal(canceled.job_id) is True
    assert any(
        event.event_type == "job.recovered"
        and event.payload["state"] == "canceled"
        and event.payload["desired_state"] == "cancel"
        for event in cancel_events
    )
    assert cancel_events[-1].event_type == "job.terminal"
    assert cancel_events[-1].payload["canceled"] == 2
    reopened.close()
    datasets.close()


def test_pause_resume_cancel_and_terminal_progress_are_all_recorded(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=3)
    paused = jobs.create(_request())
    jobs.request_pause(paused.job_id)
    assert jobs.get(paused.job_id).state == "paused"
    pause_events = jobs.list_events(paused.job_id, limit=100)
    assert any(
        event.event_type == "job.state"
        and event.payload["reason"] == "request_pause"
        and event.payload["state"] == "paused"
        for event in pause_events
    )
    assert jobs.is_terminal(paused.job_id) is False
    jobs.request_resume(paused.job_id)
    assert any(
        event.event_type == "job.state" and event.payload["reason"] == "request_resume"
        for event in jobs.list_events(paused.job_id, limit=100)
    )

    canceled = jobs.create(_request())
    assert jobs.transition_to_running(canceled.job_id)
    running_item = jobs.queued_items(canceled.job_id, 1)[0]
    claim = jobs.mark_item_running(canceled.job_id, running_item.asset_id)
    assert claim is not None
    jobs.request_cancel(canceled.job_id)
    assert jobs.finish_item(
        canceled.job_id,
        running_item.asset_id,
        claim_token=claim,
        state="succeeded",
        result={"late": True},
    ) is False
    assert jobs.cancel_running_item(canceled.job_id, running_item.asset_id, claim_token=claim)
    assert jobs.settle_cancel(canceled.job_id)

    cancel_events = jobs.list_events(canceled.job_id, limit=1000)
    terminal = cancel_events[-1]
    assert terminal.event_type == "job.terminal"
    assert terminal.payload["state"] == "canceled"
    assert terminal.payload["canceled"] == 3
    assert terminal.payload["finished"] == 3
    assert jobs.is_terminal(canceled.job_id) is True
    assert not any(
        event.event_type == "item.state" and event.payload.get("has_result") is True
        for event in cancel_events
    )
    jobs.close()
    datasets.close()


def test_retry_requeues_failed_item_with_progress_event(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=1)
    job = jobs.create(_request())
    assert jobs.transition_to_running(job.job_id)
    item = jobs.queued_items(job.job_id, 1)[0]
    claim = jobs.mark_item_running(job.job_id, item.asset_id)
    assert claim is not None
    assert jobs.finish_item(job.job_id, item.asset_id, claim_token=claim, state="failed", error="retry me")
    assert jobs.settle_completed(job.job_id)
    before_retry = jobs.latest_event_id(job.job_id)

    assert jobs.retry_failed(job.job_id) == 1
    retry_events = jobs.list_events(job.job_id, after=before_retry, limit=10)

    assert [event.event_type for event in retry_events] == ["item.state", "job.progress"]
    assert retry_events[0].payload["previous_state"] == "failed"
    assert retry_events[0].payload["state"] == "queued"
    assert retry_events[0].payload["reason"] == "retry_failed"
    assert retry_events[1].payload["queued"] == 1
    assert retry_events[1].payload["finished"] == 0
    jobs.close()
    datasets.close()
