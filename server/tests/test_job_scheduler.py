from __future__ import annotations

from threading import Thread
from time import sleep

import pytest

from labelone.jobs.scheduler import (
    FairJobScheduler,
    SchedulerClosedError,
    SchedulerError,
    SchedulerQueueFullError,
)


def _drain(scheduler: FairJobScheduler, count: int) -> list:
    items = []
    for _ in range(count):
        item = scheduler.acquire(timeout=0)
        assert item is not None
        items.append(item)
        scheduler.release(item)
    return items


def test_same_priority_jobs_round_robin_exactly() -> None:
    scheduler = FairJobScheduler({"cpu_pipeline": 1}, global_capacity=1)
    scheduler.register("a", priority="user_batch", lane="cpu_pipeline", max_inflight=1)
    scheduler.register("b", priority="user_batch", lane="cpu_pipeline", max_inflight=1)
    for index in range(6):
        scheduler.submit("a", f"a-{index}")
        scheduler.submit("b", f"b-{index}")

    dispatched = _drain(scheduler, 12)

    assert [item.job_id for item in dispatched] == ["a", "b"] * 6
    assert scheduler.snapshot().queued == 0
    scheduler.shutdown()


def test_weighted_deficit_round_robin_gives_8_3_1_share_without_starving_background() -> None:
    scheduler = FairJobScheduler({"cpu_pipeline": 1}, global_capacity=1, aging_seconds=1_000)
    for job_id, priority in (("ui", "interactive"), ("batch", "user_batch"), ("bg", "background")):
        scheduler.register(job_id, priority=priority, lane="cpu_pipeline", max_inflight=1)
        for index in range(24):
            scheduler.submit(job_id, f"{job_id}-{index}")

    first_cycle = _drain(scheduler, 12)
    second_cycle = _drain(scheduler, 12)

    for cycle in (first_cycle, second_cycle):
        counts = {job_id: sum(item.job_id == job_id for item in cycle) for job_id in ("ui", "batch", "bg")}
        assert counts == {"ui": 8, "batch": 3, "bg": 1}
    assert any(item.job_id == "bg" for item in first_cycle)
    scheduler.shutdown(wait=False)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_aging_promotes_old_background_item_and_large_cost_accumulates_deficit() -> None:
    clock = _Clock()
    scheduler = FairJobScheduler(
        {"cpu_pipeline": 1},
        global_capacity=1,
        aging_seconds=10,
        max_item_cost=10,
        clock=clock,
    )
    scheduler.register("old", priority="background", lane="cpu_pipeline", max_inflight=1)
    scheduler.submit("old", "old-token")
    clock.value = 25
    scheduler.register("new", priority="interactive", lane="cpu_pipeline", max_inflight=1)
    scheduler.submit("new", "new-token")

    promoted = scheduler.acquire(timeout=0)
    assert promoted is not None
    assert promoted.job_id == "old"
    assert promoted.base_priority == "background"
    assert promoted.effective_priority == "interactive"
    assert promoted.queued_seconds == 25
    scheduler.release(promoted)
    scheduler.cancel("new")
    scheduler.register("costly", priority="background", lane="cpu_pipeline", max_inflight=1)
    scheduler.submit("costly", "cost-2", cost=2)
    costly = scheduler.acquire(timeout=0)
    assert costly is not None and costly.token == "cost-2"
    scheduler.release(costly)
    scheduler.shutdown()


def test_global_lane_and_per_job_capacities_are_all_enforced() -> None:
    scheduler = FairJobScheduler(
        {"cpu_pipeline": 2, "model:yolo": 1},
        global_capacity=3,
        aging_seconds=1_000,
    )
    scheduler.register("model", priority="user_batch", lane="model:yolo", max_inflight=3)
    scheduler.register("cpu", priority="user_batch", lane="cpu_pipeline", max_inflight=3)
    for index in range(3):
        scheduler.submit("model", f"model-{index}")
        scheduler.submit("cpu", f"cpu-{index}")

    first = scheduler.acquire(timeout=0)
    second = scheduler.acquire(timeout=0)
    third = scheduler.acquire(timeout=0)
    assert first is not None and second is not None and third is not None
    snapshot = scheduler.snapshot()
    assert snapshot.inflight == 3
    assert snapshot.lane_inflight == {"cpu_pipeline": 2, "model:yolo": 1}
    assert scheduler.acquire(timeout=0) is None

    model_item = next(item for item in (first, second, third) if item.lane == "model:yolo")
    scheduler.release(model_item)
    replacement = scheduler.acquire(timeout=0)
    assert replacement is not None and replacement.lane == "model:yolo"
    for item in (first, second, third):
        if item is not model_item:
            scheduler.release(item)
    scheduler.release(replacement)

    serial = FairJobScheduler({"cpu_pipeline": 2}, global_capacity=2)
    serial.register("serial", priority="interactive", lane="cpu_pipeline", max_inflight=1)
    serial.submit("serial", 1)
    serial.submit("serial", 2)
    leased = serial.acquire(timeout=0)
    assert leased is not None
    assert serial.acquire(timeout=0) is None
    serial.release(leased)
    assert serial.acquire(timeout=0) is not None
    serial.shutdown(wait=False)
    scheduler.shutdown(wait=False)


def test_interactive_reserve_limits_batch_without_weakening_priority_fairness() -> None:
    scheduler = FairJobScheduler(
        {"cpu_pipeline": 4},
        global_capacity=4,
        interactive_reserves={"cpu_pipeline": 1},
        aging_seconds=1_000,
    )
    scheduler.register("batch", priority="user_batch", lane="cpu_pipeline", max_inflight=4)
    scheduler.register("preheat", priority="background", lane="cpu_pipeline", max_inflight=1)
    scheduler.register("ui", priority="interactive", lane="cpu_pipeline", max_inflight=1)
    for index in range(4):
        scheduler.submit("batch", f"batch-{index}")

    batch = [scheduler.acquire(timeout=0) for _ in range(3)]
    assert all(item is not None and item.job_id == "batch" for item in batch)
    assert scheduler.acquire(timeout=0) is None

    scheduler.submit("preheat", "neighbor-preheat")
    assert scheduler.acquire(timeout=0) is None

    scheduler.submit("ui", "preview")
    interactive = scheduler.acquire(timeout=0)
    assert interactive is not None
    assert interactive.job_id == "ui"
    snapshot = scheduler.snapshot()
    assert snapshot.lane_inflight["cpu_pipeline"] == 4
    assert snapshot.lane_interactive_reserves["cpu_pipeline"] == 1

    scheduler.release(interactive)
    for item in batch:
        assert item is not None
        scheduler.release(item)
    final_batch = scheduler.acquire(timeout=0)
    assert final_batch is not None and final_batch.job_id == "batch"
    scheduler.release(final_batch)
    preheat = scheduler.acquire(timeout=0)
    assert preheat is not None and preheat.job_id == "preheat"
    scheduler.release(preheat)
    scheduler.shutdown()


def test_interactive_reserve_validation_and_dynamic_lanes_default_to_zero() -> None:
    with pytest.raises(ValueError, match="below lane capacity"):
        FairJobScheduler({"cpu_pipeline": 1}, interactive_reserves={"cpu_pipeline": 1})
    with pytest.raises(ValueError, match="unknown lane"):
        FairJobScheduler({"cpu_pipeline": 2}, interactive_reserves={"missing": 1})

    scheduler = FairJobScheduler({"cpu_pipeline": 2}, interactive_reserves={"cpu_pipeline": 1})
    scheduler.configure_lane("model:new", 1)
    scheduler.register("model", priority="user_batch", lane="model:new", max_inflight=1)
    scheduler.submit("model", "inference")
    item = scheduler.acquire(timeout=0)
    assert item is not None and item.lane == "model:new"
    assert scheduler.snapshot().lane_interactive_reserves["model:new"] == 0
    scheduler.release(item)
    scheduler.shutdown()


def test_pause_cancel_and_unregister_prevent_future_dispatch() -> None:
    scheduler = FairJobScheduler({"cpu_pipeline": 1}, global_capacity=1)
    scheduler.register("job", priority="interactive", lane="cpu_pipeline", max_inflight=1)
    scheduler.submit("job", "first")
    scheduler.submit("job", "second")
    scheduler.pause("job")
    assert scheduler.acquire(timeout=0) is None
    assert scheduler.snapshot().job_queued["job"] == 2

    scheduler.resume("job")
    first = scheduler.acquire(timeout=0)
    assert first is not None and first.token == "first"
    dropped = scheduler.cancel("job")
    assert dropped == ["second"]
    scheduler.release(first)
    assert scheduler.acquire(timeout=0) is None
    with pytest.raises(SchedulerError, match="does not accept"):
        scheduler.submit("job", "late")

    scheduler.register("remove", priority="background", lane="cpu_pipeline", max_inflight=1)
    scheduler.submit("remove", "removed")
    assert scheduler.unregister("remove") == ["removed"]
    assert "remove" not in scheduler.snapshot().job_states
    assert scheduler.acquire(timeout=0) is None
    scheduler.shutdown()


def test_release_wakes_waiting_consumer_and_is_thread_safe() -> None:
    scheduler = FairJobScheduler({"model:sam": 1}, global_capacity=1)
    scheduler.register("sam", priority="interactive", lane="model:sam", max_inflight=1)
    scheduler.submit("sam", "one")
    scheduler.submit("sam", "two")
    first = scheduler.acquire(timeout=0)
    assert first is not None
    received: list = []

    def consume() -> None:
        received.append(scheduler.acquire(timeout=1))

    thread = Thread(target=consume)
    thread.start()
    sleep(0.05)
    assert received == []
    scheduler.release(first)
    thread.join(timeout=1)
    assert len(received) == 1 and received[0] is not None
    assert received[0].token == "two"
    scheduler.release(received[0])
    assert scheduler.snapshot().inflight == 0
    scheduler.shutdown()


def test_bounded_queue_and_shutdown_drop_pending_and_wake_waiters() -> None:
    scheduler = FairJobScheduler(
        {"cpu_pipeline": 1},
        global_capacity=1,
        max_queued=2,
        max_queued_per_job=1,
    )
    scheduler.register("a", priority="user_batch", lane="cpu_pipeline", max_inflight=1)
    scheduler.register("b", priority="user_batch", lane="cpu_pipeline", max_inflight=1)
    scheduler.submit("a", "a")
    with pytest.raises(SchedulerQueueFullError):
        scheduler.submit("a", "a-overflow")
    scheduler.submit("b", "b")
    with pytest.raises(SchedulerQueueFullError):
        scheduler.submit("b", "global-overflow")

    dropped = scheduler.shutdown(wait=False)
    assert dropped == ["a", "b"]
    assert scheduler.acquire(timeout=0) is None
    with pytest.raises(SchedulerClosedError):
        scheduler.submit("a", "after-close")
    assert scheduler.snapshot().closed is True
