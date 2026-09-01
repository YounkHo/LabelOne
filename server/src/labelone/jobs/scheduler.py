from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil, isfinite
from threading import Condition, RLock
from time import monotonic
from typing import Callable, Deque, Literal


Priority = Literal["interactive", "user_batch", "background"]
JobSchedulerState = Literal["active", "paused", "canceled", "unregistered"]

_PRIORITIES: tuple[Priority, ...] = ("interactive", "user_batch", "background")
_PRIORITY_INDEX = {priority: index for index, priority in enumerate(_PRIORITIES)}
_DEFAULT_WEIGHTS: dict[Priority, int] = {"interactive": 8, "user_batch": 3, "background": 1}


class SchedulerError(RuntimeError):
    pass


class SchedulerClosedError(SchedulerError):
    pass


class SchedulerQueueFullError(SchedulerError):
    pass


@dataclass(frozen=True, slots=True)
class ScheduledItem:
    lease_id: int
    job_id: str
    lane: str
    token: object
    base_priority: Priority
    effective_priority: Priority
    cost: float
    queued_seconds: float


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    queued: int
    inflight: int
    global_capacity: int
    lane_inflight: dict[str, int]
    lane_capacities: dict[str, int]
    lane_interactive_reserves: dict[str, int]
    job_queued: dict[str, int]
    job_inflight: dict[str, int]
    job_states: dict[str, JobSchedulerState]
    closed: bool


@dataclass(slots=True)
class _QueuedItem:
    token: object
    cost: float
    enqueued_at: float
    sequence: int


@dataclass(slots=True)
class _Job:
    job_id: str
    priority: Priority
    lane: str
    max_inflight: int
    order: int
    state: JobSchedulerState = "active"
    queue: Deque[_QueuedItem] = field(default_factory=deque)
    inflight: int = 0


class FairJobScheduler:
    """Thread-safe weighted fair dispatcher with explicit resource leases."""

    def __init__(
        self,
        lane_capacities: dict[str, int],
        *,
        global_capacity: int | None = None,
        max_queued: int = 10_000,
        max_queued_per_job: int | None = None,
        max_item_cost: float = 100.0,
        weights: dict[Priority, int] | None = None,
        interactive_reserves: dict[str, int] | None = None,
        aging_seconds: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not lane_capacities:
            raise ValueError("At least one scheduler lane is required")
        if any(not isinstance(name, str) or not name or isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0 for name, capacity in lane_capacities.items()):
            raise ValueError("Lane names must be non-empty and capacities must be positive integers")
        inferred_global = sum(lane_capacities.values())
        global_capacity = inferred_global if global_capacity is None else global_capacity
        if isinstance(global_capacity, bool) or not isinstance(global_capacity, int) or global_capacity <= 0:
            raise ValueError("global_capacity must be a positive integer")
        if isinstance(max_queued, bool) or not isinstance(max_queued, int) or max_queued <= 0:
            raise ValueError("max_queued must be a positive integer")
        max_queued_per_job = max_queued if max_queued_per_job is None else max_queued_per_job
        if isinstance(max_queued_per_job, bool) or not isinstance(max_queued_per_job, int) or max_queued_per_job <= 0:
            raise ValueError("max_queued_per_job must be a positive integer")
        if not isfinite(max_item_cost) or max_item_cost <= 0:
            raise ValueError("max_item_cost must be positive and finite")
        if not isfinite(aging_seconds) or aging_seconds <= 0:
            raise ValueError("aging_seconds must be positive and finite")
        active_weights = dict(_DEFAULT_WEIGHTS if weights is None else weights)
        if set(active_weights) != set(_PRIORITIES) or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in active_weights.values()):
            raise ValueError("weights must provide positive integer quanta for all priorities")
        active_reserves = {lane: 0 for lane in lane_capacities}
        for lane, reserve in (interactive_reserves or {}).items():
            if lane not in lane_capacities:
                raise ValueError(f"Interactive reserve refers to an unknown lane: {lane}")
            if isinstance(reserve, bool) or not isinstance(reserve, int) or not 0 <= reserve < lane_capacities[lane]:
                raise ValueError("Interactive reserves must be non-negative integers below lane capacity")
            active_reserves[lane] = reserve

        self._condition = Condition(RLock())
        self._lane_capacities = dict(lane_capacities)
        self._lane_inflight = {lane: 0 for lane in lane_capacities}
        self._lane_interactive_reserves = active_reserves
        self._global_capacity = global_capacity
        self._max_queued = max_queued
        self._max_queued_per_job = max_queued_per_job
        self._max_item_cost = float(max_item_cost)
        self._weights = active_weights
        self._aging_seconds = float(aging_seconds)
        self._clock = clock
        self._jobs: dict[str, _Job] = {}
        self._leases: dict[int, tuple[str, str]] = {}
        self._queued = 0
        self._inflight = 0
        self._job_sequence = 0
        self._item_sequence = 0
        self._lease_sequence = 0
        self._rr_cursor = {priority: 0 for priority in _PRIORITIES}
        self._deficit = {priority: 0.0 for priority in _PRIORITIES}
        self._class_cursor = 0
        self._class_entered = False
        self._closed = False

    def register(
        self,
        job_id: str,
        *,
        priority: Priority,
        lane: str,
        max_inflight: int,
    ) -> None:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        if priority not in _PRIORITIES:
            raise ValueError(f"Unsupported scheduler priority: {priority}")
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight <= 0:
            raise ValueError("max_inflight must be a positive integer")
        with self._condition:
            self._ensure_open()
            if lane not in self._lane_capacities:
                raise ValueError(f"Unknown scheduler lane: {lane}")
            if job_id in self._jobs:
                raise SchedulerError(f"Job is already registered: {job_id}")
            self._jobs[job_id] = _Job(
                job_id=job_id,
                priority=priority,
                lane=lane,
                max_inflight=max_inflight,
                order=self._job_sequence,
            )
            self._job_sequence += 1
            self._condition.notify_all()

    register_job = register

    def submit(self, job_id: str, token: object, *, cost: float = 1.0) -> int:
        try:
            normalized_cost = float(cost)
        except (TypeError, ValueError) as exc:
            raise ValueError("Item cost must be numeric") from exc
        if not isfinite(normalized_cost) or not 0 < normalized_cost <= self._max_item_cost:
            raise ValueError(f"Item cost must be in (0, {self._max_item_cost}]")
        with self._condition:
            self._ensure_open()
            job = self._job(job_id)
            if job.state in {"canceled", "unregistered"}:
                raise SchedulerError(f"Job does not accept new items in state {job.state}: {job_id}")
            if self._queued >= self._max_queued or len(job.queue) >= self._max_queued_per_job:
                raise SchedulerQueueFullError("Scheduler queue capacity has been reached")
            sequence = self._item_sequence
            self._item_sequence += 1
            job.queue.append(
                _QueuedItem(
                    token=token,
                    cost=normalized_cost,
                    enqueued_at=self._clock(),
                    sequence=sequence,
                )
            )
            self._queued += 1
            self._condition.notify_all()
            return sequence

    def acquire(self, timeout: float | None = None) -> ScheduledItem | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                if self._closed:
                    return None
                dispatched = self._dispatch_locked()
                if dispatched is not None:
                    return dispatched
                if timeout == 0:
                    return None
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def release(self, item: ScheduledItem | int) -> None:
        lease_id = item.lease_id if isinstance(item, ScheduledItem) else item
        if isinstance(lease_id, bool) or not isinstance(lease_id, int):
            raise ValueError("lease must be a ScheduledItem or integer lease id")
        with self._condition:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                raise SchedulerError(f"Unknown or already released lease: {lease_id}")
            job_id, lane = lease
            self._inflight -= 1
            self._lane_inflight[lane] -= 1
            job = self._jobs.get(job_id)
            if job is not None:
                job.inflight -= 1
                if job.state == "unregistered" and job.inflight == 0:
                    self._jobs.pop(job_id, None)
            self._condition.notify_all()

    def pause(self, job_id: str) -> None:
        with self._condition:
            job = self._job(job_id)
            if job.state != "active":
                raise SchedulerError(f"Only active jobs can be paused: {job_id}")
            job.state = "paused"
            self._condition.notify_all()

    def resume(self, job_id: str) -> None:
        with self._condition:
            self._ensure_open()
            job = self._job(job_id)
            if job.state != "paused":
                raise SchedulerError(f"Only paused jobs can be resumed: {job_id}")
            job.state = "active"
            self._condition.notify_all()

    def cancel(self, job_id: str) -> list[object]:
        with self._condition:
            job = self._job(job_id)
            if job.state == "unregistered":
                return []
            job.state = "canceled"
            dropped = self._drop_queue_locked(job)
            self._condition.notify_all()
            return dropped

    def unregister(self, job_id: str) -> list[object]:
        with self._condition:
            job = self._job(job_id)
            job.state = "unregistered"
            dropped = self._drop_queue_locked(job)
            if job.inflight == 0:
                self._jobs.pop(job_id, None)
            self._condition.notify_all()
            return dropped

    def configure_lane(self, lane: str, capacity: int) -> None:
        if not isinstance(lane, str) or not lane:
            raise ValueError("lane must be a non-empty string")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        with self._condition:
            self._ensure_open()
            current = self._lane_inflight.get(lane, 0)
            if capacity < current:
                raise SchedulerError("Lane capacity cannot be reduced below its current inflight count")
            reserve = self._lane_interactive_reserves.get(lane, 0)
            if reserve >= capacity:
                raise SchedulerError("Lane capacity must remain above its interactive reserve")
            self._lane_capacities[lane] = capacity
            self._lane_inflight.setdefault(lane, 0)
            self._lane_interactive_reserves.setdefault(lane, 0)
            self._condition.notify_all()

    def snapshot(self) -> SchedulerSnapshot:
        with self._condition:
            visible = {job_id: job for job_id, job in self._jobs.items() if job.state != "unregistered"}
            return SchedulerSnapshot(
                queued=self._queued,
                inflight=self._inflight,
                global_capacity=self._global_capacity,
                lane_inflight=dict(self._lane_inflight),
                lane_capacities=dict(self._lane_capacities),
                lane_interactive_reserves=dict(self._lane_interactive_reserves),
                job_queued={job_id: len(job.queue) for job_id, job in visible.items()},
                job_inflight={job_id: job.inflight for job_id, job in visible.items()},
                job_states={job_id: job.state for job_id, job in visible.items()},
                closed=self._closed,
            )

    def shutdown(self, *, wait: bool = True, timeout: float | None = None) -> list[object]:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            if not self._closed:
                self._closed = True
                dropped: list[object] = []
                for job in self._jobs.values():
                    dropped.extend(self._drop_queue_locked(job))
                self._condition.notify_all()
            else:
                dropped = []
            if wait:
                while self._inflight:
                    remaining = None if deadline is None else deadline - monotonic()
                    if remaining is not None and remaining <= 0:
                        break
                    self._condition.wait(remaining)
            return dropped

    def _ensure_open(self) -> None:
        if self._closed:
            raise SchedulerClosedError("Scheduler has been shut down")

    def _job(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise SchedulerError(f"Unknown scheduler job: {job_id}") from exc

    def _drop_queue_locked(self, job: _Job) -> list[object]:
        dropped = [item.token for item in job.queue]
        self._queued -= len(job.queue)
        job.queue.clear()
        return dropped

    def _effective_priority(self, job: _Job, now: float) -> Priority:
        if not job.queue:
            return job.priority
        age = max(0.0, now - job.queue[0].enqueued_at)
        promotion = int(age // self._aging_seconds)
        effective_index = max(0, _PRIORITY_INDEX[job.priority] - promotion)
        return _PRIORITIES[effective_index]

    def _eligible_jobs(self, priority: Priority, now: float) -> list[_Job]:
        if self._inflight >= self._global_capacity:
            return []
        noninteractive_inflight = {
            lane: sum(
                job.inflight
                for job in self._jobs.values()
                if job.lane == lane and job.priority != "interactive"
            )
            for lane in self._lane_capacities
        }
        jobs = [
            job
            for job in self._jobs.values()
            if job.state == "active"
            and job.queue
            and job.inflight < job.max_inflight
            and self._lane_inflight[job.lane] < self._lane_capacities[job.lane]
            and (
                job.priority == "interactive"
                or noninteractive_inflight[job.lane]
                < self._lane_capacities[job.lane] - self._lane_interactive_reserves[job.lane]
            )
            and self._effective_priority(job, now) == priority
        ]
        return sorted(jobs, key=lambda job: job.order)

    def _advance_class(self) -> None:
        self._class_cursor = (self._class_cursor + 1) % len(_PRIORITIES)
        self._class_entered = False

    def _dispatch_locked(self) -> ScheduledItem | None:
        if self._inflight >= self._global_capacity or self._queued == 0:
            return None
        now = self._clock()
        largest_rounds = ceil(self._max_item_cost / min(self._weights.values())) + 2
        for _ in range(len(_PRIORITIES) * largest_rounds):
            priority = _PRIORITIES[self._class_cursor]
            if not self._class_entered:
                self._deficit[priority] += self._weights[priority]
                self._class_entered = True
            jobs = self._eligible_jobs(priority, now)
            if not jobs:
                self._deficit[priority] = 0.0
                self._advance_class()
                continue
            start = self._rr_cursor[priority] % len(jobs)
            selected_index: int | None = None
            for offset in range(len(jobs)):
                index = (start + offset) % len(jobs)
                if jobs[index].queue[0].cost <= self._deficit[priority]:
                    selected_index = index
                    break
            if selected_index is None:
                self._advance_class()
                continue
            job = jobs[selected_index]
            queued_item = job.queue.popleft()
            self._queued -= 1
            self._deficit[priority] -= queued_item.cost
            self._rr_cursor[priority] = (selected_index + 1) % len(jobs)
            job.inflight += 1
            self._inflight += 1
            self._lane_inflight[job.lane] += 1
            lease_id = self._lease_sequence
            self._lease_sequence += 1
            self._leases[lease_id] = (job.job_id, job.lane)
            return ScheduledItem(
                lease_id=lease_id,
                job_id=job.job_id,
                lane=job.lane,
                token=queued_item.token,
                base_priority=job.priority,
                effective_priority=priority,
                cost=queued_item.cost,
                queued_seconds=max(0.0, now - queued_item.enqueued_at),
            )
        return None
