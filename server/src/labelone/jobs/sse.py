from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import json
from time import monotonic
from typing import Any, AsyncIterator, Mapping, Protocol

from fastapi import Request


TERMINAL_EVENT_TYPES = {
    "job.terminal",
    "job.succeeded",
    "job.succeeded_with_errors",
    "job.failed",
    "job.canceled",
    "terminal",
}


class EventRepository(Protocol):
    def list_events(self, job_id: str, *, after: int, limit: int) -> object: ...


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event_id: int
    event_type: str
    data: object


def parse_event_cursor(value: str | int | None, *, source: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a non-negative integer")
    if isinstance(value, int):
        cursor = value
    elif isinstance(value, str) and value.strip() and value.strip().isascii() and value.strip().isdigit():
        cursor = int(value.strip())
    else:
        raise ValueError(f"{source} must be a non-negative integer")
    if cursor < 0:
        raise ValueError(f"{source} must be a non-negative integer")
    return cursor


def resolve_event_cursor(after: str | int | None, last_event_id: str | None) -> int:
    if after is not None:
        return parse_event_cursor(after, source="after")
    return parse_event_cursor(last_event_id, source="Last-Event-ID")


def _mapping(event: object) -> dict[str, object]:
    if hasattr(event, "model_dump"):
        value = event.model_dump(mode="json")
        if isinstance(value, dict):
            return value
    if is_dataclass(event) and not isinstance(event, type):
        value = asdict(event)
        if isinstance(value, dict):
            return value
    if isinstance(event, Mapping):
        return dict(event)
    if hasattr(event, "keys"):
        try:
            return {str(key): event[key] for key in event.keys()}  # type: ignore[index]
        except Exception:
            pass
    values: dict[str, object] = {}
    for name in ("event_id", "id", "event_type", "type", "payload", "payload_json", "data", "created_at"):
        if hasattr(event, name):
            values[name] = getattr(event, name)
    if values:
        return values
    raise ValueError("Job event has an unsupported representation")


def normalize_event(event: object) -> StreamEvent:
    value = _mapping(event)
    raw_id = value.get("event_id", value.get("id"))
    event_id = parse_event_cursor(raw_id, source="event id")  # type: ignore[arg-type]
    if event_id <= 0:
        raise ValueError("event id must be positive")
    raw_type = value.get("event_type", value.get("type", value.get("kind")))
    if not isinstance(raw_type, str) or not raw_type.strip() or "\n" in raw_type or "\r" in raw_type:
        raise ValueError("Job event type is invalid")
    event_type = raw_type.strip()
    if "data" in value:
        data = value["data"]
    elif "payload" in value:
        data = value["payload"]
    elif "payload_json" in value:
        raw_payload = value["payload_json"]
        if not isinstance(raw_payload, str):
            raise ValueError("Job event payload_json must be a string")
        try:
            data = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Job event payload_json is invalid") from exc
    else:
        excluded = {"event_id", "id", "event_type", "type", "kind"}
        data = {key: item for key, item in value.items() if key not in excluded}
    return StreamEvent(event_id, event_type, data)


def _event_values(page: object) -> list[object]:
    if isinstance(page, list):
        return page
    if isinstance(page, tuple):
        return list(page)
    if isinstance(page, Mapping):
        values = page.get("events", page.get("items"))
        if isinstance(values, (list, tuple)):
            return list(values)
    for attribute in ("events", "items"):
        values = getattr(page, attribute, None)
        if isinstance(values, (list, tuple)):
            return list(values)
    raise ValueError("Job event page must be a list or contain an events array")


def normalize_event_page(page: object, *, after: int) -> list[StreamEvent]:
    events = sorted((normalize_event(event) for event in _event_values(page)), key=lambda event: event.event_id)
    unique: list[StreamEvent] = []
    seen: set[int] = set()
    for event in events:
        if event.event_id <= after or event.event_id in seen:
            continue
        seen.add(event.event_id)
        unique.append(event)
    return unique


def event_is_terminal(event: StreamEvent) -> bool:
    return event.event_type in TERMINAL_EVENT_TYPES or event.event_type.endswith(".terminal")


def encode_sse_event(event: StreamEvent) -> bytes:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"id: {event.event_id}\nevent: {event.event_type}\ndata: {data}\n\n".encode("utf-8")


def event_page_payload(job_id: str, after: int, page: object) -> dict[str, object]:
    events = normalize_event_page(page, after=after)
    return {
        "job_id": job_id,
        "after": after,
        "next_after": events[-1].event_id if events else after,
        "events": [
            {"id": event.event_id, "event": event.event_type, "data": event.data}
            for event in events
        ],
    }


async def stream_job_events(
    request: Request,
    repository: EventRepository,
    *,
    job_id: str,
    after: int,
    limit: int = 200,
    keepalive_seconds: float = 15.0,
    wait_seconds: float = 0.35,
) -> AsyncIterator[bytes]:
    cursor = after
    last_activity = monotonic()
    while True:
        if await request.is_disconnected():
            return
        page = await asyncio.to_thread(repository.list_events, job_id, after=cursor, limit=limit)
        events = normalize_event_page(page, after=cursor)
        if events:
            for event in events:
                if await request.is_disconnected():
                    return
                yield encode_sse_event(event)
                cursor = event.event_id
                last_activity = monotonic()
                if event_is_terminal(event):
                    return
            continue
        now = monotonic()
        if now - last_activity >= keepalive_seconds:
            yield b": keep-alive\n\n"
            last_activity = now
        await asyncio.sleep(wait_seconds)
