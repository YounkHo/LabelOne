from __future__ import annotations

import json
from typing import BinaryIO


PROTOCOL_VERSION = 1
ALLOWED_OPERATIONS = frozenset({"load", "layers", "predict", "unload", "close"})


class WorkerProtocolError(ValueError):
    pass


class WorkerBudgetError(WorkerProtocolError):
    pass


def encode_message(message: object, *, maximum_bytes: int) -> bytes:
    try:
        encoded = json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("Worker IPC message must be finite JSON data") from exc
    if len(encoded) > maximum_bytes:
        raise WorkerBudgetError(
            f"Worker IPC message exceeds the {maximum_bytes}-byte budget"
        )
    return encoded


def decode_message(line: bytes, *, maximum_bytes: int) -> dict[str, object]:
    if not line.endswith(b"\n"):
        raise WorkerProtocolError("Worker IPC message is not newline terminated")
    if len(line) > maximum_bytes:
        raise WorkerBudgetError(
            f"Worker IPC message exceeds the {maximum_bytes}-byte budget"
        )
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("Worker IPC message is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("Worker IPC message root must be an object")
    return value


def read_message(stream: BinaryIO, *, maximum_bytes: int) -> dict[str, object] | None:
    line = stream.readline(maximum_bytes + 1)
    if not line:
        return None
    return decode_message(line, maximum_bytes=maximum_bytes)


def write_message(stream: BinaryIO, message: object, *, maximum_bytes: int) -> None:
    stream.write(encode_message(message, maximum_bytes=maximum_bytes))
    stream.flush()
