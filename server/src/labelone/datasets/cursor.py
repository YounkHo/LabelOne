from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json

from labelone.errors import LabelOneError


class InvalidDatasetCursorError(LabelOneError):
    code = "invalid_dataset_cursor"
    status_code = 400


class StaleDatasetCursorError(LabelOneError):
    code = "stale_dataset_cursor"
    status_code = 409


@dataclass(frozen=True, slots=True)
class DatasetCursor:
    dataset_id: str
    revision: int
    query_fingerprint: str
    selectable: int
    display_path: str
    asset_id: str
    total: int
    version: int = 1


def query_fingerprint(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()[:32]


def encode_cursor(cursor: DatasetCursor) -> str:
    payload = asdict(cursor)
    encoded_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    envelope = {
        "payload": payload,
        "checksum": sha256(encoded_payload.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def decode_cursor(value: str) -> DatasetCursor:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise InvalidDatasetCursorError("Dataset cursor is empty or too long")
    try:
        padded = value + "=" * (-len(value) % 4)
        envelope = json.loads(urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidDatasetCursorError("Dataset cursor is not valid base64 JSON") from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict) or not isinstance(envelope.get("checksum"), str):
        raise InvalidDatasetCursorError("Dataset cursor envelope is malformed")
    payload = envelope["payload"]
    encoded_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = sha256(encoded_payload.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(envelope["checksum"], expected):
        raise InvalidDatasetCursorError("Dataset cursor checksum does not match")
    try:
        cursor = DatasetCursor(
            dataset_id=payload["dataset_id"],
            revision=payload["revision"],
            query_fingerprint=payload["query_fingerprint"],
            selectable=payload["selectable"],
            display_path=payload["display_path"],
            asset_id=payload["asset_id"],
            total=payload["total"],
            version=payload["version"],
        )
    except (KeyError, TypeError) as exc:
        raise InvalidDatasetCursorError("Dataset cursor payload is incomplete") from exc
    if (
        cursor.version != 1
        or not isinstance(cursor.dataset_id, str)
        or not cursor.dataset_id
        or isinstance(cursor.revision, bool)
        or not isinstance(cursor.revision, int)
        or cursor.revision < 1
        or not isinstance(cursor.query_fingerprint, str)
        or len(cursor.query_fingerprint) != 32
        or cursor.selectable not in {0, 1}
        or not isinstance(cursor.display_path, str)
        or not isinstance(cursor.asset_id, str)
        or not cursor.asset_id
        or isinstance(cursor.total, bool)
        or not isinstance(cursor.total, int)
        or cursor.total < 0
    ):
        raise InvalidDatasetCursorError("Dataset cursor payload has invalid field types")
    return cursor
