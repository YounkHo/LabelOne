from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from labelone.errors import InvalidPathError


class CompositeDefinitionStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def load(self) -> list[dict[str, object]]:
        with self._lock:
            if not self.path.is_file():
                return []
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise InvalidPathError(
                    "Pipeline composite store is unreadable",
                    details={"path": str(self.path), "error": str(exc)},
                ) from exc
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise InvalidPathError("Pipeline composite store must contain an array of definitions")
        return payload

    def append(self, definition: dict[str, object]) -> None:
        with self._lock:
            definitions = self.load()
            composite_id = definition.get("id")
            if any(item.get("id") == composite_id for item in definitions):
                raise InvalidPathError("Pipeline composite is already persisted", details={"id": composite_id})
            definitions.append(definition)
            encoded = json.dumps(definitions, ensure_ascii=False, indent=2).encode("utf-8")
            partial = self.path.parent / f".{self.path.name}.part"
            partial.unlink(missing_ok=True)
            try:
                with partial.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, self.path)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                partial.unlink(missing_ok=True)
