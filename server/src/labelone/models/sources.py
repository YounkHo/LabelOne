from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock


class ModelSourceStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def x_anylabeling_root(self) -> Path | None:
        with self._lock:
            if not self.path.is_file():
                return None
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return None
        raw = payload.get("x_anylabeling_root") if isinstance(payload, dict) else None
        return Path(raw).expanduser() if isinstance(raw, str) and raw else None

    def set_x_anylabeling_root(self, root: Path) -> None:
        resolved = root.expanduser().resolve()
        payload = json.dumps({"x_anylabeling_root": str(resolved)}, ensure_ascii=False, indent=2)
        with self._lock:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
