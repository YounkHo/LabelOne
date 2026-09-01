from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Hashable, Iterator


@dataclass(slots=True)
class _Entry:
    lock: Lock = field(default_factory=Lock)
    users: int = 0


class KeyedLockPool:
    """Reference-counted per-key locks without retaining every key forever."""

    def __init__(self) -> None:
        self._guard = RLock()
        self._entries: dict[Hashable, _Entry] = {}

    @contextmanager
    def hold(self, key: Hashable) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(key) is entry:
                    self._entries.pop(key, None)

    @property
    def active_keys(self) -> int:
        with self._guard:
            return len(self._entries)
