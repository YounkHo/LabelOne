from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from labelone.keyed_lock import KeyedLockPool


def test_keyed_lock_serializes_same_key_allows_other_keys_and_releases_entries() -> None:
    pool = KeyedLockPool()
    observation_lock = Lock()
    same_key_inflight = 0
    same_key_max = 0
    all_inflight = 0
    all_max = 0

    def work(key: str) -> None:
        nonlocal same_key_inflight, same_key_max, all_inflight, all_max
        with pool.hold(key):
            with observation_lock:
                if key == "same":
                    same_key_inflight += 1
                    same_key_max = max(same_key_max, same_key_inflight)
                all_inflight += 1
                all_max = max(all_max, all_inflight)
            sleep(0.01)
            with observation_lock:
                if key == "same":
                    same_key_inflight -= 1
                all_inflight -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(work, key) for key in ["same", "same", "same", "other-1", "other-2"]]
        for future in futures:
            future.result()

    assert same_key_max == 1
    assert all_max >= 2
    assert pool.active_keys == 0
