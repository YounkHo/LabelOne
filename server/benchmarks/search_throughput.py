from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from labelone.datasets.repository import DatasetRepository


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _seed(database_path: Path, items: int) -> None:
    repository = DatasetRepository(database_path)
    now = datetime.now(timezone.utc).isoformat()
    summary = json.dumps({
        "valid": items,
        "duplicate_match": 0,
        "orphan_annotation": 0,
        "corrupt_image": 0,
        "corrupt_annotation": 0,
        "hidden_image_only": 0,
    })
    with repository._lock, repository._connection:  # benchmark fixture setup only
        repository._connection.execute(
            """
            INSERT INTO datasets(
                dataset_id, name, root_dir, image_root, summary_json,
                created_at, updated_at, index_revision
            ) VALUES(?, ?, ?, ?, ?, ?, ?, 1)
            """,
            ("benchmark", "Benchmark", "/tmp/benchmark", "/tmp/benchmark", summary, now, now),
        )
        repository._connection.executemany(
            """
            INSERT INTO assets(
                dataset_id, asset_id, match_key, display_path, image_path,
                annotation_paths_json, status, selectable, reason, issues_json,
                width, height, annotation_count, annotation_revision,
                labels_json, shape_types_json
            ) VALUES(?, ?, ?, ?, NULL, '[]', 'valid', 1, NULL, '[]', ?, ?, ?, NULL, ?, ?)
            """,
            (
                (
                    "benchmark",
                    f"asset-{index:06d}",
                    f"nested/asset-{index:06d}",
                    f"nested/asset-{index:06d}.png",
                    4096 if index % 5 else 8192,
                    3072,
                    index % 8,
                    '["scratch"]' if index % 10 == 0 else '["particle"]',
                    '["rotation"]' if index % 7 == 0 else '["rectangle"]',
                )
                for index in range(items)
            ),
        )
    repository.close()


def _measure(call, repeats: int) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        call()
        durations.append((perf_counter() - started) * 1000)
    return {
        "p50_ms": round(median(durations), 3),
        "p95_ms": round(_percentile(durations, 0.95), 3),
        "minimum_ms": round(min(durations), 3),
    }


def _walk_keyset(repository: DatasetRepository, page_size: int) -> int:
    cursor = None
    visited = 0
    while True:
        page = repository.list_assets_cursor("benchmark", cursor=cursor, limit=page_size)
        visited += len(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return visited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="labelone-search-") as directory:
        database_path = Path(directory) / "index.sqlite3"
        _seed(database_path, args.items)
        repository = DatasetRepository(database_path)
        results = {
            "items": args.items,
            "repeats": args.repeats,
            "page": _measure(lambda: repository.list_assets("benchmark", limit=100), args.repeats),
            "offset_tail": _measure(
                lambda: repository.list_assets("benchmark", offset=max(0, args.items - 100), limit=100),
                args.repeats,
            ),
            "keyset_first": _measure(
                lambda: repository.list_assets_cursor("benchmark", limit=100),
                args.repeats,
            ),
            "keyset_walk_1000": _measure(
                lambda: _walk_keyset(repository, 1000),
                max(1, min(args.repeats, 3)),
            ),
            "text_tail": _measure(
                lambda: repository.search_assets("benchmark", query=f"asset-{args.items - 1:06d}", mode="text", limit=100),
                args.repeats,
            ),
            "regex_tail": _measure(
                lambda: repository.search_assets("benchmark", query=rf"asset-{args.items - 1:06d}\.png$", mode="regex", limit=100),
                args.repeats,
            ),
            "condition": _measure(
                lambda: repository.search_assets("benchmark", query="class:scratch type:rotation annotations>0", mode="condition", limit=100),
                args.repeats,
            ),
        }
        repository.close()
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
