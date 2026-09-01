from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
from time import perf_counter
from uuid import uuid4

import httpx
import numpy as np
from PIL import Image


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def create_dataset(root: Path, count: int, width: int, height: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        generator = np.random.default_rng(index)
        array = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(root / f"image-{index:04d}.jpg", quality=86)
        (root / f"image-{index:04d}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_id = f"job-bench-{uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="labelone-job-bench-") as directory:
        root = Path(directory)
        create_dataset(root, args.images, args.width, args.height)
        async with httpx.AsyncClient(base_url=args.api, timeout=180) as client:
            registered = await client.post("/datasets/register", json={
                "dataset_id": dataset_id,
                "root_dir": str(root),
                "layout": "same_directory",
            })
            registered.raise_for_status()
            request = {
                "kind": "pipeline",
                "dataset_id": dataset_id,
                "concurrency": args.concurrency,
                "pipeline_nodes": [
                    {"id": "crop", "kind": "crop", "parameters": {"margin_ratio": 0.03}},
                    {"id": "color", "kind": "color", "parameters": {"brightness": 1.02, "contrast": 1.08}},
                    {"id": "noise", "kind": "noise", "parameters": {"radius": 1.0, "percent": 120}},
                ],
            }
            started = perf_counter()
            created = await client.post("/jobs", headers={"Idempotency-Key": uuid4().hex}, json=request)
            created.raise_for_status()
            job_id = created.json()["job_id"]
            polls = 0
            while True:
                polls += 1
                response = await client.get(f"/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
                if job["state"] in {"succeeded", "succeeded_with_errors", "failed", "canceled"}:
                    break
                await asyncio.sleep(0.1)
            elapsed = perf_counter() - started
            items_response = await client.get(f"/jobs/{job_id}/items?limit=1000")
            items_response.raise_for_status()
            items = items_response.json()["items"]
            removed = await client.delete(f"/datasets/{dataset_id}")
            removed.raise_for_status()
    durations = [
        (datetime.fromisoformat(item["finished_at"]) - datetime.fromisoformat(item["started_at"])).total_seconds() * 1000
        for item in items
        if item.get("started_at") and item.get("finished_at")
    ]
    return {
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "workload": {"images": args.images, "source_size": [args.width, args.height], "concurrency": args.concurrency, "operators": ["crop", "color", "noise"]},
        "result": {
            "state": job["state"],
            "completed": job["completed"],
            "failed": job["failed"],
            "elapsed_seconds": round(elapsed, 4),
            "items_per_second": round(args.images / elapsed, 3),
            "polls": polls,
            "service_latency_ms": {
                "mean": round(statistics.fmean(durations), 3),
                "p50": round(percentile(durations, 0.50), 3),
                "p95": round(percentile(durations, 0.95), 3),
                "p99": round(percentile(durations, 0.99), 3),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8767/api/v1")
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
