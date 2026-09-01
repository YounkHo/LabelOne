from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import platform
from time import monotonic, perf_counter

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


async def run(args: argparse.Namespace) -> dict[str, object]:
    timeout = httpx.Timeout(30.0, connect=5.0)
    limits = httpx.Limits(max_connections=max(8, args.concurrency * 2), max_keepalive_connections=max(4, args.concurrency))
    latencies: list[float] = []
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    samples: list[str] = []
    pipeline_runs = 0
    started_at = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(base_url=args.api, timeout=timeout, limits=limits) as client:
        datasets = (await client.get("/datasets")).raise_for_status().json()["datasets"]
        if not datasets:
            raise RuntimeError("Mixed soak requires at least one registered dataset")
        dataset = next((item for item in datasets if item["name"] == args.dataset), datasets[0])
        dataset_id = dataset["dataset_id"]
        page = (await client.get(f"/datasets/{dataset_id}/assets-cursor?limit=100")).raise_for_status().json()
        asset = next((item for item in page["items"] if item["selectable"]), None)
        if asset is None:
            raise RuntimeError("Mixed soak dataset has no selectable asset")
        asset_id = asset["asset_id"]
        metadata_response = await client.get(f"/datasets/{dataset_id}/assets/{asset_id}/tiles/metadata")
        tile_url = None
        if metadata_response.status_code == 200:
            metadata = metadata_response.json()
            tile_url = f"/datasets/{dataset_id}/assets/{asset_id}/tiles/{metadata['max_level']}/0/0?format=webp"
        jobs = (await client.get("/jobs?limit=1")).raise_for_status().json()["jobs"]
        event_url = f"/jobs/{jobs[0]['job_id']}/events?format=json&after=0&limit=20" if jobs else "/jobs?limit=1"
        requests: list[tuple[str, str, dict[str, str] | None]] = [
            ("health", "/health", None),
            ("datasets", "/datasets", None),
            ("assets_cursor", f"/datasets/{dataset_id}/assets-cursor?limit=50", None),
            ("search_cursor", f"/datasets/{dataset_id}/search-cursor?q=&mode=smart&limit=50", None),
            ("annotation", f"/datasets/{dataset_id}/assets/{asset_id}/annotation", None),
            ("thumbnail", f"/datasets/{dataset_id}/assets/{asset_id}/thumbnail?max_size=256&format=webp", None),
            ("image_range", f"/datasets/{dataset_id}/assets/{asset_id}/image", {"Range": "bytes=0-65535"}),
            ("jobs", "/jobs?limit=100", None),
            ("scheduler", "/jobs-scheduler", None),
            ("events_json", event_url, None),
        ]
        if tile_url:
            requests.append(("tile", tile_url, None))
        deadline = monotonic() + args.duration

        async def requester(worker_id: int) -> None:
            index = worker_id
            while monotonic() < deadline:
                name, url, headers = requests[index % len(requests)]
                index += 1
                began = perf_counter()
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    counts[name] += 1
                except Exception as exc:  # noqa: BLE001 - benchmark records every transport/server failure
                    errors[name] += 1
                    if len(samples) < 20:
                        samples.append(f"{name}: {exc}")
                finally:
                    latencies.append((perf_counter() - began) * 1000)
                await asyncio.sleep(args.delay)

        async def previewer() -> None:
            nonlocal pipeline_runs
            request = {
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "nodes": [
                    {"id": "soak-crop", "kind": "crop", "parameters": {"margin_ratio": 0.02}},
                    {"id": "soak-color", "kind": "color", "parameters": {"brightness": 1.01, "contrast": 1.02}},
                ],
            }
            while monotonic() < deadline:
                began = perf_counter()
                try:
                    response = await client.post("/pipelines/preview", json=request)
                    response.raise_for_status()
                    counts["pipeline_preview"] += 1
                    pipeline_runs += 1
                except Exception as exc:  # noqa: BLE001 - benchmark records every failure
                    errors["pipeline_preview"] += 1
                    if len(samples) < 20:
                        samples.append(f"pipeline_preview: {exc}")
                finally:
                    latencies.append((perf_counter() - began) * 1000)
                await asyncio.sleep(args.pipeline_interval)

        await asyncio.gather(
            *(requester(worker_id) for worker_id in range(args.concurrency)),
            previewer(),
        )

    total_requests = sum(counts.values()) + sum(errors.values())
    duration = args.duration
    finite = [value for value in latencies if math.isfinite(value)]
    return {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "workload": {
            "duration_seconds": duration,
            "concurrency": args.concurrency,
            "request_delay_seconds": args.delay,
            "pipeline_interval_seconds": args.pipeline_interval,
            "dataset_name": args.dataset,
        },
        "result": {
            "requests": total_requests,
            "successful": sum(counts.values()),
            "errors": sum(errors.values()),
            "requests_per_second": round(total_requests / duration, 3),
            "pipeline_runs": pipeline_runs,
            "latency_ms": {
                "mean": round(sum(finite) / len(finite), 3) if finite else 0.0,
                "p50": round(percentile(finite, 0.50), 3),
                "p95": round(percentile(finite, 0.95), 3),
                "p99": round(percentile(finite, 0.99), 3),
                "max": round(max(finite), 3) if finite else 0.0,
            },
            "success_by_endpoint": dict(sorted(counts.items())),
            "errors_by_endpoint": dict(sorted(errors.items())),
            "sample_errors": samples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8766/api/v1")
    parser.add_argument("--dataset", default="final-demo")
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--pipeline-interval", type=float, default=20.0)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
