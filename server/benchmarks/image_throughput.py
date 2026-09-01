from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
from time import perf_counter
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def create_dataset(root: Path, *, images: int, width: int, height: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(images):
        image = Image.new("RGB", (width, height), (20 + index % 80, 40, 70))
        draw = ImageDraw.Draw(image)
        draw.rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), outline=(100, 220, 190), width=8)
        image.save(root / f"image-{index:04d}.jpg", quality=88)
        (root / f"image-{index:04d}.json").write_text(
            json.dumps({"imagePath": f"image-{index:04d}.jpg", "shapes": []}),
            encoding="utf-8",
        )


async def request_batch(client: httpx.AsyncClient, urls: list[str], concurrency: int) -> dict[str, object]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: list[str] = []
    hits = 0

    async def request(url: str) -> None:
        nonlocal hits
        async with semaphore:
            started = perf_counter()
            try:
                response = await client.get(url)
                response.raise_for_status()
                if response.headers.get("x-labelone-cache") == "hit":
                    hits += 1
            except Exception as exc:
                errors.append(str(exc))
            finally:
                latencies.append((perf_counter() - started) * 1000)

    started = perf_counter()
    await asyncio.gather(*(request(url) for url in urls))
    elapsed = perf_counter() - started
    return {
        "requests": len(urls),
        "errors": len(errors),
        "cache_hits": hits,
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(len(urls) / elapsed, 2) if elapsed else 0,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
        },
        "sample_errors": errors[:3],
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_id = f"bench-{uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="labelone-bench-") as directory:
        root = Path(directory)
        create_dataset(root, images=args.images, width=args.width, height=args.height)
        async with httpx.AsyncClient(base_url=args.api, timeout=120) as client:
            registered = await client.post("/datasets/register", json={
                "dataset_id": dataset_id,
                "root_dir": str(root),
                "layout": "same_directory",
                "validate_images": True,
                "validate_annotations": True,
            })
            registered.raise_for_status()
            items = registered.json()["items"]
            urls = [f"/datasets/{dataset_id}/assets/{item['asset_id']}/thumbnail?max_size={args.thumbnail}" for item in items if item["selectable"]]
            annotation_urls = [f"/datasets/{dataset_id}/assets/{item['asset_id']}/annotation" for item in items if item["selectable"]]
            page_started = perf_counter()
            page = await client.get(f"/datasets/{dataset_id}/assets?limit=200")
            page.raise_for_status()
            asset_page_ms = (perf_counter() - page_started) * 1000
            cold = await request_batch(client, urls, args.concurrency)
            warm = await request_batch(client, urls, args.concurrency)
            annotations = await request_batch(client, annotation_urls, args.concurrency)
            removed = await client.delete(f"/datasets/{dataset_id}")
            removed.raise_for_status()
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "workload": {
            "images": args.images,
            "source_size": [args.width, args.height],
            "thumbnail": args.thumbnail,
            "concurrency": args.concurrency,
        },
        "cold": cold,
        "warm": warm,
        "asset_page": {"items": len(page.json()["items"]), "latency_ms": round(asset_page_ms, 3)},
        "annotation_reads": annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8766/api/v1")
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--thumbnail", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
