from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, sleep

from fastapi.testclient import TestClient
from PIL import Image

from labelone.config import Settings
from labelone.main import create_app


def _four_visualization_nodes() -> list[dict[str, object]]:
    return [
        {"id": "source", "kind": "source"},
        {"id": "original", "kind": "visualize", "parameters": {"label": "Original"}},
        {"id": "flip", "kind": "flip", "parameters": {"axis": "horizontal"}},
        {"id": "flipped", "kind": "visualize", "parameters": {"label": "Flipped"}},
        {"id": "resize", "kind": "resize", "parameters": {"width": 50, "height": 25}},
        {"id": "resized", "kind": "visualize", "parameters": {"label": "Resized"}},
        {"id": "crop", "kind": "crop", "parameters": {"x": 25, "y": 2, "width": 20, "height": 10}},
        {"id": "detail", "kind": "visualize", "parameters": {"label": "Detail"}},
    ]


def test_pipeline_validation_accepts_exactly_four_visualizations_and_api_rejects_five(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data"))
    four = _four_visualization_nodes()
    five = [
        *four[:-1],
        {"id": "detail", "kind": "visualize", "parameters": {"label": "Detail"}},
        {"id": "color", "kind": "color"},
        {"id": "fifth", "kind": "visualize", "parameters": {"label": "Fifth"}},
    ]

    with TestClient(app) as client:
        accepted = client.post("/api/v1/pipelines/validate", json={
            "nodes": four,
            "mode": "preview",
            "width": 100,
            "height": 50,
        })
        rejected = client.post("/api/v1/pipelines/validate", json={
            "nodes": five,
            "mode": "preview",
            "width": 100,
            "height": 50,
        })

    assert accepted.status_code == 200
    assert accepted.json()["visualization_count"] == 4
    assert rejected.status_code == 400
    assert rejected.json()["code"] == "pipeline_validation_error"
    assert "between one and four" in rejected.json()["message"]


def test_four_visualization_batch_result_survives_sqlite_and_service_restart(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (100, 50), (40, 60, 80)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({
        "shapes": [{
            "label": "box",
            "shape_type": "rectangle",
            "points": [[20, 10], [40, 10], [40, 20], [20, 20]],
        }],
    }), encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings)

    with TestClient(app) as client:
        registered = client.post("/api/v1/datasets/register", json={
            "dataset_id": "dataset",
            "root_dir": str(root),
            "layout": "same_directory",
        })
        asset_id = registered.json()["items"][0]["asset_id"]
        ensured = client.post("/api/v1/pipelines/precompute/ensure", json={
            "kind": "pipeline",
            "dataset_id": "dataset",
            "priority": "background",
            "concurrency": 1,
            "preferred_asset_ids": [asset_id],
            "pipeline_nodes": _four_visualization_nodes(),
            "output_policy": {"mode": "preview", "image_format": "png", "conflict": "reuse"},
        })
        assert ensured.status_code == 200
        job_id = ensured.json()["job"]["job_id"]
        deadline = monotonic() + 5
        while monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["state"] in {"succeeded", "succeeded_with_errors", "failed", "canceled"}:
                break
            sleep(0.01)
        lookup = client.post(f"/api/v1/jobs/{job_id}/items/lookup", json={"asset_ids": [asset_id]})
        before_restart = lookup.json()["items"][0]["result"]

    assert job["state"] == "succeeded"
    assert lookup.status_code == 200
    assert [
        (item["visualization_id"], item["width"], item["height"])
        for item in before_restart["visualizations"]
    ] == [
        ("original", 100, 50),
        ("flipped", 100, 50),
        ("resized", 50, 25),
        ("detail", 20, 10),
    ]
    assert len({item["artifact_id"] for item in before_restart["visualizations"]}) == 4
    assert before_restart["artifact_id"] == before_restart["visualizations"][-1]["artifact_id"]

    expected_points = [
        [[20.0, 10.0], [40.0, 10.0], [40.0, 20.0], [20.0, 20.0]],
        [[80.0, 10.0], [60.0, 10.0], [60.0, 20.0], [80.0, 20.0]],
        [[40.0, 5.0], [30.0, 5.0], [30.0, 10.0], [40.0, 10.0]],
        [[15.0, 3.0], [5.0, 3.0], [5.0, 8.0], [15.0, 8.0]],
    ]
    expected_timing_keys = [
        {"source", "original"},
        {"source", "flip", "flipped"},
        {"source", "flip", "resize", "resized"},
        {"source", "flip", "resize", "crop", "detail"},
    ]
    for visualization, points, timing_keys in zip(
        before_restart["visualizations"],
        expected_points,
        expected_timing_keys,
        strict=True,
    ):
        document = visualization["annotation_document"]
        assert (document["imageWidth"], document["imageHeight"]) == (
            visualization["width"],
            visualization["height"],
        )
        assert document["shapes"], visualization["visualization_id"]
        assert document["shapes"][0]["points"] == points
        assert set(visualization["operator_timings_ms"]) == timing_keys

    restarted = create_app(settings)
    with TestClient(restarted) as client:
        recovered = client.post(f"/api/v1/jobs/{job_id}/items/lookup", json={"asset_ids": [asset_id]})
        after_restart = recovered.json()["items"][0]["result"]
        artifacts = [
            client.get(f"/api/v1/pipeline-artifacts/{item['artifact_id']}")
            for item in after_restart["visualizations"]
        ]

    assert recovered.status_code == 200
    assert after_restart == before_restart
    assert all(response.status_code == 200 for response in artifacts)
    assert all(response.headers["content-type"] == "image/png" for response in artifacts)
