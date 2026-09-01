from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

from PIL import Image
import pytest

from labelone.annotations import AnnotationStore
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.pipelines import PipelineCancelled, PipelineEngine, PipelineNode, PipelinePreviewRequest
from labelone.pipelines.registry import validate_nodes


class _FeatureManagerFixture:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def state(self, model_id: str):
        return SimpleNamespace(state="loaded", model_id=model_id)

    def load(self, model_id: str, providers: list[str]):
        raise AssertionError("already-loaded fixture must not reload")

    def layers(self, model_id: str):
        return SimpleNamespace(layers=[SimpleNamespace(id="backbone.3", captureable=True, spatial=True)])

    def predict(self, model_id: str, image_path: Path, capture_layers: list[str], parameters: dict[str, object]):
        with Image.open(image_path) as image:
            size = image.size
        self.calls.append({
            "model_id": model_id,
            "size": size,
            "capture_layers": capture_layers,
            "parameters": parameters,
        })
        return SimpleNamespace(artifacts=[SimpleNamespace(id="tensor-preview", preview_available=True)])


class _FeatureArtifactFixture:
    def __init__(self, preview_path: Path) -> None:
        self.preview = preview_path
        self.discarded: list[str] = []

    def preview_path(self, artifact_id: str):
        assert artifact_id == "tensor-preview"
        return self.preview, "image/png"

    def discard(self, artifact_id: str) -> None:
        self.discarded.append(artifact_id)


def test_pipeline_transforms_image_and_annotation_points(tmp_path: Path) -> None:
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
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    annotations = AnnotationStore(repository, tmp_path / "backups")
    engine = PipelineEngine(repository, annotations, tmp_path / "artifacts", maximum_preview_cache_entries=1)
    progress: list[dict[str, object]] = []

    result = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[
            PipelineNode(id="crop", kind="crop", parameters={"x": 10, "y": 5, "width": 80, "height": 40}),
            PipelineNode(id="resize", kind="resize", parameters={"width": 40, "height": 20}),
            PipelineNode(id="flip", kind="flip", parameters={"axis": "horizontal"}),
        ],
    ), progress=progress.append)

    assert (result.width, result.height) == (40, 20)
    assert result.annotation_document["shapes"][0]["points"] == [
        [35.0, 2.5], [25.0, 2.5], [25.0, 7.5], [35.0, 7.5]
    ]
    artifact_path, media_type = engine.artifact_path(result.artifact_id)
    assert artifact_path.is_file()
    assert media_type == "image/webp"
    assert set(result.operator_timings_ms) == {"source", "crop", "resize", "flip", "visualize"}
    assert result.operator_average_timings_ms == pytest.approx(result.operator_timings_ms)
    assert result.timing_sample_count == {"source": 1, "crop": 1, "resize": 1, "flip": 1, "visualize": 1}
    assert [item["completed_steps"] for item in progress] == [1, 2, 3, 4, 5, 6]
    assert all(item["total_steps"] == 6 for item in progress)
    assert [item["phase"] for item in progress] == [
        "loaded", "operator", "operator", "operator", "visualization", "completed",
    ]
    assert progress[1]["node_id"] == "crop"
    assert progress[-1]["phase"] == "completed"
    assert len(result.visualizations) == 1
    assert result.visualizations[0].artifact_id == result.artifact_id
    mapping = result.visualizations[0].coordinate_mapping
    assert mapping.kind == "affine"
    assert mapping.source_to_output == pytest.approx((-0.5, 0.0, 0.0, 0.5, 45.0, -2.5))
    assert mapping.output_to_source == pytest.approx((-2.0, 0.0, 0.0, 2.0, 90.0, 5.0))
    assert (mapping.source_width, mapping.source_height) == (100, 50)
    assert (mapping.output_width, mapping.output_height) == (40, 20)
    assert mapping.topology_safe is False
    assert len(mapping.coordinate_space_id) == 24
    assert result.cache_hit is False

    second = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[
            PipelineNode(id="crop", kind="crop", parameters={"x": 10, "y": 5, "width": 80, "height": 40}),
            PipelineNode(id="resize", kind="resize", parameters={"width": 40, "height": 20}),
            PipelineNode(id="flip", kind="flip", parameters={"axis": "horizontal"}),
        ],
    ))
    assert second.cache_hit is True
    assert second.artifact_id == result.artifact_id
    assert second.timing_sample_count == {"source": 1, "crop": 1, "resize": 1, "flip": 1, "visualize": 1}

    third = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        output_format="png",
        nodes=[
            PipelineNode(id="crop", kind="crop", parameters={"x": 10, "y": 5, "width": 80, "height": 40}),
            PipelineNode(id="resize", kind="resize", parameters={"width": 40, "height": 20}),
            PipelineNode(id="flip", kind="flip", parameters={"axis": "horizontal"}),
        ],
    ))
    assert third.cache_hit is False
    assert third.timing_sample_count == {"source": 2, "crop": 2, "resize": 2, "flip": 2, "visualize": 2}

    evicted = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[
            PipelineNode(id="crop", kind="crop", parameters={"x": 10, "y": 5, "width": 80, "height": 40}),
            PipelineNode(id="resize", kind="resize", parameters={"width": 40, "height": 20}),
            PipelineNode(id="flip", kind="flip", parameters={"axis": "horizontal"}),
        ],
    ))
    assert evicted.cache_hit is False
    assert evicted.timing_sample_count == {"source": 3, "crop": 3, "resize": 3, "flip": 3, "visualize": 3}


def test_frequency_pipeline_suppresses_spatial_annotations_and_overlay_mapping(tmp_path: Path) -> None:
    root = tmp_path / "frequency-dataset"
    root.mkdir()
    Image.new("RGB", (32, 24), (40, 60, 80)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": [{"label": "box", "shape_type": "rectangle", "points": [[2, 2], [8, 8]]}]}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="frequency"))
    repository = DatasetRepository(tmp_path / "frequency.sqlite3")
    repository.register(scan)
    engine = PipelineEngine(repository, AnnotationStore(repository, tmp_path / "frequency-backups"), tmp_path / "frequency-artifacts")

    result = engine.preview(PipelinePreviewRequest(dataset_id="frequency", asset_id=scan.items[0].asset_id, nodes=[
        PipelineNode(id="fft", kind="opencv.fourier_transform", parameters={"mode": "magnitude", "center": True}),
        PipelineNode(id="display", kind="visualize", parameters={"label": "频谱"}),
    ]))

    view = result.visualizations[0]
    assert view.content_kind == "frequency_spectrum"
    assert view.overlay_compatible is False
    assert view.coordinate_mapping.kind == "unavailable"
    assert view.annotation_document["shapes"] == []


def test_model_feature_node_produces_overlay_sized_visualization_and_keeps_comparison_display(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (80, 40), (20, 40, 60)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    annotations = AnnotationStore(repository, tmp_path / "backups")
    preview_path = tmp_path / "feature.png"
    Image.new("RGB", (16, 8), (220, 30, 40)).save(preview_path)
    manager = _FeatureManagerFixture()
    artifacts = _FeatureArtifactFixture(preview_path)
    engine = PipelineEngine(
        repository,
        annotations,
        tmp_path / "artifacts",
        model_manager=manager,  # type: ignore[arg-type]
        model_artifacts=artifacts,  # type: ignore[arg-type]
    )
    request = PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        output_format="png",
        nodes=[
            PipelineNode(id="before", kind="visualize", parameters={"label": "原图对照"}),
            PipelineNode(id="feature", kind="model_feature", parameters={"model_id": "fixture", "layer_id": "backbone.3", "projection": "channel", "normalization": "zscore", "channel": 3, "clip": "p5p95"}),
            PipelineNode(id="after", kind="visualize", parameters={"label": "中间层"}),
        ],
    )

    result = engine.preview(request)

    assert [(item.label, item.width, item.height) for item in result.visualizations] == [
        ("原图对照", 80, 40),
        ("中间层", 80, 40),
    ]
    assert [item.content_kind for item in result.visualizations] == ["image", "model_feature"]
    assert all(item.overlay_compatible for item in result.visualizations)
    before_path, _ = engine.artifact_path(result.visualizations[0].artifact_id)
    feature_path, _ = engine.artifact_path(result.visualizations[1].artifact_id)
    assert Image.open(before_path).getpixel((0, 0)) == (20, 40, 60)
    assert Image.open(feature_path).getpixel((0, 0)) == (220, 30, 40)
    assert manager.calls[0]["size"] == (80, 40)
    assert manager.calls[0]["capture_layers"] == ["backbone.3"]
    assert manager.calls[0]["parameters"] == {"feature_transform": {
        "projection": "channel",
        "normalization": "zscore",
        "interpolation": "bilinear",
        "spatial_scale": 1,
        "channel": 3,
        "gain": 1,
        "gamma": 1,
        "clip_percentiles": [5, 95],
    }}
    assert artifacts.discarded == ["tensor-preview"]

    cached = engine.preview(request)
    assert cached.cache_hit is True
    assert len(manager.calls) == 1


def test_identical_concurrent_previews_share_one_computation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (32, 24), (40, 60, 80)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    annotations = AnnotationStore(repository, tmp_path / "backups")
    engine = PipelineEngine(repository, annotations, tmp_path / "artifacts")
    request = PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[PipelineNode(id="display", kind="visualize", parameters={"label": "Display"})],
    )
    started = Event()
    release = Event()
    call_lock = Lock()
    call_count = 0
    original = engine._compute_preview

    def delayed_compute(*args, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_compute_preview", delayed_compute)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(engine.preview, request)
        assert started.wait(2)
        second_future = executor.submit(engine.preview, request)
        assert second_future.running()
        release.set()
        first = first_future.result(timeout=3)
        second = second_future.result(timeout=3)

    assert call_count == 1
    assert first.artifact_id == second.artifact_id
    assert {first.cache_hit, second.cache_hit} == {False, True}
    assert list((tmp_path / "artifacts").rglob("*.part")) == []


def test_preview_cancellation_stops_at_operator_boundary_and_allows_retry(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (32, 24), (40, 60, 80)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    annotations = AnnotationStore(repository, tmp_path / "backups")
    engine = PipelineEngine(repository, annotations, tmp_path / "artifacts")
    request = PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[
            PipelineNode(id="color", kind="color", parameters={"brightness": 1.1}),
            PipelineNode(id="resize", kind="resize", parameters={"width": 16, "height": 12}),
            PipelineNode(id="display", kind="visualize", parameters={"label": "Display"}),
        ],
    )
    canceled = Event()

    def cancel_after_first_operator(payload: dict[str, object]) -> None:
        if payload.get("phase") == "operator":
            canceled.set()

    with pytest.raises(PipelineCancelled):
        engine.preview(request, progress=cancel_after_first_operator, canceled=canceled.is_set)

    retry = engine.preview(request)
    assert retry.cache_hit is False
    assert (retry.width, retry.height) == (16, 12)


def test_crop_clips_partial_shapes_and_drops_shapes_outside() -> None:
    image = Image.new("RGB", (100, 80), "black")
    document = {
        "imagePath": "image.png",
        "shapes": [
            {"label": "partial", "shape_type": "rectangle", "points": [[5, 15], [30, 15], [30, 35], [5, 35]]},
            {"label": "outside", "shape_type": "rectangle", "points": [[0, 0], [5, 0], [5, 5], [0, 5]]},
            {"label": "line", "shape_type": "line", "points": [[0, 25], [50, 25]]},
        ],
    }
    output = PipelineEngine._crop(image, document, {"x": 10, "y": 10, "width": 40, "height": 30})
    assert output.size == (40, 30)
    assert [shape["label"] for shape in document["shapes"]] == ["partial", "line"]
    assert document["shapes"][0]["shape_type"] == "polygon"
    assert document["shapes"][0]["points"] == [[0, 5.0], [20.0, 5.0], [20.0, 25.0], [0, 25.0]]
    assert document["shapes"][1]["points"] == [[0.0, 15.0], [40.0, 15.0]]
    assert (document["imageWidth"], document["imageHeight"]) == (40, 30)


def test_anisotropic_resize_promotes_circle_and_rotation_to_polygon() -> None:
    image = Image.new("RGB", (100, 100), "black")
    document = {
        "shapes": [
            {"label": "circle", "shape_type": "circle", "points": [[50, 50], [60, 50]]},
            {"label": "rotation", "shape_type": "rotation", "direction": 0.2, "points": [[10, 10], [30, 10], [30, 20], [10, 20]]},
        ],
    }
    output = PipelineEngine._resize(image, document, {"width": 200, "height": 50})
    assert output.size == (200, 50)
    circle, rotation = document["shapes"]
    assert circle["shape_type"] == "polygon" and len(circle["points"]) == 32
    assert rotation["shape_type"] == "polygon" and "direction" not in rotation
    assert rotation["points"] == [[20.0, 5.0], [60.0, 5.0], [60.0, 10.0], [20.0, 10.0]]


@pytest.mark.parametrize(
    ("degrees", "expected_size", "expected_point"),
    [
        (0, (100, 50), [20.0, 10.0]),
        (90, (50, 100), [40.0, 20.0]),
        (180, (100, 50), [80.0, 40.0]),
        (270, (50, 100), [10.0, 80.0]),
    ],
)
def test_right_angle_rotation_updates_image_and_annotation_coordinates(
    degrees: int, expected_size: tuple[int, int], expected_point: list[float]
) -> None:
    image = Image.new("RGB", (100, 50), "black")
    document = {"shapes": [{"label": "point", "shape_type": "point", "points": [[20, 10]]}]}
    output = PipelineEngine._rotate(image, document, {"degrees": degrees})
    assert output.size == expected_size
    assert document["shapes"][0]["points"][0] == expected_point


def test_vertical_flip_and_non_spatial_operators_preserve_annotation_semantics() -> None:
    image = Image.new("RGB", (100, 50), (40, 60, 80))
    document = {"shapes": [{"label": "line", "shape_type": "line", "points": [[10, 5], [30, 20]], "flags": {"reviewed": True}}]}
    flipped = PipelineEngine._flip(image, document, {"axis": "vertical"})
    assert flipped.size == image.size
    assert document["shapes"][0]["points"] == [[10, 45.0], [30, 30.0]]
    before = json.loads(json.dumps(document))
    assert PipelineEngine._color(image, {"brightness": 1, "contrast": 1, "saturation": 1}).size == image.size
    assert PipelineEngine._noise(image, {"radius": 0, "percent": 100, "threshold": 0}).size == image.size
    assert document == before


@pytest.mark.parametrize(("axis", "expected_point"), [("horizontal", [80.0, 10.0]), ("vertical", [20.0, 40.0])])
def test_pipeline_preview_flip_returns_transformed_annotations_without_mutating_source(
    tmp_path: Path, axis: str, expected_point: list[float]
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (100, 50), "black").save(root / "image.png")
    source_document = {"shapes": [{"label": "point", "shape_type": "point", "points": [[20, 10]]}]}
    (root / "image.json").write_text(json.dumps(source_document), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    annotations = AnnotationStore(repository, tmp_path / "backups")
    engine = PipelineEngine(repository, annotations, tmp_path / "artifacts")
    result = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[
            PipelineNode(id="source", kind="source"),
            PipelineNode(id="flip", kind="flip", parameters={"axis": axis}),
            PipelineNode(id="display", kind="visualize", parameters={"label": axis}),
        ],
        output_format="png",
    ))
    assert (result.width, result.height) == (100, 50)
    assert result.annotation_document["shapes"][0]["points"][0] == expected_point
    assert (result.annotation_document["imageWidth"], result.annotation_document["imageHeight"]) == (100, 50)
    artifact_path, _media_type = engine.artifact_path(result.artifact_id)
    with Image.open(artifact_path) as artifact:
        assert artifact.size == (100, 50)
    assert annotations.load("dataset", scan.items[0].asset_id).document["shapes"][0]["points"][0] == [20, 10]

def test_pipeline_visualizations_tap_distinct_upstream_stages_and_write_distinct_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (100, 50), (40, 60, 80)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    repository = DatasetRepository(tmp_path / "index.sqlite3")
    repository.register(scan)
    annotations = AnnotationStore(repository, tmp_path / "backups")
    engine = PipelineEngine(repository, annotations, tmp_path / "artifacts")

    result = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=scan.items[0].asset_id,
        nodes=[
            PipelineNode(id="source", kind="source"),
            PipelineNode(id="resize", kind="resize", parameters={"width": 40, "height": 20}),
            PipelineNode(id="resized", kind="visualize", parameters={"label": "Resized"}),
            PipelineNode(id="crop", kind="crop", parameters={"x": 5, "y": 2, "width": 20, "height": 10}),
            PipelineNode(id="detail", kind="visualize", parameters={"label": "Detail"}),
        ],
    ))

    assert [(item.visualization_id, item.label, item.width, item.height) for item in result.visualizations] == [
        ("resized", "Resized", 40, 20),
        ("detail", "Detail", 20, 10),
    ]
    assert len({item.artifact_id for item in result.visualizations}) == 2
    assert result.artifact_id == result.visualizations[-1].artifact_id
    assert (result.width, result.height) == (20, 10)
    assert set(result.visualizations[0].operator_timings_ms) == {"source", "resize", "resized"}
    assert set(result.visualizations[1].operator_timings_ms) == {"source", "resize", "crop", "detail"}
    assert result.visualizations[0].coordinate_mapping.source_to_output == pytest.approx((0.4, 0, 0, 0.4, 0, 0))
    assert result.visualizations[1].coordinate_mapping.source_to_output == pytest.approx((0.4, 0, 0, 0.4, -5, -2))
    assert result.visualizations[0].coordinate_mapping.coordinate_space_id != result.visualizations[1].coordinate_mapping.coordinate_space_id
    for visualization in result.visualizations:
        artifact_path, media_type = engine.artifact_path(visualization.artifact_id)
        assert artifact_path.is_file()
        assert media_type == visualization.media_type


def test_pipeline_timing_aggregation_is_thread_safe_and_lru_bounded(tmp_path: Path) -> None:
    engine = PipelineEngine(None, None, tmp_path / "artifacts", maximum_timing_signatures=2)  # type: ignore[arg-type]

    def nodes(node_id: str):
        return validate_nodes([
            {"id": "source", "kind": "source"},
            {"id": node_id, "kind": "flip"},
            {"id": "display", "kind": "visualize"},
        ])

    first = nodes("first")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: engine._record_operator_timings(first, {"first": 2.0}), range(80)))

    assert all(averages == {"first": 2.0} for averages, _ in results)
    assert max(counts["first"] for _, counts in results) == 80
    first_signature = engine._pipeline_signature(first)
    second = nodes("second")
    third = nodes("third")
    engine._record_operator_timings(second, {"second": 1.0})
    engine._record_operator_timings(third, {"third": 1.0})

    assert len(engine._timing_history) == 2
    assert first_signature not in engine._timing_history
