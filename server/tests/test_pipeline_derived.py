from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

from PIL import Image
from pydantic import ValidationError
import pytest

from labelone.annotations import AnnotationStore
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError
from labelone.jobs import BatchJobRequest, JobRepository, JobService
from labelone.pipelines import PipelineEngine, PipelineNode, PipelineOutputPolicy, PipelinePreviewRequest
from labelone.pipelines.derived import DerivedDatasetWriter, PreparedDerivedOutput, clip_annotation_document


def _runtime(tmp_path: Path, *, size: tuple[int, int] = (100, 60)):
    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    image_path = root / "nested" / "image.png"
    Image.new("RGB", size, (40, 80, 120)).save(image_path)
    (root / "nested" / "image.json").write_text(json.dumps({
        "imagePath": "image.png",
        "shapes": [
            {"label": "cross", "shape_type": "rectangle", "points": [[40, 10], [80, 30]]},
            {"label": "outside", "shape_type": "rectangle", "points": [[90, 45], [99, 59]]},
            {"label": "point", "shape_type": "point", "points": [[95, 55]]},
        ],
    }), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(dataset_id="dataset", root_dir=root, layout="same_directory"))
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    datasets.register(scan)
    annotations = AnnotationStore(datasets, tmp_path / "backups")
    engine = PipelineEngine(datasets, annotations, tmp_path / "artifacts")
    jobs = JobRepository(tmp_path / "index.sqlite3", datasets)
    return root, scan.items[0], datasets, jobs, engine


def _wait(repository: JobRepository, job_id: str, states: set[str], timeout: float = 6.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = repository.get(job_id)
        if job.state in states:
            return job
        sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {states}")


def _derived_request(output_root: Path, *, width: int = 50) -> BatchJobRequest:
    return BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        concurrency=1,
        pipeline_nodes=[PipelineNode(id="resize", kind="resize", parameters={"width": width, "height": 30})],
        output_policy={
            "mode": "derived_dataset",
            "output_root": output_root,
            "image_format": "png",
            "conflict": "reuse",
        },
    )


def test_derived_execution_contributes_real_samples_to_pipeline_timing_average(tmp_path: Path) -> None:
    _, asset, datasets, jobs, engine = _runtime(tmp_path)
    nodes = [PipelineNode(id="color", kind="color", parameters={"contrast": 1.1})]
    engine.export_derived_item(
        job_id="timing-job",
        dataset_id="dataset",
        asset_id=asset.asset_id,
        nodes=nodes,
        policy=PipelineOutputPolicy(
            mode="derived_dataset",
            output_root=(tmp_path / "timing-output").resolve(),
        ),
    )

    preview = engine.preview(PipelinePreviewRequest(
        dataset_id="dataset",
        asset_id=asset.asset_id,
        nodes=nodes,
    ))

    assert preview.timing_sample_count == {"source": 2, "color": 2, "visualize": 1}
    assert set(preview.operator_average_timings_ms) == {"source", "color", "visualize"}
    jobs.close()
    datasets.close()


def test_batch_derived_dataset_is_atomically_published_and_reused_by_fingerprint(tmp_path: Path) -> None:
    _, _, datasets, jobs, engine = _runtime(tmp_path)
    output_root = (tmp_path / "derived").resolve()
    service = JobService(jobs, datasets, engine, None)  # type: ignore[arg-type]
    created = service.create(_derived_request(output_root))
    finished = _wait(jobs, created.job_id, {"succeeded"})

    image_path = output_root / "nested" / "image.png"
    annotation_path = output_root / "nested" / "image.json"
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert output_root.is_dir()
    assert Image.open(image_path).size == (50, 30)
    assert document["imagePath"] == "image.png"
    assert (document["imageWidth"], document["imageHeight"]) == (50, 30)
    cross = next(shape for shape in document["shapes"] if shape["label"] == "cross")
    assert cross["points"] == [[20.0, 5.0], [40.0, 5.0], [40.0, 15.0], [20.0, 15.0]]
    manifest = json.loads((output_root / ".labelone-derived.json").read_text(encoding="utf-8"))
    assert manifest["item_count"] == manifest["output_count"] == 1
    assert finished.items[0].result["outputs"][0]["image_relative_path"] == "nested/image.png"

    reused_job = service.create(_derived_request(output_root))
    assert _wait(jobs, reused_job.job_id, {"succeeded"}).completed == 1
    conflicting = service.create(_derived_request(output_root, width=40))
    conflict = _wait(jobs, conflicting.job_id, {"failed"})
    assert "different content" in (conflict.error or "")
    assert Image.open(image_path).size == (50, 30)
    assert not list(tmp_path.glob(".*.part"))
    service.close()
    jobs.close()
    datasets.close()


def test_tile_batch_outputs_overlap_edge_dimensions_and_clipped_translated_annotations(tmp_path: Path) -> None:
    source_root, asset, datasets, jobs, engine = _runtime(tmp_path)
    output_root = (tmp_path / "tiles").resolve()
    policy = PipelineOutputPolicy(mode="derived_dataset", output_root=output_root, image_format="png")
    nodes = [PipelineNode(id="tile", kind="tile", parameters={
        "tile_width": 64,
        "tile_height": 40,
        "overlap_x": 16,
        "overlap_y": 10,
        "include_partial": True,
    })]

    item = engine.export_derived_item(
        job_id="tile-job",
        dataset_id="dataset",
        asset_id=asset.asset_id,
        nodes=nodes,
        policy=policy,
    )
    published = engine.finalize_derived(
        job_id="tile-job",
        dataset_id="dataset",
        policy=policy,
        item_results=[item.model_dump(mode="json")],
        expected_item_count=1,
    )

    assert published.output_count == 4
    dimensions = {(output.tile["column"], output.tile["row"]): (output.width, output.height) for output in item.outputs}
    assert dimensions == {(0, 0): (64, 40), (1, 0): (52, 40), (0, 1): (64, 30), (1, 1): (52, 30)}
    top_left = json.loads((output_root / "nested" / "image__tile_r0000_c0000.json").read_text(encoding="utf-8"))
    cross = next(shape for shape in top_left["shapes"] if shape["label"] == "cross")
    assert cross["shape_type"] == "polygon"
    assert min(point[0] for point in cross["points"]) == 40
    assert max(point[0] for point in cross["points"]) == 64
    bottom_right = json.loads((output_root / "nested" / "image__tile_r0001_c0001.json").read_text(encoding="utf-8"))
    translated_point = next(shape for shape in bottom_right["shapes"] if shape["label"] == "point")
    assert translated_point["points"] == [[47.0, 25.0]]
    assert bottom_right["imagePath"] == "image__tile_r0001_c0001.png"
    assert (bottom_right["imageWidth"], bottom_right["imageHeight"]) == (52, 30)
    assert source_root.is_dir()
    jobs.close()
    datasets.close()


def test_linestrip_clipping_preserves_open_runs() -> None:
    document = {
        "shapes": [{
            "label": "trace",
            "shape_type": "linestrip",
            "points": [[-10, 10], [10, 10], [30, 10], [50, 10]],
        }],
    }

    clipped = clip_annotation_document(document, x=0, y=0, width=40, height=30, image_name="tile.png")

    assert clipped["shapes"] == [{
        "label": "trace",
        "shape_type": "linestrip",
        "points": [[0.0, 10.0], [10.0, 10.0], [30.0, 10.0], [40.0, 10.0]],
    }]
    assert clipped["shapes"][0]["points"][0] != clipped["shapes"][0]["points"][-1]


def test_preview_rejects_tile_and_output_policy_requires_safe_absolute_nonoverlapping_root(tmp_path: Path) -> None:
    source_root, asset, datasets, jobs, engine = _runtime(tmp_path)
    tile = PipelineNode(id="tile", kind="tile", parameters={
        "tile_width": 64, "tile_height": 40, "overlap_x": 0, "overlap_y": 0,
    })
    with pytest.raises(InvalidPathError, match="multiple images"):
        engine.preview(PipelinePreviewRequest(dataset_id="dataset", asset_id=asset.asset_id, nodes=[tile]))
    with pytest.raises(ValidationError, match="absolute"):
        _derived_request(Path("relative-output"))
    with pytest.raises(ValidationError, match="Tile pipeline jobs require"):
        BatchJobRequest(kind="pipeline", dataset_id="dataset", pipeline_nodes=[tile])
    with pytest.raises(InvalidPathError, match="overlap"):
        engine.export_derived_item(
            job_id="unsafe",
            dataset_id="dataset",
            asset_id=asset.asset_id,
            nodes=[PipelineNode(id="color", kind="color")],
            policy=PipelineOutputPolicy(mode="derived_dataset", output_root=source_root / "inside"),
        )

    target = tmp_path / "symlink-target"
    target.mkdir()
    symlink = tmp_path / "symlink-output"
    try:
        symlink.symlink_to(target, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(InvalidPathError, match="symlink"):
            engine.export_derived_item(
                job_id="symlink",
                dataset_id="dataset",
                asset_id=asset.asset_id,
                nodes=[PipelineNode(id="color", kind="color")],
                policy=PipelineOutputPolicy(mode="derived_dataset", output_root=symlink),
            )
    jobs.close()
    datasets.close()


def test_writer_rejects_traversal_and_cleans_files_when_item_write_fails(tmp_path: Path, monkeypatch) -> None:
    from labelone.pipelines import derived as derived_module

    writer = DerivedDatasetWriter()
    source = tmp_path / "source"
    source.mkdir()
    output = (tmp_path / "output").resolve()
    policy = PipelineOutputPolicy(mode="derived_dataset", output_root=output, image_format="png")
    unsafe = PreparedDerivedOutput(
        image_relative_path="../escape.png",
        annotation_relative_path="escape.json",
        image=Image.new("RGB", (4, 4)),
        document={"shapes": []},
    )
    with pytest.raises(InvalidPathError, match="unsafe"):
        writer.write_item(
            job_id="unsafe",
            dataset_id="dataset",
            asset_id="asset",
            source_root=source,
            policy=policy,
            item_fingerprint="fingerprint",
            outputs=[unsafe],
        )

    original_atomic = derived_module._atomic_bytes
    writes = 0

    def fail_second(path, content):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("fixture write failure")
        return original_atomic(path, content)

    monkeypatch.setattr(derived_module, "_atomic_bytes", fail_second)
    prepared = PreparedDerivedOutput(
        image_relative_path="safe.png",
        annotation_relative_path="safe.json",
        image=Image.new("RGB", (4, 4)),
        document={"shapes": []},
    )
    with pytest.raises(OSError, match="fixture"):
        writer.write_item(
            job_id="failure",
            dataset_id="dataset",
            asset_id="asset",
            source_root=source,
            policy=policy,
            item_fingerprint="fingerprint",
            outputs=[prepared],
        )
    staging = writer.staging_root(output, "failure")
    assert not staging.exists()
    assert not output.exists()


def test_cancel_after_staging_and_restart_never_publish_incomplete_directory(tmp_path: Path, monkeypatch) -> None:
    _, _, datasets, jobs, engine = _runtime(tmp_path)
    output_root = (tmp_path / "cancel-output").resolve()
    service = JobService(jobs, datasets, engine, None)  # type: ignore[arg-type]
    started = Event()
    release = Event()
    original_write = engine.derived_writer.write_item

    def delayed_write(**kwargs):
        result = original_write(**kwargs)
        started.set()
        assert release.wait(timeout=3)
        return result

    monkeypatch.setattr(engine.derived_writer, "write_item", delayed_write)
    created = service.create(_derived_request(output_root))
    assert started.wait(timeout=3)
    service.cancel(created.job_id)
    release.set()
    canceled = _wait(jobs, created.job_id, {"canceled"})
    assert canceled.canceled == 1
    assert not output_root.exists()
    assert not engine.derived_writer.staging_root(output_root, created.job_id).exists()
    service.close()
    jobs.close()
    datasets.close()


def test_pause_after_staging_requeues_item_and_resume_publishes_atomically(tmp_path: Path, monkeypatch) -> None:
    _, _, datasets, jobs, engine = _runtime(tmp_path)
    output_root = (tmp_path / "pause-output").resolve()
    service = JobService(jobs, datasets, engine, None)  # type: ignore[arg-type]
    started = Event()
    release = Event()
    original_write = engine.derived_writer.write_item

    def delayed_write(**kwargs):
        result = original_write(**kwargs)
        started.set()
        assert release.wait(timeout=3)
        return result

    monkeypatch.setattr(engine.derived_writer, "write_item", delayed_write)
    created = service.create(_derived_request(output_root))
    assert started.wait(timeout=3)
    service.pause(created.job_id)
    release.set()
    paused = _wait(jobs, created.job_id, {"paused"})

    assert paused.items[0].state == "queued"
    assert not output_root.exists()
    service.resume(created.job_id)
    succeeded = _wait(jobs, created.job_id, {"succeeded"})
    assert succeeded.items[0].attempts == 2
    assert succeeded.items[0].result["cache_hit"] is True
    assert output_root.is_dir()
    service.close()
    jobs.close()
    datasets.close()


def test_restart_reuses_complete_staging_item_before_atomic_publish(tmp_path: Path, monkeypatch) -> None:
    _, _, datasets, jobs, engine = _runtime(tmp_path)
    output_root = (tmp_path / "restart-output").resolve()
    first_service = JobService(jobs, datasets, engine, None)  # type: ignore[arg-type]
    staged = Event()
    release = Event()
    original_write = engine.derived_writer.write_item

    def delayed_write(**kwargs):
        result = original_write(**kwargs)
        staged.set()
        assert release.wait(timeout=3)
        return result

    monkeypatch.setattr(engine.derived_writer, "write_item", delayed_write)
    created = first_service.create(_derived_request(output_root))
    assert staged.wait(timeout=3)
    close_thread = Thread(target=first_service.close)
    close_thread.start()
    sleep(0.05)
    release.set()
    close_thread.join(timeout=4)
    interrupted = jobs.get(created.job_id)
    assert interrupted.state == "interrupted"
    assert not output_root.exists()
    assert engine.derived_writer.staging_root(output_root, created.job_id).is_dir()
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    second_service = JobService(reopened, datasets, engine, None)  # type: ignore[arg-type]
    second_service.start()
    finished = _wait(reopened, created.job_id, {"succeeded"})

    assert output_root.is_dir()
    assert finished.items[0].attempts == 2
    assert finished.items[0].result["cache_hit"] is True
    assert not engine.derived_writer.staging_root(output_root, created.job_id).exists()
    second_service.close()
    reopened.close()
    datasets.close()
