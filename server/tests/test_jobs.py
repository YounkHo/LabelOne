from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, sleep
from types import SimpleNamespace

from PIL import Image
import pytest

from labelone.annotations import AnnotationStore
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError, RevisionConflictError
from labelone.jobs import BatchJobRequest, JobRepository, JobService
from labelone.pipelines import PipelineCancelled, PipelineNode


class SlowPipeline:
    def preview(self, request, progress=None, canceled=None):
        del canceled
        if progress is not None:
            progress({
                "kind": "pipeline", "progress": 1.0, "phase": "completed",
                "completed_steps": 3, "total_steps": 3,
            })
        sleep(0.04)
        return SimpleNamespace(model_dump=lambda mode: {"asset_id": request.asset_id, "mode": mode})


class BlockingPipeline:
    def __init__(self, expected_started: int = 3) -> None:
        self.expected_started = expected_started
        self.started = 0
        self.started_lock = Lock()
        self.capacity_reached = Event()
        self.release = Event()

    def preview(self, request, progress=None, canceled=None):
        del progress, canceled
        with self.started_lock:
            self.started += 1
            if self.started >= self.expected_started:
                self.capacity_reached.set()
        assert self.release.wait(timeout=3)
        return SimpleNamespace(model_dump=lambda mode: {"asset_id": request.asset_id, "mode": mode})


class BlockingDerivedPipeline:
    def __init__(self) -> None:
        self.publishing = Event()
        self.release = Event()

    def export_derived_item(self, *, asset_id, **kwargs):
        del kwargs
        return SimpleNamespace(model_dump=lambda mode: {"asset_id": asset_id, "mode": mode})

    def finalize_derived(self, **kwargs):
        del kwargs
        self.publishing.set()
        assert self.release.wait(timeout=3)

    def abort_derived(self, **kwargs):
        del kwargs


class TrackingModels:
    def __init__(self) -> None:
        self.lock = Lock()
        self.inflight = 0
        self.max_inflight = 0

    def predict(self, model_id, image_path, capture_layers, parameters):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            sleep(0.02)
            return SimpleNamespace(model_dump=lambda mode: {"model_id": model_id, "image_path": str(image_path), "mode": mode})
        finally:
            with self.lock:
                self.inflight -= 1


def _repositories(tmp_path: Path, count: int = 8):
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(count):
        Image.new("RGB", (32, 24), (index, 20, 30)).save(root / f"image-{index}.png")
        (root / f"image-{index}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(root_dir=root, layout="same_directory", dataset_id="dataset"))
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    datasets.register(scan)
    jobs = JobRepository(tmp_path / "index.sqlite3", datasets)
    return datasets, jobs


def _request(concurrency: int = 2, *, preferred_asset_ids: list[str] | None = None) -> BatchJobRequest:
    return BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        concurrency=concurrency,
        preferred_asset_ids=preferred_asset_ids or [],
        pipeline_nodes=[PipelineNode(id="color", kind="color")],
    )


def _wait_for(repository: JobRepository, job_id: str, states: set[str], timeout: float = 5.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = repository.get(job_id)
        if job.state in states:
            return job
        sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {states}")


def test_batch_feature_capture_requires_an_explicit_bounded_asset_scope() -> None:
    with pytest.raises(ValueError, match="explicit scope"):
        BatchJobRequest(kind="inference", dataset_id="dataset", model_id="model", capture_layers=["hidden"])

    request = BatchJobRequest(
        kind="inference",
        dataset_id="dataset",
        model_id="model",
        asset_ids=["asset-1"],
        capture_layers=["hidden"],
    )
    assert request.capture_layers == ["hidden"]


def test_pipeline_job_runs_all_items_and_idempotency_reuses_job(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=6)
    service = JobService(jobs, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    created = service.create(_request(concurrency=3), idempotency_key="same")
    duplicate = service.create(_request(concurrency=3), idempotency_key="same")
    finished = _wait_for(jobs, created.job_id, {"succeeded"})

    assert duplicate.job_id == created.job_id
    assert created.items == []
    assert created.request.asset_ids == []
    assert finished.completed == 6
    assert finished.failed == 0
    assert all(item.state == "succeeded" for item in finished.items)
    assert all(item.progress and item.progress["progress"] == 1.0 for item in finished.items)
    service.close()
    jobs.close()
    datasets.close()


def test_category_rename_job_updates_every_asset_and_is_idempotent(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=4)
    for asset_id in datasets.selectable_asset_ids("dataset"):
        asset = datasets.get_asset("dataset", asset_id, require_selectable=True)
        asset.annotation_paths[0].write_text(json.dumps({
            "shapes": [
                {"label": " scratch ", "shape_type": "rectangle", "points": [[1, 1], [8, 1], [8, 8], [1, 8]]},
                {"label": "keep", "shape_type": "point", "points": [[4, 4]]},
            ],
        }), encoding="utf-8")
        datasets.update_annotation_metadata(
            "dataset",
            asset_id,
            annotation_count=2,
            revision="fixture",
            labels=[" scratch ", "keep"],
            shape_types=["rectangle", "point"],
        )
    annotations = AnnotationStore(datasets, tmp_path / "annotation-backups")
    service = JobService(jobs, datasets, SlowPipeline(), None, annotations)  # type: ignore[arg-type]
    request = BatchJobRequest(
        kind="category_rename",
        dataset_id="dataset",
        concurrency=2,
        source_category="scratch",
        target_category="defect",
    )

    created = service.create(request, idempotency_key="rename-scratch-defect")
    duplicate = service.create(request, idempotency_key="rename-scratch-defect")
    finished = _wait_for(jobs, created.job_id, {"succeeded"})
    completed_duplicate = service.create(request, idempotency_key="rename-scratch-defect")

    assert duplicate.job_id == created.job_id
    assert completed_duplicate.job_id == created.job_id
    assert finished.completed == 4
    assert finished.failed == 0
    assert all(item.result and item.result["renamed"] == 1 for item in finished.items)
    for asset_id in datasets.selectable_asset_ids("dataset"):
        loaded = annotations.load("dataset", asset_id)
        assert [shape["label"] for shape in loaded.document["shapes"]] == ["defect", "keep"]
        asset = datasets.get_asset("dataset", asset_id, require_selectable=True)
        assert asset.labels == ["defect", "keep"]

    service.close()
    jobs.close()
    datasets.close()


def test_category_rename_request_requires_distinct_normalized_names() -> None:
    with pytest.raises(ValueError, match="source and target must differ"):
        BatchJobRequest(
            kind="category_rename",
            dataset_id="dataset",
            source_category=" scratch ",
            target_category="scratch",
        )


def test_category_rename_committed_item_remains_succeeded_after_cancel_request(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=1)
    asset_id = datasets.selectable_asset_ids("dataset")[0]
    datasets.update_annotation_metadata(
        "dataset",
        asset_id,
        annotation_count=1,
        revision="fixture",
        labels=["source"],
        shape_types=["point"],
    )
    created = jobs.create(BatchJobRequest(
        kind="category_rename",
        dataset_id="dataset",
        source_category="source",
        target_category="target",
    ))
    assert jobs.transition_to_running(created.job_id)
    claim = jobs.mark_item_running(created.job_id, asset_id)
    assert claim
    jobs.request_cancel(created.job_id)

    assert jobs.finish_committed_item(created.job_id, asset_id, claim_token=claim, result={"renamed": 1})
    item = jobs.get(created.job_id).items[0]
    assert item.state == "succeeded"
    assert item.result == {"renamed": 1}

    jobs.close()
    datasets.close()


def test_preferred_assets_run_current_and_neighbors_first_then_keep_dataset_order(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=7)
    stable = datasets.selectable_asset_ids("dataset")
    preferred = [stable[3], stable[2], stable[4]]
    request = _request(preferred_asset_ids=preferred)

    created = jobs.create(request, idempotency_key="preferred-neighborhood")
    items = jobs.get(created.job_id).items

    assert [item.asset_id for item in items] == [
        *preferred,
        *(asset_id for asset_id in stable if asset_id not in set(preferred)),
    ]
    assert [item.position for item in items] == list(range(len(stable)))
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    assert [item.asset_id for item in reopened.get(created.job_id).items] == [item.asset_id for item in items]
    duplicate = reopened.create(request, idempotency_key="preferred-neighborhood")
    assert duplicate.job_id == created.job_id
    different = _request(preferred_asset_ids=list(reversed(preferred)))
    with pytest.raises(RevisionConflictError, match="different job request"):
        reopened.create(different, idempotency_key="preferred-neighborhood")
    reopened.close()
    datasets.close()


def test_preferred_assets_apply_to_explicit_scope_without_reordering_the_remainder(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=7)
    stable = datasets.selectable_asset_ids("dataset")
    explicit = [stable[5], stable[1], stable[4], stable[0]]
    preferred = [stable[4], stable[5]]
    request = BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        asset_ids=explicit,
        preferred_asset_ids=preferred,
        pipeline_nodes=[PipelineNode(id="color", kind="color")],
    )

    created = jobs.create(request)

    assert [item.asset_id for item in jobs.get(created.job_id).items] == [
        stable[4], stable[5], stable[1], stable[0],
    ]
    jobs.close()
    datasets.close()


def test_active_job_reprioritizes_new_current_and_neighbor_items_by_distance(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=7)
    stable = datasets.selectable_asset_ids("dataset")
    created = jobs.create(_request(concurrency=1))

    jobs.prioritize_queued_items(created.job_id, [stable[5], stable[4], stable[6]])
    queued = jobs.queued_items(created.job_id, 7)

    assert [item.asset_id for item in queued[:3]] == [stable[5], stable[4], stable[6]]
    assert [item.asset_id for item in queued[3:]] == stable[:4]
    jobs.close()
    datasets.close()


def test_preferred_asset_validation_rejects_duplicates_out_of_scope_and_oversized_hints(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=3)
    stable = datasets.selectable_asset_ids("dataset")

    with pytest.raises(ValueError, match="preferred_asset_ids must contain unique"):
        BatchJobRequest(
            kind="pipeline",
            dataset_id="dataset",
            preferred_asset_ids=[stable[0], stable[0]],
            pipeline_nodes=[PipelineNode(id="color", kind="color")],
        )
    with pytest.raises(ValueError, match="preferred_asset_ids must be contained"):
        BatchJobRequest(
            kind="pipeline",
            dataset_id="dataset",
            asset_ids=stable[:1],
            preferred_asset_ids=stable[1:2],
            pipeline_nodes=[PipelineNode(id="color", kind="color")],
        )
    with pytest.raises(ValueError, match="at most 64"):
        BatchJobRequest(
            kind="pipeline",
            dataset_id="dataset",
            preferred_asset_ids=[f"asset-{index}" for index in range(65)],
            pipeline_nodes=[PipelineNode(id="color", kind="color")],
        )
    with pytest.raises(ValueError, match="asset_ids must contain unique"):
        BatchJobRequest(
            kind="pipeline",
            dataset_id="dataset",
            asset_ids=[stable[0], stable[0]],
            pipeline_nodes=[PipelineNode(id="color", kind="color")],
        )

    request = _request(preferred_asset_ids=["missing-asset"])
    with pytest.raises(InvalidPathError, match="must belong to the batch job scope"):
        jobs.create(request)
    jobs.close()
    datasets.close()


def test_empty_preferred_assets_preserve_legacy_serialization_recovery_and_idempotency(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=3)
    request = _request()
    created = jobs.create(request, idempotency_key="legacy-empty-preferred")
    with jobs._lock:
        stored = jobs._connection.execute(
            "SELECT request_json FROM jobs WHERE job_id=?", (created.job_id,)
        ).fetchone()
    assert stored is not None
    assert "preferred_asset_ids" not in json.loads(str(stored["request_json"]))
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    recovered = reopened.get(created.job_id, include_items=False)
    service = JobService(reopened, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    duplicate = service.create(_request(), idempotency_key="legacy-empty-preferred")
    assert recovered.request.preferred_asset_ids == []
    assert recovered.request.pipeline_context is None
    assert duplicate.job_id == created.job_id
    service.close()
    reopened.close()
    datasets.close()


def test_pipeline_precompute_context_binds_revision_registry_nodes_and_output_format(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=3)
    stable = datasets.selectable_asset_ids("dataset")
    service = JobService(jobs, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    first = _request(concurrency=1, preferred_asset_ids=[stable[1], stable[0]])
    equivalent = _request(concurrency=4, preferred_asset_ids=[stable[2]])
    canonical = BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        pipeline_nodes=[
            PipelineNode(id="source", kind="source"),
            PipelineNode(id="color", kind="color"),
            PipelineNode(id="visualize", kind="visualize", parameters={"label": "显示"}),
        ],
    )

    first_context = service.pipeline_precompute_context(first)
    equivalent_context = service.pipeline_precompute_context(equivalent)
    canonical_context = service.pipeline_precompute_context(canonical)
    assert first_context is not None
    assert first_context.signature == equivalent_context.signature == canonical_context.signature  # type: ignore[union-attr]
    assert first_context.dataset_index_revision == datasets.get_dataset("dataset").index_revision
    assert first_context.output_format == "png"

    created = service.create(first, idempotency_key="pipeline-context")
    reusable = service.find_pipeline_precompute(equivalent)
    assert created.request.pipeline_context == first_context
    assert reusable is not None and reusable.job_id == created.job_id

    changed_nodes = BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        pipeline_nodes=[PipelineNode(id="color", kind="color", parameters={"contrast": 1.2})],
    )
    changed_format = BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        pipeline_nodes=[PipelineNode(id="color", kind="color")],
        output_policy={"mode": "preview", "image_format": "webp", "conflict": "reuse"},
    )
    assert service.pipeline_precompute_context(changed_nodes).signature != first_context.signature  # type: ignore[union-attr]
    assert service.pipeline_precompute_context(changed_format).signature != first_context.signature  # type: ignore[union-attr]

    datasets.update_annotation_metadata(
        "dataset",
        stable[0],
        annotation_count=0,
        revision="updated-revision",
        labels=[],
        shape_types=[],
    )
    revised = service.pipeline_precompute_context(first)
    assert revised is not None
    assert revised.dataset_index_revision > first_context.dataset_index_revision
    assert revised.signature != first_context.signature
    assert service.find_pipeline_precompute(first) is None
    service.close()
    jobs.close()
    datasets.close()


def test_new_background_precompute_only_supersedes_different_background_signature(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=2)
    service = JobService(jobs, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    old = _request()
    old.priority = "background"
    current = BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        priority="background",
        pipeline_nodes=[PipelineNode(id="color", kind="color", parameters={"contrast": 1.2})],
    )
    old_context = service.pipeline_precompute_context(old)
    current_context = service.pipeline_precompute_context(current)
    assert old_context is not None and current_context is not None
    assert old_context.signature != current_context.signature

    superseded = jobs.create(old.model_copy(update={"pipeline_context": old_context}))
    same_signature = jobs.create(current.model_copy(update={"pipeline_context": current_context}))
    user_batch = jobs.create(old.model_copy(update={"priority": "user_batch", "pipeline_context": old_context}))

    canceled = service.cancel_superseded_background_precomputes(current)
    created = service.create(current, idempotency_key="new-background-precompute")

    assert canceled == [superseded.job_id]
    assert jobs.get(superseded.job_id, include_items=False).state == "canceled"
    assert jobs.get(same_signature.job_id, include_items=False).desired_state == "run"
    assert jobs.get(user_batch.job_id, include_items=False).desired_state == "run"
    assert created.request.pipeline_context == current_context
    service.close()
    jobs.close()
    datasets.close()


@pytest.mark.parametrize("state", ["paused", "interrupted"])
def test_ensure_pipeline_precompute_resumes_reusable_stopped_job(tmp_path: Path, state: str) -> None:
    datasets, jobs = _repositories(tmp_path, count=2)
    service = JobService(jobs, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    service.start()
    request = _request()
    request.priority = "background"
    context = service.pipeline_precompute_context(request)
    assert context is not None
    stopped = jobs.create(request.model_copy(update={"pipeline_context": context}))
    if state == "paused":
        jobs.request_pause(stopped.job_id)
    else:
        jobs.set_state(stopped.job_id, "interrupted")
    assert jobs.get(stopped.job_id, include_items=False).state == state

    ensured = service.ensure_pipeline_precompute(request)

    assert ensured.reused is True
    assert ensured.resumed is True
    assert ensured.job.job_id == stopped.job_id
    assert ensured.job.state in {"queued", "running", "succeeded"}
    assert _wait_for(jobs, stopped.job_id, {"succeeded"}).completed == 2
    service.close()
    jobs.close()
    datasets.close()


def test_dataset_with_active_job_cannot_be_safely_unregistered(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=2)
    created = jobs.create(_request())

    assert jobs.has_active_dataset_jobs("dataset") is True
    jobs.set_state(created.job_id, "failed", error="fixture")
    assert jobs.has_active_dataset_jobs("dataset") is False
    datasets.delete_dataset("dataset")
    assert datasets.list_datasets().datasets == []
    jobs.close()
    datasets.close()


def test_pause_resume_cancel_and_restart_recovery(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=12)
    service = JobService(jobs, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    paused_job = service.create(_request(concurrency=2))
    _wait_for(jobs, paused_job.job_id, {"running"})
    service.pause(paused_job.job_id)
    paused = _wait_for(jobs, paused_job.job_id, {"paused"})
    assert paused.completed < paused.total
    service.resume(paused_job.job_id)
    assert _wait_for(jobs, paused_job.job_id, {"succeeded"}).completed == 12

    canceled_job = service.create(_request(concurrency=1))
    _wait_for(jobs, canceled_job.job_id, {"running"})
    service.cancel(canceled_job.job_id)
    canceled = _wait_for(jobs, canceled_job.job_id, {"canceled"})
    assert canceled.canceled > 0
    service.close()

    interrupted = jobs.create(_request())
    jobs.set_state(interrupted.job_id, "running")
    first = jobs.queued_items(interrupted.job_id, 1)[0]
    jobs.mark_item_running(interrupted.job_id, first.asset_id)
    jobs.close()
    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    recovered = reopened.get(interrupted.job_id)
    assert recovered.state == "interrupted"
    assert recovered.items[0].state == "queued"
    reopened.close()
    datasets.close()


def test_cancel_generation_rejects_late_result_and_survives_restart(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=3)
    job = jobs.create(_request(concurrency=1))
    assert jobs.transition_to_running(job.job_id)
    first = jobs.queued_items(job.job_id, 1)[0]
    claim = jobs.mark_item_running(job.job_id, first.asset_id)
    assert claim is not None

    jobs.request_cancel(job.job_id)
    canceling = jobs.get(job.job_id)
    assert canceling.state == "canceling"
    assert canceling.desired_state == "cancel"
    assert not jobs.finish_item(
        job.job_id,
        first.asset_id,
        claim_token=claim,
        state="succeeded",
        result={"late": True},
    )
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    recovered = reopened.get(job.job_id)
    assert recovered.state == "canceled"
    assert recovered.desired_state == "cancel"
    assert recovered.canceled == recovered.total
    assert all(item.result is None for item in recovered.items)
    reopened.close()
    datasets.close()


def test_pausing_recovery_preserves_pause_and_job_service_resumes_interrupted(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=4)
    paused_job = jobs.create(_request(concurrency=1))
    assert jobs.transition_to_running(paused_job.job_id)
    first = jobs.queued_items(paused_job.job_id, 1)[0]
    assert jobs.mark_item_running(paused_job.job_id, first.asset_id)
    jobs.request_pause(paused_job.job_id)
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    recovered = reopened.get(paused_job.job_id)
    assert recovered.state == "paused"
    assert recovered.desired_state == "pause"
    assert all(item.state == "queued" for item in recovered.items)

    interrupted = reopened.create(_request(concurrency=2))
    reopened.set_state(interrupted.job_id, "running")
    reopened.close()
    auto_repository = JobRepository(tmp_path / "index.sqlite3", datasets)
    assert auto_repository.get(interrupted.job_id).state == "interrupted"
    service = JobService(auto_repository, datasets, SlowPipeline(), None)  # type: ignore[arg-type]
    service.start()
    assert _wait_for(auto_repository, interrupted.job_id, {"succeeded"}).completed == 4
    service.close()
    auto_repository.close()
    datasets.close()


def test_same_model_batch_jobs_share_a_single_scheduler_lane(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=10)
    models = TrackingModels()
    service = JobService(jobs, datasets, SlowPipeline(), models)  # type: ignore[arg-type]
    asset_ids = datasets.selectable_asset_ids("dataset")
    requests = [
        BatchJobRequest(
            kind="inference",
            dataset_id="dataset",
            asset_ids=selection,
            concurrency=4,
            model_id="same-model",
        )
        for selection in (asset_ids[:5], asset_ids[5:])
    ]

    created = [service.create(request) for request in requests]
    finished = [_wait_for(jobs, job.job_id, {"succeeded"}) for job in created]

    assert [job.completed for job in finished] == [5, 5]
    assert models.max_inflight == 1
    assert service.scheduler_snapshot().lane_capacities["model:same-model"] == 1
    service.close()
    jobs.close()
    datasets.close()


def test_interactive_preview_uses_reserved_cpu_capacity_while_batch_is_saturated(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=4)
    pipeline = BlockingPipeline()
    service = JobService(jobs, datasets, pipeline, None)  # type: ignore[arg-type]
    created = service.create(_request(concurrency=4))
    assert pipeline.capacity_reached.wait(timeout=2)
    assert pipeline.started == 3

    background_result: list[object] = []
    background_finished = Event()

    def preheat() -> None:
        background_result.append(service.run_adhoc(
            "cpu_pipeline",
            lambda: "background-ready",
            priority="background",
        ))
        background_finished.set()

    background_thread = Thread(target=preheat)
    background_thread.start()
    assert not background_finished.wait(timeout=0.1)

    interactive_result: list[object] = []
    interactive_finished = Event()
    started_at = monotonic()

    def preview() -> None:
        interactive_result.append(service.run_interactive("cpu_pipeline", lambda: "interactive-ready"))
        interactive_finished.set()

    thread = Thread(target=preview)
    thread.start()
    try:
        assert interactive_finished.wait(timeout=0.5)
        assert monotonic() - started_at < 0.5
        assert interactive_result == ["interactive-ready"]
        assert not background_finished.is_set()
    finally:
        pipeline.release.set()
        thread.join(timeout=2)
        background_thread.join(timeout=2)

    assert background_finished.is_set()
    assert background_result == ["background-ready"]
    assert _wait_for(jobs, created.job_id, {"succeeded"}).completed == 4
    service.close()
    jobs.close()
    datasets.close()


def test_canceled_queued_adhoc_pipeline_work_never_executes(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=4)
    pipeline = BlockingPipeline()
    service = JobService(jobs, datasets, pipeline, None)  # type: ignore[arg-type]
    service.create(_request(concurrency=4))
    assert pipeline.capacity_reached.wait(timeout=2)

    canceled = Event()
    callback_started = Event()
    finished = Event()
    errors: list[BaseException] = []

    def queued_work() -> None:
        try:
            service.run_adhoc(
                "cpu_pipeline",
                callback_started.set,
                priority="background",
                canceled=canceled.is_set,
            )
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = Thread(target=queued_work)
    thread.start()
    try:
        sleep(0.1)
        canceled.set()
        assert finished.wait(timeout=0.5)
        assert not callback_started.is_set()
        assert len(errors) == 1 and isinstance(errors[0], PipelineCancelled)
    finally:
        pipeline.release.set()
        thread.join(timeout=2)

    service.close()
    jobs.close()
    datasets.close()


def test_derived_publish_does_not_hold_repository_lock_and_reports_phase(tmp_path: Path) -> None:
    datasets, jobs = _repositories(tmp_path, count=1)
    pipeline = BlockingDerivedPipeline()
    service = JobService(jobs, datasets, pipeline, None)  # type: ignore[arg-type]
    request = BatchJobRequest(
        kind="pipeline",
        dataset_id="dataset",
        concurrency=1,
        pipeline_nodes=[PipelineNode(id="color", kind="color")],
        output_policy={
            "mode": "derived_dataset",
            "output_root": (tmp_path / "derived").resolve(),
            "image_format": "png",
            "conflict": "reuse",
        },
    )
    created = service.create(request)
    assert pipeline.publishing.wait(timeout=2)

    read_finished = Event()
    snapshots: list[object] = []

    def read_during_publish() -> None:
        snapshots.append(jobs.get(created.job_id, include_items=False))
        snapshots.append(jobs.list_events(created.job_id, limit=1000))
        read_finished.set()

    reader = Thread(target=read_during_publish)
    reader.start()
    try:
        assert read_finished.wait(timeout=0.5)
        assert snapshots[0].state == "running"  # type: ignore[union-attr]
        assert any(
            event.event_type == "job.progress" and event.payload.get("phase") == "publishing"
            for event in snapshots[1]  # type: ignore[union-attr]
        )
        with pytest.raises(InvalidPathError, match="while publishing"):
            service.pause(created.job_id)
        with pytest.raises(InvalidPathError, match="while publishing"):
            service.cancel(created.job_id)
    finally:
        pipeline.release.set()
        reader.join(timeout=2)

    assert _wait_for(jobs, created.job_id, {"succeeded"}).completed == 1
    service.close()
    jobs.close()
    datasets.close()
