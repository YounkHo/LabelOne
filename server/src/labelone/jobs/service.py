from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import Event, RLock
from time import monotonic
from typing import Callable
from unicodedata import normalize
from uuid import uuid4

from labelone.annotations import AnnotationStore
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError
from labelone.models import ModelManager
from labelone.models.weights import DownloadProgress, ModelWeightCancelled, ModelWeightError
from labelone.pipelines import (
    PipelineCancelled,
    PipelineEngine,
    PipelinePreviewRequest,
    normalize_legacy_nodes,
    operator_registry_hash,
    validate_nodes,
)

from .models import BatchJobRequest, JobRecord, PipelinePrecomputeContext, PipelinePrecomputeEnsureResponse
from .repository import JobRepository
from .scheduler import FairJobScheduler, Priority, SchedulerError


@dataclass(frozen=True, slots=True)
class _ScheduledWork:
    job: JobRecord
    asset_id: str
    claim_token: str
    future: Future


@dataclass(frozen=True, slots=True)
class _AdhocWork:
    callback: Callable[[], object]
    future: Future


class JobService:
    def __init__(
        self,
        repository: JobRepository,
        datasets: DatasetRepository,
        pipelines: PipelineEngine,
        models: ModelManager,
        annotations: AnnotationStore | None = None,
    ) -> None:
        self.repository = repository
        self.datasets = datasets
        self.pipelines = pipelines
        self.models = models
        self.annotations = annotations
        self._job_executor = ThreadPoolExecutor(max_workers=64, thread_name_prefix="labelone-job")
        self._item_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="labelone-item")
        self._scheduler = FairJobScheduler(
            {"cpu_pipeline": 4, "model_download": 4},
            global_capacity=8,
            max_queued=10_000,
            max_queued_per_job=16,
            interactive_reserves={"cpu_pipeline": 1},
        )
        self._active: set[str] = set()
        self._publishing: set[str] = set()
        self._lock = RLock()
        self._closing = Event()
        self._started = False
        self._scheduler_workers: list[Future] = []

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._closing.is_set():
                raise RuntimeError("Job service is closing")
            self._started = True
            self._scheduler_workers = [self._item_executor.submit(self._scheduler_worker) for _ in range(8)]
        weight_store = getattr(self.models, "weight_store", None)
        if weight_store is not None:
            weight_store.cleanup_all_stale_parts()
        for job_id in self.repository.resumable_job_ids():
            self._submit(job_id)

    def close(self) -> None:
        self._closing.set()
        self._job_executor.shutdown(wait=True, cancel_futures=False)
        self._scheduler.shutdown(wait=True)
        self._item_executor.shutdown(wait=True, cancel_futures=False)

    def scheduler_snapshot(self):
        return self._scheduler.snapshot()

    def run_adhoc(
        self,
        lane: str,
        callback: Callable[[], object],
        priority: Priority,
        canceled: Callable[[], bool] | None = None,
    ) -> object:
        """Run one callback through the shared scheduler at an explicit priority."""
        self.start()
        if lane != "cpu_pipeline":
            self._scheduler.configure_lane(lane, 1)
        job_id = f"adhoc:{priority}:{uuid4().hex}"
        future: Future = Future()
        self._scheduler.register(job_id, priority=priority, lane=lane, max_inflight=1)
        try:
            self._scheduler.submit(job_id, _AdhocWork(callback=callback, future=future))
            if canceled is None:
                return future.result()
            while True:
                try:
                    return future.result(timeout=0.05)
                except FutureTimeoutError:
                    if canceled() and future.cancel():
                        raise PipelineCancelled("Ad-hoc pipeline work was canceled before execution")
        finally:
            try:
                dropped = self._scheduler.unregister(job_id)
                for token in dropped:
                    if isinstance(token, _AdhocWork):
                        token.future.cancel()
            except SchedulerError:
                pass

    def run_interactive(self, lane: str, callback: Callable[[], object]) -> object:
        return self.run_adhoc(lane, callback, priority="interactive")

    def pipeline_precompute_context(self, request: BatchJobRequest) -> PipelinePrecomputeContext | None:
        if request.kind != "pipeline" or request.output_policy.mode != "preview":
            return None
        dataset_revision = self.datasets.get_dataset(request.dataset_id).index_revision
        registry_hash = operator_registry_hash()
        nodes = validate_nodes(normalize_legacy_nodes(request.pipeline_nodes))
        payload = {
            "schema_version": 1,
            "dataset_id": request.dataset_id,
            "dataset_index_revision": dataset_revision,
            "registry_hash": registry_hash,
            "output_format": request.output_policy.image_format,
            "nodes": [node.as_dict() for node in nodes],
        }
        signature = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return PipelinePrecomputeContext(
            signature=signature,
            dataset_index_revision=dataset_revision,
            registry_hash=registry_hash,
            output_format=request.output_policy.image_format,
        )

    def find_pipeline_precompute(self, request: BatchJobRequest) -> JobRecord | None:
        context = self.pipeline_precompute_context(request)
        if context is None:
            return None
        return self.repository.find_pipeline_precompute(request.dataset_id, context.signature)

    def cancel_superseded_background_precomputes(self, request: BatchJobRequest) -> list[str]:
        context = self.pipeline_precompute_context(request)
        if context is None or request.priority != "background" or request.asset_ids:
            return []
        job_ids = self.repository.superseded_background_pipeline_job_ids(
            request.dataset_id,
            context.signature,
        )
        for job_id in job_ids:
            self.cancel(job_id)
        return job_ids

    def ensure_pipeline_precompute(self, request: BatchJobRequest) -> PipelinePrecomputeEnsureResponse:
        if request.kind != "pipeline":
            raise InvalidPathError("Pipeline precompute requires a pipeline job request")
        if request.output_policy.mode != "preview":
            raise InvalidPathError("Pipeline precompute only supports preview output")
        if request.priority != "background":
            raise InvalidPathError("Pipeline precompute requires background priority")
        if request.asset_ids:
            raise InvalidPathError("Pipeline precompute must target the complete dataset")
        self.start()
        with self._lock:
            context = self.pipeline_precompute_context(request)
            assert context is not None
            existing = self.repository.find_pipeline_precompute(
                request.dataset_id,
                context.signature,
                full_dataset_only=True,
            )
            if existing is not None:
                resumed = existing.state in {"paused", "interrupted"}
                if resumed:
                    existing = self.resume(existing.job_id)
                return PipelinePrecomputeEnsureResponse(job=existing, reused=True, resumed=resumed)
            canceled_job_ids = self.cancel_superseded_background_precomputes(request)
            prepared = request.model_copy(update={"pipeline_context": context})
            job = self.repository.create(prepared)
            if job.state == "queued":
                self._submit(job.job_id)
            return PipelinePrecomputeEnsureResponse(
                job=job,
                reused=False,
                canceled_job_ids=canceled_job_ids,
            )

    def create(self, request: BatchJobRequest, *, idempotency_key: str | None = None) -> JobRecord:
        context = self.pipeline_precompute_context(request)
        prepared = request.model_copy(update={"pipeline_context": context})
        self.start()
        job = self.repository.create(prepared, idempotency_key=idempotency_key)
        if job.state == "queued":
            self._submit(job.job_id)
        return job

    def create_model_download(
        self,
        model_id: str,
        url_indices: list[int],
        *,
        expected_sha256: dict[int, str] | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        if self.models is None or self.models.weight_store is None:
            raise ModelWeightError("Model weight downloads are not configured")
        record = self.models.catalog.get(model_id)
        available = {weight.url_index for weight in self.models.weight_store.list_remote(model_id, record)}
        if (
            not url_indices
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in url_indices)
            or len(set(url_indices)) != len(url_indices)
        ):
            raise ModelWeightError("Weight download requires unique URL indices")
        unknown = sorted(set(url_indices) - available)
        if unknown:
            raise ModelWeightError(
                "Weight URL index is not an allowlisted remote model weight",
                details={"model_id": model_id, "unknown_indices": unknown},
            )
        request = BatchJobRequest(
            kind="model_download",
            model_id=model_id,
            concurrency=min(4, len(url_indices)),
            weight_url_indices=url_indices,
            expected_sha256=expected_sha256 or {},
        )
        return self.create(request, idempotency_key=idempotency_key)

    def _submit(self, job_id: str) -> None:
        if not self._started:
            self.start()
        if self._closing.is_set():
            return
        with self._lock:
            if job_id in self._active:
                return
            self._active.add(job_id)
        self._job_executor.submit(self._run, job_id)

    def _execute_item(self, job: JobRecord, asset_id: str, claim_token: str) -> dict:
        if job.kind == "pipeline":
            visualization_count = sum(
                1 for node in job.request.pipeline_nodes if node.enabled and node.kind in {"output", "visualize"}
            ) or 1

            def report(payload: dict[str, object]) -> None:
                self.repository.update_item_progress(
                    job.job_id,
                    asset_id,
                    claim_token=claim_token,
                    payload=payload,
                )

            report({
                "kind": "pipeline",
                "progress": 0.0,
                "phase": "starting",
                "completed_steps": 0,
                "total_steps": len([
                    node for node in job.request.pipeline_nodes
                    if node.enabled and node.kind not in {"source", "output", "visualize", "tile"}
                ]) + (
                    3
                    if job.request.output_policy.mode == "derived_dataset"
                    else visualization_count + 2
                ),
                "node_id": None,
                "node_kind": None,
            })
            def canceled() -> bool:
                if self._closing.is_set():
                    return True
                current = self.repository.get(job.job_id, include_items=False)
                return current.desired_state != "run" or current.generation != job.generation

            if job.request.output_policy.mode == "derived_dataset":
                result = self.pipelines.export_derived_item(
                    job_id=job.job_id,
                    dataset_id=job.dataset_id,
                    asset_id=asset_id,
                    nodes=job.request.pipeline_nodes,
                    policy=job.request.output_policy,
                    canceled=canceled,
                    progress=report,
                )
                return result.model_dump(mode="json")
            result = self.pipelines.preview(PipelinePreviewRequest(
                dataset_id=job.dataset_id,
                asset_id=asset_id,
                nodes=job.request.pipeline_nodes,
                output_format=job.request.output_policy.image_format,
            ), progress=report, canceled=canceled)
            return result.model_dump(mode="json")
        if job.kind == "inference":
            asset = self.datasets.get_asset(job.dataset_id, asset_id, require_selectable=True)
            if asset.image_path is None or job.request.model_id is None:
                raise InvalidPathError("Inference job input is incomplete")
            result = self.models.predict(
                job.request.model_id,
                asset.image_path,
                job.request.capture_layers,
                job.request.parameters,
            )
            return result.model_dump(mode="json")
        if job.kind == "category_rename":
            if self.annotations is None or job.request.source_category is None or job.request.target_category is None:
                raise InvalidPathError("Category rename job is not configured")
            envelope = self.annotations.load(job.dataset_id, asset_id)
            source_category = normalize("NFC", job.request.source_category.strip())
            target_category = normalize("NFC", job.request.target_category.strip())
            document = deepcopy(envelope.document)
            renamed = 0
            for shape in document.get("shapes", []):
                if not isinstance(shape, dict):
                    continue
                current_label = normalize("NFC", str(shape.get("label", "")).strip())
                if current_label != source_category:
                    continue
                shape["label"] = target_category
                renamed += 1
            if renamed == 0:
                return {
                    "source_category": source_category,
                    "target_category": target_category,
                    "renamed": 0,
                    "skipped": True,
                    "revision": envelope.revision,
                }
            control = self.repository.get(job.job_id, include_items=False)
            if control.desired_state != "run" or control.generation != job.generation:
                raise InvalidPathError("Category rename item was canceled before save")
            saved = self.annotations.save(job.dataset_id, asset_id, document, if_match=envelope.revision)
            return {
                "source_category": source_category,
                "target_category": target_category,
                "renamed": renamed,
                "skipped": False,
                "previous_revision": saved.previous_revision,
                "revision": saved.revision,
            }
        if job.kind == "model_download":
            if self.models is None or self.models.weight_store is None or job.request.model_id is None:
                raise ModelWeightError("Model download job is not configured")
            if not asset_id.startswith("weight:"):
                raise ModelWeightError("Model download work item is invalid", details={"asset_id": asset_id})
            try:
                url_index = int(asset_id.split(":", 1)[1])
            except ValueError as exc:
                raise ModelWeightError("Model download work item is invalid", details={"asset_id": asset_id}) from exc
            if url_index not in job.request.weight_url_indices:
                raise ModelWeightError("Model download work item was not requested", details={"url_index": url_index})
            record = self.models.catalog.get(job.request.model_id)
            last_emitted_at = 0.0
            last_emitted_bytes = -1

            def canceled() -> bool:
                if self._closing.is_set():
                    return True
                current = self.repository.get(job.job_id, include_items=False)
                return current.desired_state != "run"

            def report(update: DownloadProgress) -> None:
                nonlocal last_emitted_at, last_emitted_bytes
                now = monotonic()
                finished = update.total_bytes is not None and update.received_bytes >= update.total_bytes
                if (
                    not finished
                    and update.received_bytes != 0
                    and update.received_bytes - last_emitted_bytes < 1024 * 1024
                    and now - last_emitted_at < 0.25
                ):
                    return
                accepted = self.repository.update_item_progress(
                    job.job_id,
                    asset_id,
                    claim_token=claim_token,
                    payload={
                        "model_id": update.model_id,
                        "url_index": update.url_index,
                        "received_bytes": update.received_bytes,
                        "total_bytes": update.total_bytes,
                        "progress": update.progress,
                        "cache_hit": update.cache_hit,
                    },
                )
                if accepted:
                    last_emitted_at = now
                    last_emitted_bytes = update.received_bytes

            downloaded = self.models.weight_store.download(
                job.request.model_id,
                record,
                url_index,
                expected_sha256=job.request.expected_sha256.get(url_index),
                progress=report,
                cancel=canceled,
            )
            return {
                "model_id": downloaded.model_id,
                "url_index": downloaded.url_index,
                "source_url": downloaded.source_url,
                "final_url": downloaded.final_url,
                "local_path": str(downloaded.local_path),
                "size_bytes": downloaded.size_bytes,
                "sha256": downloaded.sha256,
                "cache_hit": downloaded.cache_hit,
            }
        raise InvalidPathError("Unsupported job kind", details={"kind": job.kind})

    def _scheduler_worker(self) -> None:
        while not self._closing.is_set():
            leased = self._scheduler.acquire(timeout=0.2)
            if leased is None:
                continue
            work = leased.token
            try:
                if isinstance(work, _ScheduledWork) and work.future.set_running_or_notify_cancel():
                    try:
                        work.future.set_result(self._execute_item(work.job, work.asset_id, work.claim_token))
                    except BaseException as exc:
                        work.future.set_exception(exc)
                elif isinstance(work, _AdhocWork) and work.future.set_running_or_notify_cancel():
                    try:
                        work.future.set_result(work.callback())
                    except BaseException as exc:
                        work.future.set_exception(exc)
                elif not isinstance(work, (_ScheduledWork, _AdhocWork)):
                    raise RuntimeError("Scheduler received an unknown work token")
            finally:
                self._scheduler.release(leased)

    def _lane_for(self, job: JobRecord) -> str:
        if job.kind == "pipeline":
            return "cpu_pipeline"
        if job.kind == "category_rename":
            self._scheduler.configure_lane("annotation_write", 2)
            return "annotation_write"
        if job.kind == "model_download":
            return "model_download"
        assert job.request.model_id is not None
        lane = f"model:{job.request.model_id}"
        self._scheduler.configure_lane(lane, 1)
        return lane

    def _unregister_scheduled(self, job_id: str, *, mode: str) -> None:
        try:
            dropped = self._scheduler.unregister(job_id)
        except SchedulerError:
            return
        for token in dropped:
            if not isinstance(token, _ScheduledWork):
                continue
            if mode == "cancel":
                self.repository.cancel_running_item(job_id, token.asset_id, claim_token=token.claim_token)
            else:
                self.repository.requeue_running_item(job_id, token.asset_id, claim_token=token.claim_token)
            token.future.cancel()

    def _begin_publishing(self, job_id: str) -> JobRecord | None:
        """Claim the non-cancelable atomic publish phase without holding repository locks."""
        with self._lock:
            current = self.repository.get(job_id, include_items=False)
            if (
                current.state != "running"
                or current.desired_state != "run"
                or current.failed
                or current.completed != current.total
            ):
                return None
            if not self.repository.report_job_phase(job_id, "publishing"):
                return None
            self._publishing.add(job_id)
            return current

    def _end_publishing(self, job_id: str) -> None:
        with self._lock:
            self._publishing.discard(job_id)

    def _run(self, job_id: str) -> None:
        active: dict[Future, tuple[str, str]] = {}
        scheduler_registered = False
        try:
            job = self.repository.get(job_id, include_items=False)
            if not self.repository.transition_to_running(job_id):
                return
            self._scheduler.register(
                job_id,
                priority=job.request.priority,
                lane=self._lane_for(job),
                max_inflight=job.request.concurrency,
            )
            scheduler_registered = True
            while True:
                control = self.repository.get(job_id, include_items=False).state
                if control == "canceling":
                    self.repository.cancel_queued(job_id)
                if self._closing.is_set():
                    control = "interrupted"
                if scheduler_registered and control in {"pausing", "canceling", "interrupted"}:
                    self._unregister_scheduled(
                        job_id,
                        mode="cancel" if control == "canceling" else "requeue",
                    )
                    scheduler_registered = False
                if control not in {"pausing", "canceling", "interrupted"}:
                    for item in self.repository.queued_items(job_id, max(0, job.request.concurrency - len(active))):
                        claim_token = self.repository.mark_item_running(job_id, item.asset_id)
                        if claim_token:
                            future: Future = Future()
                            work = _ScheduledWork(job=job, asset_id=item.asset_id, claim_token=claim_token, future=future)
                            try:
                                self._scheduler.submit(job_id, work)
                            except Exception:
                                self.repository.requeue_running_item(job_id, item.asset_id, claim_token=claim_token)
                                raise
                            active[future] = (item.asset_id, claim_token)
                if not active:
                    remaining = self.repository.queued_items(job_id, 1)
                    if control == "interrupted":
                        self.repository.interrupt_running(job_id)
                        return
                    if control == "pausing":
                        self.repository.settle_pause(job_id)
                        return
                    if control == "canceling":
                        if job.kind == "pipeline" and job.request.output_policy.mode == "derived_dataset":
                            self.pipelines.abort_derived(
                                job_id=job_id,
                                dataset_id=job.dataset_id,
                                policy=job.request.output_policy,
                            )
                        self.repository.settle_cancel(job_id)
                        return
                    if control in {"paused", "canceled", "succeeded", "succeeded_with_errors", "failed"}:
                        return
                    if not remaining:
                        if job.kind == "pipeline" and job.request.output_policy.mode == "derived_dataset":
                            current = self.repository.get(job_id, include_items=False)
                            if current.state != "running" or current.desired_state != "run":
                                continue
                            if current.failed:
                                self.pipelines.abort_derived(
                                    job_id=job_id,
                                    dataset_id=job.dataset_id,
                                    policy=job.request.output_policy,
                                )
                                self.repository.settle_completed(job_id)
                                return
                            publishing = self._begin_publishing(job_id)
                            if publishing is None:
                                continue
                            try:
                                self.pipelines.finalize_derived(
                                    job_id=job_id,
                                    dataset_id=job.dataset_id,
                                    policy=job.request.output_policy,
                                    item_results=self.repository.iter_item_results(job_id, page_size=1000),
                                    expected_item_count=publishing.total,
                                )
                            except Exception as exc:
                                self.pipelines.abort_derived(
                                    job_id=job_id,
                                    dataset_id=job.dataset_id,
                                    policy=job.request.output_policy,
                                )
                                self.repository.set_state(job_id, "failed", error=str(exc))
                                return
                            else:
                                self.repository.settle_completed(job_id)
                                return
                            finally:
                                self._end_publishing(job_id)
                        self.repository.settle_completed(job_id)
                        return
                done, _ = wait(active, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    asset_id, claim_token = active.pop(future)
                    if future.cancelled():
                        continue
                    current = self.repository.get(job_id, include_items=False).state
                    if current == "canceling" and job.kind != "category_rename":
                        self.repository.cancel_running_item(job_id, asset_id, claim_token=claim_token)
                        continue
                    if (current == "pausing" or self._closing.is_set()) and job.kind != "category_rename":
                        self.repository.requeue_running_item(job_id, asset_id, claim_token=claim_token)
                        continue
                    try:
                        result = future.result()
                        finished = (
                            self.repository.finish_committed_item(
                                job_id, asset_id, claim_token=claim_token, result=result
                            )
                            if job.kind == "category_rename"
                            else self.repository.finish_item(
                                job_id, asset_id, claim_token=claim_token, state="succeeded", result=result
                            )
                        )
                        if not finished:
                            self.repository.cancel_running_item(job_id, asset_id, claim_token=claim_token)
                    except (ModelWeightCancelled, PipelineCancelled):
                        interrupted = self.repository.get(job_id, include_items=False).state
                        if interrupted == "pausing" or self._closing.is_set():
                            self.repository.requeue_running_item(job_id, asset_id, claim_token=claim_token)
                        else:
                            self.repository.cancel_running_item(job_id, asset_id, claim_token=claim_token)
                    except Exception as exc:
                        if not self.repository.finish_item(
                            job_id, asset_id, claim_token=claim_token, state="failed", error=str(exc)
                        ):
                            self.repository.cancel_running_item(job_id, asset_id, claim_token=claim_token)
        except Exception as exc:
            try:
                failed_job = self.repository.get(job_id, include_items=False)
                if failed_job.kind == "pipeline" and failed_job.request.output_policy.mode == "derived_dataset":
                    self.pipelines.abort_derived(
                        job_id=job_id,
                        dataset_id=failed_job.dataset_id,
                        policy=failed_job.request.output_policy,
                    )
            except Exception:
                pass
            self.repository.set_state(job_id, "failed", error=str(exc))
        finally:
            if scheduler_registered:
                self._unregister_scheduled(job_id, mode="requeue")
            with self._lock:
                self._active.discard(job_id)

    def pause(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id in self._publishing:
                raise InvalidPathError("Job cannot be paused while publishing its derived dataset")
            state = self.repository.get(job_id, include_items=False).state
            if state not in {"queued", "running"}:
                raise InvalidPathError("Only queued/running jobs can be paused", details={"state": state})
            self.repository.request_pause(job_id)
        return self.repository.get(job_id, include_items=False)

    def resume(self, job_id: str) -> JobRecord:
        state = self.repository.get(job_id, include_items=False).state
        if state not in {"paused", "interrupted", "failed", "succeeded_with_errors"}:
            raise InvalidPathError("Job cannot be resumed", details={"state": state})
        job = self.repository.get(job_id, include_items=False)
        if job.kind == "pipeline" and job.request.output_policy.mode == "derived_dataset":
            self.repository.retry_all_items(job_id)
        else:
            self.repository.retry_failed(job_id)
        self.repository.request_resume(job_id)
        self._submit(job_id)
        return self.repository.get(job_id, include_items=False)

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id in self._publishing:
                raise InvalidPathError("Job cannot be canceled while publishing its derived dataset")
            state = self.repository.get(job_id, include_items=False).state
            if state in {"succeeded", "succeeded_with_errors", "failed", "canceled"}:
                return self.repository.get(job_id, include_items=False)
            self.repository.request_cancel(job_id)
        return self.repository.get(job_id, include_items=False)
