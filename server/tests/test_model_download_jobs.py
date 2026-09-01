from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from threading import Barrier, Event
from time import monotonic, sleep
from urllib.request import Request

from labelone.datasets.repository import DatasetRepository
from labelone.jobs import BatchJobRequest, JobRepository, JobService
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor
from labelone.models.weights import ModelWeightError, ModelWeightStore


class _Response:
    def __init__(
        self,
        url: str,
        body: bytes,
        *,
        fail_after_reads: int | None = None,
        started: Event | None = None,
        release: Event | None = None,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.body = body
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}
        self.fail_after_reads = fail_after_reads
        self.started = started
        self.release = release
        self.offset = 0
        self.reads = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.fail_after_reads is not None and self.reads > self.fail_after_reads:
            raise OSError("simulated transport failure")
        if self.offset >= len(self.body):
            return b""
        end = len(self.body) if size < 0 else min(len(self.body), self.offset + size)
        chunk = self.body[self.offset:end]
        self.offset = end
        if self.started is not None and self.reads == 1:
            self.started.set()
            assert self.release is not None
            assert self.release.wait(5)
        return chunk

    def close(self) -> None:
        self.closed = True

    def geturl(self) -> str:
        return self.url


class _BarrierResponse(_Response):
    def __init__(self, url: str, body: bytes, barrier: Barrier) -> None:
        super().__init__(url, body)
        self.barrier = barrier

    def read(self, size: int = -1) -> bytes:
        if self.reads == 0:
            self.barrier.wait(5)
        return super().read(size)


class _Opener:
    def __init__(self, url: str, responses: list[_Response]) -> None:
        self.url = url
        self.responses = responses
        self.calls: list[str] = []
        self.request_headers: list[dict[str, str]] = []

    def __call__(self, request: Request, timeout: float) -> _Response:
        del timeout
        assert request.full_url == self.url
        self.calls.append(request.full_url)
        self.request_headers.append({key.casefold(): value for key, value in request.header_items()})
        if not self.responses:
            raise AssertionError("Unexpected model weight network call")
        return self.responses.pop(0)


class _ConcurrentOpener:
    def __init__(self, responses: dict[str, _Response]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, request: Request, timeout: float) -> _Response:
        del timeout
        self.calls.append(request.full_url)
        return self.responses[request.full_url]


class _Catalog:
    def __init__(self, record: ModelRecord) -> None:
        self.record = record

    def get(self, model_id: str) -> ModelRecord:
        if model_id != self.record.descriptor.id:
            raise AssertionError(model_id)
        return self.record


class _Models:
    def __init__(self, record: ModelRecord, store: ModelWeightStore) -> None:
        self.catalog = _Catalog(record)
        self.weight_store = store


def _record(tmp_path: Path, url: str) -> ModelRecord:
    return _record_urls(tmp_path, [url])


def _record_urls(tmp_path: Path, urls: list[str]) -> ModelRecord:
    descriptor = ModelDescriptor(
        id="fixture",
        name="fixture",
        display_name="Fixture",
        model_type="fixture",
        task="detection",
        family="fixture",
        adapter="onnx_raw",
        runtime=["ONNX Runtime"],
        config_path=tmp_path / "fixture.yaml",
        weight_locations=urls,
        availability=Availability(state=AvailabilityState.MISSING_WEIGHTS),
        capabilities=ModelCapabilities(),
    )
    return ModelRecord(descriptor=descriptor, config={"model_path": urls[0]})


def _runtime(tmp_path: Path, store: ModelWeightStore, record: ModelRecord):
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    jobs = JobRepository(tmp_path / "index.sqlite3", datasets)
    service = JobService(jobs, datasets, None, _Models(record, store))  # type: ignore[arg-type]
    return datasets, jobs, service


def _wait(repository: JobRepository, job_id: str, states: set[str], timeout: float = 5.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = repository.get(job_id)
        if job.state in states:
            return job
        sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {states}")


def test_confirmed_download_is_persistent_idempotent_and_emits_progress(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"a" * (2 * 1024 * 1024 + 17)
    response = _Response(url, payload)
    opener = _Opener(url, [response])
    store = ModelWeightStore(tmp_path, opener=opener, max_bytes=4 * 1024 * 1024)
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    created = service.create_model_download(
        "fixture",
        [0],
        expected_sha256={0: sha256(payload).hexdigest()},
        idempotency_key="confirmed-download",
    )
    duplicate = service.create_model_download(
        "fixture",
        [0],
        expected_sha256={0: sha256(payload).hexdigest()},
        idempotency_key="confirmed-download",
    )
    finished = _wait(jobs, created.job_id, {"succeeded"})

    assert duplicate.job_id == created.job_id
    assert finished.total == finished.completed == 1
    assert finished.items[0].asset_id == "weight:0"
    assert finished.items[0].result["sha256"] == sha256(payload).hexdigest()
    assert Path(finished.items[0].result["local_path"]).read_bytes() == payload
    progress = [event.payload for event in jobs.list_events(created.job_id) if event.event_type == "item.progress"]
    assert progress
    assert [event["received_bytes"] for event in progress] == sorted(event["received_bytes"] for event in progress)
    assert progress[-1]["progress"] == 1.0
    assert len(opener.calls) == 1
    assert response.closed
    service.close()
    jobs.close()
    datasets.close()


def test_multiple_weight_files_download_concurrently(tmp_path: Path) -> None:
    urls = [
        "https://github.com/org/model-a.onnx",
        "https://github.com/org/model-b.onnx",
    ]
    barrier = Barrier(2)
    opener = _ConcurrentOpener({url: _BarrierResponse(url, url.encode(), barrier) for url in urls})
    store = ModelWeightStore(tmp_path, opener=opener, max_bytes=1024)
    record = _record_urls(tmp_path, urls)
    datasets, jobs, service = _runtime(tmp_path, store, record)

    created = service.create_model_download("fixture", [0, 1])
    finished = _wait(jobs, created.job_id, {"succeeded"})

    assert created.request.concurrency == 2
    assert finished.total == finished.completed == 2
    assert set(opener.calls) == set(urls)
    directory = tmp_path / "model-weights" / "fixture"
    assert (directory / "model-a.onnx").read_bytes() == urls[0].encode()
    assert (directory / "model-b.onnx").read_bytes() == urls[1].encode()
    service.close()
    jobs.close()
    datasets.close()


def test_failed_download_preserves_partial_and_manual_resume_reuses_it(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    failed_response = _Response(url, b"partial", fail_after_reads=1)
    success_response = _Response(url, b"complete")
    opener = _Opener(url, [failed_response, success_response])
    store = ModelWeightStore(tmp_path, opener=opener, max_bytes=1024)
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    created = service.create_model_download("fixture", [0])
    failed = _wait(jobs, created.job_id, {"succeeded_with_errors"})
    directory = tmp_path / "model-weights" / "fixture"

    assert failed.failed == 1
    assert (directory / ".model.onnx.part").read_bytes() == b"partial"
    assert (directory / ".model.onnx.part.json").is_file()
    service.resume(created.job_id)
    succeeded = _wait(jobs, created.job_id, {"succeeded"})
    assert succeeded.items[0].attempts == 2
    assert Path(succeeded.items[0].result["local_path"]).read_bytes() == b"complete"
    assert len(opener.calls) == 2
    assert opener.request_headers[1]["range"] == "bytes=7-"
    assert not list(directory.glob("*.part"))
    service.close()
    jobs.close()
    datasets.close()


def test_cancel_during_stream_closes_response_and_preserves_resumable_partial(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    started = Event()
    release = Event()
    response = _Response(url, b"x" * (2 * 1024 * 1024), started=started, release=release)
    store = ModelWeightStore(tmp_path, opener=_Opener(url, [response]), max_bytes=4 * 1024 * 1024)
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    created = service.create_model_download("fixture", [0])
    assert started.wait(5)
    service.cancel(created.job_id)
    release.set()
    canceled = _wait(jobs, created.job_id, {"canceled"})
    directory = tmp_path / "model-weights" / "fixture"

    assert canceled.canceled == 1
    assert response.closed
    assert (directory / ".model.onnx.part").stat().st_size == 1024 * 1024
    assert (directory / ".model.onnx.part.json").is_file()
    assert not (directory / "model.onnx").exists()
    service.close()
    jobs.close()
    datasets.close()


def test_pause_during_stream_requeues_cleanly_and_resume_uses_range(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    started = Event()
    release = Event()
    payload = b"x" * (2 * 1024 * 1024)
    interrupted_response = _Response(
        url,
        payload,
        started=started,
        release=release,
        headers={"Content-Length": str(len(payload)), "ETag": '"pause-v1"'},
    )
    remaining = payload[1024 * 1024:]
    resumed_response = _Response(
        url,
        remaining,
        status=206,
        headers={
            "Content-Length": str(len(remaining)),
            "Content-Range": f"bytes {1024 * 1024}-{len(payload) - 1}/{len(payload)}",
            "ETag": '"pause-v1"',
        },
    )
    opener = _Opener(url, [interrupted_response, resumed_response])
    store = ModelWeightStore(tmp_path, opener=opener, max_bytes=4 * 1024 * 1024)
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    created = service.create_model_download("fixture", [0])
    assert started.wait(5)
    service.pause(created.job_id)
    release.set()
    paused = _wait(jobs, created.job_id, {"paused"})
    directory = tmp_path / "model-weights" / "fixture"

    assert paused.items[0].state == "queued"
    assert interrupted_response.closed
    assert (directory / ".model.onnx.part").stat().st_size == 1024 * 1024
    service.resume(created.job_id)
    succeeded = _wait(jobs, created.job_id, {"succeeded"})
    assert succeeded.items[0].attempts == 2
    assert Path(succeeded.items[0].result["local_path"]).read_bytes() == payload
    assert len(opener.calls) == 2
    assert opener.request_headers[1]["range"] == f"bytes={1024 * 1024}-"
    assert opener.request_headers[1]["if-range"] == '"pause-v1"'
    service.close()
    jobs.close()
    datasets.close()


def test_restart_requeues_download_and_cleans_crashed_part_before_retry(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"recovered"
    store = ModelWeightStore(tmp_path, opener=_Opener(url, [_Response(url, payload)]), max_bytes=1024)
    record = _record(tmp_path, url)
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    jobs = JobRepository(tmp_path / "index.sqlite3", datasets)
    request = BatchJobRequest(kind="model_download", model_id="fixture", concurrency=1, weight_url_indices=[0])
    created = jobs.create(request, idempotency_key="restart-download")
    assert jobs.transition_to_running(created.job_id)
    item = jobs.queued_items(created.job_id, 1)[0]
    assert jobs.mark_item_running(created.job_id, item.asset_id)
    directory = tmp_path / "model-weights" / "fixture"
    directory.mkdir(parents=True)
    stale = directory / ".model.onnx.part"
    stale.write_bytes(b"crashed")
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    assert reopened.get(created.job_id).state == "interrupted"
    service = JobService(reopened, datasets, None, _Models(record, store))  # type: ignore[arg-type]
    service.start()
    finished = _wait(reopened, created.job_id, {"succeeded"})

    assert finished.completed == 1
    assert not stale.exists()
    assert (directory / "model.onnx").read_bytes() == payload
    service.close()
    reopened.close()
    datasets.close()


def test_service_restart_preserves_valid_partial_and_resumes_range(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    payload = b"r" * (1024 * 1024) + b"restart-tail"
    remaining = payload[1024 * 1024:]
    response = _Response(
        url,
        remaining,
        status=206,
        headers={
            "Content-Length": str(len(remaining)),
            "Content-Range": f"bytes {1024 * 1024}-{len(payload) - 1}/{len(payload)}",
            "ETag": '"restart-v1"',
        },
    )
    opener = _Opener(url, [response])
    store = ModelWeightStore(tmp_path, opener=opener, max_bytes=2 * 1024 * 1024)
    record = _record(tmp_path, url)
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    jobs = JobRepository(tmp_path / "index.sqlite3", datasets)
    created = jobs.create(
        BatchJobRequest(kind="model_download", model_id="fixture", concurrency=1, weight_url_indices=[0]),
        idempotency_key="restart-resume",
    )
    assert jobs.transition_to_running(created.job_id)
    item = jobs.queued_items(created.job_id, 1)[0]
    assert jobs.mark_item_running(created.job_id, item.asset_id)
    directory = tmp_path / "model-weights" / "fixture"
    directory.mkdir(parents=True)
    partial = directory / ".model.onnx.part"
    partial.write_bytes(payload[:1024 * 1024])
    partial.with_name(f"{partial.name}.json").write_text(json.dumps({
        "version": 1,
        "source_url": url,
        "final_url": url,
        "etag": '"restart-v1"',
        "last_modified": None,
        "total_bytes": len(payload),
    }), encoding="utf-8")
    jobs.close()

    reopened = JobRepository(tmp_path / "index.sqlite3", datasets)
    service = JobService(reopened, datasets, None, _Models(record, store))  # type: ignore[arg-type]
    service.start()
    finished = _wait(reopened, created.job_id, {"succeeded"})

    assert finished.completed == 1
    assert (directory / "model.onnx").read_bytes() == payload
    assert opener.request_headers[0]["range"] == f"bytes={1024 * 1024}-"
    assert opener.request_headers[0]["if-range"] == '"restart-v1"'
    service.close()
    reopened.close()
    datasets.close()


def test_hash_mismatch_fails_without_publishing_file_or_partial(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    store = ModelWeightStore(tmp_path, opener=_Opener(url, [_Response(url, b"content")]), max_bytes=1024)
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    created = service.create_model_download("fixture", [0], expected_sha256={0: "0" * 64})
    failed = _wait(jobs, created.job_id, {"succeeded_with_errors"})
    directory = tmp_path / "model-weights" / "fixture"

    assert "SHA-256" in failed.items[0].error
    assert not (directory / "model.onnx").exists()
    assert not list(directory.glob("*.part"))
    assert not list(directory.glob(".*.part"))
    service.close()
    jobs.close()
    datasets.close()


def test_unsafe_remote_is_rejected_before_job_creation_or_transport(tmp_path: Path) -> None:
    url = "https://evil.example/model.onnx"
    opener = _Opener(url, [])
    store = ModelWeightStore(tmp_path, opener=opener, max_bytes=1024)
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    try:
        service.create_model_download("fixture", [0])
    except ModelWeightError as exc:
        assert "allowlisted" in exc.message
    else:
        raise AssertionError("unsafe model download was accepted")

    assert jobs.list().jobs == []
    assert opener.calls == []
    service.close()
    jobs.close()
    datasets.close()


def test_service_start_cleans_stale_parts_even_without_a_resumable_job(tmp_path: Path) -> None:
    url = "https://github.com/org/model.onnx"
    store = ModelWeightStore(tmp_path, opener=_Opener(url, []), max_bytes=1024)
    directory = tmp_path / "model-weights" / "fixture"
    directory.mkdir(parents=True)
    stale = directory / ".abandoned.onnx.part"
    stale.write_bytes(b"abandoned")
    datasets, jobs, service = _runtime(tmp_path, store, _record(tmp_path, url))

    service.start()

    assert not stale.exists()
    service.close()
    jobs.close()
    datasets.close()
