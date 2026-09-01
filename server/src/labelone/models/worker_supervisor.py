from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import RLock, Thread
from typing import Mapping, Sequence

from labelone.errors import ModelRuntimeError

from .worker_protocol import (
    PROTOCOL_VERSION,
    WorkerBudgetError,
    WorkerProtocolError,
    decode_message,
    encode_message,
)


_WORKER_MODULES = {
    "labelone.models.worker_process",
    "labelone.models.worker_fixture",
}


class ModelWorkerError(ModelRuntimeError):
    code = "model_worker_error"


class ModelWorkerTimeout(ModelWorkerError):
    code = "model_worker_timeout"


class ModelWorkerCrashed(ModelWorkerError):
    code = "model_worker_crashed"


class ModelWorkerProtocolError(ModelWorkerError):
    code = "model_worker_protocol_error"


class ModelWorkerInvalidRequest(ModelWorkerProtocolError):
    code = "model_worker_invalid_request"


class ModelWorkerBudgetExceeded(ModelWorkerError):
    code = "model_worker_budget_exceeded"


class ModelWorkerRequestBudgetExceeded(ModelWorkerBudgetExceeded):
    code = "model_worker_request_budget_exceeded"


class ModelWorkerRemoteError(ModelWorkerError):
    code = "model_worker_remote_error"


class ModelWorkerClosed(ModelWorkerError):
    code = "model_worker_closed"


@dataclass(frozen=True, slots=True)
class _ReaderEof:
    pid: int


class ModelWorkerSupervisor:
    """One-model subprocess supervisor using bounded JSON-lines IPC."""

    def __init__(
        self,
        model_id: str,
        *,
        catalog_root: Path,
        data_dir: Path,
        model_weights_dir: Path | None = None,
        worker_module: str = "labelone.models.worker_process",
        worker_options: Mapping[str, object] | None = None,
        python_executable: str | None = None,
        startup_timeout: float = 15.0,
        request_timeout: float = 120.0,
        close_timeout: float = 2.0,
        maximum_request_bytes: int = 2 * 1024 * 1024,
        maximum_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not isinstance(model_id, str) or not model_id or len(model_id) > 256:
            raise ValueError("model_id must be a bounded non-empty string")
        if worker_module not in _WORKER_MODULES:
            raise ValueError("worker_module is not allowlisted")
        if startup_timeout <= 0 or request_timeout <= 0 or close_timeout <= 0:
            raise ValueError("worker timeouts must be positive")
        if not 1024 <= maximum_request_bytes <= 64 * 1024 * 1024:
            raise ValueError("maximum_request_bytes is outside the supported range")
        if not 1024 <= maximum_response_bytes <= 64 * 1024 * 1024:
            raise ValueError("maximum_response_bytes is outside the supported range")
        self.model_id = model_id
        self.catalog_root = catalog_root.expanduser().resolve()
        self.data_dir = data_dir.expanduser().resolve()
        self.model_weights_dir = (
            model_weights_dir.expanduser().resolve()
            if model_weights_dir is not None
            else self.data_dir / "model-weights"
        )
        self.worker_module = worker_module
        self.worker_options = dict(worker_options or {})
        self.python_executable = python_executable or sys.executable
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.close_timeout = float(close_timeout)
        self.maximum_request_bytes = int(maximum_request_bytes)
        self.maximum_response_bytes = int(maximum_response_bytes)
        self.restart_count = 0
        self.last_crash: dict[str, object] | None = None
        self._request_id = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: Queue[bytes | _ReaderEof] = Queue()
        self._stdout_thread: Thread | None = None
        self._stderr_thread: Thread | None = None
        self._stderr_tail = bytearray()
        self._loaded_providers: list[str] | None = None
        self._lock = RLock()
        self._stderr_lock = RLock()
        self._closed = False

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    @property
    def is_alive(self) -> bool:
        return self.pid is not None

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        current = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source_root if not current else f"{source_root}{os.pathsep}{current}"
        return environment

    def _read_stdout(self, process: subprocess.Popen[bytes], queue: Queue[bytes | _ReaderEof]) -> None:
        assert process.stdout is not None
        while True:
            line = process.stdout.readline(self.maximum_response_bytes + 1)
            if not line:
                queue.put(_ReaderEof(process.pid))
                return
            queue.put(line)

    def _read_stderr(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stderr is not None
        while chunk := process.stderr.read(4096):
            with self._stderr_lock:
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > 64 * 1024:
                    del self._stderr_tail[: len(self._stderr_tail) - 64 * 1024]

    def _write(self, message: object) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise self._crashed("Model worker is not running")
        try:
            encoded = encode_message(message, maximum_bytes=self.maximum_request_bytes)
        except WorkerBudgetError as exc:
            raise ModelWorkerRequestBudgetExceeded(str(exc)) from exc
        except WorkerProtocolError as exc:
            raise ModelWorkerInvalidRequest(str(exc)) from exc
        try:
            process.stdin.write(encoded)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._crashed("Model worker IPC pipe closed") from exc

    def _next_message(self, timeout: float) -> dict[str, object]:
        process = self._process
        if process is None:
            raise self._crashed("Model worker is not running")
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except Empty as exc:
            if process.poll() is not None:
                raise self._crashed("Model worker exited while waiting for a response") from exc
            raise ModelWorkerTimeout(
                "Model worker request timed out",
                details={"model_id": self.model_id, "timeout_seconds": timeout, "pid": process.pid},
            ) from exc
        if isinstance(item, _ReaderEof):
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            raise self._crashed("Model worker exited before sending a response")
        if len(item) > self.maximum_response_bytes:
            raise ModelWorkerBudgetExceeded(
                "Model worker response exceeded the byte budget",
                details={"maximum": self.maximum_response_bytes},
            )
        try:
            return decode_message(item, maximum_bytes=self.maximum_response_bytes)
        except WorkerBudgetError as exc:
            raise ModelWorkerBudgetExceeded(str(exc)) from exc
        except WorkerProtocolError as exc:
            raise ModelWorkerProtocolError(str(exc)) from exc

    def _crashed(self, message: str) -> ModelWorkerCrashed:
        process = self._process
        exit_code = process.poll() if process is not None else None
        with self._stderr_lock:
            stderr = bytes(self._stderr_tail).decode("utf-8", errors="replace")[-4096:]
        return ModelWorkerCrashed(
            message,
            details={
                "model_id": self.model_id,
                "pid": process.pid if process is not None else None,
                "exit_code": exit_code,
                "suspected_oom": exit_code in {-9, 9, 137},
                "stderr_tail": stderr,
            },
        )

    def _start(self) -> None:
        self._terminate()
        command: Sequence[str] = [self.python_executable, "-m", self.worker_module]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._environment(),
            bufsize=0,
        )
        self._process = process
        self._stdout_queue = Queue()
        with self._stderr_lock:
            self._stderr_tail = bytearray()
        self._stdout_thread = Thread(
            target=self._read_stdout,
            args=(process, self._stdout_queue),
            name=f"labelone-worker-stdout-{process.pid}",
            daemon=True,
        )
        self._stderr_thread = Thread(
            target=self._read_stderr,
            args=(process,),
            name=f"labelone-worker-stderr-{process.pid}",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._write({
                "type": "init",
                "protocol": PROTOCOL_VERSION,
                "model_id": self.model_id,
                "catalog_root": str(self.catalog_root),
                "data_dir": str(self.data_dir),
                "model_weights_dir": str(self.model_weights_dir),
                "max_request_bytes": self.maximum_request_bytes,
                "max_response_bytes": self.maximum_response_bytes,
                "options": self.worker_options,
            })
            ready = self._next_message(self.startup_timeout)
            if ready.get("type") == "startup_error":
                error = ready.get("error")
                raise ModelWorkerRemoteError(
                    "Model worker failed to initialize",
                    details=error if isinstance(error, dict) else {},
                )
            if ready.get("type") != "ready" or ready.get("protocol") != PROTOCOL_VERSION:
                raise ModelWorkerProtocolError("Model worker did not send a valid ready handshake")
        except Exception:
            self._terminate()
            raise

    def _ensure_started(self) -> bool:
        if self._closed:
            raise ModelWorkerClosed("Model worker supervisor is closed")
        if self._process is None:
            self._start()
            return True
        if self._process.poll() is not None:
            raise self._crashed("Model worker exited between requests")
        return False

    def _exchange(self, operation: str, payload: dict[str, object], timeout: float) -> object:
        self._request_id += 1
        request_id = self._request_id
        self._write({"id": request_id, "op": operation, "payload": payload})
        response = self._next_message(timeout)
        if response.get("id") != request_id or not isinstance(response.get("ok"), bool):
            raise ModelWorkerProtocolError(
                "Model worker response id or status is invalid",
                details={"expected_id": request_id, "response_id": response.get("id")},
            )
        if response["ok"]:
            return response.get("result")
        error = response.get("error")
        details = error.get("details") if isinstance(error, dict) else None
        raise ModelWorkerRemoteError(
            str(error.get("message") if isinstance(error, dict) else "Model worker request failed"),
            details={
                "remote_code": error.get("code") if isinstance(error, dict) else None,
                "remote_details": details if isinstance(details, dict) else {},
            },
        )

    def _call(self, operation: str, payload: dict[str, object], *, timeout: float | None = None) -> object:
        request_timeout = self.request_timeout if timeout is None else timeout
        if request_timeout <= 0:
            raise ValueError("request timeout must be positive")
        with self._lock:
            for attempt in range(2):
                try:
                    started = self._ensure_started()
                    if started and self._loaded_providers is not None and operation != "load":
                        self._exchange("load", {"providers": self._loaded_providers}, request_timeout)
                    result = self._exchange(operation, payload, request_timeout)
                    if operation == "load":
                        self._loaded_providers = list(payload["providers"])  # type: ignore[arg-type]
                    elif operation == "unload":
                        self._loaded_providers = None
                    return result
                except ModelWorkerTimeout:
                    self._terminate()
                    raise
                except (ModelWorkerInvalidRequest, ModelWorkerRequestBudgetExceeded):
                    raise
                except ModelWorkerBudgetExceeded:
                    self._terminate()
                    raise
                except (ModelWorkerCrashed, ModelWorkerProtocolError) as exc:
                    self.last_crash = dict(exc.details)
                    self._terminate()
                    if attempt == 0:
                        self.restart_count += 1
                        continue
                    raise
            raise ModelWorkerCrashed("Model worker restart budget was exhausted")

    def load(self, providers: list[str], *, timeout: float | None = None) -> dict[str, object]:
        if not isinstance(providers, list) or len(providers) > 16 or any(not isinstance(item, str) for item in providers):
            raise ValueError("providers must be a bounded string array")
        result = self._call("load", {"providers": providers}, timeout=timeout)
        if not isinstance(result, dict):
            raise ModelWorkerProtocolError("Model worker load result must be an object")
        return result

    def layers(self, *, timeout: float | None = None) -> dict[str, object]:
        result = self._call("layers", {}, timeout=timeout)
        if not isinstance(result, dict):
            raise ModelWorkerProtocolError("Model worker layers result must be an object")
        return result

    def predict(
        self,
        image_path: Path,
        capture_layers: list[str],
        parameters: Mapping[str, object],
        *,
        timeout: float | None = None,
    ) -> dict[str, object]:
        if len(capture_layers) > 256 or any(not isinstance(item, str) for item in capture_layers):
            raise ValueError("capture_layers must be a bounded string array")
        if not isinstance(parameters, Mapping) or len(parameters) > 512:
            raise ValueError("parameters must be a bounded object")
        result = self._call(
            "predict",
            {
                "image_path": str(image_path.expanduser().resolve()),
                "capture_layers": list(capture_layers),
                "parameters": dict(parameters),
            },
            timeout=timeout,
        )
        if not isinstance(result, dict):
            raise ModelWorkerProtocolError("Model worker prediction result must be an object")
        return result

    def unload(self, *, timeout: float | None = None) -> dict[str, object]:
        result = self._call("unload", {}, timeout=timeout)
        if not isinstance(result, dict):
            raise ModelWorkerProtocolError("Model worker unload result must be an object")
        return result

    def _terminate(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        else:
            process.wait(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        self._stdout_thread = None
        self._stderr_thread = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._exchange("close", {}, self.close_timeout)
                except ModelWorkerError:
                    pass
            self._closed = True
            self._loaded_providers = None
            self._terminate()

    def __enter__(self) -> "ModelWorkerSupervisor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()
