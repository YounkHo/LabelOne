from __future__ import annotations

from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
from typing import Any

from labelone.errors import LabelOneError

from .artifacts import ArtifactStore
from .catalog import ModelCatalog
from .manager import ModelManager
from .weights import ModelWeightStore
from .worker_protocol import (
    ALLOWED_OPERATIONS,
    PROTOCOL_VERSION,
    WorkerBudgetError,
    WorkerProtocolError,
    read_message,
    write_message,
)


DEFAULT_REQUEST_BYTES = 2 * 1024 * 1024
DEFAULT_RESPONSE_BYTES = 8 * 1024 * 1024


def _write_response(protocol_output, response: dict[str, Any], *, maximum_bytes: int) -> None:  # noqa: ANN001
    try:
        write_message(protocol_output, response, maximum_bytes=maximum_bytes)
    except WorkerBudgetError:
        write_message(
            protocol_output,
            {
                "id": response.get("id", 0),
                "ok": False,
                "error": {
                    "code": "worker_response_budget_exceeded",
                    "message": "Worker result exceeded the IPC response budget",
                    "details": {"maximum_bytes": maximum_bytes},
                },
            },
            maximum_bytes=maximum_bytes,
        )
    except WorkerProtocolError:
        write_message(
            protocol_output,
            {
                "id": response.get("id", 0),
                "ok": False,
                "error": {
                    "code": "worker_response_protocol_error",
                    "message": "Worker result contained non-finite or non-JSON data",
                    "details": {},
                },
            },
            maximum_bytes=maximum_bytes,
        )


def _integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise WorkerProtocolError(f"{name} is outside the supported range")
    return value


def _string_list(value: object, *, name: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum or any(not isinstance(item, str) for item in value):
        raise WorkerProtocolError(f"{name} must be a bounded string array")
    return value


def _request(message: dict[str, object]) -> tuple[int, str, dict[str, object]]:
    if set(message) != {"id", "op", "payload"}:
        raise WorkerProtocolError("Worker request fields are invalid")
    request_id = _integer(message["id"], name="request id", minimum=1, maximum=2**63 - 1)
    operation = message["op"]
    payload = message["payload"]
    if operation not in ALLOWED_OPERATIONS or not isinstance(operation, str):
        raise WorkerProtocolError("Worker operation is not allowlisted")
    if not isinstance(payload, dict):
        raise WorkerProtocolError("Worker payload must be an object")
    return request_id, operation, payload


def _initialize(message: dict[str, object]) -> tuple[ModelManager, str, int, int]:
    allowed = {
        "type",
        "protocol",
        "model_id",
        "catalog_root",
        "data_dir",
        "model_weights_dir",
        "max_request_bytes",
        "max_response_bytes",
        "options",
    }
    if set(message) - allowed or message.get("type") != "init" or message.get("protocol") != PROTOCOL_VERSION:
        raise WorkerProtocolError("Worker initialization message is invalid")
    model_id = message.get("model_id")
    catalog_root = message.get("catalog_root")
    data_dir = message.get("data_dir")
    model_weights_dir = message.get("model_weights_dir")
    if not all(isinstance(value, str) and value for value in (model_id, catalog_root, data_dir, model_weights_dir)):
        raise WorkerProtocolError("Worker initialization paths and model id are required")
    request_budget = _integer(
        message.get("max_request_bytes", DEFAULT_REQUEST_BYTES),
        name="request byte budget",
        minimum=1024,
        maximum=64 * 1024 * 1024,
    )
    response_budget = _integer(
        message.get("max_response_bytes", DEFAULT_RESPONSE_BYTES),
        name="response byte budget",
        minimum=1024,
        maximum=64 * 1024 * 1024,
    )
    root = Path(catalog_root).expanduser().resolve()
    worker_data = Path(data_dir).expanduser().resolve()
    worker_model_weights = Path(model_weights_dir).expanduser().resolve()
    catalog = ModelCatalog()
    catalog.import_x_anylabeling(root)
    catalog.get(model_id)
    artifacts = ArtifactStore(worker_data / "artifacts")
    weights = ModelWeightStore(worker_data, root_dir=worker_model_weights)
    return ModelManager(catalog, artifacts, weights), model_id, request_budget, response_budget


def _execute(manager: ModelManager, model_id: str, operation: str, payload: dict[str, object]) -> object:
    if operation == "load":
        if set(payload) != {"providers"}:
            raise WorkerProtocolError("load payload fields are invalid")
        providers = _string_list(payload["providers"], name="providers", maximum=16)
        return manager.load(model_id, providers).model_dump(mode="json")
    if operation == "layers":
        if payload:
            raise WorkerProtocolError("layers payload must be empty")
        return manager.layers(model_id).model_dump(mode="json")
    if operation == "predict":
        if set(payload) != {"image_path", "capture_layers", "parameters"}:
            raise WorkerProtocolError("predict payload fields are invalid")
        image_path = payload["image_path"]
        parameters = payload["parameters"]
        if not isinstance(image_path, str) or not image_path:
            raise WorkerProtocolError("predict image_path must be a string")
        capture_layers = _string_list(payload["capture_layers"], name="capture_layers", maximum=256)
        if not isinstance(parameters, dict) or len(parameters) > 512:
            raise WorkerProtocolError("predict parameters must be a bounded object")
        return manager.predict(
            model_id,
            Path(image_path),
            capture_layers,
            parameters,
        ).model_dump(mode="json")
    if operation == "unload":
        if payload:
            raise WorkerProtocolError("unload payload must be empty")
        return manager.unload(model_id).model_dump(mode="json")
    if operation == "close":
        if payload:
            raise WorkerProtocolError("close payload must be empty")
        return manager.unload(model_id).model_dump(mode="json")
    raise WorkerProtocolError("Worker operation is not allowlisted")


def main() -> int:
    protocol_output = sys.stdout.buffer
    request_budget = DEFAULT_REQUEST_BYTES
    response_budget = DEFAULT_RESPONSE_BYTES
    manager: ModelManager | None = None
    model_id = ""
    try:
        init = read_message(sys.stdin.buffer, maximum_bytes=DEFAULT_REQUEST_BYTES)
        if init is None:
            return 2
        with redirect_stdout(sys.stderr):
            manager, model_id, request_budget, response_budget = _initialize(init)
        write_message(
            protocol_output,
            {"type": "ready", "protocol": PROTOCOL_VERSION, "pid": os.getpid()},
            maximum_bytes=response_budget,
        )
    except Exception as exc:
        try:
            write_message(
                protocol_output,
                {
                    "type": "startup_error",
                    "error": {"code": "worker_startup_error", "message": str(exc), "details": {}},
                },
                maximum_bytes=response_budget,
            )
        except Exception:
            pass
        return 2

    while True:
        try:
            message = read_message(sys.stdin.buffer, maximum_bytes=request_budget)
            if message is None:
                break
            request_id, operation, payload = _request(message)
            try:
                with redirect_stdout(sys.stderr):
                    result = _execute(manager, model_id, operation, payload)
                response: dict[str, Any] = {"id": request_id, "ok": True, "result": result}
            except LabelOneError as exc:
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": {"code": exc.code, "message": exc.message, "details": exc.details},
                }
            except WorkerProtocolError as exc:
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": {"code": "worker_protocol_error", "message": str(exc), "details": {}},
                }
            except Exception as exc:
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": {"code": "worker_error", "message": str(exc), "details": {}},
                }
            _write_response(protocol_output, response, maximum_bytes=response_budget)
            if operation == "close":
                break
        except Exception as exc:
            try:
                write_message(
                    protocol_output,
                    {
                        "id": 0,
                        "ok": False,
                        "error": {"code": "worker_protocol_error", "message": str(exc), "details": {}},
                    },
                    maximum_bytes=response_budget,
                )
            except Exception:
                pass
            break
    if manager is not None and model_id:
        try:
            with redirect_stdout(sys.stderr):
                manager.unload(model_id)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
