from __future__ import annotations

from pathlib import Path
from threading import RLock

import yaml

from labelone.errors import ModelRuntimeError

from .adapters import (
    OnnxRuntimeAdapter,
    DetrDetectionOnnxAdapter,
    DepthAnythingOnnxAdapter,
    HypirSd2SubprocessAdapter,
    RmbgMattingOnnxAdapter,
    RamTaggingOnnxAdapter,
    PpOcrOnnxAdapter,
    SegmentAnythingOnnxAdapter,
    TrustedRemoteHttpAdapter,
    YoloClassificationOnnxAdapter,
    YoloDetectionOnnxAdapter,
    YoloObbOnnxAdapter,
    YoloPoseOnnxAdapter,
    YoloSegmentationOnnxAdapter,
)
from .adapters.base import ModelAdapter
from .artifacts import ArtifactStore
from .catalog import ModelCatalog
from .types import FeatureCaptureMode, InferenceResult, ModelRuntimeState
from .weights import ModelWeightStore
from .worker_supervisor import ModelWorkerSupervisor


class ModelManager:
    def __init__(
        self,
        catalog: ModelCatalog,
        artifact_store: ArtifactStore,
        weight_store: ModelWeightStore | None = None,
        *,
        isolate_processes: bool = False,
        data_dir: Path | None = None,
        model_weights_dir: Path | None = None,
        worker_startup_timeout: float = 15.0,
        worker_request_timeout: float = 120.0,
        worker_close_timeout: float = 2.0,
        worker_max_request_bytes: int = 2 * 1024 * 1024,
        worker_max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if isolate_processes and data_dir is None:
            raise ValueError("data_dir is required when model process isolation is enabled")
        self.catalog = catalog
        self.artifact_store = artifact_store
        self.weight_store = weight_store
        self.isolate_processes = isolate_processes
        self.data_dir = data_dir.expanduser().resolve() if data_dir is not None else None
        self.model_weights_dir = (
            model_weights_dir.expanduser().resolve()
            if model_weights_dir is not None
            else weight_store.root
            if weight_store is not None
            else None
        )
        self.worker_startup_timeout = worker_startup_timeout
        self.worker_request_timeout = worker_request_timeout
        self.worker_close_timeout = worker_close_timeout
        self.worker_max_request_bytes = worker_max_request_bytes
        self.worker_max_response_bytes = worker_max_response_bytes
        self._adapters: dict[str, ModelAdapter] = {}
        self._supervisors: dict[str, ModelWorkerSupervisor] = {}
        self._states: dict[str, ModelRuntimeState] = {}
        self._lock = RLock()
        self._model_locks: dict[str, RLock] = {}
        self._closed = False

    def _model_lock(self, model_id: str) -> RLock:
        with self._lock:
            return self._model_locks.setdefault(model_id, RLock())

    def state(self, model_id: str) -> ModelRuntimeState:
        with self._lock:
            return self._states.get(model_id, ModelRuntimeState(model_id=model_id, state="unloaded"))

    @staticmethod
    def _loaded_state(model_id: str, adapter: ModelAdapter, layers) -> ModelRuntimeState:  # noqa: ANN001
        declared_mode = adapter.record.descriptor.capabilities.feature_capture.mode
        capture_mode = getattr(adapter, "runtime_capture_mode", declared_mode)
        if not isinstance(capture_mode, FeatureCaptureMode):
            capture_mode = declared_mode
        warning = getattr(adapter, "graph_warning", None)
        return ModelRuntimeState(
            model_id=model_id,
            state="loaded",
            layers=layers,
            capture_mode=capture_mode,
            capture_warning=warning if isinstance(warning, str) and warning else None,
        )

    @staticmethod
    def _catalog_root(record) -> Path:
        config_path = record.descriptor.config_path.expanduser().resolve()
        if not config_path.is_file():
            raise ModelRuntimeError(
                "Model worker cannot recover a missing catalog config",
                details={"config_path": str(config_path)},
            )
        config_dir = config_path.parent.resolve()
        models_path = config_dir.parent / "models.yaml"
        if not models_path.is_file():
            raise ModelRuntimeError(
                "Model process isolation requires a recoverable models.yaml catalog",
                details={"config_path": str(config_path), "models_path": str(models_path)},
            )
        try:
            catalog_entries = yaml.safe_load(models_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ModelRuntimeError(
                "Model process isolation could not read models.yaml",
                details={"models_path": str(models_path), "error": str(exc)},
            ) from exc
        if not isinstance(catalog_entries, list):
            raise ModelRuntimeError("Model process isolation requires models.yaml to contain a list")
        referenced = False
        for entry in catalog_entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("config_file"), str):
                continue
            raw = entry["config_file"]
            relative = raw[2:] if raw.startswith(":/") else raw
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                continue
            if (config_dir / candidate).resolve() == config_path:
                referenced = True
                break
        if not referenced:
            raise ModelRuntimeError(
                "Model config is not referenced by its recoverable models.yaml catalog",
                details={"config_path": str(config_path), "models_path": str(models_path)},
            )
        return config_dir

    def _supervisor(self, model_id: str) -> ModelWorkerSupervisor:
        with self._lock:
            if self._closed:
                raise ModelRuntimeError("Model manager is closed")
            existing = self._supervisors.get(model_id)
            if existing is not None:
                return existing
            record = self.catalog.get(model_id)
            if not record.descriptor.capabilities.predict:
                raise ModelRuntimeError(
                    "Model does not declare a runnable predict capability",
                    details={"model_id": model_id, "adapter": record.descriptor.adapter},
                )
            assert self.data_dir is not None
            supervisor = ModelWorkerSupervisor(
                model_id,
                catalog_root=self._catalog_root(record),
                data_dir=self.data_dir,
                model_weights_dir=self.model_weights_dir,
                startup_timeout=self.worker_startup_timeout,
                request_timeout=self.worker_request_timeout,
                close_timeout=self.worker_close_timeout,
                maximum_request_bytes=self.worker_max_request_bytes,
                maximum_response_bytes=self.worker_max_response_bytes,
            )
            self._supervisors[model_id] = supervisor
            return supervisor

    def _create_adapter(self, model_id: str) -> ModelAdapter:
        record = self.catalog.get(model_id)
        if self.weight_store is not None:
            record = self.weight_store.effective_record(record)
        factories = {
            "onnx_raw": OnnxRuntimeAdapter,
            "detr_detection_onnx": DetrDetectionOnnxAdapter,
            "depth_anything_onnx": DepthAnythingOnnxAdapter,
            "hypir_sd2_pytorch": HypirSd2SubprocessAdapter,
            "rmbg_matting_onnx": RmbgMattingOnnxAdapter,
            "ram_tagging_onnx": RamTaggingOnnxAdapter,
            "ppocr_onnx": PpOcrOnnxAdapter,
            "segment_anything_onnx": SegmentAnythingOnnxAdapter,
            "trusted_remote_http": TrustedRemoteHttpAdapter,
            "yolo_classification_onnx": YoloClassificationOnnxAdapter,
            "yolo_detection_onnx": YoloDetectionOnnxAdapter,
            "yolo_obb_onnx": YoloObbOnnxAdapter,
            "yolo_pose_onnx": YoloPoseOnnxAdapter,
            "yolo_segmentation_onnx": YoloSegmentationOnnxAdapter,
        }
        factory = factories.get(record.descriptor.adapter)
        if factory is None:
            raise ModelRuntimeError(
                "Model adapter is not implemented",
                details={"model_id": model_id, "adapter": record.descriptor.adapter},
            )
        return factory(record, self.artifact_store)

    def load(self, model_id: str, providers: list[str]) -> ModelRuntimeState:
        with self._model_lock(model_id):
            with self._lock:
                self._states[model_id] = ModelRuntimeState(model_id=model_id, state="loading")
            try:
                if self.isolate_processes:
                    timeout = self._configured_request_timeout(model_id, "load_timeout")
                    state = ModelRuntimeState.model_validate(self._supervisor(model_id).load(providers, timeout=timeout))
                else:
                    with self._lock:
                        adapter = self._adapters.get(model_id) or self._create_adapter(model_id)
                    layers = adapter.load(providers)
                    with self._lock:
                        self._adapters[model_id] = adapter
                    state = self._loaded_state(model_id, adapter, layers)
            except Exception as exc:
                state = ModelRuntimeState(model_id=model_id, state="failed", error=str(exc))
                with self._lock:
                    self._states[model_id] = state
                raise
            with self._lock:
                self._states[model_id] = state
            return state

    def unload(self, model_id: str) -> ModelRuntimeState:
        with self._model_lock(model_id):
            if self.isolate_processes:
                with self._lock:
                    supervisor = self._supervisors.pop(model_id, None)
                if supervisor is None:
                    state = ModelRuntimeState(model_id=model_id, state="unloaded")
                else:
                    supervisor.close()
                    state = ModelRuntimeState(model_id=model_id, state="unloaded")
                with self._lock:
                    self._states[model_id] = state
                return state
            with self._lock:
                adapter = self._adapters.pop(model_id, None)
            if adapter:
                adapter.unload()
            state = ModelRuntimeState(model_id=model_id, state="unloaded")
            with self._lock:
                self._states[model_id] = state
            return state

    def layers(self, model_id: str) -> ModelRuntimeState:
        with self._model_lock(model_id):
            state = self.state(model_id)
            if state.state != "loaded":
                return state
            if self.isolate_processes:
                return ModelRuntimeState.model_validate(self._supervisor(model_id).layers())
            with self._lock:
                adapter = self._adapters[model_id]
            return self._loaded_state(model_id, adapter, adapter.list_layers())

    def predict(self, model_id: str, image_path: Path, capture_layers: list[str], parameters: dict[str, object]) -> InferenceResult:
        with self._model_lock(model_id):
            if self.isolate_processes:
                if self.state(model_id).state != "loaded":
                    raise ModelRuntimeError("Model is not loaded", details={"model_id": model_id})
                timeout = self._configured_request_timeout(model_id, "inference_timeout")
                return InferenceResult.model_validate(
                    self._supervisor(model_id).predict(image_path, capture_layers, parameters, timeout=timeout)
                )
            with self._lock:
                adapter = self._adapters.get(model_id)
            if not adapter or not adapter.loaded:
                raise ModelRuntimeError("Model is not loaded", details={"model_id": model_id})
            return adapter.predict(image_path, capture_layers, parameters)

    def _configured_request_timeout(self, model_id: str, key: str) -> float | None:
        record = self.catalog.get(model_id)
        if record.descriptor.adapter != "hypir_sd2_pytorch":
            return None
        raw = record.config.get(key, 600)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError(f"HYPIR {key} must be numeric") from exc
        if not 1 <= value <= 3600:
            raise ModelRuntimeError(f"HYPIR {key} is outside the supported range")
        return value

    def close_all(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            model_ids = sorted(set(self._supervisors) | set(self._adapters))
        for model_id in model_ids:
            with self._model_lock(model_id):
                with self._lock:
                    supervisor = self._supervisors.pop(model_id, None)
                    adapter = self._adapters.pop(model_id, None)
                if supervisor is not None:
                    supervisor.close()
                if adapter is not None:
                    adapter.unload()
                with self._lock:
                    self._states[model_id] = ModelRuntimeState(model_id=model_id, state="unloaded")
