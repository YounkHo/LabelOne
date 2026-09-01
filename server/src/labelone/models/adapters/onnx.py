from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult, ClassificationResult, FeatureLayer, InferenceResult, RasterArtifact
from ..features import FeatureTransformOptions, transform_feature
from ..onnx_graph import inspect_onnx_graph, instrument_onnx_outputs
from ..types import FeatureCaptureMode
from .base import ModelAdapter


MAX_SOURCE_FEATURE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _ImageTransform:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_x: float
    pad_y: float


def _shape(value: list[Any] | tuple[Any, ...]) -> list[int | str | None]:
    result: list[int | str | None] = []
    for item in value:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str):
            result.append(item)
        else:
            result.append(None)
    return result


def _axes_for(shape: list[int | str | None]) -> list[str]:
    if len(shape) == 1:
        return ["C"]
    if len(shape) == 2:
        return ["N", "C"]
    if len(shape) == 4:
        return ["N", "C", "H", "W"]
    if len(shape) == 3:
        return ["N", "T", "C"]
    return [f"D{index}" for index in range(len(shape))]


class OnnxRuntimeAdapter(ModelAdapter):
    def __init__(self, record, artifact_store) -> None:
        super().__init__(record, artifact_store)
        self.session = None
        self.input_meta = None
        self.output_meta: list[Any] = []
        self.feature_layers: list[FeatureLayer] = []
        self.feature_output_names: set[str] = set()
        self.graph_warning: str | None = None
        self.runtime_capture_mode = FeatureCaptureMode.EXPORTED_OUTPUTS
        self.model_path: Path | None = None
        self.providers: list[str] = []
        self.instrumented_capture_names: frozenset[str] = frozenset()

    def load(self, providers: list[str]) -> list[FeatureLayer]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelRuntimeError("onnxruntime is not installed") from exc
        model_path = self.resolve_local_weight(".onnx")
        available = set(ort.get_available_providers())
        selected = [provider for provider in providers if provider in available] or ["CPUExecutionProvider"]
        self.model_path = model_path
        self.providers = selected
        graph = None
        try:
            graph = inspect_onnx_graph(model_path)
        except ModelRuntimeError as exc:
            self.graph_warning = str(exc)
        try:
            self.session = ort.InferenceSession(str(model_path), providers=selected)
        except Exception as exc:
            raise ModelRuntimeError("Failed to load ONNX model", details={"path": str(model_path), "error": str(exc)}) from exc
        inputs = self.session.get_inputs()
        self._configure_inputs(inputs)
        runtime_outputs = list(self.session.get_outputs())
        output_by_name = {str(output.name): output for output in runtime_outputs}
        original_names = graph.original_output_names if graph else [str(output.name) for output in runtime_outputs]
        self.output_meta = [output_by_name[name] for name in original_names if name in output_by_name]
        if not self.output_meta:
            self.output_meta = runtime_outputs
        declared_layers = graph.layers if graph else []
        self.feature_layers = []
        for layer in declared_layers:
            output = output_by_name.get(layer.id)
            output_shape = _shape(list(output.shape)) if output is not None else layer.shape
            self.feature_layers.append(layer.model_copy(update={
                "shape": output_shape,
                "axes": _axes_for(output_shape),
                "dtype": str(output.type) if output is not None else layer.dtype,
                "spatial": len(output_shape) == 4,
            }))
        if not self.feature_layers:
            for output in self.output_meta:
                output_shape = _shape(list(output.shape))
                captureable = len(output_shape) in {1, 2, 3, 4} and "float" in str(output.type).casefold()
                self.feature_layers.append(FeatureLayer(
                    id=output.name,
                    group="模型输出",
                    name=output.name,
                    shape=output_shape,
                    axes=_axes_for(output_shape),
                    dtype=str(output.type),
                    spatial=len(output_shape) == 4,
                    captureable=captureable,
                    reason=None if captureable else "通用特征预览仅支持有界浮点向量、矩阵、NTC 或 NCHW Tensor",
                ))
        self.feature_output_names = {layer.id for layer in self.feature_layers if layer.captureable}
        if graph and graph.rewrite_supported and any(layer.captureable and layer.id not in set(original_names) for layer in self.feature_layers):
            self.runtime_capture_mode = FeatureCaptureMode.GRAPH_REWRITE
        if graph and graph.warning:
            self.graph_warning = graph.warning
        self.loaded = True
        return self.list_layers()

    def _configure_inputs(self, inputs: list[Any]) -> None:
        if len(inputs) != 1:
            raise ModelRuntimeError(
                "This ONNX adapter requires exactly one image input",
                details={"input_count": len(inputs)},
            )
        self.input_meta = inputs[0]

    def unload(self) -> None:
        self.session = None
        self.input_meta = None
        self.output_meta = []
        self.feature_layers = []
        self.feature_output_names = set()
        self.graph_warning = None
        self.runtime_capture_mode = FeatureCaptureMode.EXPORTED_OUTPUTS
        self.model_path = None
        self.providers = []
        self.instrumented_capture_names = frozenset()
        self.loaded = False

    def list_layers(self) -> list[FeatureLayer]:
        return [layer.model_copy(deep=True) for layer in self.feature_layers]

    def _ensure_capture_session(self, capture_layers: list[str]) -> None:
        original_names = [str(meta.name) for meta in self.output_meta]
        original_name_set = set(original_names)
        intermediate_names = frozenset(layer for layer in capture_layers if layer not in original_name_set)
        if not intermediate_names or intermediate_names == self.instrumented_capture_names:
            return
        if self.model_path is None:
            raise ModelRuntimeError("ONNX model path is unavailable for feature instrumentation")
        try:
            import onnxruntime as ort
            model_bytes = instrument_onnx_outputs(self.model_path, sorted(intermediate_names))
            session = ort.InferenceSession(model_bytes, providers=self.providers)
            inputs = session.get_inputs()
            previous_input_meta = self.input_meta
            try:
                self._configure_inputs(inputs)
                configured_input_meta = self.input_meta
            finally:
                self.input_meta = previous_input_meta
            output_by_name = {str(output.name): output for output in session.get_outputs()}
            original_meta = [output_by_name[name] for name in original_names if name in output_by_name]
            if len(original_meta) != len(original_names):
                raise ModelRuntimeError("Instrumented ONNX session lost a model output")
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise ModelRuntimeError("Failed to load instrumented ONNX model", details={"error": str(exc)}) from exc
        self.session = session
        self.input_meta = configured_input_meta
        self.output_meta = original_meta
        self.instrumented_capture_names = intermediate_names

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, _ImageTransform]:
        if self.input_meta is None:
            raise ModelRuntimeError("Model is not loaded")
        input_shape = list(self.input_meta.shape)
        if len(input_shape) != 4:
            raise ModelRuntimeError("Image input must have four dimensions", details={"shape": input_shape})
        nchw = input_shape[1] in {1, 3, 4, "C", "channels"}
        input_height = input_shape[2] if nchw and isinstance(input_shape[2], int) else input_shape[1] if not nchw and isinstance(input_shape[1], int) else 640
        input_width = input_shape[3] if nchw and isinstance(input_shape[3], int) else input_shape[2] if not nchw and isinstance(input_shape[2], int) else 640
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            original_width, original_height = image.size
            if self._image_resize_mode() == "stretch":
                scale = input_width / original_width
                resized_width, resized_height = input_width, input_height
            else:
                scale = min(input_width / original_width, input_height / original_height)
                resized_width = max(1, round(original_width * scale))
                resized_height = max(1, round(original_height * scale))
            resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_x = (input_width - resized_width) / 2
        pad_y = (input_height - resized_height) / 2
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        canvas[top : top + resized_height, left : left + resized_width] = np.asarray(resized)
        tensor = canvas.astype(np.float32) / 255.0
        if nchw:
            tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        if "float16" in str(self.input_meta.type):
            tensor = tensor.astype(np.float16)
        transform = _ImageTransform(
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
            scale=scale,
            pad_x=left,
            pad_y=top,
        )
        return tensor, transform

    def _image_resize_mode(self) -> str:
        return "letterbox"

    def _input_feed(self, tensor: np.ndarray, transform: _ImageTransform) -> dict[str, np.ndarray]:
        if self.input_meta is None:
            raise ModelRuntimeError("Model is not loaded")
        return {self.input_meta.name: tensor}

    def _annotations(self, outputs: dict[str, np.ndarray], transform: _ImageTransform, parameters: dict[str, object]) -> list[AnnotationResult]:
        return []

    def _classifications(
        self,
        outputs: dict[str, np.ndarray],
        parameters: dict[str, object],
    ) -> list[ClassificationResult]:
        return []

    def _rasters(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        image_path: Path,
        parameters: dict[str, object],
    ) -> list[RasterArtifact]:
        return []

    def predict(self, image_path: Path, capture_layers: list[str], parameters: dict[str, object]) -> InferenceResult:
        if not self.loaded or self.session is None or self.input_meta is None:
            raise ModelRuntimeError("Model is not loaded", details={"model_id": self.record.descriptor.id})
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            raise ModelRuntimeError("Inference image does not exist", details={"image_path": str(image_path)})
        started = perf_counter()
        tensor, transform = self._prepare_image(image_path)
        preprocessed = perf_counter()
        unknown = sorted(set(capture_layers) - self.feature_output_names)
        if unknown:
            raise ModelRuntimeError("Requested capture layer is not available in the ONNX graph", details={"unknown_layers": unknown})
        if len(capture_layers) > 1:
            raise ModelRuntimeError("Only one ONNX feature layer may be captured per inference run")
        self._ensure_capture_session(capture_layers)
        requested_names = list(dict.fromkeys([*(str(meta.name) for meta in self.output_meta), *capture_layers]))
        try:
            values = self.session.run(requested_names, self._input_feed(tensor, transform))
        except Exception as exc:
            raise ModelRuntimeError("ONNX inference failed", details={"error": str(exc)}) from exc
        inferred = perf_counter()
        requested_outputs = {name: np.asarray(value) for name, value in zip(requested_names, values, strict=True)}
        outputs = {str(meta.name): requested_outputs[str(meta.name)] for meta in self.output_meta}
        try:
            transform_options = FeatureTransformOptions.model_validate(parameters.get("feature_transform") or {})
        except Exception as exc:
            raise ModelRuntimeError("Invalid feature transform options", details={"error": str(exc)}) from exc
        artifacts = []
        for layer_id in capture_layers:
            source = requested_outputs[layer_id]
            if not np.issubdtype(source.dtype, np.number) or source.size == 0:
                raise ModelRuntimeError("Captured ONNX feature must be a non-empty numeric tensor", details={"layer_id": layer_id})
            if source.nbytes > MAX_SOURCE_FEATURE_BYTES:
                raise ModelRuntimeError(
                    "Captured ONNX feature exceeds the source byte budget",
                    details={"layer_id": layer_id, "nbytes": int(source.nbytes), "maximum_bytes": MAX_SOURCE_FEATURE_BYTES},
                )
            transformed = transform_feature(source, transform_options)
            artifacts.append(self.artifact_store.put_tensor(
                model_id=self.record.descriptor.id,
                image_path=image_path,
                layer_id=layer_id,
                tensor=transformed,
                source_shape=list(source.shape),
                transform=transform_options.model_dump(mode="json"),
            ))
        annotations = self._annotations(outputs, transform, parameters)
        classifications = self._classifications(outputs, parameters)
        rasters = self._rasters(outputs, transform, image_path, parameters)
        finished = perf_counter()
        return InferenceResult(
            model_id=self.record.descriptor.id,
            image_path=image_path,
            annotations=annotations,
            classifications=classifications,
            artifacts=artifacts,
            rasters=rasters,
            timings_ms={
                "preprocess": (preprocessed - started) * 1000,
                "inference": (inferred - preprocessed) * 1000,
                "postprocess": (finished - inferred) * 1000,
                "total": (finished - started) * 1000,
            },
        )


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-7)


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
    keep: list[int] = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return keep


class YoloDetectionOnnxAdapter(OnnxRuntimeAdapter):
    def _annotations(self, outputs: dict[str, np.ndarray], transform: _ImageTransform, parameters: dict[str, object]) -> list[AnnotationResult]:
        if not outputs:
            return []
        prediction = np.asarray(next(iter(outputs.values())))
        prediction = np.squeeze(prediction)
        if prediction.ndim != 2:
            raise ModelRuntimeError("Unsupported YOLO output rank", details={"shape": list(prediction.shape)})
        classes = list(self.record.config.get("classes") or [])
        expected_widths = {6, 4 + len(classes), 5 + len(classes)}
        if prediction.shape[0] in expected_widths and prediction.shape[1] not in expected_widths:
            prediction = prediction.T
        confidence = float(parameters.get("conf_threshold", self.record.config.get("conf_threshold", 0.25)))
        iou_threshold = float(parameters.get("iou_threshold", self.record.config.get("iou_threshold", 0.45)))

        end_to_end = prediction.shape[1] == 6 and prediction.shape[1] != 4 + len(classes)
        if end_to_end:
            boxes = prediction[:, :4].astype(np.float32)
            scores = prediction[:, 4].astype(np.float32)
            class_ids = prediction[:, 5].astype(np.int64)
        else:
            class_count = len(classes)
            if class_count <= 0 or prediction.shape[1] < 4 + class_count:
                raise ModelRuntimeError("YOLO classes do not match output shape", details={"shape": list(prediction.shape), "class_count": class_count})
            has_objectness = prediction.shape[1] >= 5 + class_count
            class_start = 5 if has_objectness else 4
            class_scores = prediction[:, class_start : class_start + class_count]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            if has_objectness:
                scores = scores * prediction[:, 4]
            boxes = _xywh_to_xyxy(prediction[:, :4].astype(np.float32))
        mask = scores >= confidence
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if boxes.size == 0:
            return []
        keep = _nms(boxes, scores, class_ids, iou_threshold)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_x) / transform.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_y) / transform.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, transform.original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, transform.original_height)
        results: list[AnnotationResult] = []
        for box, score, class_id in zip(boxes, scores, class_ids, strict=True):
            label = classes[int(class_id)] if 0 <= int(class_id) < len(classes) else str(int(class_id))
            results.append(AnnotationResult(
                label=label,
                score=float(score),
                points=[[float(box[0]), float(box[1])], [float(box[2]), float(box[3])]],
            ))
        return results
