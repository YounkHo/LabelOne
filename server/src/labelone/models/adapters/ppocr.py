from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult, FeatureLayer, InferenceResult
from ..features import FeatureTransformOptions, transform_feature
from .base import ModelAdapter


@dataclass(slots=True)
class _Stage:
    name: str
    session: Any
    input_meta: Any
    output_meta: list[Any]


@dataclass(frozen=True, slots=True)
class _DetectedQuad:
    points: np.ndarray
    score: float


def _shape(value: object) -> list[int | str | None]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item if isinstance(item, (int, str)) else None for item in value]


def _numeric_parameter(
    parameters: dict[str, object],
    config: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = parameters.get(name, config.get(name, default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"PP-OCR {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"PP-OCR {name} is outside the supported range",
            details={"value": raw, "minimum": minimum, "maximum": maximum},
        )
    return value


def _integer_parameter(
    parameters: dict[str, object],
    config: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = parameters.get(name, config.get(name, default))
    if isinstance(raw, bool):
        raise ModelRuntimeError(f"PP-OCR {name} must be an integer", details={"value": raw})
    try:
        value = int(raw)
        exact = float(raw) == value
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"PP-OCR {name} must be an integer", details={"value": raw}) from exc
    if not exact or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"PP-OCR {name} is outside the supported range",
            details={"value": raw, "minimum": minimum, "maximum": maximum},
        )
    return value


def _boolean_parameter(
    parameters: dict[str, object],
    config: dict[str, Any],
    name: str,
    default: bool,
) -> bool:
    raw = parameters.get(name, config.get(name, default))
    if not isinstance(raw, bool):
        raise ModelRuntimeError(f"PP-OCR {name} must be boolean", details={"value": raw})
    return raw


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _probability_tensor(values: np.ndarray, *, role: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ModelRuntimeError(f"PP-OCR {role} output must contain finite numeric values")
    array = array.astype(np.float32, copy=False)
    if np.all((array >= 0) & (array <= 1)):
        return array
    return _sigmoid(array).astype(np.float32, copy=False)


def _order_quad(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(4, 2)
    sums = array.sum(axis=1)
    differences = array[:, 1] - array[:, 0]
    indices = [int(np.argmin(sums)), int(np.argmin(differences)), int(np.argmax(sums)), int(np.argmax(differences))]
    if len(set(indices)) != 4:
        center = array.mean(axis=0)
        angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
        array = array[np.argsort(angles)]
        start = int(np.argmin(array.sum(axis=1)))
        array = np.roll(array, -start, axis=0)
        if array[1, 0] < array[-1, 0]:
            array = array[[0, 3, 2, 1]]
        return array
    return array[indices]


def _component_quad(coordinates: np.ndarray, unclip_ratio: float) -> np.ndarray:
    xy = coordinates[:, [1, 0]].astype(np.float64)
    center = xy.mean(axis=0)
    centered = xy - center
    if len(xy) >= 3:
        covariance = centered.T @ centered / max(1, len(xy) - 1)
        _, vectors = np.linalg.eigh(covariance)
        axes = vectors[:, ::-1]
    else:
        axes = np.eye(2)
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    projected = centered @ axes
    low = projected.min(axis=0) - 0.5
    high = projected.max(axis=0) + 0.5
    width, height = np.maximum(high - low, 1.0)
    expansion = width * height * unclip_ratio / max(2.0 * (width + height), 1e-7)
    low -= expansion
    high += expansion
    local = np.array([
        [low[0], low[1]],
        [high[0], low[1]],
        [high[0], high[1]],
        [low[0], high[1]],
    ])
    return _order_quad(local @ axes.T + center)


def _connected_components(bitmap: np.ndarray, *, maximum: int, minimum_pixels: int) -> list[np.ndarray]:
    height, width = bitmap.shape
    visited = np.zeros(bitmap.size, dtype=bool)
    flat = bitmap.ravel()
    components: list[np.ndarray] = []
    for start_value in np.flatnonzero(flat):
        start = int(start_value)
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        points: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            y, x = divmod(current, width)
            points.append((y, x))
            if x > 0:
                neighbor = current - 1
                if flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            if x + 1 < width:
                neighbor = current + 1
                if flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            if y > 0:
                neighbor = current - width
                if flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            if y + 1 < height:
                neighbor = current + width
                if flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(points) >= minimum_pixels:
            components.append(np.asarray(points, dtype=np.int32))
            if len(components) >= maximum:
                break
    return components


def _db_quads(
    probability: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    maximum_candidates: int,
    maximum_boxes: int,
    minimum_pixels: int,
) -> list[_DetectedQuad]:
    bitmap = probability >= threshold
    components = _connected_components(bitmap, maximum=maximum_candidates, minimum_pixels=minimum_pixels)
    found: list[_DetectedQuad] = []
    height, width = probability.shape
    for component in components:
        score = float(probability[component[:, 0], component[:, 1]].mean())
        if score < box_threshold:
            continue
        quad = _component_quad(component, unclip_ratio)
        quad[:, 0] = np.clip(quad[:, 0] / width * original_width, 0, original_width - 1)
        quad[:, 1] = np.clip(quad[:, 1] / height * original_height, 0, original_height - 1)
        quad = _order_quad(quad)
        horizontal = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
        vertical = min(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        if horizontal <= 3 or vertical <= 3:
            continue
        found.append(_DetectedQuad(quad, score))
    found.sort(key=lambda item: (float(item.points[0, 1]), float(item.points[0, 0])))
    return found[:maximum_boxes]


def _perspective_coefficients(destination: np.ndarray, source: np.ndarray) -> tuple[float, ...]:
    matrix: list[list[float]] = []
    target: list[float] = []
    for (x, y), (u, v) in zip(destination, source, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        target.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        target.append(v)
    try:
        coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(target))
    except np.linalg.LinAlgError as exc:
        raise ModelRuntimeError("PP-OCR detected a degenerate text quadrilateral") from exc
    return tuple(float(value) for value in coefficients)


def _perspective_crop(image: Image.Image, points: np.ndarray) -> Image.Image:
    quad = _order_quad(points)
    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    target_width = max(1, int(round(width)))
    target_height = max(1, int(round(height)))
    destination = np.asarray([
        [0.0, 0.0],
        [target_width, 0.0],
        [target_width, target_height],
        [0.0, target_height],
    ])
    crop = image.transform(
        (target_width, target_height),
        Image.Transform.PERSPECTIVE,
        _perspective_coefficients(destination, quad),
        resample=Image.Resampling.BICUBIC,
    )
    if crop.height / max(1, crop.width) >= 1.5:
        crop = crop.transpose(Image.Transpose.ROTATE_90)
    return crop


def _decode_ctc(probabilities: np.ndarray, characters: list[str], maximum_characters: int) -> list[tuple[str, float]]:
    if probabilities.ndim != 3:
        raise ModelRuntimeError(
            "PP-OCR recognition output must have shape [B,T,C]",
            details={"shape": list(probabilities.shape)},
        )
    if probabilities.shape[2] != len(characters) + 1:
        raise ModelRuntimeError(
            "PP-OCR recognition class count does not match its dictionary",
            details={"classes": probabilities.shape[2], "dictionary_characters": len(characters)},
        )
    if not np.all(np.isfinite(probabilities)):
        raise ModelRuntimeError("PP-OCR recognition output contains non-finite values")
    if not np.all((probabilities >= 0) & (probabilities <= 1)) or not np.allclose(
        probabilities.sum(axis=2), 1.0, atol=1e-3, rtol=1e-3
    ):
        probabilities = _softmax(probabilities, axis=2)
    indices = np.argmax(probabilities, axis=2)
    scores = np.max(probabilities, axis=2)
    decoded: list[tuple[str, float]] = []
    for sequence, sequence_scores in zip(indices, scores, strict=True):
        text: list[str] = []
        confidence: list[float] = []
        previous = -1
        for class_id, score in zip(sequence, sequence_scores, strict=True):
            current = int(class_id)
            if current != 0 and current != previous and len(text) < maximum_characters:
                text.append(characters[current - 1])
                confidence.append(float(score))
            previous = current
        decoded.append(("".join(text), float(np.mean(confidence)) if confidence else 0.0))
    return decoded


class PpOcrOnnxAdapter(ModelAdapter):
    """Clean-room PP-OCR v4/v5/v6 DB + optional angle classifier + CTC adapter."""

    def __init__(self, record, artifact_store) -> None:
        super().__init__(record, artifact_store)
        self.detector: _Stage | None = None
        self.classifier: _Stage | None = None
        self.recognizer: _Stage | None = None
        self.characters: list[str] = []

    def _local_path(self, key: str, *, required: bool) -> Path | None:
        raw = self.record.config.get(key)
        if raw is None:
            if required:
                raise ModelRuntimeError(f"PP-OCR configuration requires {key}")
            return None
        if not isinstance(raw, str):
            raise ModelRuntimeError(f"PP-OCR {key} must be a path string")
        if urlparse(raw).scheme in {"http", "https"}:
            if required:
                raise ModelRuntimeError(f"PP-OCR {key} has no downloaded local file", details={"location": raw})
            return None
        path = Path(raw).expanduser()
        resolved = (path if path.is_absolute() else self.record.descriptor.config_path.parent / path).resolve()
        if not resolved.is_file():
            if required:
                raise ModelRuntimeError(f"PP-OCR {key} file is missing", details={"path": str(resolved)})
            return None
        return resolved

    def _dictionary_path(self) -> Path:
        configured = self.record.config.get("rec_char_dict_path")
        default_names = {
            "ch": "ppocr_keys_v1.txt",
            "japan": "japan_dict.txt",
            "ppocrv5_dict": "ppocrv5_dict.txt",
        }
        filename = str(configured) if configured else default_names.get(str(self.record.config.get("lang", "ch")))
        if not filename:
            raise ModelRuntimeError("PP-OCR could not determine its recognition dictionary")
        raw = Path(filename).expanduser()
        config_path = self.record.descriptor.config_path.resolve()
        source_root = next(
            (
                parent
                for parent in (config_path.parent, *config_path.parents)
                if (parent / "services" / "auto_labeling" / "configs" / "ppocr").is_dir()
            ),
            None,
        )
        allowed_roots = [config_path.parent]
        if source_root is not None:
            allowed_roots.append(source_root.resolve())
        candidates = [
            raw if raw.is_absolute() else config_path.parent / raw,
            config_path.parent / "configs" / "ppocr" / raw.name,
            config_path.parent.parent.parent / "services" / "auto_labeling" / "configs" / "ppocr" / raw.name,
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
                continue
            if resolved.is_file():
                return resolved
        raise ModelRuntimeError(
            "PP-OCR recognition dictionary is missing",
            details={"dictionary": filename, "searched": [str(path.resolve()) for path in candidates]},
        )

    def _load_dictionary(self) -> list[str]:
        path = self._dictionary_path()
        try:
            characters = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError) as exc:
            raise ModelRuntimeError("Could not read PP-OCR recognition dictionary", details={"path": str(path)}) from exc
        characters = [character for character in characters if character]
        if _boolean_parameter({}, self.record.config, "use_space_char", True) and " " not in characters:
            characters.append(" ")
        if not characters or len(characters) > 100_000:
            raise ModelRuntimeError(
                "PP-OCR recognition dictionary size is invalid",
                details={"path": str(path), "characters": len(characters)},
            )
        return characters

    @staticmethod
    def _create_stage(ort: Any, name: str, path: Path, providers: list[str]) -> _Stage:
        try:
            session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise ModelRuntimeError(
                f"Failed to load PP-OCR {name} ONNX model",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if len(inputs) != 1 or not outputs:
            raise ModelRuntimeError(
                f"PP-OCR {name} model must have one input and at least one output",
                details={"inputs": len(inputs), "outputs": len(outputs)},
            )
        shape = list(inputs[0].shape)
        if len(shape) != 4 or shape[1] not in {3, "C", "channels", None}:
            raise ModelRuntimeError(
                f"PP-OCR {name} input must be NCHW RGB/BGR",
                details={"shape": shape},
            )
        return _Stage(name, session, inputs[0], outputs)

    def load(self, providers: list[str]) -> list[FeatureLayer]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelRuntimeError("onnxruntime is not installed") from exc
        available = set(ort.get_available_providers())
        selected = [provider for provider in providers if provider in available] or ["CPUExecutionProvider"]
        use_classifier = _boolean_parameter({}, self.record.config, "use_angle_cls", True)
        try:
            self.characters = self._load_dictionary()
            detector_path = self._local_path("det_model_path", required=True)
            recognizer_path = self._local_path("rec_model_path", required=True)
            assert detector_path is not None and recognizer_path is not None
            self.detector = self._create_stage(ort, "det", detector_path, selected)
            self.recognizer = self._create_stage(ort, "rec", recognizer_path, selected)
            classifier_path = self._local_path("cls_model_path", required=use_classifier)
            self.classifier = self._create_stage(ort, "cls", classifier_path, selected) if classifier_path else None
            self.loaded = True
            return self.list_layers()
        except Exception:
            self.unload()
            raise

    def unload(self) -> None:
        self.detector = None
        self.classifier = None
        self.recognizer = None
        self.characters = []
        self.loaded = False

    def list_layers(self) -> list[FeatureLayer]:
        layers: list[FeatureLayer] = []
        for stage in (self.detector, self.classifier, self.recognizer):
            if stage is None:
                continue
            for output in stage.output_meta:
                output_shape = _shape(output.shape)
                layers.append(FeatureLayer(
                    id=f"{stage.name}:{output.name}",
                    group=stage.name.upper(),
                    name=output.name,
                    shape=output_shape,
                    axes=[f"D{index}" for index in range(len(output_shape))],
                    dtype=str(output.type),
                    spatial=len(output_shape) == 4,
                ))
        return layers

    @staticmethod
    def _stage_dimensions(stage: _Stage, *, default_height: int, default_width: int) -> tuple[int, int]:
        shape = list(stage.input_meta.shape)
        height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else default_height
        width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else default_width
        return height, width

    @staticmethod
    def _run(stage: _Stage, tensor: np.ndarray) -> dict[str, np.ndarray]:
        if "float16" in str(stage.input_meta.type):
            tensor = tensor.astype(np.float16)
        try:
            values = stage.session.run(None, {stage.input_meta.name: tensor})
        except Exception as exc:
            raise ModelRuntimeError(f"PP-OCR {stage.name} inference failed", details={"error": str(exc)}) from exc
        if len(values) != len(stage.output_meta):
            raise ModelRuntimeError(f"PP-OCR {stage.name} returned an unexpected output count")
        return {meta.name: np.asarray(value) for meta, value in zip(stage.output_meta, values, strict=True)}

    def _prepare_detection(self, image: Image.Image, parameters: dict[str, object]) -> np.ndarray:
        assert self.detector is not None
        shape = list(self.detector.input_meta.shape)
        original_width, original_height = image.size
        if isinstance(shape[2], int) and shape[2] > 0 and isinstance(shape[3], int) and shape[3] > 0:
            height, width = shape[2], shape[3]
        else:
            limit = _integer_parameter(parameters, self.record.config, "det_limit_side_len", 960, minimum=32, maximum=8192)
            ratio = min(1.0, limit / max(original_width, original_height))
            height = max(32, round(round(original_height * ratio) / 32) * 32)
            width = max(32, round(round(original_width * ratio) / 32) * 32)
        maximum = _integer_parameter(
            parameters,
            self.record.config,
            "max_detection_input_pixels",
            16_777_216,
            minimum=1024,
            maximum=268_435_456,
        )
        if height * width > maximum:
            raise ModelRuntimeError(
                "PP-OCR detection input exceeds the pixel budget",
                details={"size": [width, height], "pixels": width * height, "maximum": maximum},
            )
        resized = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        bgr = resized[:, :, ::-1] / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return np.transpose((bgr - mean) / std, (2, 0, 1))[None, ...]

    @staticmethod
    def _single_map(outputs: dict[str, np.ndarray], configured: object) -> tuple[str, np.ndarray]:
        items = outputs.items()
        if configured is not None:
            if not isinstance(configured, str) or configured not in outputs:
                raise ModelRuntimeError(
                    "Configured PP-OCR detection output was not returned",
                    details={"output_name": configured, "available": sorted(outputs)},
                )
            items = [(configured, outputs[configured])]
        candidates: list[tuple[str, np.ndarray]] = []
        for name, value in items:
            array = np.asarray(value)
            if array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == 1:
                candidates.append((name, array[0, 0]))
            elif array.ndim == 3 and array.shape[0] == 1:
                candidates.append((name, array[0]))
            elif array.ndim == 2:
                candidates.append((name, array))
        if not candidates:
            raise ModelRuntimeError(
                "No supported PP-OCR DB detection map was found",
                details={"outputs": {name: list(value.shape) for name, value in outputs.items()}},
            )
        if len(candidates) > 1:
            raise ModelRuntimeError(
                "Multiple PP-OCR detection maps are ambiguous",
                details={"candidates": [name for name, _ in candidates]},
            )
        name, probability = candidates[0]
        return name, _probability_tensor(probability, role="detection")

    @staticmethod
    def _prepare_crop(crop: Image.Image, height: int, width: int) -> np.ndarray:
        ratio = crop.width / max(1, crop.height)
        resized_width = min(width, max(1, int(np.ceil(height * ratio))))
        resized = np.asarray(crop.resize((resized_width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        bgr = resized[:, :, ::-1]
        normalized = (np.transpose(bgr, (2, 0, 1)) / 255.0 - 0.5) / 0.5
        tensor = np.zeros((3, height, width), dtype=np.float32)
        tensor[:, :, :resized_width] = normalized
        return tensor

    def _run_crop_batches(
        self,
        stage: _Stage,
        crops: list[Image.Image],
        *,
        batch_size: int,
        default_height: int,
        default_width: int,
    ) -> dict[str, np.ndarray]:
        height, width = self._stage_dimensions(stage, default_height=default_height, default_width=default_width)
        static_batch = stage.input_meta.shape[0] if isinstance(stage.input_meta.shape[0], int) else None
        actual_limit = min(batch_size, static_batch) if static_batch and static_batch > 0 else batch_size
        collected: dict[str, list[np.ndarray]] = {meta.name: [] for meta in stage.output_meta}
        for start in range(0, len(crops), actual_limit):
            selected = crops[start : start + actual_limit]
            tensor = np.stack([self._prepare_crop(crop, height, width) for crop in selected])
            padded = len(selected)
            if static_batch and len(selected) < static_batch:
                tensor = np.concatenate([tensor, np.repeat(tensor[-1:], static_batch - len(selected), axis=0)])
            outputs = self._run(stage, tensor)
            for name, value in outputs.items():
                if value.ndim == 0 or value.shape[0] < padded:
                    raise ModelRuntimeError(
                        f"PP-OCR {stage.name} output batch does not match its input",
                        details={"output": name, "shape": list(value.shape), "batch": padded},
                    )
                collected[name].append(value[:padded])
        return {name: np.concatenate(values, axis=0) for name, values in collected.items()}

    @staticmethod
    def _select_stage_output(
        outputs: dict[str, np.ndarray],
        configured: object,
        *,
        stage: str,
        rank: int,
    ) -> tuple[str, np.ndarray]:
        if configured is not None:
            if not isinstance(configured, str) or configured not in outputs:
                raise ModelRuntimeError(
                    f"Configured PP-OCR {stage} output was not returned",
                    details={"output_name": configured, "available": sorted(outputs)},
                )
            value = outputs[configured]
            if value.ndim != rank:
                raise ModelRuntimeError(f"PP-OCR {stage} output rank is unsupported", details={"shape": list(value.shape)})
            return configured, value
        candidates = [(name, value) for name, value in outputs.items() if value.ndim == rank]
        if len(candidates) != 1:
            raise ModelRuntimeError(
                f"PP-OCR {stage} output is ambiguous",
                details={"candidates": [name for name, _ in candidates], "expected_rank": rank},
            )
        return candidates[0]

    def predict(
        self,
        image_path: Path,
        capture_layers: list[str],
        parameters: dict[str, object],
    ) -> InferenceResult:
        if not self.loaded or self.detector is None or self.recognizer is None:
            raise ModelRuntimeError("PP-OCR model is not loaded", details={"model_id": self.record.descriptor.id})
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            raise ModelRuntimeError("Inference image does not exist", details={"image_path": str(image_path)})
        available_layers = {layer.id for layer in self.list_layers()}
        unknown = sorted(set(capture_layers) - available_layers)
        if unknown:
            raise ModelRuntimeError("Requested PP-OCR capture layer does not exist", details={"unknown_layers": unknown})
        skip_classifier = _boolean_parameter(parameters, self.record.config, "skip_angle_cls", False)
        if skip_classifier and any(layer.startswith("cls:") for layer in capture_layers):
            raise ModelRuntimeError("Cannot capture PP-OCR classifier outputs while skip_angle_cls is enabled")

        started = perf_counter()
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise ModelRuntimeError("Could not read PP-OCR source image", details={"error": str(exc)}) from exc
        maximum_image_pixels = _integer_parameter(
            parameters,
            self.record.config,
            "max_image_pixels",
            268_435_456,
            minimum=1,
            maximum=1_073_741_824,
        )
        if image.width * image.height > maximum_image_pixels:
            raise ModelRuntimeError("PP-OCR source image exceeds the pixel budget")
        det_tensor = self._prepare_detection(image, parameters)
        preprocessed = perf_counter()
        det_outputs = self._run(self.detector, det_tensor)
        det_name, det_map = self._single_map(
            det_outputs,
            parameters.get("det_output_name", self.record.config.get("det_output_name")),
        )
        if det_map.size > _integer_parameter(
            parameters,
            self.record.config,
            "max_detection_map_values",
            16_777_216,
            minimum=1,
            maximum=268_435_456,
        ):
            raise ModelRuntimeError("PP-OCR detection map exceeds the value budget")
        quads = _db_quads(
            det_map,
            original_width=image.width,
            original_height=image.height,
            threshold=_numeric_parameter(parameters, self.record.config, "det_db_thresh", 0.3, minimum=0, maximum=1),
            box_threshold=_numeric_parameter(parameters, self.record.config, "det_db_box_thresh", 0.6, minimum=0, maximum=1),
            unclip_ratio=_numeric_parameter(parameters, self.record.config, "det_db_unclip_ratio", 1.5, minimum=0, maximum=10),
            maximum_candidates=_integer_parameter(
                parameters,
                self.record.config,
                "det_db_max_candidates",
                1000,
                minimum=1,
                maximum=100_000,
            ),
            maximum_boxes=_integer_parameter(parameters, self.record.config, "max_boxes", 300, minimum=1, maximum=10_000),
            minimum_pixels=_integer_parameter(parameters, self.record.config, "det_min_component_pixels", 3, minimum=1, maximum=1_000_000),
        )
        maximum_crop_pixels = _integer_parameter(
            parameters,
            self.record.config,
            "max_crop_pixels",
            32_000_000,
            minimum=1,
            maximum=536_870_912,
        )
        crops: list[Image.Image] = []
        crop_pixels = 0
        for detection in quads:
            crop = _perspective_crop(image, detection.points)
            crop_pixels += crop.width * crop.height
            if crop_pixels > maximum_crop_pixels:
                raise ModelRuntimeError(
                    "PP-OCR text crops exceed the pixel budget",
                    details={"pixels": crop_pixels, "maximum": maximum_crop_pixels},
                )
            crops.append(crop)

        captured: dict[str, np.ndarray] = {f"det:{name}": value for name, value in det_outputs.items()}
        if crops and self.classifier is not None and not skip_classifier:
            cls_outputs = self._run_crop_batches(
                self.classifier,
                crops,
                batch_size=_integer_parameter(parameters, self.record.config, "cls_batch_size", 6, minimum=1, maximum=128),
                default_height=48,
                default_width=192,
            )
            captured.update({f"cls:{name}": value for name, value in cls_outputs.items()})
            _, cls_scores = self._select_stage_output(
                cls_outputs,
                parameters.get("cls_output_name", self.record.config.get("cls_output_name")),
                stage="classifier",
                rank=2,
            )
            if not np.all(np.isfinite(cls_scores)) or cls_scores.shape[1] < 2:
                raise ModelRuntimeError("PP-OCR classifier output must have shape [B,2+] with finite values")
            if not np.all((cls_scores >= 0) & (cls_scores <= 1)) or not np.allclose(
                cls_scores.sum(axis=1), 1.0, atol=1e-3, rtol=1e-3
            ):
                cls_scores = _softmax(cls_scores, axis=1)
            cls_threshold = _numeric_parameter(parameters, self.record.config, "cls_threshold", 0.9, minimum=0, maximum=1)
            for index, scores in enumerate(cls_scores):
                class_id = int(np.argmax(scores))
                if class_id == 1 and float(scores[class_id]) >= cls_threshold:
                    crops[index] = crops[index].transpose(Image.Transpose.ROTATE_180)

        rec_outputs: dict[str, np.ndarray] = {}
        decoded: list[tuple[str, float]] = []
        if crops:
            rec_outputs = self._run_crop_batches(
                self.recognizer,
                crops,
                batch_size=_integer_parameter(parameters, self.record.config, "rec_batch_size", 6, minimum=1, maximum=128),
                default_height=48,
                default_width=320,
            )
            captured.update({f"rec:{name}": value for name, value in rec_outputs.items()})
            _, rec_scores = self._select_stage_output(
                rec_outputs,
                parameters.get("rec_output_name", self.record.config.get("rec_output_name")),
                stage="recognition",
                rank=3,
            )
            decoded = _decode_ctc(
                rec_scores,
                self.characters,
                _integer_parameter(parameters, self.record.config, "max_characters", 256, minimum=1, maximum=4096),
            )
        inferred = perf_counter()
        drop_score = _numeric_parameter(parameters, self.record.config, "drop_score", 0.5, minimum=0, maximum=1)
        maximum_total_characters = _integer_parameter(
            parameters,
            self.record.config,
            "max_total_characters",
            4096,
            minimum=1,
            maximum=1_000_000,
        )
        total_characters = 0
        annotations: list[AnnotationResult] = []
        for detection, (text, score) in zip(quads, decoded, strict=True):
            if not text or score < drop_score:
                continue
            total_characters += len(text)
            if total_characters > maximum_total_characters:
                raise ModelRuntimeError("PP-OCR decoded text exceeds the character budget")
            annotations.append(AnnotationResult(
                label=text,
                score=score,
                shape_type="polygon",
                points=[[float(x), float(y)] for x, y in detection.points],
            ))

        artifacts = []
        try:
            feature_options = FeatureTransformOptions.model_validate(parameters.get("feature_transform") or {})
        except Exception as exc:
            raise ModelRuntimeError("Invalid PP-OCR feature transform options", details={"error": str(exc)}) from exc
        for layer_id in capture_layers:
            tensor = captured.get(layer_id)
            if tensor is None:
                raise ModelRuntimeError(
                    "Requested PP-OCR stage output was not executed",
                    details={"layer_id": layer_id, "detected_boxes": len(quads)},
                )
            transformed = transform_feature(tensor, feature_options)
            artifacts.append(self.artifact_store.put_tensor(
                model_id=self.record.descriptor.id,
                image_path=image_path,
                layer_id=layer_id,
                tensor=transformed,
                source_shape=list(tensor.shape),
                transform={"stage": layer_id.split(":", 1)[0], "explicit_output": True, **feature_options.model_dump(mode="json")},
            ))
        finished = perf_counter()
        return InferenceResult(
            model_id=self.record.descriptor.id,
            image_path=image_path,
            annotations=annotations,
            artifacts=artifacts,
            timings_ms={
                "preprocess": (preprocessed - started) * 1000,
                "inference": (inferred - preprocessed) * 1000,
                "postprocess": (finished - inferred) * 1000,
                "total": (finished - started) * 1000,
            },
        )
