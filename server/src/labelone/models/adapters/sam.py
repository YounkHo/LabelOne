from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult, FeatureLayer, InferenceResult
from ..features import FeatureTransformOptions, transform_feature
from .base import ModelAdapter
from .yolo_segmentation import _polygons_from_mask


_SUPPORTED_MODEL_TYPES = {"segment_anything", "sam_hq"}
_INPUT_ALIASES: dict[str, set[str]] = {
    "embedding": {"image_embeddings", "image_embedding", "embeddings"},
    "intermediate": {"interm_embeddings", "intermediate_embeddings", "intermediate_features"},
    "coords": {"point_coords", "point_coordinates"},
    "labels": {"point_labels"},
    "mask": {"mask_input", "mask_inputs"},
    "has_mask": {"has_mask_input", "has_mask"},
    "original_size": {"orig_im_size", "original_image_size", "original_size"},
}
_MASK_OUTPUT_ALIASES = {"masks", "mask", "high_res_masks", "output_masks"}
_IOU_OUTPUT_ALIASES = {"iou_predictions", "iou_scores", "scores", "mask_scores"}


@dataclass(frozen=True, slots=True)
class _SamTransform:
    original_width: int
    original_height: int
    target_size: int
    resized_width: int
    resized_height: int
    scale: float


def _canonical(name: str) -> str:
    return name.rsplit("/", 1)[-1].split(":", 1)[0].casefold()


def _shape(value: list[Any] | tuple[Any, ...]) -> list[int | str | None]:
    return [item if isinstance(item, (int, str)) else None for item in value]


def _axes(shape: list[int | str | None]) -> list[str]:
    if len(shape) == 4:
        return ["N", "C", "H", "W"]
    if len(shape) == 3:
        return ["N", "T", "C"]
    return [f"D{index}" for index in range(len(shape))]


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
        raise ModelRuntimeError(f"SAM {name} must be an integer", details={"value": raw})
    try:
        value = int(raw)
        exact = float(raw) == value
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"SAM {name} must be an integer", details={"value": raw}) from exc
    if not exact or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"SAM {name} is outside the supported range",
            details={"value": raw, "minimum": minimum, "maximum": maximum},
        )
    return value


def _number_parameter(
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
        raise ModelRuntimeError(f"SAM {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"SAM {name} is outside the supported range",
            details={"value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def _normalization_vector(config: dict[str, Any], name: str, default: list[float]) -> np.ndarray:
    raw = config.get(name, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ModelRuntimeError(f"SAM pixel {name} must contain three values", details={"value": raw})
    try:
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"SAM pixel {name} must be numeric", details={"value": raw}) from exc
    if not np.all(np.isfinite(values)) or (name == "std" and np.any(values <= 0)):
        raise ModelRuntimeError(f"SAM pixel {name} is invalid", details={"value": list(values)})
    return values


def _local_model_path(record, key: str) -> Path:
    raw = record.config.get(key)
    if not isinstance(raw, str) or not raw:
        raise ModelRuntimeError(f"SAM config requires {key}")
    if raw.startswith(("http://", "https://")):
        raise ModelRuntimeError(f"SAM {key} is not available locally", details={"location": raw})
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else record.descriptor.config_path.parent / path).resolve()
    if not resolved.is_file() or resolved.suffix.casefold() != ".onnx":
        raise ModelRuntimeError(f"SAM {key} does not resolve to a local ONNX file", details={"path": str(resolved)})
    return resolved


def _single_named_output(metadata: list[Any], aliases: set[str], configured: object, role: str) -> str | None:
    names = [str(item.name) for item in metadata]
    if configured is not None:
        if not isinstance(configured, str) or configured not in names:
            raise ModelRuntimeError(
                f"Configured SAM {role} output was not exported",
                details={"output_name": configured, "available": names},
            )
        return configured
    matches = [name for name in names if _canonical(name) in aliases]
    if len(matches) > 1:
        raise ModelRuntimeError(f"Multiple SAM outputs match {role}", details={"matches": matches})
    return matches[0] if matches else None


def _resolve_decoder_inputs(metadata: list[Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    unknown: list[str] = []
    for item in metadata:
        canonical = _canonical(str(item.name))
        roles = [role for role, aliases in _INPUT_ALIASES.items() if canonical in aliases]
        if len(roles) != 1:
            unknown.append(str(item.name))
            continue
        role = roles[0]
        if role in resolved:
            raise ModelRuntimeError("Multiple SAM decoder inputs have the same role", details={"role": role})
        resolved[role] = item
    required = {"embedding", "coords", "labels", "mask", "has_mask", "original_size"}
    missing = sorted(required - set(resolved))
    if missing or unknown:
        raise ModelRuntimeError(
            "SAM decoder input signature is unsupported",
            details={"missing_roles": missing, "unknown_inputs": unknown},
        )
    return resolved


def _prepare_prompts(
    parameters: dict[str, object],
    transform: _SamTransform,
    *,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    raw_points = parameters.get("points", [])
    raw_boxes = parameters.get("boxes", [])
    if not isinstance(raw_points, list) or not isinstance(raw_boxes, list):
        raise ModelRuntimeError("SAM points and boxes must be lists")
    coordinates: list[list[float]] = []
    labels: list[float] = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, dict):
            raise ModelRuntimeError("SAM point must be an object", details={"index": index})
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRuntimeError("SAM point requires numeric x and y", details={"index": index}) from exc
        label = point.get("label")
        if isinstance(label, bool) or label not in {0, 1}:
            raise ModelRuntimeError("SAM point label must be 0 or 1", details={"index": index, "label": label})
        if not np.isfinite(x) or not np.isfinite(y) or not 0 <= x <= transform.original_width or not 0 <= y <= transform.original_height:
            raise ModelRuntimeError("SAM point is outside the image", details={"index": index, "point": [x, y]})
        coordinates.append(
            [
                x * transform.resized_width / transform.original_width,
                y * transform.resized_height / transform.original_height,
            ]
        )
        labels.append(float(label))
    for index, box in enumerate(raw_boxes):
        if not isinstance(box, list) or len(box) != 4:
            raise ModelRuntimeError("SAM box must contain [x1,y1,x2,y2]", details={"index": index})
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError("SAM box coordinates must be numeric", details={"index": index}) from exc
        values = np.asarray([x1, y1, x2, y2])
        if not np.all(np.isfinite(values)) or not (0 <= x1 < x2 <= transform.original_width and 0 <= y1 < y2 <= transform.original_height):
            raise ModelRuntimeError("SAM box is invalid or outside the image", details={"index": index, "box": list(values)})
        x_scale = transform.resized_width / transform.original_width
        y_scale = transform.resized_height / transform.original_height
        coordinates.extend(([x1 * x_scale, y1 * y_scale], [x2 * x_scale, y2 * y_scale]))
        labels.extend((2.0, 3.0))
    prompt_count = len(coordinates)
    if prompt_count == 0:
        raise ModelRuntimeError("SAM prediction requires at least one point or box prompt")
    if prompt_count > maximum:
        raise ModelRuntimeError("SAM prompt count exceeds the budget", details={"count": prompt_count, "maximum": maximum})
    coordinates.append([0.0, 0.0])
    labels.append(-1.0)
    return (
        np.asarray(coordinates, dtype=np.float32)[None],
        np.asarray(labels, dtype=np.float32)[None],
        {"points": len(raw_points), "boxes": len(raw_boxes)},
    )


def _mask_input_shape(metadata: Any) -> tuple[int, int]:
    shape = list(metadata.shape)
    if len(shape) != 4:
        raise ModelRuntimeError("SAM decoder mask_input must have four dimensions", details={"shape": shape})
    height = shape[2] if isinstance(shape[2], int) else 256
    width = shape[3] if isinstance(shape[3], int) else 256
    if height <= 0 or width <= 0:
        raise ModelRuntimeError("SAM decoder mask_input has invalid dimensions", details={"shape": shape})
    return height, width


def _decoder_masks(value: np.ndarray, output_name: str) -> np.ndarray:
    masks = np.asarray(value)
    original_shape = list(masks.shape)
    if masks.ndim == 4:
        if masks.shape[0] != 1:
            raise ModelRuntimeError("SAM masks output must have one batch", details={"shape": original_shape})
        masks = masks[0]
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or not all(int(item) > 0 for item in masks.shape):
        raise ModelRuntimeError(
            "SAM masks output must have shape [1,K,H,W], [K,H,W], or [H,W]",
            details={"output_name": output_name, "shape": original_shape},
        )
    if not np.issubdtype(masks.dtype, np.number) or not np.all(np.isfinite(masks)):
        raise ModelRuntimeError("SAM masks output must contain finite numeric values", details={"output_name": output_name})
    return masks.astype(np.float32, copy=False)


def _iou_scores(value: np.ndarray, count: int, output_name: str) -> np.ndarray:
    scores = np.asarray(value).reshape(-1)
    if scores.size != count or not np.issubdtype(scores.dtype, np.number) or not np.all(np.isfinite(scores)):
        raise ModelRuntimeError(
            "SAM IoU output must contain one finite score per mask",
            details={"output_name": output_name, "shape": list(np.asarray(value).shape), "mask_count": count},
        )
    return scores.astype(np.float32, copy=False)


def _restore_mask(mask: np.ndarray, transform: _SamTransform) -> np.ndarray:
    if mask.shape == (transform.original_height, transform.original_width):
        return mask.astype(np.float32, copy=False)
    image = Image.fromarray(mask.astype(np.float32, copy=False))
    if image.size != (transform.target_size, transform.target_size):
        image = image.resize((transform.target_size, transform.target_size), Image.Resampling.BILINEAR)
    image = image.crop((0, 0, transform.resized_width, transform.resized_height))
    if image.size != (transform.original_width, transform.original_height):
        image = image.resize((transform.original_width, transform.original_height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


class SegmentAnythingOnnxAdapter(ModelAdapter):
    """Clean-room standard SAM/SAM-HQ encoder-decoder ONNX adapter."""

    def __init__(self, record, artifact_store) -> None:
        super().__init__(record, artifact_store)
        self.encoder_session = None
        self.decoder_session = None
        self.encoder_input_meta = None
        self.encoder_output_meta: list[Any] = []
        self.decoder_input_meta: list[Any] = []
        self.decoder_output_meta: list[Any] = []

    def load(self, providers: list[str]) -> list[FeatureLayer]:
        model_type = str(self.record.descriptor.model_type).casefold()
        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ModelRuntimeError(
                "SAM adapter supports only segment_anything and sam_hq",
                details={"model_type": model_type},
            )
        encoder_path = _local_model_path(self.record, "encoder_model_path")
        decoder_path = _local_model_path(self.record, "decoder_model_path")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelRuntimeError("onnxruntime is not installed") from exc
        available = set(ort.get_available_providers())
        selected = [provider for provider in providers if provider in available] or ["CPUExecutionProvider"]
        try:
            encoder = ort.InferenceSession(str(encoder_path), providers=selected)
            decoder = ort.InferenceSession(str(decoder_path), providers=selected)
            encoder_inputs = list(encoder.get_inputs())
            if len(encoder_inputs) != 1:
                raise ModelRuntimeError("SAM encoder requires exactly one image input", details={"input_count": len(encoder_inputs)})
            decoder_inputs = list(decoder.get_inputs())
            _resolve_decoder_inputs(decoder_inputs)
            self.encoder_session = encoder
            self.decoder_session = decoder
            self.encoder_input_meta = encoder_inputs[0]
            self.encoder_output_meta = list(encoder.get_outputs())
            self.decoder_input_meta = decoder_inputs
            self.decoder_output_meta = list(decoder.get_outputs())
            if not self.encoder_output_meta or not self.decoder_output_meta:
                raise ModelRuntimeError("SAM encoder and decoder must expose outputs")
        except ModelRuntimeError:
            self.unload()
            raise
        except Exception as exc:
            self.unload()
            raise ModelRuntimeError(
                "Failed to load SAM ONNX sessions",
                details={"encoder": str(encoder_path), "decoder": str(decoder_path), "error": str(exc)},
            ) from exc
        self.loaded = True
        return self.list_layers()

    def unload(self) -> None:
        self.encoder_session = None
        self.decoder_session = None
        self.encoder_input_meta = None
        self.encoder_output_meta = []
        self.decoder_input_meta = []
        self.decoder_output_meta = []
        self.loaded = False

    def list_layers(self) -> list[FeatureLayer]:
        layers: list[FeatureLayer] = []
        for stage, metadata in (("encoder", self.encoder_output_meta), ("decoder", self.decoder_output_meta)):
            for output in metadata:
                shape = _shape(list(output.shape))
                layers.append(
                    FeatureLayer(
                        id=f"{stage}:{output.name}",
                        group=f"SAM {stage.title()}",
                        name=str(output.name),
                        shape=shape,
                        axes=_axes(shape),
                        dtype=str(output.type),
                        spatial=len(shape) == 4,
                    )
                )
        return layers

    def _target_size(self) -> int:
        if self.encoder_input_meta is None:
            raise ModelRuntimeError("SAM encoder is not loaded")
        shape = list(self.encoder_input_meta.shape)
        if len(shape) != 4 or shape[1] not in {3, "C", "channels"}:
            raise ModelRuntimeError("SAM encoder input must be NCHW RGB", details={"shape": shape})
        height = shape[2] if isinstance(shape[2], int) else int(self.record.config.get("input_size", 1024))
        width = shape[3] if isinstance(shape[3], int) else int(self.record.config.get("input_size", 1024))
        if height <= 0 or width <= 0 or height != width:
            raise ModelRuntimeError("SAM encoder input must be a positive square", details={"shape": shape})
        return height

    def _preprocess_image(self, image_path: Path, *, max_input_pixels: int) -> tuple[np.ndarray, _SamTransform]:
        target = self._target_size()
        if target * target > max_input_pixels:
            raise ModelRuntimeError(
                "SAM encoder input exceeds the pixel budget",
                details={"input_size": [target, target], "pixels": target * target, "maximum": max_input_pixels},
            )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            original_width, original_height = image.size
            scale = target / max(original_width, original_height)
            resized_width = max(1, int(original_width * scale + 0.5))
            resized_height = max(1, int(original_height * scale + 0.5))
            resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32)
        mean = _normalization_vector(self.record.config, "mean", [123.675, 116.28, 103.53])
        std = _normalization_vector(self.record.config, "std", [58.395, 57.12, 57.375])
        normalized = (pixels - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        canvas = np.zeros((target, target, 3), dtype=np.float32)
        canvas[:resized_height, :resized_width] = normalized
        tensor = np.ascontiguousarray(np.transpose(canvas, (2, 0, 1))[None])
        return tensor, _SamTransform(
            original_width=original_width,
            original_height=original_height,
            target_size=target,
            resized_width=resized_width,
            resized_height=resized_height,
            scale=scale,
        )

    def _encoder_embedding_name(self) -> str:
        configured = self.record.config.get("encoder_embedding_output")
        matched = _single_named_output(
            self.encoder_output_meta,
            {"image_embeddings", "image_embedding", "embeddings"},
            configured,
            "encoder embedding",
        )
        if matched is not None:
            return matched
        if not self.encoder_output_meta:
            raise ModelRuntimeError("SAM encoder exposes no outputs")
        return str(self.encoder_output_meta[0].name)

    def _decoder_outputs(self, outputs: dict[str, np.ndarray]) -> tuple[str, np.ndarray, str | None, np.ndarray]:
        mask_name = _single_named_output(
            self.decoder_output_meta,
            _MASK_OUTPUT_ALIASES,
            self.record.config.get("decoder_mask_output"),
            "decoder mask",
        )
        if mask_name is None:
            rank_candidates = [name for name, value in outputs.items() if np.asarray(value).ndim in {2, 3, 4}]
            if len(rank_candidates) != 1:
                raise ModelRuntimeError(
                    "Could not identify one SAM decoder mask output",
                    details={"candidates": rank_candidates, "available": sorted(outputs)},
                )
            mask_name = rank_candidates[0]
        masks = _decoder_masks(outputs[mask_name], mask_name)
        iou_name = _single_named_output(
            self.decoder_output_meta,
            _IOU_OUTPUT_ALIASES,
            self.record.config.get("decoder_iou_output"),
            "decoder IoU",
        )
        if iou_name is None:
            score_candidates = [
                name for name, value in outputs.items()
                if name != mask_name and np.asarray(value).size == len(masks) and np.asarray(value).ndim <= 2
            ]
            if len(score_candidates) > 1:
                raise ModelRuntimeError("Multiple SAM decoder outputs could be IoU scores", details={"candidates": score_candidates})
            iou_name = score_candidates[0] if score_candidates else None
        if iou_name is None:
            if len(masks) != 1:
                raise ModelRuntimeError("SAM decoder returned multiple masks without IoU scores")
            scores = np.ones(1, dtype=np.float32)
        else:
            scores = _iou_scores(outputs[iou_name], len(masks), iou_name)
        return mask_name, masks, iou_name, scores

    def predict(
        self,
        image_path: Path,
        capture_layers: list[str],
        parameters: dict[str, object],
    ) -> InferenceResult:
        if not self.loaded or self.encoder_session is None or self.decoder_session is None or self.encoder_input_meta is None:
            raise ModelRuntimeError("SAM model is not loaded", details={"model_id": self.record.descriptor.id})
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            raise ModelRuntimeError("SAM image does not exist", details={"image_path": str(image_path)})
        available_layers = {layer.id for layer in self.list_layers()}
        unknown = sorted(set(capture_layers) - available_layers)
        if unknown:
            raise ModelRuntimeError("Requested SAM capture output is unavailable", details={"unknown_layers": unknown})
        max_input_pixels = _integer_parameter(
            parameters, self.record.config, "max_input_pixels", 4_194_304, minimum=1, maximum=268_435_456
        )
        max_output_pixels = _integer_parameter(
            parameters, self.record.config, "max_output_pixels", 64_000_000, minimum=1, maximum=536_870_912
        )
        max_capture_values = _integer_parameter(
            parameters, self.record.config, "max_capture_values", 64_000_000, minimum=1, maximum=1_000_000_000
        )
        maximum_prompts = _integer_parameter(
            parameters, self.record.config, "max_prompt_elements", 256, minimum=1, maximum=100_000
        )
        started = perf_counter()
        tensor, transform = self._preprocess_image(image_path, max_input_pixels=max_input_pixels)
        point_coords, point_labels, prompt_summary = _prepare_prompts(
            parameters, transform, maximum=maximum_prompts
        )
        preprocessed = perf_counter()
        try:
            encoder_values = self.encoder_session.run(None, {str(self.encoder_input_meta.name): tensor})
        except Exception as exc:
            raise ModelRuntimeError("SAM encoder inference failed", details={"error": str(exc)}) from exc
        encoded = perf_counter()
        if len(encoder_values) != len(self.encoder_output_meta):
            raise ModelRuntimeError(
                "SAM encoder output count changed",
                details={"expected": len(self.encoder_output_meta), "actual": len(encoder_values)},
            )
        encoder_outputs = {
            str(meta.name): np.asarray(value)
            for meta, value in zip(self.encoder_output_meta, encoder_values, strict=True)
        }
        embedding_name = self._encoder_embedding_name()
        embedding = encoder_outputs[embedding_name]
        if not np.issubdtype(embedding.dtype, np.number) or not np.all(np.isfinite(embedding)):
            raise ModelRuntimeError("SAM encoder embedding must contain finite numeric values")
        if embedding.size > max_capture_values:
            raise ModelRuntimeError(
                "SAM encoder embedding exceeds the value budget",
                details={"values": int(embedding.size), "maximum": max_capture_values},
            )

        decoder_roles = _resolve_decoder_inputs(self.decoder_input_meta)
        mask_height, mask_width = _mask_input_shape(decoder_roles["mask"])
        decoder_feed: dict[str, np.ndarray] = {
            str(decoder_roles["embedding"].name): embedding.astype(np.float32, copy=False),
            str(decoder_roles["coords"].name): point_coords,
            str(decoder_roles["labels"].name): point_labels,
            str(decoder_roles["mask"].name): np.zeros((1, 1, mask_height, mask_width), dtype=np.float32),
            str(decoder_roles["has_mask"].name): np.zeros(1, dtype=np.float32),
            str(decoder_roles["original_size"].name): np.asarray(
                [transform.original_height, transform.original_width], dtype=np.float32
            ),
        }
        if "intermediate" in decoder_roles:
            configured = self.record.config.get("encoder_intermediate_outputs")
            if configured is not None:
                if not isinstance(configured, list) or not all(isinstance(name, str) for name in configured):
                    raise ModelRuntimeError("encoder_intermediate_outputs must be a list of output names")
                names = configured
            else:
                names = [name for name in encoder_outputs if name != embedding_name]
            if not names or any(name not in encoder_outputs for name in names):
                raise ModelRuntimeError(
                    "SAM-HQ decoder requires encoder intermediate outputs",
                    details={"configured": names, "available": sorted(encoder_outputs)},
                )
            intermediate = encoder_outputs[names[0]] if len(names) == 1 else np.stack([encoder_outputs[name] for name in names])
            if intermediate.size > max_capture_values or not np.all(np.isfinite(intermediate)):
                raise ModelRuntimeError("SAM-HQ intermediate embeddings are invalid or exceed the value budget")
            decoder_feed[str(decoder_roles["intermediate"].name)] = intermediate.astype(np.float32, copy=False)
        try:
            decoder_values = self.decoder_session.run(None, decoder_feed)
        except Exception as exc:
            raise ModelRuntimeError("SAM decoder inference failed", details={"error": str(exc)}) from exc
        decoded = perf_counter()
        if len(decoder_values) != len(self.decoder_output_meta):
            raise ModelRuntimeError(
                "SAM decoder output count changed",
                details={"expected": len(self.decoder_output_meta), "actual": len(decoder_values)},
            )
        decoder_outputs = {
            str(meta.name): np.asarray(value)
            for meta, value in zip(self.decoder_output_meta, decoder_values, strict=True)
        }
        captures = {f"encoder:{name}": value for name, value in encoder_outputs.items()}
        captures.update({f"decoder:{name}": value for name, value in decoder_outputs.items()})
        for layer_id in capture_layers:
            value = captures[layer_id]
            if value.size > max_capture_values or not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                raise ModelRuntimeError(
                    "SAM capture output is invalid or exceeds the value budget",
                    details={"layer_id": layer_id, "values": int(value.size), "maximum": max_capture_values},
                )
        mask_name, masks, iou_name, iou_scores = self._decoder_outputs(decoder_outputs)
        selected_index = int(np.argmax(iou_scores))
        output_pixels = transform.original_width * transform.original_height
        if output_pixels > max_output_pixels:
            raise ModelRuntimeError(
                "SAM mask exceeds the output pixel budget",
                details={"pixels": output_pixels, "maximum": max_output_pixels},
            )
        restored = _restore_mask(masks[selected_index], transform)
        threshold = _number_parameter(
            parameters, self.record.config, "mask_threshold", 0.0, minimum=-1e6, maximum=1e6
        )
        binary = restored > threshold
        if not np.any(binary):
            raise ModelRuntimeError("SAM selected mask contains no foreground", details={"mask_index": selected_index})

        max_components = _integer_parameter(
            parameters, self.record.config, "max_mask_components", 16, minimum=1, maximum=1024
        )
        max_polygon_points = _integer_parameter(
            parameters, self.record.config, "max_polygon_points", 1024, minimum=3, maximum=100_000
        )
        minimum_area = _number_parameter(
            parameters, self.record.config, "min_mask_area", 1.0, minimum=0.0, maximum=1e15
        )
        simplify = _number_parameter(
            parameters, self.record.config, "polygon_simplify", 1.0, minimum=0.0, maximum=1000.0
        )
        polygons = _polygons_from_mask(
            binary,
            origin_x=0,
            origin_y=0,
            target_width=transform.original_width,
            target_height=transform.original_height,
            original_width=transform.original_width,
            original_height=transform.original_height,
            max_components=max_components,
            minimum_original_area=minimum_area,
            simplify_epsilon=simplify,
            max_points=max_polygon_points,
        )
        if not polygons:
            raise ModelRuntimeError("SAM mask did not yield a polygon within the configured budgets")
        label = parameters.get("label", self.record.config.get("label", "sam-mask"))
        if not isinstance(label, str) or not label:
            raise ModelRuntimeError("SAM label must be a non-empty string")
        score = float(np.clip(iou_scores[selected_index], 0.0, 1.0))
        annotations = [
            AnnotationResult(
                label=label,
                score=score,
                shape_type="polygon",
                points=[[float(x), float(y)] for x, y in polygon],
            )
            for polygon in polygons
        ]
        raster = Image.fromarray(binary.astype(np.uint8) * 255)
        try:
            raster_artifact = self.artifact_store.put_raster(
                model_id=self.record.descriptor.id,
                image_path=image_path,
                role="sam-mask",
                image=raster,
                format_name="png",
                metadata={
                    "mask_output": mask_name,
                    "iou_output": iou_name,
                    "selected_mask": selected_index,
                    "predicted_iou": score,
                    "mask_threshold": threshold,
                    "prompt_summary": prompt_summary,
                    "encoder_size": [transform.target_size, transform.target_size],
                    "resized_size": [transform.resized_width, transform.resized_height],
                    "original_size": [transform.original_width, transform.original_height],
                },
            )
        except ValueError as exc:
            raise ModelRuntimeError("Could not store SAM mask raster", details={"error": str(exc)}) from exc

        artifacts = []
        try:
            feature_options = FeatureTransformOptions.model_validate(parameters.get("feature_transform") or {})
        except Exception as exc:
            raise ModelRuntimeError("Invalid SAM feature transform options", details={"error": str(exc)}) from exc
        for layer_id in capture_layers:
            value = captures[layer_id]
            transformed = transform_feature(value, feature_options)
            artifacts.append(
                self.artifact_store.put_tensor(
                    model_id=self.record.descriptor.id,
                    image_path=image_path,
                    layer_id=layer_id,
                    tensor=transformed,
                    source_shape=list(value.shape),
                    transform={"stage": layer_id.split(":", 1)[0], **feature_options.model_dump(mode="json")},
                )
            )
        finished = perf_counter()
        return InferenceResult(
            model_id=self.record.descriptor.id,
            image_path=image_path,
            annotations=annotations,
            artifacts=artifacts,
            rasters=[raster_artifact],
            timings_ms={
                "preprocess": (preprocessed - started) * 1000,
                "encoder": (encoded - preprocessed) * 1000,
                "decoder": (decoded - encoded) * 1000,
                "postprocess": (finished - decoded) * 1000,
                "total": (finished - started) * 1000,
            },
        )
