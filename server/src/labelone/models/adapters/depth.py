from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import RasterArtifact
from .onnx import OnnxRuntimeAdapter, _ImageTransform


_TURBO_POSITIONS = np.asarray([0.0, 0.12, 0.28, 0.46, 0.64, 0.82, 1.0], dtype=np.float32)
_TURBO_COLORS = np.asarray(
    [
        [24, 18, 95],
        [43, 77, 217],
        [35, 171, 216],
        [77, 224, 112],
        [225, 221, 55],
        [244, 106, 26],
        [122, 4, 3],
    ],
    dtype=np.float32,
)


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
        raise ModelRuntimeError(f"Depth {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"Depth {name} is outside the supported range",
            details={"value": value, "minimum": minimum, "maximum": maximum},
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
        raise ModelRuntimeError(f"Depth {name} must be an integer", details={"value": raw})
    try:
        value = int(raw)
        exact = float(raw) == value
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"Depth {name} must be an integer", details={"value": raw}) from exc
    if not exact or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"Depth {name} is outside the supported range",
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
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.casefold() in {"true", "false"}:
        return raw.casefold() == "true"
    raise ModelRuntimeError(f"Depth {name} must be boolean", details={"value": raw})


def _depth_plane(value: np.ndarray, *, output_name: str) -> np.ndarray:
    array = np.asarray(value)
    original_shape = list(array.shape)
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] != 1:
            raise ModelRuntimeError(
                "Depth output must have a single batch and channel",
                details={"output_name": output_name, "shape": original_shape, "expected": "[1,1,H,W]"},
            )
        array = array[0, 0]
    elif array.ndim == 3:
        if array.shape[0] != 1:
            raise ModelRuntimeError(
                "Depth output must have a single leading dimension",
                details={"output_name": output_name, "shape": original_shape, "expected": "[1,H,W]"},
            )
        array = array[0]
    elif array.ndim != 2:
        raise ModelRuntimeError(
            "Unsupported depth output rank",
            details={"output_name": output_name, "shape": original_shape, "expected": "[1,1,H,W], [1,H,W], or [H,W]"},
        )
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ModelRuntimeError(
            "Depth output has invalid spatial dimensions",
            details={"output_name": output_name, "shape": original_shape},
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ModelRuntimeError(
            "Depth output must be numeric",
            details={"output_name": output_name, "dtype": str(array.dtype)},
        )
    if not np.all(np.isfinite(array)):
        finite_count = int(np.count_nonzero(np.isfinite(array)))
        raise ModelRuntimeError(
            "Depth output contains non-finite values",
            details={"output_name": output_name, "shape": original_shape, "finite": finite_count, "total": int(array.size)},
        )
    return array.astype(np.float32, copy=False)


def _configured_output_name(parameters: dict[str, object], config: dict[str, Any]) -> str | None:
    value = parameters.get(
        "depth_output_name",
        parameters.get("output_name", config.get("depth_output_name", config.get("output_name"))),
    )
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ModelRuntimeError("Depth output name must be a non-empty string", details={"value": value})
    return value


def _select_depth_output(
    outputs: dict[str, np.ndarray],
    parameters: dict[str, object],
    config: dict[str, Any],
) -> tuple[str, np.ndarray]:
    if not outputs:
        raise ModelRuntimeError("Depth model returned no outputs")
    configured = _configured_output_name(parameters, config)
    if configured is not None:
        if configured not in outputs:
            raise ModelRuntimeError(
                "Configured depth output was not returned",
                details={"output_name": configured, "available_outputs": sorted(outputs)},
            )
        return configured, _depth_plane(outputs[configured], output_name=configured)

    candidates: list[tuple[str, np.ndarray]] = []
    failures: dict[str, dict[str, object]] = {}
    for name, value in outputs.items():
        try:
            candidates.append((name, _depth_plane(value, output_name=name)))
        except ModelRuntimeError as exc:
            failures[name] = {"message": exc.message, **exc.details}
    if not candidates:
        raise ModelRuntimeError(
            "No supported depth output was found",
            details={"outputs": {name: list(np.asarray(value).shape) for name, value in outputs.items()}, "failures": failures},
        )
    if len(candidates) > 1:
        raise ModelRuntimeError(
            "Multiple depth outputs are ambiguous; configure depth_output_name",
            details={"candidates": [name for name, _ in candidates]},
        )
    return candidates[0]


def _normalization_vector(config: dict[str, Any], name: str, default: list[float]) -> np.ndarray:
    raw = config.get(name, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ModelRuntimeError(f"Depth input {name} must contain three values", details={"value": raw})
    try:
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"Depth input {name} must be numeric", details={"value": raw}) from exc
    if not np.all(np.isfinite(values)) or (name == "std" and np.any(values <= 0)):
        raise ModelRuntimeError(f"Depth input {name} is invalid", details={"value": list(values)})
    return values


def _raw_statistics(depth: np.ndarray, low_percentile: float, high_percentile: float) -> tuple[dict[str, object], float, float]:
    requested = np.percentile(depth, [low_percentile, high_percentile])
    standard = np.percentile(depth, [1.0, 5.0, 50.0, 95.0, 99.0])
    statistics: dict[str, object] = {
        "raw_min": float(depth.min()),
        "raw_max": float(depth.max()),
        "raw_percentiles": {
            "p01": float(standard[0]),
            "p05": float(standard[1]),
            "p50": float(standard[2]),
            "p95": float(standard[3]),
            "p99": float(standard[4]),
        },
        "clip_percentiles": [low_percentile, high_percentile],
        "clip_values": [float(requested[0]), float(requested[1])],
    }
    return statistics, float(requested[0]), float(requested[1])


def _normalized_depth(depth: np.ndarray, low: float, high: float, *, inverse: bool) -> np.ndarray:
    if high <= low:
        normalized = np.zeros(depth.shape, dtype=np.float32)
    else:
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)
    return 1.0 - normalized if inverse else normalized


def _resize_scalar(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32, copy=False))
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _turbo_like(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    channels = [
        np.interp(flat, _TURBO_POSITIONS, _TURBO_COLORS[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=1).reshape((*values.shape, 3)).clip(0, 255).astype(np.uint8)


class DepthAnythingOnnxAdapter(OnnxRuntimeAdapter):
    """Clean-room Depth Anything ONNX adapter producing a default depth raster."""

    def _image_resize_mode(self) -> str:
        return "stretch"

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, _ImageTransform]:
        if self.input_meta is None:
            raise ModelRuntimeError("Depth model is not loaded")
        input_shape = list(self.input_meta.shape)
        if len(input_shape) != 4 or input_shape[1] not in {3, "C", "channels"}:
            raise ModelRuntimeError(
                "Depth Anything input must use NCHW with three RGB channels",
                details={"shape": input_shape},
            )
        model_type = str(self.record.descriptor.model_type).casefold()
        if "depth_anything_v2" in model_type:
            input_height = input_shape[2] if isinstance(input_shape[2], int) else 518
            input_width = input_shape[3] if isinstance(input_shape[3], int) else 518
        else:
            input_height = input_width = 518
            for axis, expected in ((input_shape[2], input_height), (input_shape[3], input_width)):
                if isinstance(axis, int) and axis != expected:
                    raise ModelRuntimeError(
                        "Depth Anything v1 expects a 518x518 ONNX input",
                        details={"shape": input_shape, "expected": [1, 3, 518, 518]},
                    )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            original_width, original_height = image.size
            resized = image.resize((input_width, input_height), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = _normalization_vector(self.record.config, "mean", [0.485, 0.456, 0.406])
        std = _normalization_vector(self.record.config, "std", [0.229, 0.224, 0.225])
        normalized = (array - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        tensor = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None]).astype(np.float32, copy=False)
        return tensor, _ImageTransform(
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
            scale=input_width / original_width,
            pad_x=0,
            pad_y=0,
        )

    def _rasters(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        image_path: Path,
        parameters: dict[str, object],
    ) -> list[RasterArtifact]:
        output_name, depth = _select_depth_output(outputs, parameters, self.record.config)
        max_depth_values = _integer_parameter(
            parameters,
            self.record.config,
            "max_depth_values",
            64_000_000,
            minimum=1,
            maximum=1_000_000_000,
        )
        if depth.size > max_depth_values:
            raise ModelRuntimeError(
                "Depth output exceeds the raw value budget",
                details={"values": int(depth.size), "maximum": max_depth_values, "shape": list(depth.shape)},
            )
        max_raster_pixels = _integer_parameter(
            parameters,
            self.record.config,
            "max_raster_pixels",
            16_777_216,
            minimum=1,
            maximum=268_435_456,
        )
        output_pixels = transform.original_width * transform.original_height
        if output_pixels > max_raster_pixels:
            raise ModelRuntimeError(
                "Depth raster exceeds the output pixel budget",
                details={
                    "output_size": [transform.original_width, transform.original_height],
                    "pixels": output_pixels,
                    "maximum": max_raster_pixels,
                },
            )
        low_percentile = _number_parameter(
            parameters, self.record.config, "percentile_low", 2.0, minimum=0.0, maximum=100.0
        )
        high_percentile = _number_parameter(
            parameters, self.record.config, "percentile_high", 98.0, minimum=0.0, maximum=100.0
        )
        if high_percentile <= low_percentile:
            raise ModelRuntimeError(
                "Depth percentile_high must be greater than percentile_low",
                details={"percentile_low": low_percentile, "percentile_high": high_percentile},
            )
        inverse = _boolean_parameter(parameters, self.record.config, "inverse", False)
        color_map = str(
            parameters.get(
                "color_map",
                parameters.get("colormap", self.record.config.get("color_map", self.record.config.get("colormap", "turbo"))),
            )
        ).strip().casefold()
        aliases = {"gray": "grayscale", "grey": "grayscale", "turbo-like": "turbo", "turbo_like": "turbo"}
        color_map = aliases.get(color_map, color_map)
        if color_map not in {"grayscale", "turbo"}:
            raise ModelRuntimeError(
                "Unsupported depth color map",
                details={"color_map": color_map, "supported": ["grayscale", "turbo"]},
            )
        raster_format = str(parameters.get("raster_format", self.record.config.get("raster_format", "png"))).strip().casefold()
        if raster_format == "jpg":
            raster_format = "jpeg"
        if raster_format not in {"png", "webp", "jpeg"}:
            raise ModelRuntimeError(
                "Unsupported depth raster format",
                details={"raster_format": raster_format, "supported": ["png", "webp", "jpeg"]},
            )

        statistics, low_value, high_value = _raw_statistics(depth, low_percentile, high_percentile)
        normalized = _normalized_depth(depth, low_value, high_value, inverse=inverse)
        resized = _resize_scalar(normalized, (transform.original_width, transform.original_height))
        if color_map == "grayscale":
            raster_image = Image.fromarray(np.rint(resized * 255.0).clip(0, 255).astype(np.uint8))
        else:
            raster_image = Image.fromarray(_turbo_like(resized))
        metadata: dict[str, object] = {
            **statistics,
            "output_name": output_name,
            "source_shape": list(depth.shape),
            "output_size": [transform.original_width, transform.original_height],
            "model_input_size": [transform.input_width, transform.input_height],
            "resize_mode": self._image_resize_mode(),
            "normalization": "percentile",
            "inverse": inverse,
            "color_map": color_map,
            "constant": high_value <= low_value,
        }
        return [
            self.artifact_store.put_raster(
                model_id=self.record.descriptor.id,
                image_path=image_path,
                role="depth-map",
                image=raster_image,
                format_name=raster_format,
                metadata=metadata,
            )
        ]
