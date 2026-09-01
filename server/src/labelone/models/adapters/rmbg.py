from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import RasterArtifact
from .onnx import OnnxRuntimeAdapter, _ImageTransform


def _model_version(config: dict[str, Any]) -> float:
    raw = config.get("version", 1.4)
    try:
        version = float(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError("RMBG version must be 1.4 or 2.0", details={"version": raw}) from exc
    if not any(np.isclose(version, supported) for supported in (1.4, 2.0)):
        raise ModelRuntimeError("RMBG version must be 1.4 or 2.0", details={"version": raw})
    return 2.0 if np.isclose(version, 2.0) else 1.4


def _resize_mode(config: dict[str, Any]) -> str:
    value = str(config.get("resize_mode", "stretch")).strip().casefold()
    if value not in {"stretch", "letterbox"}:
        raise ModelRuntimeError(
            "RMBG resize_mode must be stretch or letterbox",
            details={"resize_mode": value},
        )
    return value


def _single_channel_mask(value: np.ndarray, *, output_name: str) -> tuple[np.ndarray, list[int]]:
    array = np.asarray(value)
    source_shape = list(array.shape)
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] != 1:
            raise ModelRuntimeError(
                "RMBG output must have one batch and one channel",
                details={"output_name": output_name, "shape": source_shape},
            )
        array = array[0, 0]
    elif array.ndim == 3:
        if array.shape[0] != 1:
            raise ModelRuntimeError(
                "RMBG three-dimensional output must have shape [1,H,W]",
                details={"output_name": output_name, "shape": source_shape},
            )
        array = array[0]
    elif array.ndim != 2:
        raise ModelRuntimeError(
            "Unsupported RMBG output shape",
            details={"output_name": output_name, "shape": source_shape, "expected": ["[1,1,H,W]", "[1,H,W]", "[H,W]"]},
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ModelRuntimeError(
            "RMBG mask dimensions must be positive",
            details={"output_name": output_name, "shape": source_shape},
        )
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ModelRuntimeError(
            "RMBG output must contain only finite numeric values",
            details={"output_name": output_name, "dtype": str(array.dtype)},
        )
    return array.astype(np.float32, copy=False), source_shape


def _select_mask_output(
    outputs: dict[str, np.ndarray],
    *,
    parameters: dict[str, object],
    config: dict[str, Any],
) -> tuple[str, np.ndarray, list[int]]:
    if not outputs:
        raise ModelRuntimeError("RMBG model returned no outputs")
    configured = parameters.get(
        "mask_output_name",
        parameters.get("output_name", config.get("mask_output_name", config.get("output_name"))),
    )
    if configured is not None:
        if not isinstance(configured, str):
            raise ModelRuntimeError("RMBG mask_output_name must be a string", details={"value": configured})
        if configured not in outputs:
            raise ModelRuntimeError(
                "Configured RMBG mask output was not returned",
                details={"output_name": configured, "available_outputs": sorted(outputs)},
            )
        mask, source_shape = _single_channel_mask(outputs[configured], output_name=configured)
        return configured, mask, source_shape

    candidates: list[tuple[str, np.ndarray, list[int]]] = []
    failures: dict[str, dict[str, object]] = {}
    for output_name, value in outputs.items():
        try:
            mask, source_shape = _single_channel_mask(value, output_name=output_name)
            candidates.append((output_name, mask, source_shape))
        except ModelRuntimeError as exc:
            failures[output_name] = {"message": exc.message, **exc.details}
    if not candidates:
        raise ModelRuntimeError(
            "No supported single-channel RMBG output was found",
            details={"outputs": {name: list(np.asarray(value).shape) for name, value in outputs.items()}, "failures": failures},
        )
    if len(candidates) > 1:
        raise ModelRuntimeError(
            "Multiple RMBG mask outputs are ambiguous",
            details={"candidates": [name for name, _, _ in candidates], "hint": "Set mask_output_name explicitly"},
        )
    return candidates[0]


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
        raise ModelRuntimeError(f"RMBG {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or value < minimum or value > maximum:
        raise ModelRuntimeError(
            f"RMBG {name} is outside the supported range",
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
        raise ModelRuntimeError(f"RMBG {name} must be an integer", details={"value": raw})
    try:
        value = int(raw)
        exact = float(raw) == value
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"RMBG {name} must be an integer", details={"value": raw}) from exc
    if not exact or value < minimum or value > maximum:
        raise ModelRuntimeError(
            f"RMBG {name} is outside the supported range",
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
        raise ModelRuntimeError(f"RMBG {name} must be boolean", details={"value": raw})
    return raw


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _probabilities(mask: np.ndarray, requested: object) -> tuple[np.ndarray, str]:
    activation = str(requested).strip().casefold()
    aliases = {"probabilities": "probability", "none": "probability", "logits": "sigmoid"}
    activation = aliases.get(activation, activation)
    if activation == "auto":
        activation = "probability" if np.all((mask >= 0) & (mask <= 1)) else "sigmoid"
    if activation == "sigmoid":
        return _sigmoid(mask).astype(np.float32, copy=False), activation
    if activation == "minmax":
        return mask.astype(np.float32, copy=False), activation
    if activation != "probability":
        raise ModelRuntimeError(
            "RMBG mask_activation must be auto, probability, sigmoid, or minmax",
            details={"mask_activation": requested},
        )
    if np.any(mask < 0) or np.any(mask > 1):
        raise ModelRuntimeError("RMBG probability output contains values outside zero and one")
    return mask.astype(np.float32, copy=False), activation


def _restore_probability(
    probability: np.ndarray,
    transform: _ImageTransform,
    *,
    resize_mode: str,
    clip: bool,
) -> np.ndarray:
    if transform.scale <= 0 or transform.input_width <= 0 or transform.input_height <= 0:
        raise ModelRuntimeError("RMBG image transform is invalid")
    if resize_mode == "stretch":
        restored = Image.fromarray(probability.astype(np.float32, copy=False)).resize(
            (transform.original_width, transform.original_height),
            Image.Resampling.BILINEAR,
        )
        values = np.asarray(restored, dtype=np.float32)
        return np.clip(values, 0.0, 1.0) if clip else values
    resized_width = max(1, round(transform.original_width * transform.scale))
    resized_height = max(1, round(transform.original_height * transform.scale))
    mask_height, mask_width = probability.shape
    source_left = transform.pad_x * mask_width / transform.input_width
    source_top = transform.pad_y * mask_height / transform.input_height
    source_width = resized_width * mask_width / transform.input_width
    source_height = resized_height * mask_height / transform.input_height
    if source_width <= 0 or source_height <= 0:
        raise ModelRuntimeError("RMBG letterbox content area is empty")
    restored = Image.fromarray(probability.astype(np.float32, copy=False)).transform(
        (transform.original_width, transform.original_height),
        Image.Transform.AFFINE,
        (
            source_width / transform.original_width,
            0.0,
            source_left,
            0.0,
            source_height / transform.original_height,
            source_top,
        ),
        resample=Image.Resampling.BILINEAR,
    )
    values = np.asarray(restored, dtype=np.float32)
    return np.clip(values, 0.0, 1.0) if clip else values


def _artifact_metadata(
    *,
    output_name: str,
    source_shape: list[int],
    probability: np.ndarray,
    transform: _ImageTransform,
    activation: str,
    threshold: float,
    pixel_budget: int,
    model_version: float,
    resize_mode: str,
) -> dict[str, object]:
    return {
        "kind": "alpha_mask",
        "output_name": output_name,
        "source_shape": source_shape,
        "activation": activation,
        "threshold": threshold,
        "minimum": float(probability.min()),
        "maximum": float(probability.max()),
        "mean": float(probability.mean()),
        "output_size": [transform.original_width, transform.original_height],
        "max_output_pixels": pixel_budget,
        "preprocessing": {
            "version": model_version,
            "resize_mode": resize_mode,
            "normalization": "imagenet" if model_version >= 2.0 else "center_0.5",
        },
        "letterbox": {
            "input_size": [transform.input_width, transform.input_height],
            "scale": transform.scale,
            "pad": [transform.pad_x, transform.pad_y],
        },
    }


class RmbgMattingOnnxAdapter(OnnxRuntimeAdapter):
    """Clean-room raster adapter for single-channel RMBG and matting ONNX models."""

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, _ImageTransform]:
        if self.input_meta is None:
            raise ModelRuntimeError("Model is not loaded")
        input_shape = list(self.input_meta.shape)
        if len(input_shape) != 4 or input_shape[0] not in {1, "N", "batch", None}:
            raise ModelRuntimeError(
                "RMBG input must have NCHW shape with one batch",
                details={"shape": input_shape},
            )
        if input_shape[1] not in {3, "C", "channels", None}:
            raise ModelRuntimeError(
                "RMBG input must have three RGB channels in NCHW order",
                details={"shape": input_shape},
            )
        version = _model_version(self.record.config)
        if version >= 2.0:
            input_height = input_width = 1024
            for axis, expected in ((input_shape[2], input_height), (input_shape[3], input_width)):
                if isinstance(axis, int) and axis != expected:
                    raise ModelRuntimeError(
                        "RMBG 2.0 ONNX input must be 1024 by 1024",
                        details={"shape": input_shape},
                    )
        else:
            if not isinstance(input_shape[2], int) or not isinstance(input_shape[3], int):
                raise ModelRuntimeError(
                    "RMBG 1.4 ONNX input requires static spatial dimensions",
                    details={"shape": input_shape},
                )
            input_height, input_width = input_shape[2], input_shape[3]
            if input_height <= 0 or input_width <= 0:
                raise ModelRuntimeError("RMBG input dimensions must be positive", details={"shape": input_shape})

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                original_width, original_height = image.size
                mode = _resize_mode(self.record.config)
                if mode == "stretch":
                    resized_width, resized_height = input_width, input_height
                    left = top = 0
                    canvas = image.resize((input_width, input_height), Image.Resampling.BILINEAR)
                    scale = input_width / original_width
                else:
                    scale = min(input_width / original_width, input_height / original_height)
                    resized_width = max(1, round(original_width * scale))
                    resized_height = max(1, round(original_height * scale))
                    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
                    left = round((input_width - resized_width) / 2 - 0.1)
                    top = round((input_height - resized_height) / 2 - 0.1)
                    letterbox_value = _integer_parameter(
                        {},
                        self.record.config,
                        "letterbox_value",
                        114,
                        minimum=0,
                        maximum=255,
                    )
                    canvas = Image.new("RGB", (input_width, input_height), (letterbox_value,) * 3)
                    canvas.paste(resized, (left, top))
        except ModelRuntimeError:
            raise
        except (OSError, ValueError) as exc:
            raise ModelRuntimeError(
                "Could not preprocess the RMBG source image",
                details={"image_path": str(image_path), "error": str(exc)},
            ) from exc

        tensor = np.asarray(canvas, dtype=np.float32) / 255.0
        if version >= 2.0:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = (tensor - mean) / std
        else:
            tensor = tensor - 0.5
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        if "float16" in str(self.input_meta.type):
            tensor = tensor.astype(np.float16)
        return tensor, _ImageTransform(
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
            scale=scale,
            pad_x=left,
            pad_y=top,
        )

    def _rasters(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        image_path: Path,
        parameters: dict[str, object],
    ) -> list[RasterArtifact]:
        config = self.record.config
        version = _model_version(config)
        resize_mode = _resize_mode(config)
        output_name, mask, source_shape = _select_mask_output(
            outputs,
            parameters=parameters,
            config=config,
        )
        maximum_pixels = _integer_parameter(
            parameters,
            config,
            "max_output_pixels",
            64_000_000,
            minimum=1,
            maximum=536_870_912,
        )
        output_pixels = transform.original_width * transform.original_height
        if output_pixels > maximum_pixels:
            raise ModelRuntimeError(
                "RMBG raster output exceeds the pixel budget",
                details={
                    "output_size": [transform.original_width, transform.original_height],
                    "pixels": output_pixels,
                    "maximum": maximum_pixels,
                },
            )
        probability, activation = _probabilities(
            mask,
            parameters.get("mask_activation", config.get("mask_activation", "minmax")),
        )
        restored = _restore_probability(
            probability,
            transform,
            resize_mode=resize_mode,
            clip=activation != "minmax",
        )
        if activation == "minmax":
            minimum = float(restored.min())
            maximum = float(restored.max())
            if maximum - minimum <= np.finfo(np.float32).eps:
                raise ModelRuntimeError(
                    "RMBG minmax output has no dynamic range",
                    details={"minimum": minimum, "maximum": maximum},
                )
            restored = ((restored - minimum) / (maximum - minimum)).astype(np.float32, copy=False)
        threshold = _number_parameter(
            parameters,
            config,
            "mask_threshold",
            0.0,
            minimum=0.0,
            maximum=1.0,
        )
        if threshold > 0:
            restored = np.where(restored >= threshold, restored, 0.0).astype(np.float32, copy=False)
        metadata = _artifact_metadata(
            output_name=output_name,
            source_shape=source_shape,
            probability=restored,
            transform=transform,
            activation=activation,
            threshold=threshold,
            pixel_budget=maximum_pixels,
            model_version=version,
            resize_mode=resize_mode,
        )
        alpha = np.rint(restored * 255.0).clip(0, 255).astype(np.uint8)
        mask_artifact = self.artifact_store.put_raster(
            model_id=self.record.descriptor.id,
            image_path=image_path,
            role="alpha-mask",
            image=Image.fromarray(alpha),
            format_name="png",
            metadata=metadata,
        )
        artifacts = [mask_artifact]
        if _boolean_parameter(parameters, config, "output_cutout", False):
            try:
                with Image.open(image_path) as source:
                    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
            except (OSError, ValueError) as exc:
                raise ModelRuntimeError(
                    "Could not read the source image for the RMBG cutout",
                    details={"image_path": str(image_path), "error": str(exc)},
                ) from exc
            if rgba.shape[:2] != alpha.shape:
                raise ModelRuntimeError(
                    "RMBG source image dimensions changed during inference",
                    details={"source_size": [rgba.shape[1], rgba.shape[0]], "mask_size": [alpha.shape[1], alpha.shape[0]]},
                )
            rgba[:, :, 3] = np.rint(rgba[:, :, 3].astype(np.float32) * restored).clip(0, 255).astype(np.uint8)
            artifacts.append(self.artifact_store.put_raster(
                model_id=self.record.descriptor.id,
                image_path=image_path,
                role="foreground-cutout",
                image=Image.fromarray(rgba, mode="RGBA"),
                format_name="webp",
                metadata={
                    **metadata,
                    "kind": "foreground_cutout",
                    "mask_artifact_id": mask_artifact.id,
                    "source_mode": "RGBA",
                },
            ))
        return artifacts
