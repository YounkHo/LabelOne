from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult
from .onnx import OnnxRuntimeAdapter, _ImageTransform, _nms, _xywh_to_xyxy


_OutputLayout = Literal["raw", "raw_objectness", "end_to_end"]


@dataclass(frozen=True, slots=True)
class _PoseLayout:
    prediction: np.ndarray
    output_layout: _OutputLayout
    keypoint_count: int
    keypoint_dimensions: int


def _names(value: object, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        def key_order(item: tuple[object, object]) -> tuple[int, object]:
            key = item[0]
            try:
                return (0, int(str(key)))
            except ValueError:
                return (1, str(key))

        return [str(label) for _, label in sorted(value.items(), key=key_order)]
    if isinstance(value, (list, tuple)):
        return [str(label) for label in value]
    raise ModelRuntimeError(f"YOLO pose {field} must be a list or mapping")


def _class_names(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value]
    return _names(value, field="classes")


def _class_keypoint_names(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for class_name, names in value.items():
        if not isinstance(names, (list, tuple)):
            raise ModelRuntimeError(
                "YOLO pose class keypoints must be lists",
                details={"class": str(class_name)},
            )
        result[str(class_name)] = [str(name) for name in names]
    return result


def _float_parameter(parameters: dict[str, object], config: dict[str, Any], name: str, default: float) -> float:
    value = parameters.get(name, config.get(name, default))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError("Invalid YOLO pose numeric parameter", details={"parameter": name, "value": value}) from exc
    if not np.isfinite(result) or result < 0 or result > 1:
        raise ModelRuntimeError(
            "YOLO pose thresholds must be between zero and one",
            details={"parameter": name, "value": value},
        )
    return result


def _configured_keypoint_shape(config: dict[str, Any]) -> tuple[int, int] | None:
    value = config.get("kpt_shape", config.get("keypoint_shape"))
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ModelRuntimeError("YOLO pose kpt_shape must contain [count, dimensions]")
    try:
        count, dimensions = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError("YOLO pose kpt_shape values must be integers", details={"kpt_shape": value}) from exc
    if count <= 0 or dimensions not in {2, 3}:
        raise ModelRuntimeError(
            "YOLO pose kpt_shape requires a positive count and two or three dimensions",
            details={"kpt_shape": value},
        )
    return count, dimensions


def _output_matrix(output: np.ndarray) -> np.ndarray:
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ModelRuntimeError(
                "YOLO pose adapter supports batch size one",
                details={"shape": list(prediction.shape)},
            )
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ModelRuntimeError("Unsupported YOLO pose output rank", details={"shape": list(prediction.shape)})
    if 0 in prediction.shape:
        raise ModelRuntimeError("YOLO pose output cannot contain an empty axis", details={"shape": list(prediction.shape)})
    if not np.issubdtype(prediction.dtype, np.number) or not np.all(np.isfinite(prediction)):
        raise ModelRuntimeError("YOLO pose output must contain only finite numeric values")
    return prediction.astype(np.float32, copy=False)


def _layout_for_width(
    width: int,
    *,
    class_count: int,
    keypoint_shape: tuple[int, int] | None,
    requested_format: str,
) -> tuple[_OutputLayout, int, int] | None:
    allowed: tuple[_OutputLayout, ...]
    if requested_format == "raw":
        allowed = ("raw", "raw_objectness")
    elif requested_format == "end_to_end":
        allowed = ("end_to_end",)
    elif requested_format == "auto":
        # Raw Ultralytics exports are the common case. For one-class pose models,
        # objectness and end-to-end rows can have the same width, so auto favors raw.
        allowed = ("raw", "raw_objectness", "end_to_end")
    else:
        raise ModelRuntimeError(
            "YOLO pose output_format must be auto, raw, or end_to_end",
            details={"output_format": requested_format},
        )

    layouts: list[tuple[_OutputLayout, int, int]] = []
    for output_layout in allowed:
        header_width = 4 + class_count
        if output_layout == "raw_objectness":
            header_width += 1
        elif output_layout == "end_to_end":
            header_width = 6
        remaining = width - header_width
        if keypoint_shape is not None:
            count, dimensions = keypoint_shape
            if remaining == count * dimensions:
                layouts.append((output_layout, count, dimensions))
        elif remaining >= 3 and remaining % 3 == 0:
            count = remaining // 3
            if count <= 256:
                layouts.append((output_layout, count, 3))

    if not layouts:
        return None
    if requested_format == "auto":
        for preferred in ("raw", "raw_objectness", "end_to_end"):
            for layout in layouts:
                if layout[0] == preferred:
                    return layout
    if len(layouts) > 1:
        raise ModelRuntimeError(
            "Ambiguous YOLO pose output layout",
            details={"width": width, "layouts": [layout[0] for layout in layouts]},
        )
    return layouts[0]


def _resolve_layout(
    output: np.ndarray,
    *,
    class_count: int,
    keypoint_shape: tuple[int, int] | None,
    requested_format: str,
) -> _PoseLayout:
    matrix = _output_matrix(output)
    candidates: list[_PoseLayout] = []
    for prediction in (matrix, matrix.T):
        resolved = _layout_for_width(
            prediction.shape[1],
            class_count=class_count,
            keypoint_shape=keypoint_shape,
            requested_format=requested_format,
        )
        if resolved is not None:
            output_layout, count, dimensions = resolved
            candidates.append(_PoseLayout(prediction, output_layout, count, dimensions))
    if not candidates:
        raise ModelRuntimeError(
            "YOLO pose output width does not match its classes and keypoints",
            details={
                "shape": list(matrix.shape),
                "class_count": class_count,
                "kpt_shape": list(keypoint_shape) if keypoint_shape else None,
            },
        )
    if len(candidates) > 1:
        raise ModelRuntimeError(
            "Ambiguous YOLO pose output orientation",
            details={"shape": list(matrix.shape)},
        )
    return candidates[0]


def _select_output(outputs: dict[str, np.ndarray], output_name: object) -> np.ndarray:
    if not outputs:
        raise ModelRuntimeError("YOLO pose model returned no outputs")
    if output_name is not None:
        name = str(output_name)
        if name not in outputs:
            raise ModelRuntimeError(
                "Configured YOLO pose output was not returned",
                details={"output_name": name, "available_outputs": sorted(outputs)},
            )
        return outputs[name]
    if len(outputs) != 1:
        raise ModelRuntimeError(
            "YOLO pose model returned multiple outputs; configure output_name",
            details={"available_outputs": sorted(outputs)},
        )
    return next(iter(outputs.values()))


class YoloPoseOnnxAdapter(OnnxRuntimeAdapter):
    """Post-process common Ultralytics YOLO pose ONNX tensors.

    Bounding boxes are emitted as rectangle annotations. Until the shared result
    contract gains a grouped keypoint type, visible keypoints are emitted directly
    after their box as point annotations.
    """

    def _annotations(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        parameters: dict[str, object],
    ) -> list[AnnotationResult]:
        config = self.record.config
        raw_classes = config.get("classes", config.get("names"))
        classes = _class_names(raw_classes)
        if not classes:
            raise ModelRuntimeError("YOLO pose configuration must define at least one class")
        keypoint_shape = _configured_keypoint_shape(config)
        requested_format = str(parameters.get("output_format", config.get("output_format", "auto"))).casefold()
        output = _select_output(outputs, parameters.get("output_name", config.get("output_name")))
        layout = _resolve_layout(
            output,
            class_count=len(classes),
            keypoint_shape=keypoint_shape,
            requested_format=requested_format,
        )
        prediction = layout.prediction
        confidence = _float_parameter(parameters, config, "conf_threshold", 0.25)
        iou_threshold = _float_parameter(parameters, config, "iou_threshold", 0.45)
        keypoint_threshold = _float_parameter(
            parameters,
            config,
            "keypoint_threshold",
            config.get("kpt_threshold", 0.25),
        )

        if layout.output_layout == "end_to_end":
            boxes = prediction[:, :4].copy()
            scores = prediction[:, 4].copy()
            raw_class_ids = prediction[:, 5]
            if not np.allclose(raw_class_ids, np.round(raw_class_ids), atol=1e-4):
                raise ModelRuntimeError("YOLO pose end-to-end class IDs must be integers")
            class_ids = np.round(raw_class_ids).astype(np.int64)
            keypoints = prediction[:, 6:]
        else:
            boxes = _xywh_to_xyxy(prediction[:, :4])
            class_start = 5 if layout.output_layout == "raw_objectness" else 4
            class_scores = prediction[:, class_start : class_start + len(classes)]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            if layout.output_layout == "raw_objectness":
                scores = scores * prediction[:, 4]
            keypoints = prediction[:, class_start + len(classes) :]

        expected_keypoint_width = layout.keypoint_count * layout.keypoint_dimensions
        if keypoints.shape[1] != expected_keypoint_width:
            raise ModelRuntimeError(
                "YOLO pose keypoint tensor width is inconsistent",
                details={"actual": keypoints.shape[1], "expected": expected_keypoint_width},
            )
        keypoints = keypoints.reshape(-1, layout.keypoint_count, layout.keypoint_dimensions)
        mask = scores >= confidence
        boxes, scores, class_ids, keypoints = boxes[mask], scores[mask], class_ids[mask], keypoints[mask]
        if boxes.size == 0:
            return []
        if np.any(class_ids < 0) or np.any(class_ids >= len(classes)):
            raise ModelRuntimeError(
                "YOLO pose output contains an unknown class ID",
                details={"class_ids": sorted({int(item) for item in class_ids})},
            )
        keep = _nms(boxes, scores, class_ids, iou_threshold)
        boxes, scores, class_ids, keypoints = boxes[keep], scores[keep], class_ids[keep], keypoints[keep]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_x) / transform.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_y) / transform.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, transform.original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, transform.original_height)
        keypoints[:, :, 0] = ((keypoints[:, :, 0] - transform.pad_x) / transform.scale).clip(
            0, transform.original_width
        )
        keypoints[:, :, 1] = ((keypoints[:, :, 1] - transform.pad_y) / transform.scale).clip(
            0, transform.original_height
        )

        keypoint_names = _names(
            config.get("keypoint_names", config.get("kpt_names")),
            field="keypoint_names",
        )
        if keypoint_names and len(keypoint_names) != layout.keypoint_count:
            raise ModelRuntimeError(
                "YOLO pose keypoint_names length does not match kpt_shape",
                details={"name_count": len(keypoint_names), "keypoint_count": layout.keypoint_count},
            )
        if not keypoint_names:
            keypoint_names = [f"keypoint_{index}" for index in range(layout.keypoint_count)]
        per_class_keypoint_names = _class_keypoint_names(raw_classes)
        for class_name, names in per_class_keypoint_names.items():
            if len(names) != layout.keypoint_count:
                raise ModelRuntimeError(
                    "YOLO pose class keypoint count does not match output",
                    details={"class": class_name, "name_count": len(names), "keypoint_count": layout.keypoint_count},
                )

        results: list[AnnotationResult] = []
        for box, score, class_id, pose in zip(boxes, scores, class_ids, keypoints, strict=True):
            class_label = classes[int(class_id)]
            active_keypoint_names = per_class_keypoint_names.get(class_label, keypoint_names)
            results.append(
                AnnotationResult(
                    label=class_label,
                    score=float(score),
                    shape_type="rectangle",
                    points=[[float(box[0]), float(box[1])], [float(box[2]), float(box[3])]],
                )
            )
            for name, keypoint in zip(active_keypoint_names, pose, strict=True):
                keypoint_score = float(keypoint[2]) if layout.keypoint_dimensions == 3 else float(score)
                if keypoint_score < keypoint_threshold:
                    continue
                results.append(
                    AnnotationResult(
                        label=f"{class_label}:{name}",
                        score=keypoint_score,
                        shape_type="point",
                        points=[[float(keypoint[0]), float(keypoint[1])]],
                    )
                )
        return results
