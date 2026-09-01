from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult
from .onnx import OnnxRuntimeAdapter, _ImageTransform


_OutputMode = Literal["explicit", "logits"]

_OUTPUT_ALIASES: dict[str, set[str]] = {
    "boxes": {"boxes", "pred_boxes", "bbox", "bboxes"},
    "labels": {"labels", "pred_labels", "classes", "class_ids"},
    "scores": {"scores", "pred_scores", "confidence", "confidences"},
    "logits": {"logits", "pred_logits", "class_logits"},
}


@dataclass(frozen=True, slots=True)
class _ResolvedOutputs:
    mode: _OutputMode
    boxes: np.ndarray
    labels: np.ndarray | None = None
    scores: np.ndarray | None = None
    logits: np.ndarray | None = None


def _names(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        def key_order(item: tuple[object, object]) -> tuple[int, object]:
            try:
                return (0, int(str(item[0])))
            except ValueError:
                return (1, str(item[0]))

        return [str(label) for _, label in sorted(value.items(), key=key_order)]
    if isinstance(value, (list, tuple)):
        return [str(label) for label in value]
    raise ModelRuntimeError("DETR classes must be a list or mapping")


def _canonical_output_name(name: str) -> str:
    return name.rsplit("/", 1)[-1].split(":", 1)[0].casefold()


def _configured_output_name(
    role: str,
    parameters: dict[str, object],
    config: dict[str, Any],
) -> object:
    direct_key = f"{role}_output"
    if direct_key in parameters:
        return parameters[direct_key]
    if direct_key in config:
        return config[direct_key]
    output_names = config.get("output_names")
    if isinstance(output_names, dict):
        return output_names.get(role)
    return None


def _resolve_output_name(
    outputs: dict[str, np.ndarray],
    role: str,
    configured: object,
) -> str | None:
    if configured is not None:
        requested = str(configured)
        if requested not in outputs:
            raise ModelRuntimeError(
                f"Configured DETR {role} output was not returned",
                details={"output_name": requested, "available_outputs": sorted(outputs)},
            )
        return requested
    matches = [
        name
        for name in outputs
        if _canonical_output_name(name) in _OUTPUT_ALIASES[role]
    ]
    if len(matches) > 1:
        raise ModelRuntimeError(
            f"Multiple DETR outputs match {role}; configure {role}_output",
            details={"matches": sorted(matches)},
        )
    return matches[0] if matches else None


def _boxes(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    original_shape = list(array.shape)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ModelRuntimeError(
                "DETR boxes output must have a single batch",
                details={"shape": original_shape},
            )
        array = array[0]
    elif array.ndim == 1 and array.size == 4:
        array = array.reshape(1, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ModelRuntimeError(
            "DETR boxes output must have shape [N,4] or [1,N,4]",
            details={"shape": original_shape},
        )
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ModelRuntimeError("DETR boxes output must contain only finite numeric values")
    return array.astype(np.float64, copy=False)


def _vector(value: np.ndarray, *, role: str, expected: int) -> np.ndarray:
    array = np.asarray(value)
    original_shape = list(array.shape)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != expected:
        raise ModelRuntimeError(
            f"DETR {role} output must contain one value per box",
            details={"shape": original_shape, "box_count": expected},
        )
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ModelRuntimeError(f"DETR {role} output must contain only finite numeric values")
    return array.astype(np.float64, copy=False)


def _logits(value: np.ndarray, *, box_count: int) -> np.ndarray:
    array = np.asarray(value)
    original_shape = list(array.shape)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ModelRuntimeError(
                "DETR logits output must have a single batch",
                details={"shape": original_shape},
            )
        array = array[0]
    if array.ndim != 2:
        raise ModelRuntimeError(
            "DETR logits output must have shape [N,C] or [1,N,C]",
            details={"shape": original_shape},
        )
    if array.shape[0] != box_count:
        if array.shape[1] == box_count and array.shape[0] != box_count:
            array = array.T
        else:
            raise ModelRuntimeError(
                "DETR logits query count does not match boxes",
                details={"shape": original_shape, "box_count": box_count},
            )
    if array.shape[1] <= 0:
        raise ModelRuntimeError("DETR logits output must contain at least one class")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ModelRuntimeError("DETR logits output must contain only finite numeric values")
    return array.astype(np.float64, copy=False)


def _resolve_outputs(
    outputs: dict[str, np.ndarray],
    parameters: dict[str, object],
    config: dict[str, Any],
) -> _ResolvedOutputs:
    if not outputs:
        raise ModelRuntimeError("DETR model returned no outputs")
    resolved_names = {
        role: _resolve_output_name(
            outputs,
            role,
            _configured_output_name(role, parameters, config),
        )
        for role in ("boxes", "labels", "scores", "logits")
    }
    boxes_name = resolved_names["boxes"]
    if boxes_name is None:
        raise ModelRuntimeError(
            "Could not identify the DETR boxes output; configure boxes_output",
            details={"available_outputs": sorted(outputs)},
        )
    parsed_boxes = _boxes(outputs[boxes_name])
    labels_name = resolved_names["labels"]
    scores_name = resolved_names["scores"]
    logits_name = resolved_names["logits"]
    has_explicit_piece = labels_name is not None or scores_name is not None
    if has_explicit_piece:
        if labels_name is None or scores_name is None:
            raise ModelRuntimeError(
                "DETR postprocessed output requires both labels and scores",
                details={"labels_output": labels_name, "scores_output": scores_name},
            )
        return _ResolvedOutputs(
            mode="explicit",
            boxes=parsed_boxes,
            labels=_vector(outputs[labels_name], role="labels", expected=len(parsed_boxes)),
            scores=_vector(outputs[scores_name], role="scores", expected=len(parsed_boxes)),
        )
    if logits_name is not None:
        return _ResolvedOutputs(
            mode="logits",
            boxes=parsed_boxes,
            logits=_logits(outputs[logits_name], box_count=len(parsed_boxes)),
        )
    raise ModelRuntimeError(
        "Could not identify DETR labels/scores or logits outputs; configure output names",
        details={"available_outputs": sorted(outputs)},
    )


def _unit_interval_parameter(
    parameters: dict[str, object],
    config: dict[str, Any],
    name: str,
    default: float,
) -> float:
    raw = parameters.get(name, config.get(name, default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"DETR {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ModelRuntimeError(
            f"DETR {name} must be between zero and one",
            details={"value": raw},
        )
    return value


def _top_k(parameters: dict[str, object], config: dict[str, Any], available: int) -> int:
    raw = parameters.get("top_k", config.get("top_k", config.get("max_det", 300)))
    if isinstance(raw, bool):
        raise ModelRuntimeError("DETR top_k must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError("DETR top_k must be a positive integer", details={"value": raw}) from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise ModelRuntimeError("DETR top_k must be a positive integer", details={"value": raw})
    if value <= 0:
        raise ModelRuntimeError("DETR top_k must be a positive integer", details={"value": raw})
    return min(value, available)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (1.0 + exponentials)
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _explicit_probabilities(scores: np.ndarray, activation: str) -> np.ndarray:
    if activation == "auto":
        activation = "none" if np.all((scores >= 0) & (scores <= 1)) else "sigmoid"
    if activation == "sigmoid":
        return _sigmoid(scores)
    if activation != "none":
        raise ModelRuntimeError(
            "DETR score_activation must be auto, none, or sigmoid",
            details={"score_activation": activation},
        )
    if np.any(scores < 0) or np.any(scores > 1):
        raise ModelRuntimeError("DETR explicit scores must be probabilities or use sigmoid activation")
    return scores


def _background_index(raw: object, class_count: int) -> int | None:
    if raw is None or (isinstance(raw, str) and raw.casefold() == "none"):
        return None
    if isinstance(raw, str) and raw.casefold() == "last":
        return class_count - 1
    if isinstance(raw, bool):
        raise ModelRuntimeError("DETR background_class must be an index, last, or none")
    try:
        index = int(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError("DETR background_class must be an index, last, or none") from exc
    if index < 0:
        index += class_count
    if index < 0 or index >= class_count:
        raise ModelRuntimeError(
            "DETR background_class is outside the logits width",
            details={"background_class": raw, "class_count": class_count},
        )
    return index


def _logit_probabilities(
    logits: np.ndarray,
    *,
    activation: str,
    configured_background: object,
    label_count: int,
) -> tuple[np.ndarray, int | None, str]:
    class_count = logits.shape[1]
    background = _background_index(configured_background, class_count)
    if configured_background is None and label_count == class_count - 1:
        background = class_count - 1

    if activation == "auto":
        already_probabilities = np.all((logits >= 0) & (logits <= 1))
        row_sums = logits.sum(axis=1)
        if background is not None or (already_probabilities and np.allclose(row_sums, 1.0, atol=1e-3, rtol=1e-3)):
            activation = "softmax_probabilities" if already_probabilities else "softmax"
        else:
            activation = "sigmoid_probabilities" if already_probabilities else "sigmoid"

    if activation == "sigmoid":
        probabilities = _sigmoid(logits)
        selection_default = "global_topk"
    elif activation == "softmax":
        probabilities = _softmax(logits)
        selection_default = "query_best"
        if configured_background is None and background is None and label_count == 0:
            background = class_count - 1
    elif activation == "none" or activation == "sigmoid_probabilities":
        if np.any(logits < 0) or np.any(logits > 1):
            raise ModelRuntimeError("DETR logits are not probabilities; configure logits_activation")
        probabilities = logits
        selection_default = "global_topk"
    elif activation == "softmax_probabilities":
        if np.any(logits < 0) or np.any(logits > 1) or not np.allclose(
            logits.sum(axis=1), 1.0, atol=1e-3, rtol=1e-3
        ):
            raise ModelRuntimeError("DETR softmax probabilities must sum to one per query")
        probabilities = logits
        selection_default = "query_best"
    else:
        raise ModelRuntimeError(
            "DETR logits_activation must be auto, none, sigmoid, or softmax",
            details={"logits_activation": activation},
        )
    return probabilities, background, selection_default


def _class_label(
    class_id: int,
    labels: list[str],
    *,
    output_class_count: int | None = None,
    background: int | None = None,
) -> str:
    if not labels:
        return str(class_id)
    if output_class_count is None or len(labels) == output_class_count:
        if class_id < 0 or class_id >= len(labels):
            raise ModelRuntimeError(
                "DETR output contains an unknown class ID",
                details={"class_id": class_id, "class_count": len(labels)},
            )
        return labels[class_id]
    if background is not None and len(labels) == output_class_count - 1:
        foreground_ids = [index for index in range(output_class_count) if index != background]
        try:
            return labels[foreground_ids.index(class_id)]
        except ValueError as exc:
            raise ModelRuntimeError("DETR background class cannot be emitted as a detection") from exc
    raise ModelRuntimeError(
        "DETR class count does not match logits width",
        details={"class_count": len(labels), "logits_width": output_class_count},
    )


def _coordinate_options(
    parameters: dict[str, object],
    config: dict[str, Any],
    mode: _OutputMode,
) -> tuple[str, str]:
    default_format = "xyxy" if mode == "explicit" else "cxcywh"
    default_space = "original" if mode == "explicit" else "normalized_model"
    box_format = str(parameters.get("box_format", config.get("box_format", default_format))).casefold()
    coordinate_space = str(
        parameters.get("coordinate_space", config.get("coordinate_space", default_space))
    ).casefold()
    format_aliases = {"center": "cxcywh", "corners": "xyxy"}
    space_aliases = {"normalized": "normalized_model", "input": "model"}
    box_format = format_aliases.get(box_format, box_format)
    coordinate_space = space_aliases.get(coordinate_space, coordinate_space)
    if box_format not in {"cxcywh", "xyxy"}:
        raise ModelRuntimeError(
            "DETR box_format must be cxcywh or xyxy",
            details={"box_format": box_format},
        )
    if coordinate_space not in {"normalized_model", "normalized_original", "model", "original"}:
        raise ModelRuntimeError(
            "Unsupported DETR coordinate_space",
            details={"coordinate_space": coordinate_space},
        )
    return box_format, coordinate_space


def _restore_boxes(
    boxes: np.ndarray,
    *,
    box_format: str,
    coordinate_space: str,
    transform: _ImageTransform,
) -> np.ndarray:
    restored = boxes.copy()
    if box_format == "cxcywh":
        centers = restored[:, :2].copy()
        sizes = restored[:, 2:4].copy()
        restored[:, :2] = centers - sizes / 2
        restored[:, 2:4] = centers + sizes / 2

    if coordinate_space == "normalized_model":
        restored[:, [0, 2]] *= transform.input_width
        restored[:, [1, 3]] *= transform.input_height
        coordinate_space = "model"
    elif coordinate_space == "normalized_original":
        restored[:, [0, 2]] *= transform.original_width
        restored[:, [1, 3]] *= transform.original_height
        coordinate_space = "original"

    if coordinate_space == "model":
        restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.pad_x) / transform.scale
        restored[:, [1, 3]] = (restored[:, [1, 3]] - transform.pad_y) / transform.scale
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, transform.original_width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, transform.original_height)
    return restored


class DetrDetectionOnnxAdapter(OnnxRuntimeAdapter):
    """Clean-room post-processing for common RT-DETR, D-FINE, and DEIM outputs."""

    def __init__(self, record, artifact_store) -> None:
        super().__init__(record, artifact_store)
        self.size_input_meta = None

    def _configure_inputs(self, inputs: list[Any]) -> None:
        image_inputs = [item for item in inputs if len(list(item.shape)) == 4]
        auxiliary = [item for item in inputs if item not in image_inputs]
        if len(image_inputs) != 1 or len(auxiliary) > 1:
            raise ModelRuntimeError(
                "DETR ONNX model requires one image input and at most one size input",
                details={"inputs": {item.name: list(item.shape) for item in inputs}},
            )
        if auxiliary:
            shape = list(auxiliary[0].shape)
            if len(shape) != 2 or (isinstance(shape[-1], int) and shape[-1] != 2):
                raise ModelRuntimeError(
                    "DETR auxiliary input must contain an image-size pair",
                    details={"name": auxiliary[0].name, "shape": shape},
                )
            self.size_input_meta = auxiliary[0]
        self.input_meta = image_inputs[0]

    def unload(self) -> None:
        super().unload()
        self.size_input_meta = None

    def _image_resize_mode(self) -> str:
        model_type = self.record.descriptor.model_type.casefold()
        return "stretch" if model_type in {"rtdetr", "rtdetrv2", "rt_detr", "rt_detrv2"} else "letterbox"

    def _input_feed(self, tensor: np.ndarray, transform: _ImageTransform) -> dict[str, np.ndarray]:
        feed = super()._input_feed(tensor, transform)
        if self.size_input_meta is None:
            return feed
        model_type = self.record.descriptor.model_type.casefold()
        default_space = "original_wh" if model_type in {"rtdetr", "rtdetrv2", "rt_detr", "rt_detrv2"} else "model_hw"
        size_space = str(self.record.config.get("size_input_space", default_space)).casefold()
        values = {
            "original_wh": [transform.original_width, transform.original_height],
            "original_hw": [transform.original_height, transform.original_width],
            "model_wh": [transform.input_width, transform.input_height],
            "model_hw": [transform.input_height, transform.input_width],
        }
        if size_space not in values:
            raise ModelRuntimeError(
                "Unsupported DETR size_input_space",
                details={"size_input_space": size_space},
            )
        dtype = np.int64 if "int64" in str(self.size_input_meta.type) else np.int32 if "int32" in str(self.size_input_meta.type) else np.float32
        feed[self.size_input_meta.name] = np.asarray([values[size_space]], dtype=dtype)
        return feed

    def _annotations(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        parameters: dict[str, object],
    ) -> list[AnnotationResult]:
        config = self.record.config
        resolved = _resolve_outputs(outputs, parameters, config)
        labels = _names(config.get("classes", config.get("names")))
        confidence = _unit_interval_parameter(parameters, config, "conf_threshold", 0.25)
        effective_config = dict(config)
        if "coordinate_space" not in effective_config:
            model_type = self.record.descriptor.model_type.casefold()
            if model_type in {"dfine", "d_fine", "deim", "deimv2"} and resolved.mode == "explicit":
                effective_config["coordinate_space"] = "model"
        box_format, coordinate_space = _coordinate_options(parameters, effective_config, resolved.mode)
        restored_boxes = _restore_boxes(
            resolved.boxes,
            box_format=box_format,
            coordinate_space=coordinate_space,
            transform=transform,
        )

        if resolved.mode == "explicit":
            assert resolved.labels is not None and resolved.scores is not None
            raw_labels = resolved.labels
            if not np.allclose(raw_labels, np.round(raw_labels), atol=1e-4):
                raise ModelRuntimeError("DETR explicit labels must be integer class IDs")
            class_ids = np.round(raw_labels).astype(np.int64)
            if np.any(class_ids < 0) or (labels and np.any(class_ids >= len(labels))):
                raise ModelRuntimeError(
                    "DETR output contains an unknown class ID",
                    details={"class_ids": sorted({int(class_id) for class_id in class_ids})},
                )
            activation = str(
                parameters.get("score_activation", config.get("score_activation", "auto"))
            ).casefold()
            scores = _explicit_probabilities(resolved.scores, activation)
            eligible = np.flatnonzero(scores >= confidence)
            order = eligible[np.argsort(-scores[eligible], kind="stable")]
            keep = order[:_top_k(parameters, config, len(order))]
            selected_boxes = keep
            selected_class_ids = class_ids[keep]
            selected_scores = scores[keep]
            output_class_count = None
            background = None
        else:
            assert resolved.logits is not None
            activation = str(
                parameters.get("logits_activation", config.get("logits_activation", "auto"))
            ).casefold()
            configured_background = parameters.get(
                "background_class",
                config.get("background_class"),
            )
            probabilities, background, default_selection = _logit_probabilities(
                resolved.logits,
                activation=activation,
                configured_background=configured_background,
                label_count=len(labels),
            )
            if labels and len(labels) not in {
                probabilities.shape[1],
                probabilities.shape[1] - (1 if background is not None else 0),
            }:
                raise ModelRuntimeError(
                    "DETR class count does not match logits width",
                    details={"class_count": len(labels), "logits_width": probabilities.shape[1]},
                )
            if background is not None:
                probabilities = probabilities.copy()
                probabilities[:, background] = -np.inf
            selection_mode = str(
                parameters.get("selection_mode", config.get("selection_mode", "auto"))
            ).casefold()
            if selection_mode == "auto":
                selection_mode = default_selection
            if selection_mode == "query_best":
                class_ids = np.argmax(probabilities, axis=1).astype(np.int64)
                scores = probabilities[np.arange(probabilities.shape[0]), class_ids]
                eligible = np.flatnonzero(scores >= confidence)
                order = eligible[np.argsort(-scores[eligible], kind="stable")]
                keep = order[:_top_k(parameters, config, len(order))]
                selected_boxes = keep
                selected_class_ids = class_ids[keep]
                selected_scores = scores[keep]
            elif selection_mode == "global_topk":
                flat_scores = probabilities.reshape(-1)
                eligible = np.flatnonzero(flat_scores >= confidence)
                order = eligible[np.argsort(-flat_scores[eligible], kind="stable")]
                selected = order[:_top_k(parameters, config, len(order))]
                selected_boxes = selected // probabilities.shape[1]
                selected_class_ids = selected % probabilities.shape[1]
                selected_scores = flat_scores[selected]
            else:
                raise ModelRuntimeError(
                    "DETR selection_mode must be auto, query_best, or global_topk",
                    details={"selection_mode": selection_mode},
                )
            output_class_count = probabilities.shape[1]

        results: list[AnnotationResult] = []
        for box_index, class_id, score in zip(
            selected_boxes,
            selected_class_ids,
            selected_scores,
            strict=True,
        ):
            box = restored_boxes[int(box_index)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results.append(
                AnnotationResult(
                    label=_class_label(
                        int(class_id),
                        labels,
                        output_class_count=output_class_count,
                        background=background,
                    ),
                    score=float(score),
                    shape_type="rectangle",
                    points=[[float(box[0]), float(box[1])], [float(box[2]), float(box[3])]],
                )
            )
        return results
