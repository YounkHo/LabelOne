from __future__ import annotations

from typing import Any

import numpy as np

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult, ClassificationResult
from .onnx import OnnxRuntimeAdapter, _ImageTransform


def _classes(value: object) -> list[str]:
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
    raise ModelRuntimeError("YOLO classification classes must be a list or mapping")


def _select_scores(outputs: dict[str, np.ndarray], output_name: object) -> np.ndarray:
    if not outputs:
        raise ModelRuntimeError("YOLO classification model returned no outputs")
    if output_name is not None:
        name = str(output_name)
        if name not in outputs:
            raise ModelRuntimeError(
                "Configured YOLO classification output was not returned",
                details={"output_name": name, "available_outputs": sorted(outputs)},
            )
        candidates = [(name, outputs[name])]
    else:
        candidates = list(outputs.items())

    valid: list[tuple[str, np.ndarray]] = []
    rejected: dict[str, list[int]] = {}
    for name, value in candidates:
        scores = np.asarray(value)
        if scores.ndim == 1:
            vector = scores
        elif scores.ndim == 2 and scores.shape[0] == 1:
            vector = scores[0]
        else:
            rejected[name] = list(scores.shape)
            continue
        if vector.size == 0:
            rejected[name] = list(scores.shape)
            continue
        valid.append((name, vector))

    if not valid:
        raise ModelRuntimeError(
            "YOLO classification output must have shape [classes] or [1, classes]",
            details={"outputs": rejected},
        )
    if len(valid) > 1:
        raise ModelRuntimeError(
            "YOLO classification model returned multiple score vectors; configure output_name",
            details={"candidate_outputs": [name for name, _ in valid]},
        )
    scores = valid[0][1]
    if not np.issubdtype(scores.dtype, np.number) or not np.all(np.isfinite(scores)):
        raise ModelRuntimeError("YOLO classification output must contain only finite numeric values")
    return scores.astype(np.float64, copy=False)


def _probabilities(scores: np.ndarray, apply_softmax: object, temperature: object) -> np.ndarray:
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(
            "YOLO classification temperature must be a positive number",
            details={"temperature": temperature},
        ) from exc
    if not np.isfinite(temperature_value) or temperature_value <= 0:
        raise ModelRuntimeError(
            "YOLO classification temperature must be a positive number",
            details={"temperature": temperature},
        )

    if isinstance(apply_softmax, bool):
        should_apply = apply_softmax
    elif isinstance(apply_softmax, str) and apply_softmax.casefold() == "auto":
        total = float(scores.sum())
        should_apply = not (
            np.all(scores >= 0)
            and np.all(scores <= 1)
            and np.isclose(total, 1.0, atol=1e-3, rtol=1e-3)
        )
    else:
        raise ModelRuntimeError(
            "YOLO classification apply_softmax must be true, false, or auto",
            details={"apply_softmax": apply_softmax},
        )

    if should_apply:
        scaled = scores / temperature_value
        shifted = scaled - np.max(scaled)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(), 1.0, atol=1e-3, rtol=1e-3):
        raise ModelRuntimeError(
            "YOLO classification scores are not probabilities; enable softmax or use auto"
        )
    return scores


class YoloClassificationOnnxAdapter(OnnxRuntimeAdapter):
    """Post-process single-label Ultralytics YOLO classification outputs."""

    def _annotations(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        parameters: dict[str, object],
    ) -> list[AnnotationResult]:
        del outputs, transform, parameters
        return []

    def _classifications(
        self,
        outputs: dict[str, np.ndarray],
        parameters: dict[str, object],
    ) -> list[ClassificationResult]:
        config: dict[str, Any] = self.record.config
        scores = _select_scores(outputs, parameters.get("output_name", config.get("output_name")))
        labels = _classes(config.get("classes", config.get("names")))
        if labels and len(labels) != scores.size:
            raise ModelRuntimeError(
                "YOLO classification class count does not match output width",
                details={"class_count": len(labels), "output_width": int(scores.size)},
            )
        if not labels:
            labels = [str(index) for index in range(scores.size)]

        raw_top_k = parameters.get("top_k", config.get("top_k", 5))
        if isinstance(raw_top_k, bool):
            raise ModelRuntimeError("YOLO classification top_k must be a positive integer")
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError(
                "YOLO classification top_k must be a positive integer",
                details={"top_k": raw_top_k},
            ) from exc
        if isinstance(raw_top_k, float) and not raw_top_k.is_integer():
            raise ModelRuntimeError(
                "YOLO classification top_k must be a positive integer",
                details={"top_k": raw_top_k},
            )
        if top_k <= 0:
            raise ModelRuntimeError(
                "YOLO classification top_k must be a positive integer",
                details={"top_k": raw_top_k},
            )
        top_k = min(top_k, scores.size)
        probabilities = _probabilities(
            scores,
            parameters.get("apply_softmax", config.get("apply_softmax", "auto")),
            parameters.get("temperature", config.get("temperature", 1.0)),
        )
        ranking = np.argsort(-probabilities, kind="stable")[:top_k]

        return [
            ClassificationResult(
                label=labels[int(class_id)],
                score=float(probabilities[class_id]),
                rank=rank,
            )
            for rank, class_id in enumerate(ranking, start=1)
        ]
