from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from labelone.errors import ModelRuntimeError

from ..types import ClassificationResult
from .onnx import OnnxRuntimeAdapter, _ImageTransform


_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)
_MAX_TAG_FILE_BYTES = 2 * 1024 * 1024
_MAX_TAGS = 100_000


def _number(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ModelRuntimeError(f"RAM {name} must be a number between {minimum} and {maximum}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(
            f"RAM {name} must be a number between {minimum} and {maximum}",
            details={name: value},
        ) from exc
    if not np.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ModelRuntimeError(
            f"RAM {name} must be a number between {minimum} and {maximum}",
            details={name: value},
        )
    return parsed


def _normalization_vector(config: dict[str, Any], name: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = config.get(name, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ModelRuntimeError(f"RAM {name} must contain exactly three numbers", details={name: raw})
    try:
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"RAM {name} must contain exactly three numbers", details={name: raw}) from exc
    if not np.all(np.isfinite(values)) or (name == "std" and np.any(values <= 0)):
        raise ModelRuntimeError(f"RAM {name} contains invalid values", details={name: list(raw)})
    return values


def _source_root(config_path: Path) -> Path | None:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "services" / "auto_labeling" / "configs" / "ram").is_dir():
            return parent.resolve()
    return None


def _safe_tag_path(config_path: Path, raw: object, *, source_root: Path | None) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ModelRuntimeError("RAM tag list path must be a non-empty string")
    configured = Path(raw).expanduser()
    candidate = (configured if configured.is_absolute() else config_path.parent / configured).resolve()
    allowed_roots = [config_path.parent.resolve()]
    if source_root is not None:
        allowed_roots.append(source_root)
    if not any(candidate == root or candidate.is_relative_to(root) for root in allowed_roots):
        raise ModelRuntimeError(
            "RAM tag list path escapes the imported model source",
            details={"path": str(candidate), "allowed_roots": [str(root) for root in allowed_roots]},
        )
    return candidate


def _validate_labels(raw: object, *, source: str) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        raise ModelRuntimeError("RAM tag list must be a list of strings", details={"source": source})
    if not raw or len(raw) > _MAX_TAGS:
        raise ModelRuntimeError(
            "RAM tag list size is invalid",
            details={"source": source, "tag_count": len(raw), "maximum": _MAX_TAGS},
        )
    labels: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str):
            raise ModelRuntimeError(
                "RAM tag list must contain only strings",
                details={"source": source, "index": index},
            )
        label = value.strip()
        if not label or "\x00" in label or len(label) > 512:
            raise ModelRuntimeError(
                "RAM tag list contains an invalid label",
                details={"source": source, "index": index},
            )
        labels.append(label)
    return labels


def _read_tag_file(path: Path) -> list[str]:
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size > _MAX_TAG_FILE_BYTES:
            raise ModelRuntimeError(
                "RAM tag list file is too large",
                details={"path": str(path), "size_bytes": size, "maximum": _MAX_TAG_FILE_BYTES},
            )
        lines = path.read_text(encoding="utf-8").splitlines()
    except ModelRuntimeError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ModelRuntimeError(
            "Could not read RAM tag list",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    return _validate_labels(lines, source=str(path))


def _tag_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("tag_mode", "en") or "en").casefold()
    if mode not in {"en", "zh"}:
        raise ModelRuntimeError("RAM tag_mode must be en or zh", details={"tag_mode": mode})
    return mode


def _labels(config: dict[str, Any], config_path: Path) -> list[str]:
    mode = _tag_mode(config)
    localized_key = "tag_list_chinese" if mode == "zh" else "tag_list"
    localized_path_key = "tag_list_chinese_path" if mode == "zh" else "tag_list_path"
    source_root = _source_root(config_path)

    inline = config.get(localized_key)
    if isinstance(inline, (list, tuple)):
        return _validate_labels(inline, source=localized_key)
    configured_path = config.get(localized_path_key, inline if isinstance(inline, str) else None)
    if configured_path is not None:
        return _read_tag_file(_safe_tag_path(config_path, configured_path, source_root=source_root))

    generic = config.get("tags", config.get("classes"))
    if generic is not None:
        return _validate_labels(generic, source="tags/classes")

    if source_root is not None:
        filename = "ram_tag_list_chinese.txt" if mode == "zh" else "ram_tag_list.txt"
        fallback = source_root / "services" / "auto_labeling" / "configs" / "ram" / filename
        return _read_tag_file(fallback.resolve())
    raise ModelRuntimeError(
        "RAM tag list is not configured and no X-AnyLabeling tag resource was found",
        details={"config_path": str(config_path), "expected_keys": [localized_key, localized_path_key, "tags", "classes"]},
    )


def _score_vector(outputs: dict[str, np.ndarray], output_name: object, tag_count: int) -> np.ndarray:
    if not outputs:
        raise ModelRuntimeError("RAM model returned no outputs")
    if output_name is not None:
        name = str(output_name)
        if name not in outputs:
            raise ModelRuntimeError(
                "Configured RAM output was not returned",
                details={"output_name": name, "available_outputs": sorted(outputs)},
            )
        candidates = [(name, outputs[name])]
    else:
        candidates = list(outputs.items())

    compatible: list[tuple[str, np.ndarray]] = []
    rejected: dict[str, list[int]] = {}
    for name, value in candidates:
        scores = np.asarray(value)
        if scores.ndim == 1 and scores.size == tag_count:
            vector = scores
        elif scores.ndim == 2 and scores.shape[1] == tag_count:
            if scores.shape[0] != 1:
                raise ModelRuntimeError(
                    "RAM output batch dimension must be one for single-image inference",
                    details={"output_name": name, "shape": list(scores.shape)},
                )
            vector = scores[0]
        else:
            rejected[name] = list(scores.shape)
            continue
        numeric_or_boolean = np.issubdtype(vector.dtype, np.number) or np.issubdtype(vector.dtype, np.bool_)
        if not numeric_or_boolean or not np.all(np.isfinite(vector)):
            raise ModelRuntimeError(
                "RAM score output must contain only finite numeric values",
                details={"output_name": name},
            )
        compatible.append((name, vector.astype(np.float64, copy=False)))

    if not compatible:
        raise ModelRuntimeError(
            "RAM score output must have shape [tags] or [1, tags]",
            details={"tag_count": tag_count, "outputs": rejected},
        )
    if len(compatible) > 1:
        raise ModelRuntimeError(
            "RAM model returned multiple tag score vectors; configure output_name",
            details={"candidate_outputs": [name for name, _ in compatible]},
        )
    return compatible[0][1]


def _validate_batch_output(outputs: dict[str, np.ndarray], configured_name: object) -> None:
    if configured_name is not None:
        name = str(configured_name)
        if name not in outputs:
            raise ModelRuntimeError(
                "Configured RAM batch output was not returned",
                details={"batch_output_name": name, "available_outputs": sorted(outputs)},
            )
        candidates = [(name, outputs[name])]
    else:
        candidates = [(name, value) for name, value in outputs.items() if name.casefold() in {"bs", "batch", "batch_size"}]
    for name, value in candidates:
        batch = np.asarray(value)
        if batch.size != 1 or not np.issubdtype(batch.dtype, np.number) or not np.all(np.isfinite(batch)):
            raise ModelRuntimeError(
                "RAM batch output must contain one finite integer",
                details={"output_name": name, "shape": list(batch.shape)},
            )
        count = float(batch.reshape(-1)[0])
        if not count.is_integer() or int(count) != 1:
            raise ModelRuntimeError(
                "RAM batch output must equal one for single-image inference",
                details={"output_name": name, "value": count},
            )


def _probabilities(scores: np.ndarray, raw_activation: object) -> np.ndarray:
    if isinstance(raw_activation, bool):
        activation = "sigmoid" if raw_activation else "probability"
    else:
        activation = str(raw_activation or "auto").casefold()
    aliases = {"none": "probability", "probabilities": "probability", "logits": "sigmoid"}
    activation = aliases.get(activation, activation)
    if activation == "auto":
        activation = "probability" if np.all((scores >= 0) & (scores <= 1)) else "sigmoid"
    if activation == "sigmoid":
        result = np.empty_like(scores, dtype=np.float64)
        positive = scores >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exponentials = np.exp(scores[~positive])
        result[~positive] = exponentials / (1.0 + exponentials)
        return result
    if activation != "probability":
        raise ModelRuntimeError(
            "RAM score_activation must be auto, probability, or sigmoid",
            details={"score_activation": raw_activation},
        )
    if np.any(scores < 0) or np.any(scores > 1):
        raise ModelRuntimeError("RAM probability output contains values outside zero and one")
    return scores


def _index_list(raw: object, *, tag_count: int) -> set[int]:
    if raw is None:
        return set()
    if not isinstance(raw, (list, tuple)):
        raise ModelRuntimeError("RAM delete_tag_index must be a list of integers")
    indices: set[int] = set()
    for value in raw:
        if isinstance(value, bool):
            raise ModelRuntimeError("RAM delete_tag_index must contain only integers")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError("RAM delete_tag_index must contain only integers") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ModelRuntimeError("RAM delete_tag_index must contain only integers")
        if not 0 <= index < tag_count:
            raise ModelRuntimeError(
                "RAM delete_tag_index contains an out-of-range index",
                details={"index": index, "tag_count": tag_count},
            )
        indices.add(index)
    return indices


def _string_list(raw: object, *, name: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ModelRuntimeError(f"RAM {name} must be a list of non-empty strings")
    return [value.strip() for value in raw]


def _deleted_indices(config: dict[str, Any], parameters: dict[str, object], labels: list[str]) -> set[int]:
    direct = parameters.get("delete_tag_index", config.get("delete_tag_index"))
    if direct is not None:
        return _index_list(direct, tag_count=len(labels))
    delete_tags = _string_list(config.get("delete_tags"), name="delete_tags")
    if delete_tags:
        unknown = sorted(set(delete_tags) - set(labels))
        if unknown:
            raise ModelRuntimeError("RAM delete_tags contains unknown labels", details={"unknown_labels": unknown})
        return {index for index, label in enumerate(labels) if label in delete_tags}
    filter_tags = _string_list(config.get("filter_tags"), name="filter_tags")
    if filter_tags:
        unknown = sorted(set(filter_tags) - set(labels))
        if unknown:
            raise ModelRuntimeError("RAM filter_tags contains unknown labels", details={"unknown_labels": unknown})
        allowed = set(filter_tags)
        return {index for index, label in enumerate(labels) if label not in allowed}
    return set()


def _top_k(config: dict[str, Any], parameters: dict[str, object], available: int) -> int:
    raw = parameters.get("top_k", config.get("top_k"))
    if raw is None:
        return available
    if isinstance(raw, bool):
        raise ModelRuntimeError("RAM top_k must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError("RAM top_k must be a positive integer", details={"top_k": raw}) from exc
    if (isinstance(raw, float) and not raw.is_integer()) or value <= 0:
        raise ModelRuntimeError("RAM top_k must be a positive integer", details={"top_k": raw})
    return min(value, available)


class RamTaggingOnnxAdapter(OnnxRuntimeAdapter):
    """Clean-room RAM/RAM++ multi-label ONNX adapter."""

    ADAPTER_ID = "ram_tagging_onnx"

    def _configure_inputs(self, inputs: list[Any]) -> None:
        super()._configure_inputs(inputs)
        assert self.input_meta is not None
        shape = list(self.input_meta.shape)
        if shape and isinstance(shape[0], int) and shape[0] != 1:
            raise ModelRuntimeError(
                "RAM image input batch dimension must be one or dynamic",
                details={"shape": shape},
            )

    def _image_resize_mode(self) -> str:
        return "stretch"

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, _ImageTransform]:
        tensor, transform = super()._prepare_image(image_path)
        original_dtype = tensor.dtype
        mean = _normalization_vector(self.record.config, "mean", _DEFAULT_MEAN)
        std = _normalization_vector(self.record.config, "std", _DEFAULT_STD)
        if tensor.ndim != 4:
            raise ModelRuntimeError("RAM image tensor must have four dimensions", details={"shape": list(tensor.shape)})
        if tensor.shape[1] == 3:
            tensor = (tensor - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
        elif tensor.shape[-1] == 3:
            tensor = (tensor - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
        else:
            raise ModelRuntimeError("RAM image input must have three RGB channels", details={"shape": list(tensor.shape)})
        return tensor.astype(original_dtype, copy=False), transform

    def _classifications(
        self,
        outputs: dict[str, np.ndarray],
        parameters: dict[str, object],
    ) -> list[ClassificationResult]:
        config: dict[str, Any] = self.record.config
        labels = _labels(config, self.record.descriptor.config_path.resolve())
        output_name = parameters.get("output_name", config.get("output_name"))
        scores = _score_vector(outputs, output_name, len(labels))
        _validate_batch_output(outputs, parameters.get("batch_output_name", config.get("batch_output_name")))

        activation: object = parameters.get(
            "score_activation",
            parameters.get(
                "apply_sigmoid",
                config.get("score_activation", config.get("apply_sigmoid", "auto")),
            ),
        )
        probabilities = _probabilities(scores, activation)
        threshold = _number(
            parameters.get(
                "threshold",
                parameters.get(
                    "tag_threshold",
                    config.get("threshold", config.get("tag_threshold", config.get("confidence_threshold", 0.5))),
                ),
            ),
            name="threshold",
            minimum=0.0,
            maximum=1.0,
        )
        deleted = _deleted_indices(config, parameters, labels)
        selected = np.flatnonzero(probabilities >= threshold)
        if deleted:
            selected = selected[~np.isin(selected, np.fromiter(deleted, dtype=np.int64))]
        if selected.size == 0:
            return []
        order = selected[np.argsort(-probabilities[selected], kind="stable")]
        order = order[:_top_k(config, parameters, len(order))]
        return [
            ClassificationResult(label=labels[int(index)], score=float(probabilities[index]), rank=rank)
            for rank, index in enumerate(order, start=1)
        ]
