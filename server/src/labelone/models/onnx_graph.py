from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto

from labelone.errors import ModelRuntimeError

from .types import FeatureLayer


MAX_ENUMERATED_LAYERS = 512
MAX_GRAPH_FILE_BYTES = 512 * 1024 * 1024
MAX_STATIC_FEATURE_ELEMENTS = 64_000_000
_FEATURE_DTYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.DOUBLE,
    TensorProto.BFLOAT16,
}


@dataclass(frozen=True, slots=True)
class OnnxGraphInspection:
    original_output_names: list[str]
    layers: list[FeatureLayer]
    rewrite_supported: bool
    warning: str | None = None


def _shape(value_info) -> list[int | str | None]:  # noqa: ANN001
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            shape.append(str(dimension.dim_param))
        else:
            shape.append(None)
    return shape


def _axes(shape: list[int | str | None]) -> list[str]:
    if len(shape) == 1:
        return ["C"]
    if len(shape) == 2:
        return ["N", "C"]
    if len(shape) == 4:
        return ["N", "C", "H", "W"]
    if len(shape) == 3:
        return ["N", "T", "C"]
    return [f"D{index}" for index in range(len(shape))]


def _layer(value_info, *, group: str, name: str, captureable: bool = True, reason: str | None = None) -> FeatureLayer:  # noqa: ANN001
    shape = _shape(value_info)
    element_type = int(value_info.type.tensor_type.elem_type)
    return FeatureLayer(
        id=str(value_info.name),
        group=group,
        name=name[:256],
        shape=shape,
        axes=_axes(shape),
        dtype=TensorProto.DataType.Name(element_type).casefold(),
        spatial=len(shape) == 4,
        captureable=captureable,
        reason=reason,
    )


def _captureable_feature(value_info) -> bool:  # noqa: ANN001
    tensor_type = value_info.type.tensor_type
    shape = _shape(value_info)
    if int(tensor_type.elem_type) not in _FEATURE_DTYPES or len(shape) not in {1, 2, 3, 4}:
        return False
    elements = 1
    for index, dimension in enumerate(shape):
        if isinstance(dimension, int) and dimension > 0:
            elements *= dimension
        elif index == 0:
            elements *= 1
        else:
            return False
        if elements > MAX_STATIC_FEATURE_ELEMENTS:
            return False
    return True


def _load_inferred_model(model_path: Path):  # noqa: ANN202
    if not model_path.is_file():
        raise ModelRuntimeError("ONNX model file does not exist", details={"path": str(model_path)})
    size_bytes = model_path.stat().st_size
    if size_bytes > MAX_GRAPH_FILE_BYTES:
        raise ModelRuntimeError(
            "ONNX graph is too large for safe feature inspection",
            details={"path": str(model_path), "size_bytes": size_bytes, "maximum_bytes": MAX_GRAPH_FILE_BYTES},
        )
    try:
        model = onnx.load_model(model_path, load_external_data=False)
    except Exception as exc:
        raise ModelRuntimeError(
            "Failed to parse ONNX graph",
            details={"path": str(model_path), "error": str(exc)},
        ) from exc
    external = any(
        initializer.data_location == TensorProto.EXTERNAL or bool(initializer.external_data)
        for initializer in model.graph.initializer
    )
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model,
            check_type=False,
            strict_mode=False,
            data_prop=False,
        )
        warning = None
    except Exception as exc:
        inferred = model
        warning = f"ONNX shape inference was incomplete: {exc}"
    return inferred, external, warning


def inspect_onnx_graph(
    model_path: Path,
    *,
    maximum_layers: int = MAX_ENUMERATED_LAYERS,
) -> OnnxGraphInspection:
    """Parse a bounded ONNX graph without modifying its runtime outputs."""

    resolved = model_path.expanduser().resolve()
    inferred, external, warning = _load_inferred_model(resolved)
    original_output_names = [str(output.name) for output in inferred.graph.output if output.name]

    value_infos = {
        str(info.name): info
        for info in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if info.name and info.type.HasField("tensor_type")
    }
    layers: list[FeatureLayer] = []
    captured_names: set[str] = set()
    for output_name in original_output_names:
        if len(output_name) > 1024:
            continue
        info = value_infos.get(output_name)
        if info is None:
            continue
        captureable = _captureable_feature(info)
        layers.append(_layer(
            info,
            group="模型输出",
            name=output_name,
            captureable=captureable,
            reason=None if captureable else "通用特征预览仅支持有界浮点向量、矩阵、NTC 或 NCHW Tensor",
        ))
        captured_names.add(output_name)

    truncated = False
    if external:
        suffix = "ONNX external-data models currently expose exported outputs only"
        warning = f"{warning}; {suffix}" if warning else suffix
        return OnnxGraphInspection(
            original_output_names=original_output_names,
            layers=layers,
            rewrite_supported=False,
            warning=warning,
        )
    for node in inferred.graph.node:
        for output_name in node.output:
            name = str(output_name)
            if not name or len(name) > 1024 or name in captured_names:
                continue
            info = value_infos.get(name)
            if info is None:
                continue
            if not _captureable_feature(info):
                continue
            if len(layers) >= maximum_layers:
                truncated = True
                break
            display_name = str(node.name).strip() or name
            layers.append(_layer(
                info,
                group=f"中间层 · {node.op_type or 'Node'}",
                name=display_name,
            ))
            captured_names.add(name)
        if truncated:
            break

    if truncated:
        suffix = f"ONNX layer list was limited to {maximum_layers} entries"
        warning = f"{warning}; {suffix}" if warning else suffix
    return OnnxGraphInspection(
        original_output_names=original_output_names,
        layers=layers,
        rewrite_supported=True,
        warning=warning,
    )


def instrument_onnx_outputs(model_path: Path, output_names: list[str]) -> bytes:
    if not output_names or len(output_names) > 1 or any(not name or len(name) > 1024 for name in output_names):
        raise ModelRuntimeError("Exactly one bounded ONNX intermediate layer may be captured at a time")
    resolved = model_path.expanduser().resolve()
    inferred, external, _ = _load_inferred_model(resolved)
    if external:
        raise ModelRuntimeError("ONNX external-data models cannot rewrite intermediate outputs safely")
    original_names = {str(output.name) for output in inferred.graph.output if output.name}
    value_infos = {
        str(info.name): info
        for info in [*inferred.graph.value_info, *inferred.graph.output]
        if info.name and info.type.HasField("tensor_type")
    }
    for name in output_names:
        if name in original_names:
            continue
        info = value_infos.get(name)
        if info is None:
            raise ModelRuntimeError("ONNX intermediate layer metadata is unavailable", details={"layer_id": name})
        inferred.graph.output.extend([deepcopy(info)])
    try:
        onnx.checker.check_model(inferred)
        return inferred.SerializeToString()
    except Exception as exc:
        raise ModelRuntimeError(
            "ONNX feature instrumentation failed",
            details={"layers": output_names, "error": str(exc)},
        ) from exc
