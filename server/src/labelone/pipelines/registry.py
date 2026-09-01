from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Literal, Mapping

from labelone.errors import LabelOneError


class PipelineValidationError(LabelOneError):
    code = "pipeline_validation_error"


@dataclass(frozen=True, slots=True)
class OperatorContract:
    kind: str
    title: str
    description: str
    version: str
    input_type: str
    output_type: str
    annotation_policy: Mapping[str, object]
    parameters_schema: Mapping[str, object]
    size_behavior: Literal["preserve", "deterministic", "dynamic"] = "preserve"
    node_role: Literal["source", "transform", "visualization", "batch_export"] = "transform"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "annotation_policy": dict(self.annotation_policy),
            "parameters_schema": deepcopy(dict(self.parameters_schema)),
            "size_behavior": self.size_behavior,
            "node_role": self.node_role,
        }


@dataclass(frozen=True, slots=True)
class ValidatedNode:
    id: str
    kind: str
    enabled: bool
    parameters: Mapping[str, object]
    operator_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "enabled": self.enabled,
            "parameters": dict(self.parameters),
            "operator_version": self.operator_version,
        }


@dataclass(frozen=True, slots=True)
class ValidatedPipeline:
    nodes: tuple[ValidatedNode, ...]
    transform_count: int
    visualization_count: int
    output_width: int | None
    output_height: int | None


_BUILTIN_PARAMETER_METADATA: dict[str, tuple[str, str]] = {
    "label": ("结果名称", "用于区分这个显示节点产生的预览结果。"),
    "margin_ratio": ("边缘比例", "未填写精确坐标时，从图像四周按比例裁去边缘。"),
    "x": ("起点 X", "裁剪区域左上角的水平像素坐标。"),
    "y": ("起点 Y", "裁剪区域左上角的垂直像素坐标。"),
    "width": ("宽度", "输出图像或裁剪区域的目标宽度（像素）。"),
    "height": ("高度", "输出图像或裁剪区域的目标高度（像素）。"),
    "axis": ("翻转方向", "选择沿水平方向或垂直方向镜像图像与标注。"),
    "degrees": ("旋转角度", "按图像坐标系将图像与标注旋转指定的直角角度。"),
    "brightness": ("亮度", "按倍率调整图像整体明暗。"),
    "contrast": ("对比度", "按倍率调整明暗区域之间的差异。"),
    "saturation": ("饱和度", "按倍率调整颜色浓淡。"),
    "radius": ("锐化半径", "控制去噪后锐化作用的邻域半径。"),
    "percent": ("锐化强度", "控制锐化效果的百分比强度。"),
    "threshold": ("锐化阈值", "仅对差异超过该值的像素应用锐化。"),
    "model_id": ("模型", "选择用于捕获中间层的本地 ONNX 模型。"),
    "layer_id": ("中间层", "选择模型运行时公开的一个可捕获 Tensor 层。"),
    "projection": ("展示方式", "将通道或 Token 投影为可显示的二维图像或向量曲线。"),
    "normalization": ("数值归一化", "把特征值映射到稳定的可视范围。"),
    "channel": ("通道索引", "使用单通道展示方式时选取的通道编号。"),
    "clip": ("数值截断", "在归一化前按百分位截断极端特征值。"),
    "tile_width": ("切片宽度", "每张输出切片的目标宽度（像素）。"),
    "tile_height": ("切片高度", "每张输出切片的目标高度（像素）。"),
    "overlap_x": ("水平重叠", "相邻切片在水平方向重叠的像素数。"),
    "overlap_y": ("垂直重叠", "相邻切片在垂直方向重叠的像素数。"),
    "include_partial": ("保留边缘切片", "是否输出尺寸不足目标大小的边缘切片。"),
}


def _schema(properties: dict[str, dict[str, object]], **extra: object) -> dict[str, object]:
    described_properties: dict[str, dict[str, object]] = {}
    for name, property_schema in properties.items():
        title, description = _BUILTIN_PARAMETER_METADATA[name]
        described_properties[name] = {"title": title, "description": description, **property_schema}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "default": {},
        "properties": described_properties,
        "additionalProperties": False,
        **extra,
    }


BUILTIN_OPERATORS: dict[str, OperatorContract] = {
    "source": OperatorContract(
        kind="source",
        title="原图像",
        description="从当前数据集读取原始图像与标注文档。",
        version="1.0.0",
        input_type="none",
        output_type="image",
        annotation_policy={"mode": "preserve", "coordinates": "unchanged"},
        parameters_schema=_schema({}),
        size_behavior="dynamic",
        node_role="source",
    ),
    "output": OperatorContract(
        kind="output",
        title="旧版输出",
        description="兼容旧版流程定义的输出节点。",
        version="1.0.0",
        input_type="image",
        output_type="none",
        annotation_policy={"mode": "preserve", "coordinates": "unchanged"},
        parameters_schema=_schema({}),
        node_role="visualization",
    ),
    "visualize": OperatorContract(
        kind="visualize",
        title="显示",
        description="显示上游阶段的图像与同步标注结果。",
        version="1.0.0",
        input_type="image",
        output_type="none",
        annotation_policy={"mode": "preserve", "coordinates": "unchanged"},
        parameters_schema=_schema(
            {"label": {"type": "string", "minLength": 1, "maxLength": 160, "default": "显示"}}
        ),
        node_role="visualization",
    ),
    "model_feature": OperatorContract(
        kind="model_feature",
        title="模型中间层",
        description="运行本地模型并把指定中间层转换为与输入图像同尺寸的可视化图像，可继续连接显示节点进行分屏或叠加。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "preserve", "coordinates": "unchanged", "spatial_behavior": "none", "synchronized": True},
        parameters_schema=_schema(
            {
                "model_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "layer_id": {"type": "string", "minLength": 1, "maxLength": 1024},
                "projection": {"type": "string", "enum": ["none", "mean", "max", "pca1", "channel", "token_grid"], "default": "mean"},
                "normalization": {"type": "string", "enum": ["minmax", "zscore", "l2", "none"], "default": "minmax"},
                "channel": {"type": "integer", "minimum": 0, "maximum": 1_000_000, "default": 0},
                "clip": {"type": "string", "enum": ["p1p99", "p5p95", "none"], "default": "p1p99"},
            }
        ),
        size_behavior="preserve",
    ),
    "crop": OperatorContract(
        kind="crop",
        title="智能裁剪",
        description="按边缘比例或精确像素区域裁剪图像，并同步裁剪标注。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "clip_translate", "drops_outside": True, "spatial_behavior": "crop_translate", "synchronized": True},
        parameters_schema=_schema(
            {
                "margin_ratio": {"type": "number", "minimum": 0.0, "maximum": 0.49, "multipleOf": 0.005, "default": 0.05, "x-ui": {"control": "slider", "role": "ratio"}},
                "x": {"type": "integer", "minimum": 0, "maximum": 1_000_000, "multipleOf": 1, "x-ui": {"control": "number", "role": "region-x", "unit": "px"}},
                "y": {"type": "integer", "minimum": 0, "maximum": 1_000_000, "multipleOf": 1, "x-ui": {"control": "number", "role": "region-y", "unit": "px"}},
                "width": {"type": "integer", "minimum": 1, "maximum": 1_000_000, "multipleOf": 1, "x-ui": {"control": "number", "role": "region-width", "unit": "px"}},
                "height": {"type": "integer", "minimum": 1, "maximum": 1_000_000, "multipleOf": 1, "x-ui": {"control": "number", "role": "region-height", "unit": "px"}},
            },
            allOf=[
                {
                    "oneOf": [
                        {"required": ["x", "y", "width", "height"]},
                        {"not": {"anyOf": [{"required": [name]} for name in ("x", "y", "width", "height")] }},
                    ]
                }
            ],
        ),
        size_behavior="deterministic",
    ),
    "resize": OperatorContract(
        kind="resize",
        title="缩放",
        description="将图像缩放到目标尺寸，并同步缩放所有标注坐标。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "scale_xy", "spatial_behavior": "scale_xy", "scale_strategy": "independent_xy", "synchronized": True, "preserves_shape_types": False, "anisotropic_shape_policy": "promote_polygon"},
        parameters_schema=_schema(
            {
                "width": {"type": "integer", "minimum": 1, "maximum": 1_000_000, "multipleOf": 1, "x-ui": {"control": "number", "role": "target-width", "unit": "px"}},
                "height": {"type": "integer", "minimum": 1, "maximum": 1_000_000, "multipleOf": 1, "x-ui": {"control": "number", "role": "target-height", "unit": "px"}},
            },
            dependentRequired={"width": ["height"], "height": ["width"]},
        ),
        size_behavior="deterministic",
    ),
    "flip": OperatorContract(
        kind="flip",
        title="翻转",
        description="沿水平或垂直方向镜像图像与标注。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "mirror", "preserves_shape_types": True},
        parameters_schema=_schema(
            {"axis": {"type": "string", "enum": ["horizontal", "vertical"], "default": "horizontal"}}
        ),
    ),
    "rotate": OperatorContract(
        kind="rotate",
        title="直角旋转",
        description="按 0°、90°、180° 或 270° 旋转图像与标注。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "rotate", "origin": "image", "preserves_shape_types": True},
        parameters_schema=_schema(
            {"degrees": {"type": "integer", "enum": [0, 90, 180, 270], "default": 90}}
        ),
        size_behavior="deterministic",
    ),
    "color": OperatorContract(
        kind="color",
        title="颜色增强",
        description="调整图像亮度、对比度和饱和度，不改变标注坐标。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "preserve", "coordinates": "unchanged"},
        parameters_schema=_schema(
            {
                "brightness": {"type": "number", "minimum": 0.0, "maximum": 4.0, "default": 1.05},
                "contrast": {"type": "number", "minimum": 0.0, "maximum": 4.0, "default": 1.10},
                "saturation": {"type": "number", "minimum": 0.0, "maximum": 4.0, "default": 1.0},
            }
        ),
    ),
    "noise": OperatorContract(
        kind="noise",
        title="去噪锐化",
        description="平滑细小噪声后增强图像边缘，不改变标注坐标。",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "preserve", "coordinates": "unchanged"},
        parameters_schema=_schema(
            {
                "radius": {"type": "number", "minimum": 0.0, "maximum": 5.0, "default": 1.0},
                "percent": {"type": "integer", "minimum": 0, "maximum": 500, "default": 130},
                "threshold": {"type": "integer", "minimum": 0, "maximum": 255, "default": 2},
            }
        ),
    ),
    "tile": OperatorContract(
        kind="tile",
        title="重叠切片",
        description="把大图切成可重叠的小图，并为每张切片生成对应标注。",
        version="1.0.0",
        input_type="image",
        output_type="images",
        annotation_policy={"mode": "tile_clip_translate", "preview": "rejected", "batch": "multiple_outputs"},
        parameters_schema=_schema(
            {
                "tile_width": {"type": "integer", "minimum": 32, "maximum": 16_384, "default": 1024},
                "tile_height": {"type": "integer", "minimum": 32, "maximum": 16_384, "default": 1024},
                "overlap_x": {"type": "integer", "minimum": 0, "maximum": 16_383, "default": 128},
                "overlap_y": {"type": "integer", "minimum": 0, "maximum": 16_383, "default": 128},
                "include_partial": {"type": "boolean", "default": True},
            }
        ),
        size_behavior="dynamic",
        node_role="batch_export",
    ),
}


_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_PARAMETER_TYPES = frozenset({"integer", "number", "string", "boolean"})
_PARAMETER_SCHEMA_FIELDS = frozenset({
    "type", "default", "enum", "minimum", "maximum", "multipleOf", "minLength", "maxLength", "title", "description", "x-ui",
})
_PARAMETER_UI_CONTROLS = frozenset({"auto", "slider", "number"})
_PARAMETER_UI_ROLES = frozenset({
    "region-x", "region-y", "region-width", "region-height", "target-width", "target-height", "scale-factor", "ratio",
})
_CONTRACT_SCHEMA_FIELDS = frozenset({"$schema", "type", "default", "properties", "additionalProperties"})
_CORE_SCHEMA_COMPOSITION_FIELDS = frozenset({"allOf", "dependentRequired"})


def _fail(message: str, *, node_id: object = None, kind: object = None, **details: object) -> None:
    payload = {"node_id": node_id, "kind": kind, **details}
    raise PipelineValidationError(message, details={key: value for key, value in payload.items() if value is not None})


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        or isinstance(value, float)
        and math.isfinite(value)
    )


def _validate_property(value: object, schema: Mapping[str, object], *, node_id: str, kind: str, name: str) -> object:
    expected = schema.get("type")
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            _fail("Pipeline parameter must be an integer", node_id=node_id, kind=kind, parameter=name, value=value)
    elif expected == "number":
        if not _is_finite_number(value):
            _fail("Pipeline parameter must be a finite number", node_id=node_id, kind=kind, parameter=name, value=value)
    elif expected == "string":
        if not isinstance(value, str):
            _fail("Pipeline parameter must be a string", node_id=node_id, kind=kind, parameter=name, value=value)
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            _fail("Pipeline parameter string is too short", node_id=node_id, kind=kind, parameter=name)
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            _fail("Pipeline parameter string is too long", node_id=node_id, kind=kind, parameter=name)
    elif expected == "boolean":
        if not isinstance(value, bool):
            _fail("Pipeline parameter must be boolean", node_id=node_id, kind=kind, parameter=name, value=value)
    else:
        _fail("Operator schema contains an unsupported parameter type", node_id=node_id, kind=kind, parameter=name)
    if "enum" in schema and value not in schema["enum"]:
        _fail("Pipeline parameter is not an allowed value", node_id=node_id, kind=kind, parameter=name, value=value)
    if "minimum" in schema and value < schema["minimum"]:
        _fail("Pipeline parameter is below its minimum", node_id=node_id, kind=kind, parameter=name, value=value)
    if "maximum" in schema and value > schema["maximum"]:
        _fail("Pipeline parameter exceeds its maximum", node_id=node_id, kind=kind, parameter=name, value=value)
    if "multipleOf" in schema and expected in {"integer", "number"}:
        multiple = float(schema["multipleOf"])
        quotient = float(value) / multiple
        if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
            _fail("Pipeline parameter is not aligned to its multipleOf step", node_id=node_id, kind=kind, parameter=name, value=value)
    return value


def validate_operator_contract_schema(
    contract: OperatorContract,
    *,
    allow_core_composition: bool = False,
) -> None:
    """Validate the supported parameter-schema subset for one operator contract."""
    if not isinstance(contract.title, str) or not contract.title.strip() or len(contract.title) > 160:
        _fail("Operator title must be a non-empty string", kind=contract.kind)
    if not isinstance(contract.description, str) or not contract.description.strip() or len(contract.description) > 500:
        _fail("Operator description must be a non-empty string", kind=contract.kind)
    schema = contract.parameters_schema
    if not isinstance(schema, Mapping):
        _fail("Operator parameters_schema must be an object", kind=contract.kind)
    allowed_root_fields = set(_CONTRACT_SCHEMA_FIELDS)
    if allow_core_composition:
        allowed_root_fields.update(_CORE_SCHEMA_COMPOSITION_FIELDS)
    root_extras = sorted(str(key) for key in schema if key not in allowed_root_fields)
    if root_extras:
        _fail("Operator parameters_schema contains unsupported fields", kind=contract.kind, fields=root_extras)
    if schema.get("type") != "object":
        _fail("Operator parameters_schema type must be object", kind=contract.kind)
    if schema.get("default", {}) != {}:
        _fail("Operator parameters_schema root default must be an empty object", kind=contract.kind)
    if schema.get("additionalProperties") is not False:
        _fail("Operator parameters_schema must forbid additional properties", kind=contract.kind)
    dialect = schema.get("$schema")
    if dialect is not None and not isinstance(dialect, str):
        _fail("Operator parameters_schema $schema must be a string", kind=contract.kind)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        _fail("Operator parameters_schema properties must be an object", kind=contract.kind)

    if allow_core_composition:
        if "allOf" in schema and not isinstance(schema["allOf"], list):
            _fail("Operator parameters_schema allOf must be an array", kind=contract.kind)
        if "dependentRequired" in schema and not isinstance(schema["dependentRequired"], Mapping):
            _fail("Operator parameters_schema dependentRequired must be an object", kind=contract.kind)

    for parameter_name, raw_property_schema in properties.items():
        if not isinstance(parameter_name, str) or not _PARAMETER_NAME.fullmatch(parameter_name):
            _fail("Operator parameter name is invalid", kind=contract.kind, parameter=parameter_name)
        if not isinstance(raw_property_schema, Mapping):
            _fail("Operator parameter schema must be an object", kind=contract.kind, parameter=parameter_name)
        property_schema = raw_property_schema
        extras = sorted(str(key) for key in property_schema if key not in _PARAMETER_SCHEMA_FIELDS)
        if extras:
            _fail(
                "Operator parameter schema contains unsupported fields",
                kind=contract.kind,
                parameter=parameter_name,
                fields=extras,
            )
        expected = property_schema.get("type")
        if expected not in _PARAMETER_TYPES:
            _fail("Operator parameter type is unsupported", kind=contract.kind, parameter=parameter_name)
        for text_field in ("title", "description"):
            if (
                text_field not in property_schema
                or not isinstance(property_schema[text_field], str)
                or not property_schema[text_field].strip()
                or len(property_schema[text_field]) > 500
            ):
                _fail(
                    f"Operator parameter {text_field} must be a non-empty string",
                    kind=contract.kind,
                    parameter=parameter_name,
                )

        numeric_bounds: dict[str, int | float] = {}
        for bound_name in ("minimum", "maximum"):
            if bound_name not in property_schema:
                continue
            if expected not in {"integer", "number"}:
                _fail(
                    "Operator parameter numeric bounds require integer or number type",
                    kind=contract.kind,
                    parameter=parameter_name,
                    field=bound_name,
                )
            bound = property_schema[bound_name]
            if not _is_finite_number(bound):
                _fail(
                    f"Operator parameter {bound_name} must be a finite number",
                    kind=contract.kind,
                    parameter=parameter_name,
                )
            numeric_bounds[bound_name] = bound
        if numeric_bounds.get("minimum", -math.inf) > numeric_bounds.get("maximum", math.inf):
            _fail(
                "Operator parameter minimum cannot exceed maximum",
                kind=contract.kind,
                parameter=parameter_name,
            )
        if (
            expected == "integer"
            and "minimum" in numeric_bounds
            and "maximum" in numeric_bounds
            and math.ceil(numeric_bounds["minimum"]) > math.floor(numeric_bounds["maximum"])
        ):
            _fail(
                "Operator integer parameter range must contain an integer",
                kind=contract.kind,
                parameter=parameter_name,
            )

        if "multipleOf" in property_schema:
            multiple = property_schema["multipleOf"]
            if expected not in {"integer", "number"}:
                _fail(
                    "Operator parameter multipleOf requires integer or number type",
                    kind=contract.kind,
                    parameter=parameter_name,
                )
            if not _is_finite_number(multiple) or float(multiple) <= 0:
                _fail(
                    "Operator parameter multipleOf must be a positive finite number",
                    kind=contract.kind,
                    parameter=parameter_name,
                )

        if "x-ui" in property_schema:
            ui = property_schema["x-ui"]
            if not isinstance(ui, Mapping):
                _fail("Operator parameter x-ui must be an object", kind=contract.kind, parameter=parameter_name)
            ui_extras = sorted(str(key) for key in ui if key not in {"control", "role", "unit"})
            if ui_extras:
                _fail("Operator parameter x-ui contains unsupported fields", kind=contract.kind, parameter=parameter_name, fields=ui_extras)
            control = ui.get("control", "auto")
            if control not in _PARAMETER_UI_CONTROLS:
                _fail("Operator parameter x-ui control is invalid", kind=contract.kind, parameter=parameter_name, control=control)
            if control in {"slider", "number"} and expected not in {"integer", "number"}:
                _fail("Operator parameter numeric x-ui control requires a numeric type", kind=contract.kind, parameter=parameter_name)
            if control == "slider" and not {"minimum", "maximum"}.issubset(property_schema):
                _fail("Operator parameter slider requires finite minimum and maximum", kind=contract.kind, parameter=parameter_name)
            role = ui.get("role")
            if role is not None and role not in _PARAMETER_UI_ROLES:
                _fail("Operator parameter x-ui role is invalid", kind=contract.kind, parameter=parameter_name, role=role)
            if role is not None and expected not in {"integer", "number"}:
                _fail("Operator parameter x-ui role requires a numeric type", kind=contract.kind, parameter=parameter_name)
            unit = ui.get("unit")
            if unit is not None and (not isinstance(unit, str) or not unit.strip() or len(unit) > 16):
                _fail("Operator parameter x-ui unit must be a short non-empty string", kind=contract.kind, parameter=parameter_name)

        length_bounds: dict[str, int] = {}
        for bound_name in ("minLength", "maxLength"):
            if bound_name not in property_schema:
                continue
            if expected != "string":
                _fail(
                    "Operator parameter length bounds require string type",
                    kind=contract.kind,
                    parameter=parameter_name,
                    field=bound_name,
                )
            bound = property_schema[bound_name]
            if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
                _fail(
                    f"Operator parameter {bound_name} must be a non-negative integer",
                    kind=contract.kind,
                    parameter=parameter_name,
                )
            length_bounds[bound_name] = bound
        if length_bounds.get("minLength", 0) > length_bounds.get("maxLength", 2**63 - 1):
            _fail(
                "Operator parameter minLength cannot exceed maxLength",
                kind=contract.kind,
                parameter=parameter_name,
            )

        if "enum" in property_schema:
            enum = property_schema["enum"]
            if not isinstance(enum, (list, tuple)) or not enum:
                _fail("Operator parameter enum must be a non-empty array", kind=contract.kind, parameter=parameter_name)
            schema_without_enum = {key: value for key, value in property_schema.items() if key != "enum"}
            for enum_value in enum:
                _validate_property(
                    enum_value,
                    schema_without_enum,
                    node_id=contract.kind,
                    kind=contract.kind,
                    name=parameter_name,
                )
        if "default" in property_schema:
            _validate_property(
                property_schema["default"],
                property_schema,
                node_id=contract.kind,
                kind=contract.kind,
                name=parameter_name,
            )


def normalize_parameters(kind: str, parameters: Mapping[str, object] | None, *, node_id: str = "node") -> dict[str, object]:
    contract = BUILTIN_OPERATORS.get(kind)
    if contract is None:
        _fail("Unknown pipeline operator", node_id=node_id, kind=kind)
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        _fail("Pipeline parameters must be an object", node_id=node_id, kind=kind)
    schema = contract.parameters_schema
    properties = schema.get("properties", {})
    assert isinstance(properties, Mapping)
    extras = sorted(str(key) for key in parameters if key not in properties)
    if extras:
        _fail("Pipeline parameters contain unknown fields", node_id=node_id, kind=kind, fields=extras)
    normalized: dict[str, object] = {}
    for name, property_schema in properties.items():
        assert isinstance(name, str) and isinstance(property_schema, Mapping)
        if name in parameters:
            normalized[name] = _validate_property(parameters[name], property_schema, node_id=node_id, kind=kind, name=name)
        elif "default" in property_schema:
            normalized[name] = _validate_property(
                deepcopy(property_schema["default"]),
                property_schema,
                node_id=node_id,
                kind=kind,
                name=name,
            )

    if kind == "crop":
        coordinates = [name for name in ("x", "y", "width", "height") if name in parameters]
        if coordinates and len(coordinates) != 4:
            _fail(
                "Crop coordinates must provide x, y, width, and height together",
                node_id=node_id,
                kind=kind,
                provided=coordinates,
            )
        if coordinates:
            normalized.pop("margin_ratio", None)
    elif kind == "resize":
        if ("width" in parameters) != ("height" in parameters):
            _fail("Resize width and height must be provided together", node_id=node_id, kind=kind)
        if "width" in normalized and int(normalized["width"]) * int(normalized["height"]) > 64_000_000:
            _fail("Resize output exceeds the built-in pixel budget", node_id=node_id, kind=kind)
    elif kind == "tile":
        if int(normalized["overlap_x"]) >= int(normalized["tile_width"]):
            _fail("Tile horizontal overlap must be smaller than tile width", node_id=node_id, kind=kind)
        if int(normalized["overlap_y"]) >= int(normalized["tile_height"]):
            _fail("Tile vertical overlap must be smaller than tile height", node_id=node_id, kind=kind)
    elif kind == "model_feature":
        for required_name in ("model_id", "layer_id"):
            if required_name not in normalized:
                _fail(
                    "Model feature nodes require a selected model and layer",
                    node_id=node_id,
                    kind=kind,
                    parameter=required_name,
                )
    return normalized


def _validate_node_records(nodes: Iterable[object], *, maximum_nodes: int) -> list[ValidatedNode]:
    if isinstance(nodes, (str, bytes, Mapping)):
        raise PipelineValidationError("Pipeline nodes must be an array")
    raw_nodes = list(nodes)
    if not raw_nodes:
        raise PipelineValidationError("Pipeline must contain at least one node")
    if len(raw_nodes) > maximum_nodes:
        raise PipelineValidationError(
            "Pipeline exceeds the node budget",
            details={"nodes": len(raw_nodes), "maximum": maximum_nodes},
        )
    normalized: list[ValidatedNode] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="python")
        if not isinstance(raw, Mapping):
            _fail("Pipeline node must be an object", node_id=index)
        extras = sorted(str(key) for key in raw if key not in {"id", "kind", "enabled", "parameters"})
        if extras:
            _fail("Pipeline node contains unknown fields", node_id=raw.get("id", index), fields=extras)
        node_id = raw.get("id")
        kind = raw.get("kind")
        enabled = raw.get("enabled", True)
        if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
            _fail("Pipeline node id is invalid", node_id=node_id, kind=kind)
        if node_id in ids:
            _fail("Pipeline node ids must be unique", node_id=node_id, kind=kind)
        ids.add(node_id)
        if not isinstance(kind, str) or kind not in BUILTIN_OPERATORS:
            _fail("Unknown pipeline operator", node_id=node_id, kind=kind)
        if not isinstance(enabled, bool):
            _fail("Pipeline node enabled must be boolean", node_id=node_id, kind=kind)
        parameters = normalize_parameters(kind, raw.get("parameters", {}), node_id=node_id)
        normalized.append(ValidatedNode(node_id, kind, enabled, parameters, BUILTIN_OPERATORS[kind].version))
    return normalized


def normalize_legacy_nodes(nodes: Iterable[object]) -> list[object]:
    """Map the legacy output node and add missing graph boundary nodes.

    This compatibility adapter belongs at API/engine boundaries. Internal graph
    editing and validation should call ``validate_nodes`` directly so missing
    boundaries cannot be hidden from new clients.
    """
    if isinstance(nodes, (str, bytes, Mapping)):
        raise PipelineValidationError("Pipeline nodes must be an array")
    normalized: list[object] = []
    used_ids: set[str] = set()
    for raw in nodes:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="python")
        if not isinstance(raw, Mapping):
            normalized.append(raw)
            continue
        item = dict(raw)
        node_id = item.get("id")
        if isinstance(node_id, str):
            used_ids.add(node_id)
        if item.get("kind") == "output":
            item["kind"] = "visualize"
            parameters = item.get("parameters", {})
            if isinstance(parameters, Mapping):
                item["parameters"] = {"label": "显示", **dict(parameters)}
        normalized.append(item)

    def boundary_id(preferred: str) -> str:
        candidate = preferred
        index = 2
        while candidate in used_ids:
            candidate = f"{preferred}-{index}"
            index += 1
        used_ids.add(candidate)
        return candidate

    has_source = any(isinstance(item, Mapping) and item.get("kind") == "source" for item in normalized)
    if not has_source:
        normalized.insert(0, {"id": boundary_id("source"), "kind": "source", "enabled": True, "parameters": {}})
    has_visualization = any(
        isinstance(item, Mapping) and item.get("kind") == "visualize" for item in normalized
    )
    if not has_visualization:
        normalized.append({
            "id": boundary_id("visualize"),
            "kind": "visualize",
            "enabled": True,
            "parameters": {"label": "显示"},
        })
    return normalized


def validate_transform_nodes(nodes: Iterable[object], *, maximum_nodes: int = 128) -> list[ValidatedNode]:
    """Validate the transform-only body used by declarative composites."""
    normalized = _validate_node_records(nodes, maximum_nodes=maximum_nodes)
    for node in normalized:
        if BUILTIN_OPERATORS[node.kind].node_role != "transform":
            raise PipelineValidationError(
                "Composite steps must contain only transform operators",
                details={"node_id": node.id, "kind": node.kind},
            )
    return normalized


def validate_nodes(nodes: Iterable[object], *, maximum_nodes: int = 128) -> list[ValidatedNode]:
    """Validate the canonical source/transform/visualization graph protocol."""
    normalized = _validate_node_records(nodes, maximum_nodes=maximum_nodes)
    if any(node.kind == "output" for node in normalized):
        raise PipelineValidationError("Legacy output nodes must be normalized to visualize nodes")

    source_positions = [index for index, node in enumerate(normalized) if node.kind == "source"]
    if source_positions != [0]:
        raise PipelineValidationError("Pipeline must contain exactly one source node and it must be first")
    if not normalized[0].enabled:
        raise PipelineValidationError("Pipeline source node must be enabled")

    model_feature_nodes = [node for node in normalized if node.kind == "model_feature"]
    if len(model_feature_nodes) > 1:
        raise PipelineValidationError(
            "每个处理流最多只能包含一个模型中间层节点",
            details={"node_ids": [node.id for node in model_feature_nodes], "maximum": 1},
        )

    visualizations = [node for node in normalized if node.kind == "visualize"]
    if not 1 <= len(visualizations) <= 4:
        raise PipelineValidationError(
            "Pipeline must contain between one and four visualization nodes",
            details={"visualizations": len(visualizations), "minimum": 1, "maximum": 4},
        )
    if any(not node.enabled for node in visualizations):
        raise PipelineValidationError("Pipeline visualization nodes must be enabled")
    for upstream, downstream in zip(normalized, normalized[1:]):
        if upstream.kind == downstream.kind == "visualize":
            raise PipelineValidationError(
                "连续显示节点无效：两个显示之间必须经过图像或模型处理节点",
                details={"upstream_node_id": upstream.id, "downstream_node_id": downstream.id},
            )
    if normalized[-1].kind != "visualize":
        raise PipelineValidationError("Pipeline final node must be a visualization")
    return normalized


def validate_pipeline_definition(
    nodes: Iterable[object],
    *,
    mode: Literal["preview", "derived_dataset"] = "preview",
    width: int | None = None,
    height: int | None = None,
    maximum_nodes: int = 128,
    maximum_output_pixels: int = 64_000_000,
) -> ValidatedPipeline:
    normalized = validate_nodes(normalize_legacy_nodes(nodes), maximum_nodes=maximum_nodes)
    if mode not in {"preview", "derived_dataset"}:
        raise PipelineValidationError("Pipeline validation mode is invalid", details={"mode": mode})
    if mode == "preview" and any(node.enabled and node.kind == "tile" for node in normalized):
        raise PipelineValidationError("Tile is unavailable in preview pipelines")
    if (width is None) != (height is None):
        raise PipelineValidationError("Pipeline validation width and height must be provided together")
    if width is not None and (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise PipelineValidationError("Pipeline validation dimensions must be positive integers")

    output_width, output_height = width, height
    if output_width is not None and output_height is not None:
        for node in normalized:
            if not node.enabled:
                continue
            parameters = node.parameters
            if node.kind == "crop":
                if "width" in parameters:
                    x = int(parameters["x"])
                    y = int(parameters["y"])
                    target_width = int(parameters["width"])
                    target_height = int(parameters["height"])
                    if x + target_width > output_width or y + target_height > output_height:
                        raise PipelineValidationError(
                            "Pipeline crop exceeds the current image dimensions",
                            details={"node_id": node.id, "input_size": [output_width, output_height]},
                        )
                else:
                    margin = float(parameters["margin_ratio"])
                    target_width = max(1, round(output_width * (1 - margin * 2)))
                    target_height = max(1, round(output_height * (1 - margin * 2)))
                output_width, output_height = target_width, target_height
            elif node.kind == "resize" and "width" in parameters:
                output_width = int(parameters["width"])
                output_height = int(parameters["height"])
            elif node.kind == "rotate" and int(parameters["degrees"]) in {90, 270}:
                output_width, output_height = output_height, output_width
            if node.kind in {"crop", "resize"} and output_width * output_height > maximum_output_pixels:
                raise PipelineValidationError(
                    "Pipeline deterministic output exceeds the pixel budget",
                    details={
                        "node_id": node.id,
                        "size": [output_width, output_height],
                        "pixels": output_width * output_height,
                        "maximum": maximum_output_pixels,
                    },
                )

    transform_count = sum(
        1
        for node in normalized
        if node.enabled and BUILTIN_OPERATORS[node.kind].node_role == "transform"
    )
    visualization_count = sum(1 for node in normalized if node.enabled and node.kind == "visualize")
    return ValidatedPipeline(
        nodes=tuple(normalized),
        transform_count=transform_count,
        visualization_count=visualization_count,
        output_width=output_width,
        output_height=output_height,
    )


def register_operator_contracts(contracts: Iterable[OperatorContract]) -> None:
    """Register an allowlisted extension catalog before serving requests."""
    pending = list(contracts)
    seen: set[str] = set()
    for contract in pending:
        if not isinstance(contract, OperatorContract):
            raise TypeError("Operator extensions must contain OperatorContract values")
        validate_operator_contract_schema(contract)
        if not _NODE_ID.fullmatch(contract.kind):
            raise ValueError(f"Operator kind is invalid: {contract.kind}")
        if contract.kind in seen:
            raise ValueError(f"Operator kind is already registered: {contract.kind}")
        existing = BUILTIN_OPERATORS.get(contract.kind)
        if existing is not None and existing != contract:
            raise ValueError(f"Operator kind is already registered: {contract.kind}")
        seen.add(contract.kind)
    BUILTIN_OPERATORS.update(
        (contract.kind, contract)
        for contract in pending
        if contract.kind not in BUILTIN_OPERATORS
    )


for _core_contract in BUILTIN_OPERATORS.values():
    validate_operator_contract_schema(_core_contract, allow_core_composition=True)


try:
    from .opencv_ops import OPENCV_OPERATORS
except ImportError:  # pragma: no cover - OpenCV catalog is optional during minimal builds
    OPENCV_OPERATORS = {}

register_operator_contracts(OPENCV_OPERATORS.values())
CORE_OPERATOR_KINDS = frozenset(BUILTIN_OPERATORS)


def unregister_operator_contracts(kinds: Iterable[str]) -> None:
    """Remove app-scoped package contracts without touching core operators."""
    for kind in kinds:
        if kind not in CORE_OPERATOR_KINDS:
            BUILTIN_OPERATORS.pop(kind, None)


def operator_registry_hash() -> str:
    payload = [BUILTIN_OPERATORS[kind].as_dict() for kind in sorted(BUILTIN_OPERATORS)]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def operator_catalog() -> list[dict[str, object]]:
    return [BUILTIN_OPERATORS[kind].as_dict() for kind in sorted(BUILTIN_OPERATORS)]
