from __future__ import annotations

import math

import pytest

from labelone.pipelines.custom import CompositeRegistry
from labelone.pipelines.registry import (
    BUILTIN_OPERATORS,
    OperatorContract,
    PipelineValidationError,
    normalize_legacy_nodes,
    normalize_parameters,
    operator_catalog,
    operator_registry_hash,
    register_operator_contracts,
    validate_nodes,
)


def test_builtin_catalog_declares_schema_versions_image_contract_and_annotation_policy() -> None:
    assert {"source", "output", "visualize", "crop", "resize", "flip", "rotate", "color", "noise", "tile"}.issubset(BUILTIN_OPERATORS)
    assert len(operator_registry_hash()) == 64
    assert [item["kind"] for item in operator_catalog()] == sorted(BUILTIN_OPERATORS)
    for contract in BUILTIN_OPERATORS.values():
        assert contract.title.strip()
        assert contract.description.strip()
        assert contract.version == "1.0.0"
        assert contract.annotation_policy["mode"]
        assert contract.parameters_schema["type"] == "object"
        assert contract.parameters_schema["default"] == {}
        assert contract.parameters_schema["additionalProperties"] is False
        assert contract.size_behavior in {"preserve", "deterministic", "dynamic"}
        assert contract.node_role in {"source", "transform", "visualization", "batch_export"}
        if contract.node_role == "transform":
            assert contract.input_type == contract.output_type == "image"
        for parameter in contract.parameters_schema["properties"].values():
            assert parameter["title"].strip()
            assert parameter["description"].strip()
    assert BUILTIN_OPERATORS["source"].node_role == "source"
    assert BUILTIN_OPERATORS["visualize"].node_role == "visualization"
    assert BUILTIN_OPERATORS["visualize"].output_type == "none"
    assert BUILTIN_OPERATORS["tile"].node_role == "batch_export"
    assert BUILTIN_OPERATORS["tile"].output_type == "images"
    crop_properties = BUILTIN_OPERATORS["crop"].parameters_schema["properties"]
    resize_properties = BUILTIN_OPERATORS["resize"].parameters_schema["properties"]
    assert crop_properties["margin_ratio"]["x-ui"] == {"control": "slider", "role": "ratio"}
    assert crop_properties["x"]["x-ui"] == {"control": "number", "role": "region-x", "unit": "px"}
    assert crop_properties["width"]["x-ui"]["role"] == "region-width"
    assert resize_properties["width"]["x-ui"] == {"control": "number", "role": "target-width", "unit": "px"}
    assert BUILTIN_OPERATORS["crop"].annotation_policy["drops_outside"] is True
    assert BUILTIN_OPERATORS["resize"].annotation_policy["scale_strategy"] == "independent_xy"


def test_validate_nodes_normalizes_defaults_and_keeps_complex_parameters() -> None:
    nodes = validate_nodes([
        {"id": "source", "kind": "source"},
        {"id": "crop", "kind": "crop", "parameters": {"x": 2, "y": 3, "width": 100, "height": 50}},
        {"id": "resize", "kind": "resize", "parameters": {"width": 640, "height": 320}},
        {"id": "flip", "kind": "flip"},
        {"id": "rotate", "kind": "rotate", "parameters": {"degrees": 270}},
        {"id": "color", "kind": "color", "parameters": {"brightness": 1.2, "contrast": 0.8}},
        {"id": "noise", "kind": "noise", "parameters": {"radius": 2.5, "percent": 220}},
        {"id": "tile", "kind": "tile", "parameters": {"tile_width": 512, "tile_height": 256, "overlap_x": 32}},
        {"id": "display", "kind": "visualize", "parameters": {"label": "Main"}},
    ])

    assert nodes[1].parameters == {"x": 2, "y": 3, "width": 100, "height": 50}
    assert nodes[3].parameters == {"axis": "horizontal"}
    assert nodes[5].parameters == {"brightness": 1.2, "contrast": 0.8, "saturation": 1.0}
    assert nodes[6].parameters == {"radius": 2.5, "percent": 220, "threshold": 2}
    assert nodes[7].parameters == {
        "tile_width": 512,
        "tile_height": 256,
        "overlap_x": 32,
        "overlap_y": 128,
        "include_partial": True,
    }
    assert nodes[-1].parameters == {"label": "Main"}
    assert all(node.operator_version == "1.0.0" for node in nodes)


def test_model_feature_operator_requires_model_and_layer_and_normalizes_visualization_defaults() -> None:
    with pytest.raises(PipelineValidationError, match="selected model and layer"):
        normalize_parameters("model_feature", {"model_id": "fixture"}, node_id="feature")

    parameters = normalize_parameters(
        "model_feature",
        {"model_id": "fixture", "layer_id": "backbone.3"},
        node_id="feature",
    )

    assert parameters == {
        "model_id": "fixture",
        "layer_id": "backbone.3",
        "projection": "mean",
        "normalization": "minmax",
        "channel": 0,
        "clip": "p1p99",
    }
    assert BUILTIN_OPERATORS["model_feature"].size_behavior == "preserve"


def _extension_contract(kind: str, property_schema: dict[str, object]) -> OperatorContract:
    return OperatorContract(
        kind=kind,
        title="Validation fixture",
        description="Validates one parameter schema.",
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "preserve"},
        parameters_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "default": {},
            "properties": {
                "value": {
                    "title": "Value",
                    "description": "Value used by the validation fixture.",
                    **property_schema,
                }
            },
            "additionalProperties": False,
        },
    )


@pytest.mark.parametrize(
    "property_schema, message",
    [
        ({"type": "number", "minimum": "0", "default": 0}, "minimum must be a finite number"),
        ({"type": "number", "minimum": math.nan, "default": 0}, "minimum must be a finite number"),
        ({"type": "integer", "minimum": 2, "maximum": 1, "default": 1}, "minimum cannot exceed maximum"),
        ({"type": "integer", "minimum": 0.2, "maximum": 0.8}, "must contain an integer"),
        ({"type": "integer", "enum": [1, 1.5], "default": 1}, "integer"),
        ({"type": "integer", "default": True}, "integer"),
        ({"type": "boolean", "enum": [True, 0], "default": True}, "boolean"),
        ({"type": "string", "enum": ["a", "b"], "default": "c"}, "allowed value"),
        ({"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0}, "positive finite"),
        ({"type": "number", "minimum": 0, "maximum": 1, "x-ui": {"control": "drag"}}, "control is invalid"),
        ({"type": "number", "minimum": 0, "maximum": 1, "x-ui": {"control": "number", "role": "crop-x"}}, "role is invalid"),
        ({"type": "string", "x-ui": {"control": "slider"}}, "numeric x-ui control"),
        ({"type": "number", "exclusiveMinimum": 0, "default": 1}, "unsupported fields"),
    ],
)
def test_register_operator_contracts_rejects_invalid_schema_atomically(
    property_schema: dict[str, object],
    message: str,
) -> None:
    contract = _extension_contract("validation.invalid", property_schema)

    with pytest.raises(PipelineValidationError, match=message):
        register_operator_contracts([contract])

    assert contract.kind not in BUILTIN_OPERATORS


def test_normalize_parameters_validates_schema_defaults_instead_of_copying_them(monkeypatch) -> None:
    contract = _extension_contract(
        "validation.invalid-default",
        {"type": "number", "minimum": 0, "maximum": 1, "default": 2},
    )
    monkeypatch.setitem(BUILTIN_OPERATORS, contract.kind, contract)

    with pytest.raises(PipelineValidationError, match="maximum"):
        normalize_parameters(contract.kind, {}, node_id="default-node")


def test_normalize_parameters_enforces_multiple_of_steps(monkeypatch) -> None:
    contract = _extension_contract(
        "validation.step",
        {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0.05},
    )
    monkeypatch.setitem(BUILTIN_OPERATORS, contract.kind, contract)

    assert normalize_parameters(contract.kind, {"value": 0.15}) == {"value": 0.15}
    with pytest.raises(PipelineValidationError, match="multipleOf"):
        normalize_parameters(contract.kind, {"value": 0.16})


@pytest.mark.parametrize(
    "node, message",
    [
        ({"id": "x", "kind": "unknown"}, "Unknown"),
        ({"id": "x", "kind": "flip", "script": "os.system('id')"}, "unknown fields"),
        ({"id": "x", "kind": "color", "parameters": {"eval": "1+1"}}, "unknown fields"),
        ({"id": "x", "kind": "flip", "enabled": 1}, "boolean"),
        ({"id": "x", "kind": "flip", "parameters": {"axis": "diagonal"}}, "allowed value"),
        ({"id": "x", "kind": "rotate", "parameters": {"degrees": "90"}}, "integer"),
        ({"id": "x", "kind": "color", "parameters": {"brightness": True}}, "finite number"),
        ({"id": "x", "kind": "color", "parameters": {"brightness": math.nan}}, "finite number"),
        ({"id": "x", "kind": "noise", "parameters": {"percent": 501}}, "maximum"),
        ({"id": "x", "kind": "crop", "parameters": {"x": 1, "y": 2}}, "together"),
        ({"id": "x", "kind": "resize", "parameters": {"width": 100}}, "together"),
        ({"id": "x", "kind": "resize", "parameters": {"width": 10_000, "height": 10_000}}, "pixel budget"),
        ({"id": "x", "kind": "tile", "parameters": {"tile_width": 128, "overlap_x": 128}}, "overlap"),
    ],
)
def test_validate_nodes_rejects_unknown_extra_wrong_type_and_out_of_range(node: dict, message: str) -> None:
    with pytest.raises(PipelineValidationError, match=message):
        validate_nodes([node])


def test_validate_nodes_enforces_unique_ids_and_strict_graph_boundaries() -> None:
    with pytest.raises(PipelineValidationError, match="unique"):
        validate_nodes([{"id": "same", "kind": "flip"}, {"id": "same", "kind": "color"}])
    with pytest.raises(PipelineValidationError, match="exactly one source"):
        validate_nodes([{"id": "flip", "kind": "flip"}, {"id": "source", "kind": "source"}, {"id": "display", "kind": "visualize"}])
    with pytest.raises(PipelineValidationError, match="Legacy output"):
        validate_nodes([{"id": "source", "kind": "source"}, {"id": "output", "kind": "output"}])
    with pytest.raises(PipelineValidationError, match="final node"):
        validate_nodes([{"id": "source", "kind": "source"}, {"id": "display", "kind": "visualize"}, {"id": "flip", "kind": "flip"}])


def test_validate_nodes_accepts_distinct_visualization_taps_and_rejects_connected_visualizations() -> None:
    nodes = validate_nodes([
        {"id": "source", "kind": "source"},
        {"id": "before", "kind": "visualize", "parameters": {"label": "Before"}},
        {"id": "flip", "kind": "flip"},
        {"id": "after", "kind": "visualize", "parameters": {"label": "After"}},
    ])

    assert [node.id for node in nodes if node.kind == "visualize"] == ["before", "after"]
    with pytest.raises(PipelineValidationError, match="连续显示节点无效"):
        validate_nodes([
            {"id": "source", "kind": "source"},
            {"id": "first-display", "kind": "visualize"},
            {"id": "second-display", "kind": "visualize"},
        ])
    with pytest.raises(PipelineValidationError, match="between one and four"):
        validate_nodes([{"id": "source", "kind": "source"}])
    with pytest.raises(PipelineValidationError, match="between one and four"):
        validate_nodes([
            {"id": "source", "kind": "source"},
            *({"id": f"display-{index}", "kind": "visualize"} for index in range(5)),
        ])


def test_validate_nodes_allows_only_one_model_feature_node() -> None:
    with pytest.raises(PipelineValidationError, match="最多只能包含一个模型中间层"):
        validate_nodes([
            {"id": "source", "kind": "source"},
            {"id": "feature-a", "kind": "model_feature", "parameters": {"model_id": "fixture", "layer_id": "a"}},
            {"id": "feature-b", "kind": "model_feature", "parameters": {"model_id": "fixture", "layer_id": "b"}},
            {"id": "display", "kind": "visualize"},
        ])


def test_legacy_normalization_maps_output_and_adds_only_missing_boundaries() -> None:
    normalized = normalize_legacy_nodes([
        {"id": "color", "kind": "color"},
        {"id": "old-output", "kind": "output"},
    ])
    nodes = validate_nodes(normalized)

    assert [node.kind for node in nodes] == ["source", "color", "visualize"]
    assert nodes[-1].id == "old-output"
    assert nodes[-1].parameters == {"label": "显示"}


def test_composite_expands_builtins_normalizes_defaults_and_tracks_dimensions() -> None:
    registry = CompositeRegistry()
    composite = registry.register({
        "id": "prepare-image",
        "name": "Prepare image",
        "description": "Resize, rotate, and adjust color",
        "steps": [
            {"id": "resize", "kind": "resize", "parameters": {"width": 200, "height": 100}},
            {"id": "rotate", "kind": "rotate", "parameters": {"degrees": 90}},
            {"id": "color", "kind": "color", "parameters": {"brightness": 1.2}},
        ],
    })

    expanded = registry.expand("prepare-image", input_width=640, input_height=480)

    assert len(composite.version_hash) == 64
    assert [node.kind for node in expanded.nodes] == ["resize", "rotate", "color"]
    assert expanded.nodes[-1].parameters == {"brightness": 1.2, "contrast": 1.1, "saturation": 1.0}
    assert (expanded.output_width, expanded.output_height) == (100, 200)


def test_composite_version_hash_is_deterministic_and_changes_with_behavior() -> None:
    definition = {
        "id": "stable",
        "name": "Stable",
        "steps": [{"kind": "flip", "parameters": {"axis": "vertical"}}],
    }
    first = CompositeRegistry().register(definition)
    second = CompositeRegistry().register({"steps": definition["steps"], "name": "Stable", "id": "stable"})
    changed = CompositeRegistry().register({
        "id": "stable",
        "name": "Stable",
        "steps": [{"kind": "flip", "parameters": {"axis": "horizontal"}}],
    })

    assert first.version_hash == second.version_hash
    assert first.version_hash != changed.version_hash


def test_composite_references_expand_only_to_registered_builtin_leaves() -> None:
    registry = CompositeRegistry()
    registry.register({"id": "base", "name": "Base", "steps": [{"kind": "flip"}]})
    wrapper = registry.register({
        "id": "wrapper",
        "name": "Wrapper",
        "steps": [{"composite": "base"}, {"composite": "base"}, {"kind": "noise", "enabled": False}],
    })

    expanded = registry.expand(wrapper.id, input_width=100, input_height=50)

    assert [node.kind for node in expanded.nodes] == ["flip", "flip", "noise"]
    assert len({node.id for node in expanded.nodes}) == len(expanded.nodes)
    assert all(node.kind in BUILTIN_OPERATORS for node in expanded.nodes)
    assert (expanded.output_width, expanded.output_height) == (100, 50)


@pytest.mark.parametrize(
    "definition, message",
    [
        ({"id": "bad", "name": "Bad", "python": "import os", "steps": [{"kind": "flip"}]}, "forbidden"),
        ({"id": "bad", "name": "Bad", "steps": [{"kind": "flip", "code": "eval('1')"}]}, "forbidden"),
        ({"id": "bad", "name": "Bad", "steps": [{"kind": "__import__", "parameters": {}}]}, "Unknown"),
        ({"id": "loop", "name": "Loop", "steps": [{"composite": "loop"}]}, "Recursive"),
        ({"id": "missing", "name": "Missing", "steps": [{"composite": "not-registered"}]}, "Unknown composite"),
    ],
)
def test_composite_rejects_code_unknown_and_recursive_fields(definition: dict, message: str) -> None:
    with pytest.raises(PipelineValidationError, match=message):
        CompositeRegistry().register(definition)


def test_composite_depth_node_and_output_pixel_budgets_are_enforced() -> None:
    depth = CompositeRegistry(maximum_depth=2)
    depth.register({"id": "one", "name": "One", "steps": [{"kind": "flip"}]})
    depth.register({"id": "two", "name": "Two", "steps": [{"composite": "one"}]})
    with pytest.raises(PipelineValidationError, match="depth"):
        depth.register({"id": "three", "name": "Three", "steps": [{"composite": "two"}]})

    nodes = CompositeRegistry(maximum_nodes=2)
    with pytest.raises(PipelineValidationError, match="node budget"):
        nodes.register({
            "id": "too-many",
            "name": "Too many",
            "steps": [{"kind": "flip"}, {"kind": "color"}, {"kind": "noise"}],
        })

    pixels = CompositeRegistry(maximum_output_pixels=10_000)
    pixels.register({
        "id": "huge",
        "name": "Huge",
        "steps": [{"kind": "resize", "parameters": {"width": 200, "height": 100}}],
    })
    with pytest.raises(PipelineValidationError, match="pixel budget"):
        pixels.expand("huge", input_width=10, input_height=10)


def test_composite_crop_is_checked_against_runtime_dimensions() -> None:
    registry = CompositeRegistry()
    registry.register({
        "id": "bounded-crop",
        "name": "Crop",
        "steps": [{"kind": "crop", "parameters": {"x": 80, "y": 0, "width": 30, "height": 20}}],
    })

    with pytest.raises(PipelineValidationError, match="current image"):
        registry.expand("bounded-crop", input_width=100, input_height=50)
