from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Mapping

from .registry import (
    BUILTIN_OPERATORS,
    PipelineValidationError,
    ValidatedNode,
    operator_registry_hash,
    validate_transform_nodes,
)


_COMPOSITE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class CompositeReference:
    composite_id: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class CompositeOperator:
    id: str
    name: str
    description: str
    steps: tuple[ValidatedNode | CompositeReference, ...]
    version_hash: str


@dataclass(frozen=True, slots=True)
class ExpandedComposite:
    composite_id: str
    version_hash: str
    nodes: tuple[ValidatedNode, ...]
    output_width: int
    output_height: int


class CompositeRegistry:
    def __init__(
        self,
        *,
        maximum_depth: int = 4,
        maximum_nodes: int = 64,
        maximum_output_pixels: int = 64_000_000,
    ) -> None:
        if maximum_depth < 1 or maximum_nodes < 1 or maximum_output_pixels < 1:
            raise ValueError("Composite budgets must be positive")
        self.maximum_depth = maximum_depth
        self.maximum_nodes = maximum_nodes
        self.maximum_output_pixels = maximum_output_pixels
        self._operators: dict[str, CompositeOperator] = {}

    def get(self, composite_id: str) -> CompositeOperator:
        try:
            return self._operators[composite_id]
        except KeyError as exc:
            raise PipelineValidationError(
                "Unknown composite operator",
                details={"composite_id": composite_id},
            ) from exc

    def register(self, definition: Mapping[str, object]) -> CompositeOperator:
        if not isinstance(definition, Mapping):
            raise PipelineValidationError("Composite definition must be an object")
        extras = sorted(str(key) for key in definition if key not in {"id", "name", "description", "steps"})
        if extras:
            raise PipelineValidationError(
                "Composite definition contains forbidden fields",
                details={"fields": extras},
            )
        composite_id = definition.get("id")
        name = definition.get("name")
        description = definition.get("description", "")
        steps = definition.get("steps")
        if not isinstance(composite_id, str) or not _COMPOSITE_ID.fullmatch(composite_id):
            raise PipelineValidationError("Composite id is invalid", details={"id": composite_id})
        if composite_id in BUILTIN_OPERATORS:
            raise PipelineValidationError("Composite id conflicts with a built-in operator", details={"id": composite_id})
        if composite_id in self._operators:
            raise PipelineValidationError("Composite id is already registered", details={"id": composite_id})
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise PipelineValidationError("Composite name is invalid")
        if not isinstance(description, str) or len(description) > 2000:
            raise PipelineValidationError("Composite description is invalid")
        if not isinstance(steps, list) or not steps:
            raise PipelineValidationError("Composite steps must be a non-empty array")

        parsed_steps: list[ValidatedNode | CompositeReference] = []
        builtin_nodes: list[dict[str, object]] = []
        builtin_positions: list[int] = []
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping):
                raise PipelineValidationError("Composite step must be an object", details={"step": index})
            if "composite" in raw_step:
                extras = sorted(str(key) for key in raw_step if key not in {"composite", "enabled"})
                if extras:
                    raise PipelineValidationError(
                        "Composite reference contains forbidden fields",
                        details={"step": index, "fields": extras},
                    )
                reference = raw_step.get("composite")
                enabled = raw_step.get("enabled", True)
                if not isinstance(reference, str) or not isinstance(enabled, bool):
                    raise PipelineValidationError("Composite reference is invalid", details={"step": index})
                if reference == composite_id:
                    raise PipelineValidationError("Recursive composite reference is forbidden", details={"id": composite_id})
                self.get(reference)
                parsed_steps.append(CompositeReference(reference, enabled))
                continue
            extras = sorted(
                str(key)
                for key in raw_step
                if key not in {"id", "kind", "enabled", "parameters"}
            )
            if extras:
                raise PipelineValidationError(
                    "Composite built-in step contains forbidden fields",
                    details={"step": index, "fields": extras},
                )
            kind = raw_step.get("kind")
            if kind in {"source", "output", "visualize"}:
                raise PipelineValidationError("Composite steps cannot contain graph boundary operators")
            node = dict(raw_step)
            node.setdefault("id", f"step-{index + 1}")
            builtin_positions.append(len(parsed_steps))
            builtin_nodes.append(node)
            parsed_steps.append(CompositeReference("__pending_builtin__"))

        validated = (
            iter(validate_transform_nodes(builtin_nodes, maximum_nodes=self.maximum_nodes))
            if builtin_nodes
            else iter(())
        )
        for position in builtin_positions:
            parsed_steps[position] = next(validated)
        dependency_hashes = {
            step.composite_id: self.get(step.composite_id).version_hash
            for step in parsed_steps
            if isinstance(step, CompositeReference)
        }
        payload = {
            "format": 1,
            "id": composite_id,
            "name": name.strip(),
            "description": description,
            "registry": operator_registry_hash(),
            "dependencies": dependency_hashes,
            "steps": [
                {"composite": step.composite_id, "enabled": step.enabled}
                if isinstance(step, CompositeReference)
                else step.as_dict()
                for step in parsed_steps
            ],
        }
        version_hash = sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        operator = CompositeOperator(
            id=composite_id,
            name=name.strip(),
            description=description,
            steps=tuple(parsed_steps),
            version_hash=version_hash,
        )
        self._operators[composite_id] = operator
        try:
            self._flatten(operator, depth=1, stack=())
        except Exception:
            self._operators.pop(composite_id, None)
            raise
        return operator

    def _flatten(
        self,
        operator: CompositeOperator,
        *,
        depth: int,
        stack: tuple[str, ...],
    ) -> list[ValidatedNode]:
        if depth > self.maximum_depth:
            raise PipelineValidationError(
                "Composite exceeds the nesting depth budget",
                details={"maximum": self.maximum_depth, "path": [*stack, operator.id]},
            )
        if operator.id in stack:
            raise PipelineValidationError(
                "Recursive composite reference is forbidden",
                details={"path": [*stack, operator.id]},
            )
        flattened: list[ValidatedNode] = []
        for step in operator.steps:
            if isinstance(step, ValidatedNode):
                flattened.append(step)
            elif step.enabled:
                flattened.extend(self._flatten(
                    self.get(step.composite_id),
                    depth=depth + 1,
                    stack=(*stack, operator.id),
                ))
            if len(flattened) > self.maximum_nodes:
                raise PipelineValidationError(
                    "Composite exceeds the expanded node budget",
                    details={"nodes": len(flattened), "maximum": self.maximum_nodes},
                )
        return flattened

    @staticmethod
    def _dimensions(node: ValidatedNode, width: int, height: int) -> tuple[int, int]:
        if not node.enabled:
            return width, height
        parameters = node.parameters
        if node.kind == "crop":
            if "width" in parameters:
                x, y = int(parameters["x"]), int(parameters["y"])
                target_width, target_height = int(parameters["width"]), int(parameters["height"])
                if x + target_width > width or y + target_height > height:
                    raise PipelineValidationError(
                        "Composite crop exceeds the current image",
                        details={"node_id": node.id, "input_size": [width, height]},
                    )
                return target_width, target_height
            margin = float(parameters["margin_ratio"])
            return max(1, round(width * (1 - margin * 2))), max(1, round(height * (1 - margin * 2)))
        if node.kind == "resize" and "width" in parameters:
            return int(parameters["width"]), int(parameters["height"])
        if node.kind == "rotate" and int(parameters["degrees"]) in {90, 270}:
            return height, width
        if node.kind == "tile":
            # The current preview contract is one image in / one image out. A
            # later multi-output exporter owns physical tile dimensions.
            return width, height
        return width, height

    def list(self) -> list[CompositeOperator]:
        return list(self._operators.values())

    def remove(self, composite_id: str) -> None:
        self._operators.pop(composite_id, None)

    def expand(self, composite_id: str, *, input_width: int, input_height: int) -> ExpandedComposite:
        if input_width <= 0 or input_height <= 0:
            raise PipelineValidationError("Composite input dimensions must be positive")
        operator = self.get(composite_id)
        flattened = self._flatten(operator, depth=1, stack=())
        nodes = [
            ValidatedNode(
                id=f"{index + 1}:{node.id}",
                kind=node.kind,
                enabled=node.enabled,
                parameters=node.parameters,
                operator_version=node.operator_version,
            )
            for index, node in enumerate(flattened)
        ]
        width, height = input_width, input_height
        for node in nodes:
            width, height = self._dimensions(node, width, height)
            if width * height > self.maximum_output_pixels:
                raise PipelineValidationError(
                    "Composite output exceeds the pixel budget",
                    details={
                        "node_id": node.id,
                        "size": [width, height],
                        "pixels": width * height,
                        "maximum": self.maximum_output_pixels,
                    },
                )
        return ExpandedComposite(operator.id, operator.version_hash, tuple(nodes), width, height)
