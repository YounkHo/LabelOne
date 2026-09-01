from __future__ import annotations

from pathlib import Path

from labelone.pipelines import CompositeDefinitionStore, CompositeRegistry
from labelone.errors import InvalidPathError
import pytest


def test_composite_definition_store_round_trips_in_dependency_order(tmp_path: Path) -> None:
    store = CompositeDefinitionStore(tmp_path / "pipeline-composites.json")
    base = {"id": "base", "name": "Base", "steps": [{"kind": "flip"}]}
    wrapper = {"id": "wrapper", "name": "Wrapper", "steps": [{"composite": "base"}, {"kind": "color"}]}
    store.append(base)
    store.append(wrapper)

    registry = CompositeRegistry()
    for definition in store.load():
        registry.register(definition)

    expanded = registry.expand("wrapper", input_width=100, input_height=50)
    assert [node.kind for node in expanded.nodes] == ["flip", "color"]
    assert not list(tmp_path.glob("*.part"))


def test_corrupt_composite_store_reports_a_recoverable_error(tmp_path: Path) -> None:
    path = tmp_path / "pipeline-composites.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(InvalidPathError, match="unreadable"):
        CompositeDefinitionStore(path).load()
