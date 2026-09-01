from __future__ import annotations

from pathlib import Path

from labelone.models.sources import ModelSourceStore


def test_model_source_store_is_atomic_and_tolerates_corruption(tmp_path: Path) -> None:
    store = ModelSourceStore(tmp_path / "settings" / "model-sources.json")
    assert store.x_anylabeling_root() is None

    source = tmp_path / "x-anylabeling"
    source.mkdir()
    store.set_x_anylabeling_root(source)
    assert store.x_anylabeling_root() == source.resolve()
    assert not list((tmp_path / "settings").glob("*.tmp"))

    (tmp_path / "settings" / "model-sources.json").write_text("{broken", encoding="utf-8")
    assert store.x_anylabeling_root() is None
