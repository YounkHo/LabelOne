from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.artifacts import ArtifactStore


def test_raster_artifact_is_atomic_and_content_is_resolved_inside_store(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_raster(
        model_id="depth",
        image_path=tmp_path / "source.png",
        role="depth-map",
        image=Image.new("L", (16, 8), 128),
        metadata={"minimum": 0.1, "maximum": 4.2},
    )

    path, media_type = store.content_path(artifact.id)

    assert path == artifact.path
    assert media_type == "image/png"
    assert Image.open(path).size == (16, 8)
    assert store.get_manifest(artifact.id)["metadata"]["maximum"] == 4.2
    assert not list(store.root.rglob("*.part"))

    limited = ArtifactStore(tmp_path / "limited", max_raster_pixels=10)
    with pytest.raises(ValueError, match="pixel budget"):
        limited.put_raster(
            model_id="depth",
            image_path=tmp_path / "source.png",
            role="too-large",
            image=Image.new("L", (4, 4)),
        )


def test_tensor_artifact_statistics_remain_finite_for_extreme_values(tmp_path: Path) -> None:
    maximum = np.finfo(np.float32).max
    tensor = np.array([maximum, -maximum], dtype=np.float32)
    store = ArtifactStore(tmp_path / "artifacts")

    artifact = store.put_tensor(
        model_id="feature",
        image_path=tmp_path / "source.png",
        layer_id="extreme",
        tensor=tensor,
    )
    manifest = store.get_manifest(artifact.id)

    assert all(np.isfinite(value) for value in artifact.statistics.values())
    assert all(np.isfinite(value) for value in manifest["statistics"].values())
    assert artifact.statistics["mean"] == 0.0


def test_tensor_artifact_rejects_non_finite_values_without_leaving_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ModelRuntimeError, match="non-finite"):
        store.put_tensor(
            model_id="feature",
            image_path=tmp_path / "source.png",
            layer_id="invalid",
            tensor=np.array([0.0, np.nan, np.inf], dtype=np.float32),
        )

    assert not list(store.root.iterdir())


def test_tensor_artifact_can_be_discarded_after_transient_pipeline_use(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_tensor(
        model_id="feature",
        image_path=tmp_path / "source.png",
        layer_id="temporary",
        tensor=np.arange(8, dtype=np.float32),
    )

    store.discard(artifact.id)

    assert not artifact.path.exists()
    assert not (store.root / artifact.id).exists()
