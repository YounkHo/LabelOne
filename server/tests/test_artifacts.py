from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.artifacts import ArtifactStore


def test_tensor_artifact_is_atomically_persisted(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    tensor = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)

    artifact = store.put_tensor(
        model_id="model",
        image_path=tmp_path / "image.png",
        layer_id="backbone.stage3/out:0",
        tensor=tensor,
    )

    assert artifact.path.is_file()
    assert artifact.shape == [1, 2, 3, 4]
    assert artifact.statistics["max"] == 23.0
    assert not list(artifact.path.parent.glob("*.part"))
    manifest = store.get_manifest(artifact.id)
    assert manifest["layer_id"] == "backbone.stage3/out:0"
    np.testing.assert_array_equal(np.load(artifact.path), tensor)


def test_scalar_spatial_tensor_persists_a_real_png_preview(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    tensor = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)

    artifact = store.put_tensor(
        model_id="model",
        image_path=tmp_path / "image.png",
        layer_id="backbone.hidden",
        tensor=tensor,
    )
    preview_path, media_type = store.preview_path(artifact.id)

    assert artifact.preview_available is True
    assert artifact.preview_width == 4
    assert artifact.preview_height == 3
    assert media_type == "image/png"
    with Image.open(preview_path) as preview:
        assert preview.mode == "RGB"
        assert preview.size == (4, 3)


def test_tensor_artifact_enforces_a_server_side_byte_budget(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", max_tensor_bytes=16)

    with pytest.raises(ModelRuntimeError, match="byte budget"):
        store.put_tensor(
            model_id="model",
            image_path=tmp_path / "image.png",
            layer_id="too-large",
            tensor=np.zeros((8,), dtype=np.float32),
        )

    assert not list((tmp_path / "artifacts").iterdir())
