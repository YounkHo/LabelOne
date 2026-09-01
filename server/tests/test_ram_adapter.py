from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.ram import RamTaggingOnnxAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


def _adapter(tmp_path: Path, config: dict[str, object] | None = None, *, config_path: Path | None = None) -> RamTaggingOnnxAdapter:
    path = config_path or tmp_path / "ram.yaml"
    descriptor = ModelDescriptor(
        id="ram-fixture",
        name="ram-fixture",
        display_name="RAM Fixture",
        model_type="ram",
        task="tagging",
        family="ram",
        adapter=RamTaggingOnnxAdapter.ADAPTER_ID,
        runtime=["ONNX Runtime"],
        config_path=path,
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True, result_kinds=["classifications"]),
    )
    return RamTaggingOnnxAdapter(
        ModelRecord(descriptor=descriptor, config=config or {"tag_list": ["cat", "dog", "tree"]}),
        ArtifactStore(tmp_path / "artifacts"),
    )


def _constant_ram_model(path: Path) -> None:
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 2, 2])
    tags = helper.make_tensor_value_info("tags", TensorProto.FLOAT, [1, 3])
    batch = helper.make_tensor_value_info("bs", TensorProto.INT64, [1])
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["tags"],
            value=helper.make_tensor("tag_values", TensorProto.FLOAT, [1, 3], [0.8, 0.2, 0.9]),
        ),
        helper.make_node(
            "Constant",
            [],
            ["bs"],
            value=helper.make_tensor("batch_value", TensorProto.INT64, [1], [1]),
        ),
    ]
    graph = helper.make_graph(nodes, "ram-fixture", [image], [tags, batch])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def test_real_onnx_prediction_returns_multilabel_classifications(tmp_path: Path) -> None:
    model_path = tmp_path / "ram.onnx"
    config_path = tmp_path / "ram.yaml"
    image_path = tmp_path / "image.png"
    _constant_ram_model(model_path)
    Image.new("RGB", (4, 3), (30, 60, 90)).save(image_path)
    adapter = _adapter(
        tmp_path,
        {"model_path": "ram.onnx", "tag_list": ["cat", "dog", "tree"], "threshold": 0.5},
        config_path=config_path,
    )
    adapter.record.descriptor.weight_locations = ["ram.onnx"]

    adapter.load(["CPUExecutionProvider"])
    result = adapter.predict(image_path, [], {})

    assert result.annotations == []
    assert [item.label for item in result.classifications] == ["tree", "cat"]
    assert [item.score for item in result.classifications] == pytest.approx([0.9, 0.8])
    assert [item.rank for item in result.classifications] == [1, 2]
    assert result.rasters == []


def test_binary_tags_batch_output_delete_index_and_top_k(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        {
            "tag_list": ["cat", "dog", "tree", "sky"],
            "delete_tag_index": [2],
        },
    )

    results = adapter._classifications(
        {
            "tags": np.array([[True, True, True, False]], dtype=np.bool_),
            "bs": np.array([1], dtype=np.int64),
        },
        {"top_k": 2},
    )

    assert [item.label for item in results] == ["cat", "dog"]
    assert [item.score for item in results] == [1.0, 1.0]


def test_logits_are_sigmoided_then_thresholded_and_ranked(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, {"tag_list": ["cat", "dog", "tree", "sky"]})

    results = adapter._classifications(
        {"logits": np.array([[-2.0, 2.0, 0.0, 1.0]], dtype=np.float32)},
        {"score_activation": "sigmoid", "threshold": 0.6, "top_k": 1},
    )

    assert [item.label for item in results] == ["dog"]
    assert results[0].score == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_probability_output_filter_tags_and_configured_output(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        {
            "tag_list": ["cat", "dog", "tree", "sky"],
            "filter_tags": ["dog", "sky"],
            "output_name": "probabilities",
            "threshold": 0.65,
        },
    )

    results = adapter._classifications(
        {
            "embedding": np.zeros((1, 8, 8), dtype=np.float32),
            "probabilities": np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        },
        {},
    )

    assert [item.label for item in results] == ["dog"]
    assert results[0].score == pytest.approx(0.8)


def test_x_anylabeling_source_tag_resource_is_discovered_without_copying_it(tmp_path: Path) -> None:
    source = tmp_path / "source" / "anylabeling"
    config_path = source / "configs" / "auto_labeling" / "ram.yaml"
    tag_path = source / "services" / "auto_labeling" / "configs" / "ram" / "ram_tag_list.txt"
    config_path.parent.mkdir(parents=True)
    tag_path.parent.mkdir(parents=True)
    config_path.write_text("type: ram\n", encoding="utf-8")
    tag_path.write_text("cat\ndog\ntree\n", encoding="utf-8")
    adapter = _adapter(tmp_path, {}, config_path=config_path)

    results = adapter._classifications(
        {"tags": np.array([[0, 1, 0]], dtype=np.int64), "bs": np.array([1])},
        {},
    )

    assert [item.label for item in results] == ["dog"]


def test_tag_file_cannot_escape_imported_source(tmp_path: Path) -> None:
    config_path = tmp_path / "models" / "ram.yaml"
    config_path.parent.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    adapter = _adapter(tmp_path, {"tag_list_path": "../outside.txt"}, config_path=config_path)

    with pytest.raises(ModelRuntimeError, match="escapes"):
        adapter._classifications({"tags": np.array([[1]], dtype=np.int64)}, {})


@pytest.mark.parametrize(
    "outputs, parameters, message",
    [
        ({}, {}, "returned no outputs"),
        ({"tags": np.ones((2, 3), dtype=np.float32)}, {}, "batch dimension"),
        (
            {"tags": np.ones((1, 3), dtype=np.float32), "bs": np.array([2])},
            {},
            "batch output must equal one",
        ),
        ({"tags": np.ones((1, 2), dtype=np.float32)}, {}, "must have shape"),
        ({"tags": np.array([[0.2, np.nan, 0.8]])}, {}, "finite numeric"),
        (
            {"first": np.ones((1, 3), dtype=np.float32), "second": np.ones((1, 3), dtype=np.float32)},
            {},
            "multiple tag score vectors",
        ),
        ({"tags": np.array([[2.0, 0.2, 0.8]])}, {"score_activation": "probability"}, "outside zero and one"),
        ({"tags": np.ones((1, 3), dtype=np.float32)}, {"score_activation": "softmax"}, "score_activation"),
        ({"tags": np.ones((1, 3), dtype=np.float32)}, {"threshold": 1.1}, "threshold"),
        ({"tags": np.ones((1, 3), dtype=np.float32)}, {"top_k": 0}, "top_k"),
        ({"tags": np.ones((1, 3), dtype=np.float32)}, {"delete_tag_index": [3]}, "out-of-range"),
    ],
)
def test_invalid_outputs_and_parameters_raise_clear_errors(
    tmp_path: Path,
    outputs: dict[str, np.ndarray],
    parameters: dict[str, object],
    message: str,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ModelRuntimeError, match=message):
        adapter._classifications(outputs, parameters)


def test_unknown_delete_or_filter_tags_are_rejected(tmp_path: Path) -> None:
    delete_adapter = _adapter(tmp_path, {"tag_list": ["cat", "dog"], "delete_tags": ["missing"]})
    with pytest.raises(ModelRuntimeError, match="unknown labels"):
        delete_adapter._classifications({"tags": np.ones((1, 2))}, {})

    filter_adapter = _adapter(tmp_path, {"tag_list": ["cat", "dog"], "filter_tags": ["missing"]})
    with pytest.raises(ModelRuntimeError, match="unknown labels"):
        filter_adapter._classifications({"tags": np.ones((1, 2))}, {})


def test_ram_preprocessing_stretches_rgb_and_applies_imagenet_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (255, 0, 128)).save(image_path)
    adapter = _adapter(tmp_path)
    adapter.input_meta = SimpleNamespace(shape=[1, 3, 2, 2], type="tensor(float)", name="image")

    tensor, transform = adapter._prepare_image(image_path)

    expected = np.array([
        (1.0 - 0.485) / 0.229,
        (0.0 - 0.456) / 0.224,
        ((128 / 255.0) - 0.406) / 0.225,
    ])
    assert tensor.shape == (1, 3, 2, 2)
    assert tensor[0, :, 0, 0] == pytest.approx(expected, abs=1e-5)
    assert transform.input_width == 2
    assert transform.input_height == 2
    assert transform.pad_x == 0
    assert transform.pad_y == 0


@pytest.mark.parametrize(
    "config, message",
    [
        ({"tag_list": ["cat"], "mean": [0.5, 0.5]}, "mean"),
        ({"tag_list": ["cat"], "std": [0.2, 0.0, 0.2]}, "std"),
    ],
)
def test_invalid_normalization_and_input_batch_raise_clear_errors(
    tmp_path: Path,
    config: dict[str, object],
    message: str,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)
    adapter = _adapter(tmp_path, config)
    adapter.input_meta = SimpleNamespace(shape=[1, 3, 2, 2], type="tensor(float)", name="image")
    with pytest.raises(ModelRuntimeError, match=message):
        adapter._prepare_image(image_path)

    with pytest.raises(ModelRuntimeError, match="input batch dimension"):
        adapter._configure_inputs([SimpleNamespace(shape=[2, 3, 2, 2], type="tensor(float)", name="image")])
