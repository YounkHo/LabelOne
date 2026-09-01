from __future__ import annotations

from pathlib import Path

from labelone.models.catalog import ModelCatalog
from labelone.models.types import AvailabilityState, FeatureCaptureMode


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_imports_official_x_anylabeling_catalog_and_reports_orphans(tmp_path: Path) -> None:
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    _write(config_root / "models.yaml", """
- model_name: yolo-test
  config_file: :/yolo.yaml
- model_name: unknown-test
  config_file: :/unknown.yaml
""")
    _write(auto / "yolo.yaml", """
type: yolov8
name: yolo-test
provider: Test
display_name: YOLO Test
model_path: model.onnx
classes: [0, person]
conf_threshold: 0.25
iou_threshold: 0.45
""")
    (auto / "model.onnx").write_bytes(b"fixture")
    _write(auto / "unknown.yaml", "type: custom_magic\nname: unknown-test\ndisplay_name: Unknown\n")
    _write(auto / "orphan.yaml", "type: yolov8\nname: orphan\n")

    catalog = ModelCatalog()
    response = catalog.import_x_anylabeling(tmp_path)

    assert len(response.models) == 2
    assert any(warning.code == "orphan_config" and warning.path.name == "orphan.yaml" for warning in response.warnings)
    yolo = catalog.get("yolo-test")
    assert yolo.descriptor.adapter == "yolo_detection_onnx"
    assert yolo.descriptor.availability.state is AvailabilityState.AVAILABLE
    assert yolo.descriptor.capabilities.feature_capture.mode is FeatureCaptureMode.GRAPH_REWRITE
    yolo_parameters = yolo.descriptor.capabilities.parameters_schema
    assert yolo_parameters["additionalProperties"] is False
    assert yolo_parameters["properties"]["conf_threshold"] == {
        "title": "置信度",
        "description": "过滤低置信度检测结果。",
        "type": "number",
        "default": 0.25,
        "minimum": 0.0,
        "maximum": 1.0,
    }
    assert yolo_parameters["properties"]["iou_threshold"]["default"] == 0.45
    assert yolo.config["classes"] == ["0", "person"]
    unknown = catalog.get("unknown-test").descriptor
    assert unknown.availability.state is AvailabilityState.UNSUPPORTED
    assert not unknown.capabilities.predict


def test_bad_yaml_does_not_block_remaining_catalog(tmp_path: Path) -> None:
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    _write(config_root / "models.yaml", """
- model_name: bad
  config_file: :/bad.yaml
- model_name: good
  config_file: :/good.yaml
""")
    _write(auto / "bad.yaml", "{invalid")
    _write(auto / "good.yaml", "type: remote_server\nname: good\ndisplay_name: Good\n")

    catalog = ModelCatalog()
    response = catalog.import_x_anylabeling(tmp_path)

    assert [model.id for model in response.models] == ["good"]
    assert any(warning.code == "invalid_yaml" for warning in response.warnings)


def test_common_yolo_onnx_families_select_runnable_adapters(tmp_path: Path) -> None:
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    entries = [
        ("obb", "yolo11_obb", "yolo_obb_onnx", "rotated_detection"),
        ("pose", "yolov8_pose", "yolo_pose_onnx", "pose"),
        ("classification", "yolov8_cls", "yolo_classification_onnx", "classification"),
        ("segmentation", "yolo11_seg", "yolo_segmentation_onnx", "segmentation"),
        ("detr", "rtdetrv2", "detr_detection_onnx", "detection"),
        ("depth", "depth_anything_v2", "depth_anything_onnx", "depth"),
        ("rmbg", "rmbg", "rmbg_matting_onnx", "segmentation"),
        ("ram", "ram", "ram_tagging_onnx", "tagging"),
    ]
    _write(
        config_root / "models.yaml",
        "\n".join(f"- model_name: {name}\n  config_file: :/{name}.yaml" for name, *_ in entries),
    )
    for name, model_type, _, _ in entries:
        _write(
            auto / f"{name}.yaml",
            f"type: {model_type}\nname: {name}\nmodel_path: {name}.onnx\nclasses: [object]\n",
        )
        (auto / f"{name}.onnx").write_bytes(b"fixture")

    catalog = ModelCatalog()
    response = catalog.import_x_anylabeling(tmp_path)

    assert not response.warnings
    for name, _, adapter, task in entries:
        descriptor = catalog.get(name).descriptor
        assert descriptor.adapter == adapter
        assert descriptor.task == task
        assert descriptor.capabilities.predict is True
    assert set(catalog.get("segmentation").descriptor.capabilities.parameters_schema["properties"]) >= {"conf_threshold", "iou_threshold", "mask_threshold", "max_det"}
    assert set(catalog.get("classification").descriptor.capabilities.parameters_schema["properties"]) == {"top_k", "temperature"}
    assert set(catalog.get("depth").descriptor.capabilities.parameters_schema["properties"]) == {"percentile_low", "percentile_high", "inverse", "color_map", "raster_format"}
    assert set(catalog.get("ram").descriptor.capabilities.parameters_schema["properties"]) == {"threshold", "top_k"}


def test_ppocr_requires_all_stage_weights_and_maps_supported_versions(tmp_path: Path) -> None:
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    _write(config_root / "models.yaml", "- model_name: ocr\n  config_file: :/ocr.yaml\n")
    _write(
        auto / "ocr.yaml",
        "type: ppocr_v6\nname: ocr\ndet_model_path: det.onnx\nrec_model_path: rec.onnx\ncls_model_path: cls.onnx\nuse_angle_cls: true\n",
    )
    for name in ("det.onnx", "rec.onnx", "cls.onnx"):
        (auto / name).write_bytes(b"fixture")

    catalog = ModelCatalog()
    catalog.import_x_anylabeling(tmp_path)
    descriptor = catalog.get("ocr").descriptor

    assert descriptor.adapter == "ppocr_onnx"
    assert descriptor.availability.state is AvailabilityState.AVAILABLE
    assert descriptor.capabilities.result_kinds == ["annotations", "tensors"]


def test_sam_requires_encoder_and_decoder_weights(tmp_path: Path) -> None:
    config_root = tmp_path / "anylabeling" / "configs"
    auto = config_root / "auto_labeling"
    _write(config_root / "models.yaml", "- model_name: sam\n  config_file: :/sam.yaml\n")
    _write(
        auto / "sam.yaml",
        "type: segment_anything\nname: sam\nencoder_model_path: encoder.onnx\ndecoder_model_path: decoder.onnx\n",
    )
    (auto / "encoder.onnx").write_bytes(b"fixture")
    (auto / "decoder.onnx").write_bytes(b"fixture")

    catalog = ModelCatalog()
    catalog.import_x_anylabeling(tmp_path)
    descriptor = catalog.get("sam").descriptor

    assert descriptor.adapter == "segment_anything_onnx"
    assert descriptor.availability.state is AvailabilityState.AVAILABLE
    assert descriptor.capabilities.result_kinds == ["annotations", "rasters", "tensors"]
