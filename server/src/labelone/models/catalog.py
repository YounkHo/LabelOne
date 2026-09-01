from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml

from labelone.errors import ModelCatalogError

from .types import (
    Availability,
    AvailabilityState,
    CatalogWarning,
    FeatureCapture,
    FeatureCaptureMode,
    ModelCapabilities,
    ModelCatalogResponse,
    ModelDescriptor,
)


@dataclass(frozen=True, slots=True)
class ModelRecord:
    descriptor: ModelDescriptor
    config: dict[str, Any]


def _config_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root / "anylabeling" / "configs" / "auto_labeling", root / "auto_labeling", root]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.yaml")):
            return candidate
    raise ModelCatalogError("Could not find X-AnyLabeling auto-labeling configs", details={"root_dir": str(root)})


def _catalog_paths(config_dir: Path) -> tuple[list[Path], list[CatalogWarning]]:
    all_configs = {path.resolve() for path in config_dir.glob("*.yaml")}
    models_path = config_dir.parent / "models.yaml"
    if not models_path.is_file():
        return sorted(all_configs), []
    try:
        payload = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ModelCatalogError("Could not read X-AnyLabeling models.yaml", details={"error": str(exc)}) from exc
    if not isinstance(payload, list):
        raise ModelCatalogError("X-AnyLabeling models.yaml must contain a list")
    official: list[Path] = []
    warnings: list[CatalogWarning] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("config_file"), str):
            warnings.append(CatalogWarning(path=models_path, code="invalid_catalog_entry", message=str(item)))
            continue
        raw = item["config_file"]
        relative = raw[2:] if raw.startswith(":/") else raw
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            warnings.append(CatalogWarning(path=models_path, code="unsafe_catalog_path", message=raw))
            continue
        resolved = (config_dir / relative).resolve()
        if config_dir.resolve() not in resolved.parents or not resolved.is_file():
            warnings.append(CatalogWarning(path=resolved, code="missing_catalog_config", message=raw))
            continue
        official.append(resolved)
    for orphan in sorted(all_configs - set(official)):
        warnings.append(CatalogWarning(path=orphan, code="orphan_config", message="YAML is not referenced by models.yaml"))
    return official, warnings


def _task_for(model_type: str) -> str:
    value = model_type.casefold()
    rules = [
        (("obb", "rotation"), "rotated_detection"),
        (("pose", "dwpose", "rtmo"), "pose"),
        (("ocr", "ppocr"), "ocr"),
        (("sam", "segment_anything"), "interactive_segmentation"),
        (("seg", "rmbg", "matting"), "segmentation"),
        (("track", "botsort", "bytetrack"), "tracking"),
        (("depth",), "depth"),
        (("ram", "tag"), "tagging"),
        (("ground", "world", "yoloe", "geco", "count"), "grounding"),
        (("vl", "florence", "vision"), "vision_language"),
        (("cls", "class"), "classification"),
    ]
    for needles, task in rules:
        if any(needle in value for needle in needles):
            return task
    return "detection"


def _adapter_for(model_type: str, config: dict[str, Any]) -> tuple[str, str, FeatureCaptureMode]:
    value = model_type.casefold()
    locations = _weight_locations(config)
    has_onnx = any(urlparse(location).path.casefold().endswith(".onnx") for location in locations)
    if value in {"remote_server", "grounding_dino_api"}:
        if _trusted_remote_reason(config, require_secret=True) is None:
            return "trusted_remote_http", "Trusted Remote HTTPS", FeatureCaptureMode.NONE
        return "remote_catalog", "Remote HTTP", FeatureCaptureMode.NONE
    if str(config.get("engine", "")).casefold() == "trt" or any(
        urlparse(location).path.casefold().endswith(".engine") for location in locations
    ):
        runtime = "TensorRT" if importlib.util.find_spec("tensorrt") is not None else "TensorRT (unavailable)"
        return "catalog_only", runtime, FeatureCaptureMode.NONE
    if has_onnx and value == "yoloe" and config.get("onnx_contract") == "ultralytics_detection_v1":
        if isinstance(config.get("classes"), (list, dict)) and not bool(config.get("with_mask", False)):
            return "yolo_detection_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and value in {"rtdetr", "rtdetrv2", "rt_detr", "rt_detrv2", "dfine", "d_fine", "deim", "deimv2"}:
        return "detr_detection_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and value in {"depth_anything", "depth_anything_v2"}:
        return "depth_anything_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and value == "rmbg":
        return "rmbg_matting_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and value == "ram":
        return "ram_tagging_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and value in {"ppocr_v4", "ppocr_v5", "ppocr_v6"}:
        required = [config.get("det_model_path"), config.get("rec_model_path")]
        if bool(config.get("use_angle_cls", True)):
            required.append(config.get("cls_model_path"))
        if all(isinstance(location, str) and urlparse(location).path.casefold().endswith(".onnx") for location in required):
            return "ppocr_onnx", "ONNX Runtime", FeatureCaptureMode.EXPORTED_OUTPUTS
    if has_onnx and value in {"segment_anything", "sam_hq"}:
        required = [config.get("encoder_model_path"), config.get("decoder_model_path")]
        if all(isinstance(location, str) and urlparse(location).path.casefold().endswith(".onnx") for location in required):
            return "segment_anything_onnx", "ONNX Runtime", FeatureCaptureMode.EXPORTED_OUTPUTS
    if has_onnx and re.fullmatch(r"(?:yolov5|yolov8|yolo11|yolo26)_obb", value):
        return "yolo_obb_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and re.fullmatch(r"(?:yolov8|yolo11|yolo26)_pose", value):
        return "yolo_pose_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and re.fullmatch(r"(?:yolov5|yolov8|yolo11)_cls", value):
        return "yolo_classification_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and re.fullmatch(r"(?:yolov5|yolov8|yolo11|yolo26)_seg", value):
        return "yolo_segmentation_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if has_onnx and re.fullmatch(r"(?:yolov[5-9]|yolov10|yolo11|yolo12|yolo26)(?:_sahi)?", value):
        return "yolo_detection_onnx", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if any(str(location).lower().endswith(".onnx") for location in _weight_locations(config)):
        return "onnx_raw", "ONNX Runtime", FeatureCaptureMode.GRAPH_REWRITE
    if "remote" in value or config.get("server_url") or config.get("api_url"):
        return "remote_catalog", "Remote HTTP", FeatureCaptureMode.NONE
    return "catalog_only", "Model-specific", FeatureCaptureMode.NONE


def _trusted_remote_reason(config: dict[str, Any], *, require_secret: bool) -> str | None:
    endpoint = config.get("remote_endpoint")
    trusted_hosts = config.get("trusted_hosts")
    credential_env = config.get("credential_env")
    protocol = config.get("remote_protocol")
    if not isinstance(endpoint, str) or not endpoint:
        return "Remote endpoint is not explicitly configured"
    parsed = urlparse(endpoint)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
        or parsed.query
    ):
        return "Remote endpoint must be credential-free HTTPS on the default port"
    hostname = (parsed.hostname or "").casefold().strip(".")
    if not isinstance(trusted_hosts, list) or not trusted_hosts or any(not isinstance(host, str) for host in trusted_hosts):
        return "Remote trusted_hosts must be an explicit non-empty list"
    if hostname not in {host.casefold().strip(".") for host in trusted_hosts}:
        return "Remote endpoint hostname is not explicitly trusted"
    if protocol not in {"labelone_v1", "grounding_dino_v2"}:
        return "Remote protocol is not supported"
    if not isinstance(credential_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", credential_env):
        return "Remote credential_env must name an environment variable"
    if require_secret and not os.getenv(credential_env):
        return "Remote credential environment variable is not set"
    if config.get("credential_header", "Token") not in {"Authorization", "Token", "X-API-Key"}:
        return "Remote credential header is not allowlisted"
    return None


def _unsupported_reason(model_type: str, config: dict[str, Any]) -> str:
    value = model_type.casefold()
    if value in {"remote_server", "grounding_dino_api"}:
        return _trusted_remote_reason(config, require_secret=True) or "Remote protocol adapter is unavailable"
    if value == "upn":
        missing = [name for name in ("torch", "torchvision", "mmengine", "chatrex") if importlib.util.find_spec(name) is None]
        if missing:
            return f"UPN requires its CUDA PyTorch/Chatrex runtime; missing: {', '.join(missing)}"
        return "UPN runtime is not supported by the local adapter contract"
    if value == "yoloe":
        if config.get("onnx_contract") != "ultralytics_detection_v1":
            return (
                "YOLOE PyTorch prompt models require Ultralytics and MobileCLIP; "
                "only explicit fixed-class ONNX detection exports are supported"
            )
        return "YOLOE ONNX configuration is missing fixed classes or requests mask output"
    if value == "florence2":
        missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
        suffix = f"; missing: {', '.join(missing)}" if missing else ""
        return f"Florence2 requires a pinned Transformers runtime and reviewed model code; trust_remote_code is not executed{suffix}"
    if value == "segment_anything_2_video":
        missing = [name for name in ("torch", "sam2") if importlib.util.find_spec(name) is None]
        suffix = f"; missing: {', '.join(missing)}" if missing else ""
        return f"SAM2 video requires the stateful SAM2 frame-session runtime, which is outside single-image predict{suffix}"
    if str(config.get("engine", "")).casefold() == "trt" or any(
        urlparse(location).path.casefold().endswith(".engine")
        for location in _weight_locations(config)
    ):
        if importlib.util.find_spec("tensorrt") is None:
            return "TensorRT engine is unavailable because the tensorrt Python runtime is not installed"
        return "TensorRT runtime is present but the engine execution adapter has not been implemented"
    return "Adapter has not been implemented"


def _weight_locations(config: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in config.items():
        if not (key.endswith("_path") or key in {"model", "weights"}):
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return list(dict.fromkeys(values))


def _local_weight_paths(config_path: Path, locations: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for location in locations:
        parsed = urlparse(location)
        if parsed.scheme in {"http", "https"}:
            continue
        path = Path(location).expanduser()
        paths.append((path if path.is_absolute() else config_path.parent / path).resolve())
    return paths


def _required_weight_locations(adapter: str, config: dict[str, Any], locations: list[str]) -> list[str]:
    if adapter == "ppocr_onnx":
        keys = ["det_model_path", "rec_model_path"]
        if bool(config.get("use_angle_cls", True)):
            keys.append("cls_model_path")
        return [str(config[key]) for key in keys if isinstance(config.get(key), str)]
    if adapter == "segment_anything_onnx":
        return [str(config[key]) for key in ("encoder_model_path", "decoder_model_path") if isinstance(config.get(key), str)]
    return locations


def _parameters_schema(config: dict[str, Any], adapter: str) -> dict[str, Any]:
    properties: dict[str, Any] = {}

    def configured(name: str, default: object, *aliases: str) -> object:
        for key in (name, *aliases):
            if key in config:
                return config[key]
        return default

    def number(name: str, title: str, default: float, minimum: float, maximum: float, description: str) -> None:
        try:
            value = float(configured(name, default))
        except (TypeError, ValueError):
            value = default
        if not minimum <= value <= maximum:
            value = default
        properties[name] = {"title": title, "description": description, "type": "number", "default": value, "minimum": minimum, "maximum": maximum}

    def integer(name: str, title: str, default: int, minimum: int, maximum: int, description: str) -> None:
        try:
            value = int(configured(name, default))
        except (TypeError, ValueError):
            value = default
        if not minimum <= value <= maximum:
            value = default
        properties[name] = {"title": title, "description": description, "type": "integer", "default": value, "minimum": minimum, "maximum": maximum}

    def boolean(name: str, title: str, default: bool, description: str) -> None:
        raw = configured(name, default)
        properties[name] = {"title": title, "description": description, "type": "boolean", "default": raw if isinstance(raw, bool) else default}

    def choice(name: str, title: str, default: str, values: list[str], description: str, *aliases: str) -> None:
        value = str(configured(name, default, *aliases)).casefold()
        properties[name] = {"title": title, "description": description, "type": "string", "default": value if value in values else default, "enum": values}

    if adapter in {"yolo_detection_onnx", "yolo_obb_onnx", "yolo_pose_onnx", "yolo_segmentation_onnx", "trusted_remote_http"}:
        number("conf_threshold", "置信度", 0.25, 0.0, 1.0, "过滤低置信度检测结果。")
        number("iou_threshold", "IoU 阈值", 0.8 if adapter == "trusted_remote_http" else 0.45, 0.0, 1.0, "非极大值抑制的重叠阈值。")
    if adapter == "yolo_obb_onnx":
        integer("max_det", "最大检测数", 300, 1, 10_000, "单张图片最多保留的旋转框数量。")
    elif adapter == "yolo_pose_onnx":
        try:
            default = float(configured("keypoint_threshold", 0.25, "kpt_threshold"))
        except (TypeError, ValueError):
            default = 0.25
        if not 0.0 <= default <= 1.0:
            default = 0.25
        properties["keypoint_threshold"] = {"title": "关键点阈值", "description": "过滤低置信度关键点。", "type": "number", "default": default, "minimum": 0.0, "maximum": 1.0}
    elif adapter == "yolo_segmentation_onnx":
        number("mask_threshold", "Mask 阈值", 0.5, 0.0, 1.0, "将实例 mask 转换为前景区域的阈值。")
        integer("max_det", "最大实例数", 100, 1, 10_000, "单张图片最多保留的实例数量。")
        number("min_mask_area", "最小 Mask 面积", 4.0, 0.0, 1_000_000_000.0, "丢弃面积过小的分割区域。")
        number("polygon_simplify", "轮廓简化", 1.0, 0.0, 1_000.0, "多边形轮廓的简化强度。")
    elif adapter == "yolo_classification_onnx":
        integer("top_k", "Top-K", 5, 1, 1_000, "返回得分最高的类别数量。")
        number("temperature", "Softmax 温度", 1.0, 0.01, 100.0, "调整分类概率分布的平滑程度。")
    elif adapter == "detr_detection_onnx":
        number("conf_threshold", "置信度", 0.25, 0.0, 1.0, "过滤低置信度 DETR 查询。")
        integer("top_k", "Top-K", int(configured("top_k", configured("max_det", 300))), 1, 10_000, "最多保留的查询结果数量。")
    elif adapter == "depth_anything_onnx":
        number("percentile_low", "低百分位", 2.0, 0.0, 100.0, "深度可视化归一化的低截断百分位。")
        number("percentile_high", "高百分位", 98.0, 0.0, 100.0, "深度可视化归一化的高截断百分位。")
        boolean("inverse", "反转深度", False, "反转近处和远处的显示亮度。")
        choice("color_map", "色图", "turbo", ["turbo", "grayscale"], "深度栅格的颜色映射。", "colormap")
        choice("raster_format", "输出格式", "png", ["png", "webp", "jpeg"], "深度预览的栅格格式。")
    elif adapter == "rmbg_matting_onnx":
        number("mask_threshold", "Mask 阈值", 0.0, 0.0, 1.0, "低于阈值的透明度会被清零。")
        boolean("output_cutout", "输出前景抠图", False, "同时生成带透明通道的前景图。")
    elif adapter == "ram_tagging_onnx":
        try:
            threshold = float(configured("threshold", configured("tag_threshold", configured("confidence_threshold", 0.5))))
        except (TypeError, ValueError):
            threshold = 0.5
        if not 0.0 <= threshold <= 1.0:
            threshold = 0.5
        properties["threshold"] = {"title": "标签阈值", "description": "过滤低概率图像标签。", "type": "number", "default": threshold, "minimum": 0.0, "maximum": 1.0}
        integer("top_k", "Top-K", int(configured("top_k", 50)), 1, 10_000, "最多返回的标签数量。")
    elif adapter == "ppocr_onnx":
        number("det_db_thresh", "文本像素阈值", 0.3, 0.0, 1.0, "DB 检测图的二值化阈值。")
        number("det_db_box_thresh", "文本框阈值", 0.6, 0.0, 1.0, "过滤低分文本框。")
        number("det_db_unclip_ratio", "文本框扩张", 1.5, 0.0, 10.0, "文本框轮廓的扩张比例。")
        integer("max_boxes", "最大文本框", 300, 1, 10_000, "单图最多识别的文本框数量。")
        number("drop_score", "识别分数阈值", 0.5, 0.0, 1.0, "过滤低分 OCR 文本。")
        boolean("skip_angle_cls", "跳过方向分类", False, "跳过文字方向分类阶段。")
    elif adapter == "segment_anything_onnx":
        number("mask_threshold", "Mask 阈值", 0.0, -100.0, 100.0, "将 SAM logits 转换为前景区域的阈值。")
        integer("max_mask_components", "最大连通区域", 16, 1, 1_024, "单次结果最多保留的连通区域数量。")
        number("min_mask_area", "最小区域面积", 1.0, 0.0, 1_000_000_000.0, "丢弃面积过小的分割区域。")
        number("polygon_simplify", "轮廓简化", 1.0, 0.0, 1_000.0, "输出多边形的简化强度。")

    return {"type": "object", "additionalProperties": False, "properties": properties}


class ModelCatalog:
    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._warnings: list[CatalogWarning] = []

    @property
    def count(self) -> int:
        return len(self._records)

    def list(self) -> ModelCatalogResponse:
        models = sorted((record.descriptor for record in self._records.values()), key=lambda item: (item.task, item.display_name.casefold()))
        return ModelCatalogResponse(models=models, warnings=list(self._warnings))

    def get(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as exc:
            raise ModelCatalogError("Unknown model", details={"model_id": model_id}) from exc

    def import_x_anylabeling(self, root: Path) -> ModelCatalogResponse:
        config_dir = _config_dir(root)
        records: dict[str, ModelRecord] = {}
        config_paths, warnings = _catalog_paths(config_dir)
        for config_path in config_paths:
            try:
                payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                warnings.append(CatalogWarning(path=config_path, code="invalid_yaml", message=str(exc)))
                continue
            if not isinstance(payload, dict):
                warnings.append(CatalogWarning(path=config_path, code="invalid_config", message="Config root must be a mapping"))
                continue
            for labels_key in ("classes", "filter_classes"):
                labels = payload.get(labels_key)
                if isinstance(labels, list):
                    payload[labels_key] = [str(label) for label in labels]
            model_type = str(payload.get("type", "")).strip()
            if not model_type:
                warnings.append(CatalogWarning(path=config_path, code="missing_type", message="Config has no model type"))
                continue
            model_id = str(payload.get("name") or config_path.stem)
            if model_id in records:
                warnings.append(CatalogWarning(path=config_path, code="duplicate_model_id", message=model_id))
                continue
            locations = _weight_locations(payload)
            adapter, runtime, capture_mode = _adapter_for(model_type, payload)
            required_locations = _required_weight_locations(adapter, payload, locations)
            local_paths = _local_weight_paths(config_path, required_locations)
            has_remote_location = any(urlparse(location).scheme in {"http", "https"} for location in required_locations)
            requires_all = adapter in {"ppocr_onnx", "segment_anything_onnx"}
            has_local_weight = bool(local_paths) and (all(path.is_file() for path in local_paths) if requires_all else any(path.is_file() for path in local_paths))
            if adapter in {"catalog_only", "remote_catalog"}:
                availability = Availability(
                    state=AvailabilityState.UNSUPPORTED,
                    reason=_unsupported_reason(model_type, payload),
                )
            elif adapter == "trusted_remote_http":
                availability = Availability(state=AvailabilityState.AVAILABLE)
            elif has_local_weight:
                availability = Availability(state=AvailabilityState.AVAILABLE)
            else:
                availability = Availability(
                    state=AvailabilityState.MISSING_WEIGHTS,
                    reason="Weights are remote" if has_remote_location else "Local weight file was not found",
                )
            annotation_adapters = {
                "detr_detection_onnx",
                "ppocr_onnx",
                "segment_anything_onnx",
                "yolo_classification_onnx",
                "yolo_detection_onnx",
                "yolo_obb_onnx",
                "yolo_pose_onnx",
                "yolo_segmentation_onnx",
                "trusted_remote_http",
            }
            result_kinds = ["tensors"] if adapter == "onnx_raw" else []
            if adapter in {"yolo_classification_onnx", "ram_tagging_onnx"}:
                result_kinds = ["classifications", "tensors"]
            elif adapter in {"depth_anything_onnx", "rmbg_matting_onnx"}:
                result_kinds = ["rasters", "tensors"]
            elif adapter == "segment_anything_onnx":
                result_kinds = ["annotations", "rasters", "tensors"]
            elif adapter in annotation_adapters:
                result_kinds = ["annotations", "tensors"]
                if adapter == "trusted_remote_http":
                    result_kinds = ["annotations"]
            descriptor = ModelDescriptor(
                id=model_id,
                name=model_id,
                display_name=str(payload.get("display_name") or model_id),
                model_type=model_type,
                provider=str(payload.get("provider") or "Unknown"),
                task=_task_for(model_type),
                family=model_type.split("_")[0],
                adapter=adapter,
                runtime=[runtime],
                config_path=config_path,
                weight_locations=locations,
                availability=availability,
                capabilities=ModelCapabilities(
                    predict=adapter in {"onnx_raw", "depth_anything_onnx", "rmbg_matting_onnx", "ram_tagging_onnx"} or adapter in annotation_adapters,
                    result_kinds=result_kinds,
                    feature_capture=FeatureCapture(
                        mode=capture_mode,
                        enumerable=capture_mode in {FeatureCaptureMode.EXPORTED_OUTPUTS, FeatureCaptureMode.GRAPH_REWRITE},
                    ),
                    parameters_schema=_parameters_schema(payload, adapter),
                ),
            )
            records[model_id] = ModelRecord(descriptor=descriptor, config=payload)
        self._records = records
        self._warnings = warnings
        return self.list()
