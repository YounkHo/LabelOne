from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Iterable
from uuid import uuid4

from PIL import Image, ImageEnhance, ImageFilter

from labelone.annotations import AnnotationStore
from labelone.annotations.codec import normalize_annotation_document
from labelone.annotations.validation import validate_annotation_document
from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError, LabelOneError

from .models import PipelineCoordinateMapping, PipelineNode, PipelinePreviewRequest, PipelinePreviewResult, PipelineVisualizationResult
from .models import (
    DerivedDatasetPublishResult,
    PipelineDerivedItemResult,
    PipelineOutputPolicy,
)
from .derived import (
    DerivedDatasetWriter,
    PreparedDerivedOutput,
    clip_annotation_document,
    tile_windows,
)
from .registry import ValidatedNode, normalize_legacy_nodes, operator_registry_hash, validate_nodes

if TYPE_CHECKING:
    from labelone.models.artifacts import ArtifactStore
    from labelone.models.manager import ModelManager
    from .operator_packages import OperatorPackageManager


class PipelineCancelled(LabelOneError):
    code = "pipeline_cancelled"


PipelineProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class _VisualizationTap:
    node: ValidatedNode
    image: Image.Image
    document: dict
    operator_timings_ms: dict[str, float]
    content_kind: str = "image"
    overlay_compatible: bool = True
    coordinate_mapping: "_CoordinateMapping | None" = None


@dataclass(frozen=True, slots=True)
class _CoordinateMapping:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_to_output: tuple[float, float, float, float, float, float] | None
    topology_safe: bool = True
    reason: str | None = None

    @classmethod
    def identity(cls, width: int, height: int) -> "_CoordinateMapping":
        return cls(width, height, width, height, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    def compose(
        self,
        transform: tuple[float, float, float, float, float, float],
        *,
        output_width: int,
        output_height: int,
        topology_safe: bool = True,
    ) -> "_CoordinateMapping":
        if self.source_to_output is None:
            return _CoordinateMapping(
                self.source_width,
                self.source_height,
                output_width,
                output_height,
                None,
                False,
                self.reason or "An earlier operator did not expose a reversible coordinate mapping",
            )
        a, b, c, d, e, f = self.source_to_output
        na, nb, nc, nd, ne, nf = transform
        return _CoordinateMapping(
            self.source_width,
            self.source_height,
            output_width,
            output_height,
            (
                na * a + nc * b,
                nb * a + nd * b,
                na * c + nc * d,
                nb * c + nd * d,
                na * e + nc * f + ne,
                nb * e + nd * f + nf,
            ),
            self.topology_safe and topology_safe,
        )

    def unavailable(self, *, output_width: int, output_height: int, reason: str) -> "_CoordinateMapping":
        return _CoordinateMapping(
            self.source_width,
            self.source_height,
            output_width,
            output_height,
            None,
            False,
            reason,
        )

    @staticmethod
    def _inverse(
        transform: tuple[float, float, float, float, float, float] | None,
    ) -> tuple[float, float, float, float, float, float] | None:
        if transform is None:
            return None
        a, b, c, d, e, f = transform
        determinant = a * d - b * c
        if abs(determinant) <= 1e-12:
            return None
        return (
            d / determinant,
            -b / determinant,
            -c / determinant,
            a / determinant,
            (c * f - d * e) / determinant,
            (b * e - a * f) / determinant,
        )

    def as_result(self) -> PipelineCoordinateMapping:
        inverse = self._inverse(self.source_to_output)
        payload = {
            "source_size": [self.source_width, self.source_height],
            "output_size": [self.output_width, self.output_height],
            "matrix": self.source_to_output,
            "topology_safe": self.topology_safe,
            "reason": self.reason,
        }
        coordinate_space_id = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        identity = self.source_to_output == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        return PipelineCoordinateMapping(
            kind="unavailable" if self.source_to_output is None or inverse is None else "identity" if identity else "affine",
            source_width=self.source_width,
            source_height=self.source_height,
            output_width=self.output_width,
            output_height=self.output_height,
            source_to_output=self.source_to_output,
            output_to_source=inverse,
            coordinate_space_id=coordinate_space_id,
            topology_safe=self.topology_safe,
            reason=self.reason,
        )


@dataclass(slots=True)
class _TimingAggregate:
    total_ms: float = 0.0
    sample_count: int = 0


@dataclass(slots=True)
class _PreviewFlight:
    future: Future[PipelinePreviewResult]
    waiters: int = 0


def _points(document: dict) -> list[tuple[dict, list[list[float]]]]:
    results: list[tuple[dict, list[list[float]]]] = []
    for shape in document.get("shapes", []):
        if isinstance(shape, dict) and isinstance(shape.get("points"), list):
            results.append((shape, shape["points"]))
    return results


class PipelineEngine:
    def __init__(
        self,
        repository: DatasetRepository,
        annotations: AnnotationStore,
        artifact_root: Path,
        operator_packages: "OperatorPackageManager | None" = None,
        model_manager: "ModelManager | None" = None,
        model_artifacts: "ArtifactStore | None" = None,
        maximum_timing_signatures: int = 128,
        maximum_preview_cache_entries: int = 64,
    ) -> None:
        if maximum_timing_signatures < 1:
            raise ValueError("maximum_timing_signatures must be positive")
        if maximum_preview_cache_entries < 1:
            raise ValueError("maximum_preview_cache_entries must be positive")
        self.repository = repository
        self.annotations = annotations
        self.operator_packages = operator_packages
        self.model_manager = model_manager
        self.model_artifacts = model_artifacts
        self.artifact_root = artifact_root.expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.derived_writer = DerivedDatasetWriter()
        self.maximum_timing_signatures = maximum_timing_signatures
        self.maximum_preview_cache_entries = maximum_preview_cache_entries
        self._timing_lock = RLock()
        self._timing_history: OrderedDict[str, dict[str, _TimingAggregate]] = OrderedDict()
        self._preview_cache_lock = RLock()
        self._preview_cache: OrderedDict[str, PipelinePreviewResult] = OrderedDict()
        self._preview_inflight: dict[str, _PreviewFlight] = {}

    @staticmethod
    def _pipeline_signature(nodes: Iterable[ValidatedNode]) -> str:
        payload = {"nodes": [node.as_dict() for node in nodes]}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _record_operator_timings(
        self,
        nodes: Iterable[ValidatedNode],
        timings: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, int]]:
        if not timings:
            return {}, {}
        signature = self._pipeline_signature(nodes)
        with self._timing_lock:
            aggregates = self._timing_history.pop(signature, None)
            if aggregates is None:
                aggregates = {}
            for node_id, duration_ms in timings.items():
                aggregate = aggregates.setdefault(node_id, _TimingAggregate())
                aggregate.total_ms += float(duration_ms)
                aggregate.sample_count += 1
            self._timing_history[signature] = aggregates
            while len(self._timing_history) > self.maximum_timing_signatures:
                self._timing_history.popitem(last=False)
            averages = {
                node_id: aggregate.total_ms / aggregate.sample_count
                for node_id, aggregate in aggregates.items()
            }
            sample_counts = {
                node_id: aggregate.sample_count
                for node_id, aggregate in aggregates.items()
            }
        return averages, sample_counts

    @staticmethod
    def _report_progress(
        callback: PipelineProgressCallback | None,
        *,
        completed_steps: int,
        total_steps: int,
        phase: str,
        node: PipelineNode | None = None,
    ) -> None:
        if callback is None:
            return
        callback({
            "kind": "pipeline",
            "progress": completed_steps / total_steps,
            "phase": phase,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "node_id": node.id if node is not None else None,
            "node_kind": node.kind if node is not None else None,
        })

    @staticmethod
    def _crop(image: Image.Image, document: dict, parameters: dict) -> Image.Image:
        default_margin = float(parameters.get("margin_ratio", 0.05))
        x = int(parameters.get("x", round(image.width * default_margin)))
        y = int(parameters.get("y", round(image.height * default_margin)))
        width = int(parameters.get("width", round(image.width * (1 - default_margin * 2))))
        height = int(parameters.get("height", round(image.height * (1 - default_margin * 2))))
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image.width or y + height > image.height:
            raise InvalidPathError("Crop parameters exceed the input image")
        clipped = clip_annotation_document(
            document,
            x=x,
            y=y,
            width=width,
            height=height,
            image_name=str(document.get("imagePath") or "image"),
        )
        document.clear()
        document.update(clipped)
        return image.crop((x, y, x + width, y + height))

    @staticmethod
    def _resize(image: Image.Image, document: dict, parameters: dict) -> Image.Image:
        target_width = int(parameters.get("width", image.width))
        target_height = int(parameters.get("height", image.height))
        if target_width <= 0 or target_height <= 0 or target_width * target_height > 64_000_000:
            raise InvalidPathError("Resize output exceeds the pixel budget")
        scale_x = target_width / image.width
        scale_y = target_height / image.height
        anisotropic = abs(scale_x - scale_y) > 1e-12
        for shape, points in _points(document):
            if anisotropic and shape.get("shape_type") == "circle" and len(points) >= 2:
                center, edge = points[0], points[1]
                radius = ((float(edge[0]) - float(center[0])) ** 2 + (float(edge[1]) - float(center[1])) ** 2) ** 0.5
                shape["shape_type"] = "polygon"
                shape.pop("direction", None)
                shape["points"] = [
                    [
                        (float(center[0]) + radius * math.cos(index * math.tau / 32)) * scale_x,
                        (float(center[1]) + radius * math.sin(index * math.tau / 32)) * scale_y,
                    ]
                    for index in range(32)
                ]
                continue
            if anisotropic and shape.get("shape_type") == "rotation":
                shape["shape_type"] = "polygon"
                shape.pop("direction", None)
            for point in points:
                point[0] = float(point[0]) * scale_x
                point[1] = float(point[1]) * scale_y
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    @staticmethod
    def _flip(image: Image.Image, document: dict, parameters: dict) -> Image.Image:
        axis = str(parameters.get("axis", "horizontal"))
        if axis == "horizontal":
            for _, points in _points(document):
                for point in points:
                    point[0] = image.width - float(point[0])
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if axis == "vertical":
            for _, points in _points(document):
                for point in points:
                    point[1] = image.height - float(point[1])
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        raise InvalidPathError("Flip axis must be horizontal or vertical")

    @staticmethod
    def _rotate(image: Image.Image, document: dict, parameters: dict) -> Image.Image:
        degrees = int(parameters.get("degrees", 90)) % 360
        if degrees not in {0, 90, 180, 270}:
            raise InvalidPathError("The built-in rotate operator accepts 0/90/180/270 degrees")
        old_width, old_height = image.size
        for _, points in _points(document):
            for point in points:
                x, y = float(point[0]), float(point[1])
                if degrees == 90:
                    point[0], point[1] = old_height - y, x
                elif degrees == 180:
                    point[0], point[1] = old_width - x, old_height - y
                elif degrees == 270:
                    point[0], point[1] = y, old_width - x
        return image.rotate(-degrees, expand=True)

    @staticmethod
    def _color(image: Image.Image, parameters: dict) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(float(parameters.get("brightness", 1.05)))
        image = ImageEnhance.Contrast(image).enhance(float(parameters.get("contrast", 1.10)))
        image = ImageEnhance.Color(image).enhance(float(parameters.get("saturation", 1.0)))
        return image

    @staticmethod
    def _noise(image: Image.Image, parameters: dict) -> Image.Image:
        radius = max(0.0, min(5.0, float(parameters.get("radius", 1.0))))
        percent = max(0, min(500, int(parameters.get("percent", 130))))
        threshold = max(0, min(255, int(parameters.get("threshold", 2))))
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    def _model_feature(self, image: Image.Image, parameters: dict) -> tuple[Image.Image, bool]:
        if self.model_manager is None or self.model_artifacts is None:
            raise InvalidPathError("Pipeline model feature runtime is unavailable")
        model_id = str(parameters.get("model_id") or "")
        layer_id = str(parameters.get("layer_id") or "")
        state = self.model_manager.state(model_id)
        if state.state != "loaded":
            self.model_manager.load(model_id, ["CPUExecutionProvider"])
        runtime = self.model_manager.layers(model_id)
        layer = next((item for item in runtime.layers if item.id == layer_id and item.captureable), None)
        if layer is None:
            raise InvalidPathError(
                "Pipeline model feature layer is unavailable or cannot be captured",
                details={"model_id": model_id, "layer_id": layer_id},
            )
        feature_transform = {
            "projection": str(parameters.get("projection", "mean")),
            "normalization": str(parameters.get("normalization", "minmax")),
            "interpolation": "bilinear",
            "spatial_scale": 1,
            "channel": int(parameters.get("channel", 0)),
            "gain": 1,
            "gamma": 1,
            "clip_percentiles": {"p1p99": [1, 99], "p5p95": [5, 95], "none": None}.get(str(parameters.get("clip", "p1p99")), [1, 99]),
        }
        artifact_id: str | None = None
        try:
            with TemporaryDirectory(prefix="model-feature-", dir=self.artifact_root) as temporary_directory:
                stage_path = Path(temporary_directory) / "stage.png"
                image.save(stage_path, "PNG")
                result = self.model_manager.predict(
                    model_id,
                    stage_path,
                    [layer_id],
                    {"feature_transform": feature_transform},
                )
                if len(result.artifacts) != 1 or not result.artifacts[0].preview_available:
                    raise InvalidPathError(
                        "Pipeline model feature did not produce a preview",
                        details={"model_id": model_id, "layer_id": layer_id},
                    )
                artifact_id = result.artifacts[0].id
                preview_path, _ = self.model_artifacts.preview_path(artifact_id)
                with Image.open(preview_path) as preview:
                    visualization = preview.convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
                    return visualization, bool(layer.spatial)
        finally:
            if artifact_id is not None:
                self.model_artifacts.discard(artifact_id)

    def _custom_operator(self, kind: str, image: Image.Image, document: dict, parameters: dict) -> Image.Image:
        if self.operator_packages is None or not self.operator_packages.has(kind):
            raise InvalidPathError(
                "Pipeline operator has no execution implementation",
                details={"kind": kind},
            )
        package = self.operator_packages.get(kind)
        spatial_behavior = str(package.contract.annotation_policy.get("spatial_behavior", ""))
        old_width, old_height = image.size
        output = self.operator_packages.execute(kind, image, parameters)
        new_width, new_height = output.size
        changed_size = (old_width, old_height) != (new_width, new_height)
        if package.contract.size_behavior == "preserve" and changed_size:
            raise InvalidPathError(
                "Custom operator violated its preserve-size contract",
                details={
                    "kind": kind,
                    "input_size": [old_width, old_height],
                    "output_size": [new_width, new_height],
                },
            )
        if spatial_behavior == "scale_xy" and package.annotation_mode == "scale":
            scale_x = new_width / old_width
            scale_y = new_height / old_height
            for _, points in _points(document):
                for point in points:
                    point[0] = float(point[0]) * scale_x
                    point[1] = float(point[1]) * scale_y
        elif spatial_behavior == "custom" and package.annotation_mode == "transform":
            transformed = self.operator_packages.transform_annotations(
                kind,
                deepcopy(document),
                parameters,
                input_size=(old_width, old_height),
                output_size=(new_width, new_height),
            )
            transformed["imageWidth"] = new_width
            transformed["imageHeight"] = new_height
            transformed = normalize_annotation_document(transformed)
            validate_annotation_document(transformed)
            for shape_index, (_shape, points) in enumerate(_points(transformed)):
                for point in points:
                    x, y = float(point[0]), float(point[1])
                    if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0 or x > new_width or y > new_height:
                        raise InvalidPathError(
                            "Custom annotation transform returned out-of-bounds coordinates",
                            details={"kind": kind, "shape_index": shape_index, "point": [x, y], "output_size": [new_width, new_height]},
                        )
            document.clear()
            document.update(transformed)
        elif spatial_behavior == "none" and package.annotation_mode == "preserve" and changed_size:
            raise InvalidPathError(
                "Custom operator changed image size without an annotation transform",
                details={"kind": kind, "annotation_policy": "preserve"},
            )
        elif spatial_behavior not in {"none", "scale_xy"}:
            raise InvalidPathError(
                "Custom operator has no verified annotation transform capability",
                details={"kind": kind, "spatial_behavior": spatial_behavior},
            )
        return output

    @staticmethod
    def _encode_preview(image: Image.Image, output_format: str) -> tuple[bytes, str, str, str]:
        encoder = "JPEG" if output_format == "jpeg" else output_format.upper()
        media_type = "image/jpeg" if encoder == "JPEG" else f"image/{output_format}"
        suffix = ".jpg" if encoder == "JPEG" else f".{output_format}"
        buffer = BytesIO()
        save_options = {"quality": 88} if encoder in {"WEBP", "JPEG"} else {}
        encoded_image = image.convert("RGB") if encoder == "JPEG" and image.mode not in {"RGB", "L"} else image
        encoded_image.save(buffer, encoder, **save_options)
        return buffer.getvalue(), encoder, media_type, suffix

    def _write_visualization_artifact(
        self,
        *,
        request: PipelinePreviewRequest,
        nodes: list[ValidatedNode],
        tap: _VisualizationTap,
        visualization_index: int,
        source_mtime_ns: int,
        source_size: int,
        annotation_revision: str,
    ) -> PipelineVisualizationResult:
        document = deepcopy(tap.document)
        document["imageWidth"] = tap.image.width
        document["imageHeight"] = tap.image.height
        document = normalize_annotation_document(document)
        coordinate_mapping = (tap.coordinate_mapping or _CoordinateMapping.identity(tap.image.width, tap.image.height)).as_result()
        content, _, media_type, suffix = self._encode_preview(tap.image, request.output_format)
        identity = sha256(
            json.dumps({
                "dataset": request.dataset_id,
                "asset": request.asset_id,
                "nodes": [node.as_dict() for node in nodes],
                "operator_registry": operator_registry_hash(),
                "source_mtime": source_mtime_ns,
                "source_size": source_size,
                "annotation_revision": annotation_revision,
                "output_format": request.output_format,
                "visualization_id": tap.node.id,
                "visualization_index": visualization_index,
            }, sort_keys=True).encode()
        ).hexdigest()[:24]
        directory = self.artifact_root / identity
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"preview{suffix}"
        partial = directory / f".preview{suffix}.{uuid4().hex}.part"
        try:
            if not output_path.is_file():
                with partial.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, output_path)
        finally:
            if partial.exists():
                partial.unlink()
        manifest_path = directory / "manifest.json"
        manifest_partial = directory / f".manifest.{uuid4().hex}.part"
        manifest_content = json.dumps({
            "artifact_id": identity,
            "visualization_id": tap.node.id,
            "path": str(output_path),
            "media_type": media_type,
            "width": tap.image.width,
            "height": tap.image.height,
            "content_kind": tap.content_kind,
            "overlay_compatible": tap.overlay_compatible,
            "coordinate_mapping": coordinate_mapping.model_dump(mode="json"),
        }, indent=2)
        try:
            with manifest_partial.open("x", encoding="utf-8") as handle:
                handle.write(manifest_content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_partial, manifest_path)
        finally:
            if manifest_partial.exists():
                manifest_partial.unlink()
        return PipelineVisualizationResult(
            visualization_id=tap.node.id,
            label=str(tap.node.parameters["label"]),
            artifact_id=identity,
            width=tap.image.width,
            height=tap.image.height,
            media_type=media_type,
            annotation_document=document,
            operator_timings_ms=tap.operator_timings_ms,
            content_kind=tap.content_kind,
            overlay_compatible=tap.overlay_compatible,
            coordinate_mapping=coordinate_mapping,
        )

    def preview(
        self,
        request: PipelinePreviewRequest,
        progress: PipelineProgressCallback | None = None,
        canceled: Callable[[], bool] | None = None,
    ) -> PipelinePreviewResult:
        if canceled is not None and canceled():
            raise PipelineCancelled("Pipeline preview was canceled before processing")
        nodes = validate_nodes(normalize_legacy_nodes(request.nodes))
        if any(node.enabled and node.kind == "tile" for node in nodes):
            raise InvalidPathError(
                "Tile produces multiple images and is only available for derived_dataset batch output"
            )
        source_asset = self.repository.get_asset(request.dataset_id, request.asset_id, require_selectable=True)
        source_annotation = self.annotations.load(request.dataset_id, request.asset_id)
        if source_asset.image_path is None:
            raise InvalidPathError("Pipeline input image is missing")
        source_stat = source_asset.image_path.stat()
        cache_key = sha256(json.dumps({
            "dataset_id": request.dataset_id,
            "asset_id": request.asset_id,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_size": source_stat.st_size,
            "annotation_revision": source_annotation.revision,
            "nodes": [node.as_dict() for node in nodes],
            "operator_registry": operator_registry_hash(),
            "output_format": request.output_format,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        with self._preview_cache_lock:
            cached = self._preview_cache.pop(cache_key, None)
            if cached is not None:
                self._preview_cache[cache_key] = cached
                return cached.model_copy(deep=True, update={"cache_hit": True})
            flight = self._preview_inflight.get(cache_key)
            owns_computation = flight is None
            if flight is None:
                flight = _PreviewFlight(future=Future())
                self._preview_inflight[cache_key] = flight
            else:
                flight.waiters += 1

        if not owns_computation:
            try:
                while True:
                    if canceled is not None and canceled():
                        raise PipelineCancelled("Pipeline preview wait was canceled")
                    try:
                        return flight.future.result(timeout=0.05).model_copy(deep=True, update={"cache_hit": True})
                    except FutureTimeoutError:
                        continue
            finally:
                with self._preview_cache_lock:
                    current_flight = self._preview_inflight.get(cache_key)
                    if current_flight is flight:
                        current_flight.waiters = max(0, current_flight.waiters - 1)

        def computation_canceled() -> bool:
            if canceled is None or not canceled():
                return False
            with self._preview_cache_lock:
                current_flight = self._preview_inflight.get(cache_key)
                return current_flight is flight and current_flight.waiters == 0

        try:
            result = self._compute_preview(request, nodes, progress, computation_canceled)
        except BaseException as error:
            with self._preview_cache_lock:
                if self._preview_inflight.pop(cache_key, None) is flight:
                    flight.future.set_exception(error)
            raise

        with self._preview_cache_lock:
            cached_result = result.model_copy(deep=True)
            self._preview_cache[cache_key] = cached_result
            while len(self._preview_cache) > self.maximum_preview_cache_entries:
                self._preview_cache.popitem(last=False)
            if self._preview_inflight.pop(cache_key, None) is flight:
                flight.future.set_result(cached_result)
        return result

    def _compute_preview(
        self,
        request: PipelinePreviewRequest,
        nodes: list[ValidatedNode],
        progress: PipelineProgressCallback | None,
        canceled: Callable[[], bool] | None,
    ) -> PipelinePreviewResult:
        if canceled is not None and canceled():
            raise PipelineCancelled("Pipeline preview was canceled before transformation")
        transform_nodes = [
            node for node in nodes if node.enabled and node.kind not in {"source", "visualize", "tile"}
        ]
        visualization_nodes = [node for node in nodes if node.kind == "visualize"]
        total_steps = len(transform_nodes) + len(visualization_nodes) + 2
        asset, annotation, _, _, _, _, taps, average_timings, timing_sample_count = self._transform(
            request.dataset_id,
            request.asset_id,
            nodes,
            progress=progress,
            progress_total_steps=total_steps,
            capture_visualizations=True,
            canceled=canceled,
        )
        if asset.image_path is None:  # guarded by _transform; keeps the type boundary explicit
            raise InvalidPathError("Pipeline input image is missing")
        visualizations: list[PipelineVisualizationResult] = []
        visualization_timings: dict[str, float] = {}
        for index, tap in enumerate(taps):
            if canceled is not None and canceled():
                raise PipelineCancelled("Pipeline preview was canceled before artifact encoding")
            visualization_started = perf_counter()
            visualization = self._write_visualization_artifact(
                request=request,
                nodes=nodes,
                tap=tap,
                visualization_index=index,
                source_mtime_ns=asset.image_path.stat().st_mtime_ns,
                source_size=asset.image_path.stat().st_size,
                annotation_revision=annotation.revision,
            )
            visualization_duration = (perf_counter() - visualization_started) * 1000
            visualization_timings[tap.node.id] = visualization_duration
            visualization.operator_timings_ms = {
                **visualization.operator_timings_ms,
                tap.node.id: visualization_duration,
            }
            visualizations.append(visualization)
            self._report_progress(
                progress,
                completed_steps=len(transform_nodes) + index + 2,
                total_steps=total_steps,
                phase="visualization",
                node=tap.node,
            )
        if visualization_timings:
            average_timings, timing_sample_count = self._record_operator_timings(nodes, visualization_timings)
        primary = visualizations[-1]
        self._report_progress(
            progress,
            completed_steps=total_steps,
            total_steps=total_steps,
            phase="completed",
        )
        result = PipelinePreviewResult(
            dataset_id=request.dataset_id,
            asset_id=request.asset_id,
            artifact_id=primary.artifact_id,
            width=primary.width,
            height=primary.height,
            media_type=primary.media_type,
            annotation_document=primary.annotation_document,
            operator_timings_ms=primary.operator_timings_ms,
            operator_average_timings_ms=average_timings,
            timing_sample_count=timing_sample_count,
            visualizations=visualizations,
        )
        return result

    def _transform(
        self,
        dataset_id: str,
        asset_id: str,
        nodes,
        *,
        progress: PipelineProgressCallback | None = None,
        progress_total_steps: int | None = None,
        capture_visualizations: bool = False,
        canceled: Callable[[], bool] | None = None,
    ) -> tuple:
        if canceled is not None and canceled():
            raise PipelineCancelled("Pipeline transformation was canceled before loading")
        asset = self.repository.get_asset(dataset_id, asset_id, require_selectable=True)
        if asset.image_path is None:
            raise InvalidPathError("Pipeline input image is missing")
        annotation = self.annotations.load(dataset_id, asset_id)
        document = deepcopy(annotation.document)
        timings: dict[str, float] = {}
        taps: list[_VisualizationTap] = []
        transform_nodes = [
            node for node in nodes if node.enabled and node.kind not in {"source", "visualize", "tile"}
        ]
        total_steps = progress_total_steps or len(transform_nodes) + 1
        source_node = next(node for node in nodes if node.kind == "source")
        source_started = perf_counter()
        with Image.open(asset.image_path) as source:
            image = source.convert("RGB")
            if canceled is not None and canceled():
                raise PipelineCancelled("Pipeline transformation was canceled after loading")
            timings[source_node.id] = (perf_counter() - source_started) * 1000
            self._report_progress(progress, completed_steps=1, total_steps=total_steps, phase="loaded")
            completed_transforms = 0
            stage_snapshot: tuple[Image.Image, dict, dict[str, float], str, bool, _CoordinateMapping] | None = None
            content_kind = "image"
            overlay_compatible = True
            coordinate_mapping = _CoordinateMapping.identity(image.width, image.height)
            for node in nodes:
                if canceled is not None and canceled():
                    raise PipelineCancelled("Pipeline transformation was canceled between operators")
                if not node.enabled or node.kind == "source":
                    continue
                if node.kind == "visualize":
                    if capture_visualizations:
                        if stage_snapshot is None:
                            stage_snapshot = (image.copy(), deepcopy(document), dict(timings), content_kind, overlay_compatible, coordinate_mapping)
                        snapshot_image, snapshot_document, snapshot_timings, snapshot_content_kind, snapshot_overlay_compatible, snapshot_coordinate_mapping = stage_snapshot
                        taps.append(_VisualizationTap(
                            node=node,
                            image=snapshot_image,
                            document=snapshot_document,
                            operator_timings_ms=snapshot_timings,
                            content_kind=snapshot_content_kind,
                            overlay_compatible=snapshot_overlay_compatible,
                            coordinate_mapping=snapshot_coordinate_mapping,
                        ))
                    continue
                if node.kind == "tile":
                    continue
                started = perf_counter()
                input_width, input_height = image.size
                affine_step: tuple[float, float, float, float, float, float] | None = None
                topology_safe = True
                unavailable_reason: str | None = None
                if node.kind == "crop":
                    margin = float(node.parameters.get("margin_ratio", 0.05))
                    crop_x = int(node.parameters.get("x", round(input_width * margin)))
                    crop_y = int(node.parameters.get("y", round(input_height * margin)))
                    image = self._crop(image, document, node.parameters)
                    affine_step = (1.0, 0.0, 0.0, 1.0, -float(crop_x), -float(crop_y))
                    topology_safe = False
                elif node.kind == "resize":
                    image = self._resize(image, document, node.parameters)
                    affine_step = (image.width / input_width, 0.0, 0.0, image.height / input_height, 0.0, 0.0)
                    topology_safe = abs(image.width / input_width - image.height / input_height) <= 1e-12
                elif node.kind == "flip":
                    image = self._flip(image, document, node.parameters)
                    axis = str(node.parameters.get("axis", "horizontal"))
                    affine_step = (-1.0, 0.0, 0.0, 1.0, float(input_width), 0.0) if axis == "horizontal" else (1.0, 0.0, 0.0, -1.0, 0.0, float(input_height))
                elif node.kind == "rotate":
                    image = self._rotate(image, document, node.parameters)
                    degrees = int(node.parameters.get("degrees", 90)) % 360
                    affine_step = {
                        0: (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                        90: (0.0, 1.0, -1.0, 0.0, float(input_height), 0.0),
                        180: (-1.0, 0.0, 0.0, -1.0, float(input_width), float(input_height)),
                        270: (0.0, -1.0, 1.0, 0.0, 0.0, float(input_width)),
                    }[degrees]
                elif node.kind == "color":
                    image = self._color(image, node.parameters)
                elif node.kind == "noise":
                    image = self._noise(image, node.parameters)
                elif node.kind == "model_feature":
                    image, feature_spatial = self._model_feature(image, dict(node.parameters))
                    content_kind = "model_feature"
                    overlay_compatible = overlay_compatible and feature_spatial
                    if not feature_spatial:
                        unavailable_reason = "The model feature is non-spatial and has no source-image coordinate mapping"
                elif node.kind.startswith("opencv."):
                    from .opencv_ops import apply_opencv_operator

                    image = apply_opencv_operator(node.kind, image, node.parameters)
                    if node.kind in {"opencv.fourier_transform", "opencv.haar_wavelet"}:
                        content_kind = "frequency_spectrum" if node.kind == "opencv.fourier_transform" else "wavelet_coefficients"
                        overlay_compatible = False
                        document["shapes"] = []
                        unavailable_reason = "Frequency-domain output has no one-to-one source-image coordinate mapping"
                elif self.operator_packages is not None and self.operator_packages.has(node.kind):
                    package = self.operator_packages.get(node.kind)
                    image = self._custom_operator(node.kind, image, document, dict(node.parameters))
                    spatial_behavior = str(package.contract.annotation_policy.get("spatial_behavior", "none"))
                    if spatial_behavior == "scale_xy" and package.annotation_mode == "scale":
                        affine_step = (image.width / input_width, 0.0, 0.0, image.height / input_height, 0.0, 0.0)
                        topology_safe = abs(image.width / input_width - image.height / input_height) <= 1e-12
                    elif spatial_behavior == "custom":
                        unavailable_reason = f"Custom spatial operator {node.kind} did not expose a point mapping"
                else:
                    raise InvalidPathError(
                        "Pipeline operator has no execution implementation",
                        details={"node_id": node.id, "kind": node.kind},
                    )
                if unavailable_reason is not None:
                    coordinate_mapping = coordinate_mapping.unavailable(
                        output_width=image.width,
                        output_height=image.height,
                        reason=unavailable_reason,
                    )
                elif affine_step is not None:
                    coordinate_mapping = coordinate_mapping.compose(
                        affine_step,
                        output_width=image.width,
                        output_height=image.height,
                        topology_safe=topology_safe,
                    )
                elif image.size != (input_width, input_height):
                    coordinate_mapping = coordinate_mapping.unavailable(
                        output_width=image.width,
                        output_height=image.height,
                        reason=f"Operator {node.kind} changed image size without a coordinate mapping",
                    )
                if canceled is not None and canceled():
                    raise PipelineCancelled("Pipeline transformation was canceled after an operator")
                timings[node.id] = (perf_counter() - started) * 1000
                completed_transforms += 1
                stage_snapshot = None
                self._report_progress(
                    progress,
                    completed_steps=completed_transforms + 1,
                    total_steps=total_steps,
                    phase="operator",
                    node=node,
                )
        tile_nodes = [node for node in nodes if node.enabled and node.kind == "tile"]
        if len(tile_nodes) > 1:
            raise InvalidPathError("Derived pipeline may contain only one tile operator")
        if tile_nodes:
            tile_index = nodes.index(tile_nodes[0])
            if any(node.enabled and node.kind != "visualize" for node in nodes[tile_index + 1:]):
                raise InvalidPathError("Tile must be the final enabled transform in a derived pipeline")
        average_timings, timing_sample_count = self._record_operator_timings(nodes, timings)
        return (
            asset,
            annotation,
            image,
            document,
            timings,
            tile_nodes[0] if tile_nodes else None,
            taps,
            average_timings,
            timing_sample_count,
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def export_derived_item(
        self,
        *,
        job_id: str,
        dataset_id: str,
        asset_id: str,
        nodes: list[PipelineNode],
        policy: PipelineOutputPolicy,
        canceled=None,
        progress: PipelineProgressCallback | None = None,
    ) -> PipelineDerivedItemResult:
        if policy.mode != "derived_dataset" or policy.output_root is None:
            raise InvalidPathError("Batch derived export requires a derived_dataset output policy")
        validated = validate_nodes(normalize_legacy_nodes(nodes))
        transform_nodes = [
            node for node in validated if node.enabled and node.kind not in {"source", "visualize", "tile"}
        ]
        total_steps = len(transform_nodes) + 3
        if canceled is not None and canceled():
            raise PipelineCancelled("Derived pipeline was canceled before processing")
        asset, annotation, image, document, _, tile_node, _, _, _ = self._transform(
            dataset_id,
            asset_id,
            validated,
            progress=progress,
            progress_total_steps=total_steps,
            canceled=canceled,
        )
        if asset.image_path is None:
            raise InvalidPathError("Derived pipeline source image is missing")
        dataset = self.repository.get_dataset(dataset_id)
        source_hash = self._file_sha256(asset.image_path)
        item_fingerprint = sha256(json.dumps({
            "dataset_id": dataset_id,
            "asset_id": asset_id,
            "source_sha256": source_hash,
            "annotation_revision": annotation.revision,
            "nodes": [node.as_dict() for node in validated],
            "operator_registry": operator_registry_hash(),
            "image_format": policy.image_format,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        relative = Path(*PurePosixPath(asset.display_path).parts)
        suffix = ".jpg" if policy.image_format == "jpeg" else f".{policy.image_format}"
        prepared: list[PreparedDerivedOutput] = []
        if tile_node is None:
            image_relative = relative.with_suffix(suffix).as_posix()
            annotation_relative = relative.with_suffix(".json").as_posix()
            output_document = deepcopy(document)
            output_document["imagePath"] = Path(image_relative).name
            output_document["imageWidth"] = image.width
            output_document["imageHeight"] = image.height
            output_document["imageData"] = None
            output_document = normalize_annotation_document(output_document)
            prepared.append(PreparedDerivedOutput(
                image_relative_path=image_relative,
                annotation_relative_path=annotation_relative,
                image=image,
                document=output_document,
            ))
        else:
            parameters = tile_node.parameters
            windows = tile_windows(
                image.width,
                image.height,
                tile_width=int(parameters["tile_width"]),
                tile_height=int(parameters["tile_height"]),
                overlap_x=int(parameters["overlap_x"]),
                overlap_y=int(parameters["overlap_y"]),
                include_partial=bool(parameters["include_partial"]),
            )
            if not windows:
                raise InvalidPathError("Tile operator produced no outputs for the current image")
            for column, row, x, y, width, height in windows:
                stem = f"{relative.stem}__tile_r{row:04d}_c{column:04d}"
                image_relative = (relative.parent / f"{stem}{suffix}").as_posix()
                annotation_relative = (relative.parent / f"{stem}.json").as_posix()
                tile_info = {"column": column, "row": row, "x": x, "y": y, "width": width, "height": height}
                output_document = clip_annotation_document(
                    document,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    image_name=Path(image_relative).name,
                )
                output_document = normalize_annotation_document(output_document)
                prepared.append(PreparedDerivedOutput(
                    image_relative_path=image_relative,
                    annotation_relative_path=annotation_relative,
                    image=image.crop((x, y, x + width, y + height)),
                    document=output_document,
                    tile=tile_info,
                ))
        if canceled is not None and canceled():
            raise PipelineCancelled("Derived pipeline was canceled before staging outputs")
        self._report_progress(
            progress,
            completed_steps=total_steps - 1,
            total_steps=total_steps,
            phase="writing",
        )
        result = self.derived_writer.write_item(
            job_id=job_id,
            dataset_id=dataset_id,
            asset_id=asset_id,
            source_root=dataset.root_dir,
            policy=policy,
            item_fingerprint=item_fingerprint,
            outputs=prepared,
        )
        self._report_progress(
            progress,
            completed_steps=total_steps,
            total_steps=total_steps,
            phase="completed",
        )
        return result

    def finalize_derived(
        self,
        *,
        job_id: str,
        dataset_id: str,
        policy: PipelineOutputPolicy,
        item_results: Iterable[dict],
        expected_item_count: int,
    ) -> DerivedDatasetPublishResult:
        dataset = self.repository.get_dataset(dataset_id)
        parsed = (PipelineDerivedItemResult.model_validate(item) for item in item_results)
        return self.derived_writer.finalize(
            job_id=job_id,
            dataset_id=dataset_id,
            source_root=dataset.root_dir,
            policy=policy,
            items=parsed,
            expected_item_count=expected_item_count,
        )

    def abort_derived(self, *, job_id: str, dataset_id: str, policy: PipelineOutputPolicy) -> None:
        dataset = self.repository.get_dataset(dataset_id)
        self.derived_writer.abort(
            job_id=job_id,
            source_root=dataset.root_dir,
            policy=policy,
        )

    def artifact_path(self, artifact_id: str) -> tuple[Path, str]:
        manifest_path = (self.artifact_root / artifact_id / "manifest.json").resolve()
        if self.artifact_root not in manifest_path.parents or not manifest_path.is_file():
            raise InvalidPathError("Unknown pipeline artifact", details={"artifact_id": artifact_id})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = Path(manifest["path"]).resolve()
        if not path.is_file() or self.artifact_root not in path.parents:
            raise InvalidPathError("Pipeline artifact file is missing", details={"artifact_id": artifact_id})
        return path, str(manifest["media_type"])
