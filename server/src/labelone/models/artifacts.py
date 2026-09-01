from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from .features import feature_preview_image
from .types import RasterArtifact, TensorArtifact


class ArtifactStore:
    def __init__(self, root: Path, *, max_raster_pixels: int = 64_000_000, max_tensor_bytes: int = 256 * 1024 * 1024) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_raster_pixels = max_raster_pixels
        self.max_tensor_bytes = max_tensor_bytes

    @staticmethod
    def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
        partial = directory / "manifest.json.part"
        final = directory / "manifest.json"
        try:
            with partial.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, final)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if partial.exists():
                partial.unlink()

    def put_tensor(
        self,
        *,
        model_id: str,
        image_path: Path,
        layer_id: str,
        tensor: np.ndarray,
        source_shape: list[int] | None = None,
        transform: dict[str, object] | None = None,
    ) -> TensorArtifact:
        if tensor.size == 0 or not np.issubdtype(tensor.dtype, np.number):
            raise ModelRuntimeError("Feature tensor must be a non-empty numeric array")
        if tensor.nbytes > self.max_tensor_bytes:
            raise ModelRuntimeError(
                "Feature tensor exceeds the artifact byte budget",
                details={"nbytes": int(tensor.nbytes), "maximum_bytes": self.max_tensor_bytes},
            )
        if not np.all(np.isfinite(tensor)):
            raise ModelRuntimeError("Feature tensor contains non-finite values")
        statistics = {
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean(dtype=np.float64)),
        }
        safe_layer = re.sub(r"[^a-zA-Z0-9_.-]+", "_", layer_id)[:80]
        identity = sha256(f"{model_id}|{image_path}|{layer_id}|{uuid4()}".encode()).hexdigest()[:24]
        directory = self.root / identity
        directory.mkdir(parents=True, exist_ok=False)
        final_path = directory / f"{safe_layer}.npy"
        partial_path = directory / f"{safe_layer}.npy.part"
        with partial_path.open("wb") as handle:
            np.save(handle, tensor, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial_path, final_path)
        preview = feature_preview_image(tensor)
        preview_path = directory / "preview.png"
        if preview is not None:
            preview_partial = directory / "preview.png.part"
            try:
                with preview_partial.open("xb") as handle:
                    preview.save(handle, "PNG", compress_level=4)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(preview_partial, preview_path)
            finally:
                preview_partial.unlink(missing_ok=True)
        metadata = {
            "id": identity,
            "kind": "tensor",
            "model_id": model_id,
            "image_path": str(image_path),
            "layer_id": layer_id,
            "path": str(final_path),
            "media_type": "application/x-npy",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "statistics": statistics,
            "source_shape": source_shape or list(tensor.shape),
            "transform": transform or {},
            "preview_path": str(preview_path) if preview is not None else None,
            "preview_media_type": "image/png" if preview is not None else None,
            "preview_width": preview.width if preview is not None else None,
            "preview_height": preview.height if preview is not None else None,
        }
        self._write_manifest(directory, metadata)
        return TensorArtifact(
            id=identity,
            layer_id=layer_id,
            path=final_path,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            size_bytes=final_path.stat().st_size,
            statistics=statistics,
            source_shape=source_shape or list(tensor.shape),
            transform=transform or {},
            preview_available=preview is not None,
            preview_width=preview.width if preview is not None else None,
            preview_height=preview.height if preview is not None else None,
        )

    def get_manifest(self, artifact_id: str) -> dict[str, object]:
        manifest = (self.root / artifact_id / "manifest.json").resolve()
        if self.root not in manifest.parents or not manifest.is_file():
            raise FileNotFoundError(artifact_id)
        return json.loads(manifest.read_text(encoding="utf-8"))

    def put_raster(
        self,
        *,
        model_id: str,
        image_path: Path,
        role: str,
        image: Image.Image,
        format_name: str = "png",
        metadata: dict[str, object] | None = None,
    ) -> RasterArtifact:
        encodings = {
            "png": ("PNG", "image/png", ".png", {"compress_level": 4}),
            "webp": ("WEBP", "image/webp", ".webp", {"quality": 90, "method": 4}),
            "jpeg": ("JPEG", "image/jpeg", ".jpg", {"quality": 92}),
        }
        normalized = format_name.casefold()
        if normalized not in encodings:
            raise ValueError(f"Unsupported raster artifact format: {format_name}")
        encoder, media_type, suffix, options = encodings[normalized]
        if image.width <= 0 or image.height <= 0 or image.width * image.height > self.max_raster_pixels:
            raise ValueError(
                f"Raster artifact exceeds the pixel budget: {image.width}x{image.height} > {self.max_raster_pixels}"
            )
        safe_role = re.sub(r"[^a-zA-Z0-9_.-]+", "_", role)[:80]
        identity = sha256(f"{model_id}|{image_path}|{role}|{uuid4()}".encode()).hexdigest()[:24]
        directory = self.root / identity
        directory.mkdir(parents=True, exist_ok=False)
        final_path = directory / f"{safe_role}{suffix}"
        partial_path = directory / f"{safe_role}{suffix}.part"
        output = image
        if encoder == "JPEG" and output.mode not in {"RGB", "L"}:
            output = output.convert("RGB")
        try:
            with partial_path.open("xb") as handle:
                output.save(handle, encoder, **options)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial_path, final_path)
        finally:
            if partial_path.exists():
                partial_path.unlink()
        payload = {
            "id": identity,
            "kind": "raster",
            "model_id": model_id,
            "image_path": str(image_path),
            "role": role,
            "path": str(final_path),
            "media_type": media_type,
            "width": output.width,
            "height": output.height,
            "metadata": metadata or {},
        }
        self._write_manifest(directory, payload)
        return RasterArtifact(
            id=identity,
            role=role,
            path=final_path,
            media_type=media_type,
            width=output.width,
            height=output.height,
            size_bytes=final_path.stat().st_size,
            metadata=metadata or {},
        )

    def content_path(self, artifact_id: str) -> tuple[Path, str]:
        manifest = self.get_manifest(artifact_id)
        raw_path = manifest.get("path")
        if not isinstance(raw_path, str):
            raise FileNotFoundError(artifact_id)
        path = Path(raw_path).resolve()
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(artifact_id)
        media_type = str(manifest.get("media_type") or "application/octet-stream")
        return path, media_type

    def preview_path(self, artifact_id: str) -> tuple[Path, str]:
        manifest = self.get_manifest(artifact_id)
        raw_path = manifest.get("preview_path")
        if manifest.get("kind") != "tensor" or not isinstance(raw_path, str):
            raise FileNotFoundError(artifact_id)
        path = Path(raw_path).resolve()
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(artifact_id)
        return path, str(manifest.get("preview_media_type") or "image/png")

    def discard(self, artifact_id: str) -> None:
        directory = (self.root / artifact_id).resolve()
        if self.root not in directory.parents:
            raise FileNotFoundError(artifact_id)
        if directory.is_dir():
            shutil.rmtree(directory)
