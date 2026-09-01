from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError
from labelone.keyed_lock import KeyedLockPool


@dataclass(frozen=True, slots=True)
class RenderedImage:
    content: bytes
    media_type: str
    etag: str
    width: int
    height: int
    cache_hit: bool


class ImageService:
    def __init__(self, repository: DatasetRepository, cache_root: Path) -> None:
        self.repository = repository
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._key_locks = KeyedLockPool()

    def image_path(self, dataset_id: str, asset_id: str) -> Path:
        asset = self.repository.get_asset(dataset_id, asset_id, require_selectable=True)
        if asset.image_path is None or not asset.image_path.is_file():
            raise InvalidPathError("Image file is missing", details={"dataset_id": dataset_id, "asset_id": asset_id})
        return asset.image_path.resolve()

    @staticmethod
    def source_etag(path: Path) -> str:
        stat = path.stat()
        return sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()

    @staticmethod
    def _display_image(image: Image.Image) -> Image.Image:
        if image.mode in {"I", "I;16", "I;16L", "I;16B", "F"}:
            array = np.asarray(image, dtype=np.float32)
            finite = array[np.isfinite(array)]
            if finite.size:
                low, high = np.percentile(finite, [1, 99])
                if high <= low:
                    high = low + 1
                array = np.clip((array - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
            else:
                array = np.zeros(array.shape, dtype=np.uint8)
            return Image.fromarray(array, mode="L")
        if image.mode not in {"RGB", "RGBA", "L"}:
            return image.convert("RGB")
        return image.copy()

    @staticmethod
    def _encoding(format_name: str) -> tuple[str, str]:
        normalized = format_name.casefold()
        if normalized == "png":
            return "PNG", "image/png"
        if normalized in {"jpg", "jpeg"}:
            return "JPEG", "image/jpeg"
        if normalized == "webp":
            return "WEBP", "image/webp"
        raise InvalidPathError("Unsupported preview format", details={"format": format_name})

    def _render_cached(self, path: Path, *, operation: str, format_name: str, render) -> RenderedImage:
        encoder, media_type = self._encoding(format_name)
        source_etag = self.source_etag(path)
        key = sha256(f"v1|{source_etag}|{operation}|{encoder}".encode()).hexdigest()
        suffix = ".jpg" if encoder == "JPEG" else f".{encoder.lower()}"
        cache_path = self.cache_root / key[:2] / f"{key}{suffix}"
        with self._key_locks.hold(key):
            if cache_path.is_file():
                with Image.open(cache_path) as cached:
                    width, height = cached.size
                return RenderedImage(cache_path.read_bytes(), media_type, key, width, height, True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(path) as source:
                    output = render(source)
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise InvalidPathError("Could not render image", details={"path": str(path), "error": str(exc)}) from exc
            if encoder == "JPEG" and output.mode not in {"RGB", "L"}:
                output = output.convert("RGB")
            buffer = BytesIO()
            save_options = {"quality": 86, "method": 4} if encoder == "WEBP" else {"quality": 90} if encoder == "JPEG" else {"compress_level": 4}
            output.save(buffer, encoder, **save_options)
            content = buffer.getvalue()
            partial = cache_path.parent / f".{cache_path.name}.{uuid4().hex}.part"
            try:
                with partial.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, cache_path)
            finally:
                if partial.exists():
                    partial.unlink()
            return RenderedImage(content, media_type, key, output.width, output.height, False)

    def thumbnail(self, dataset_id: str, asset_id: str, *, max_size: int = 256, format_name: str = "webp") -> RenderedImage:
        max_size = max(32, min(max_size, 2048))
        path = self.image_path(dataset_id, asset_id)

        def render(image: Image.Image) -> Image.Image:
            try:
                image.draft("RGB", (max_size, max_size))
            except (AttributeError, ValueError):
                pass
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return self._display_image(image)

        return self._render_cached(path, operation=f"thumbnail:{max_size}", format_name=format_name, render=render)

    def region(
        self,
        dataset_id: str,
        asset_id: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float = 1.0,
        format_name: str = "webp",
    ) -> RenderedImage:
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise InvalidPathError("Region coordinates must be positive")
        scale = max(0.05, min(scale, 8.0))
        max_source_pixels = 64_000_000
        max_output_pixels = 16_000_000
        if width * height > max_source_pixels or round(width * scale) * round(height * scale) > max_output_pixels:
            raise InvalidPathError(
                "Requested region exceeds the preview pixel budget",
                details={"max_source_pixels": max_source_pixels, "max_output_pixels": max_output_pixels},
            )
        path = self.image_path(dataset_id, asset_id)

        def render(image: Image.Image) -> Image.Image:
            if x >= image.width or y >= image.height:
                raise InvalidPathError("Region origin is outside the image")
            right = min(image.width, x + width)
            bottom = min(image.height, y + height)
            cropped = image.crop((x, y, right, bottom))
            if scale != 1.0:
                target = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
                cropped = cropped.resize(target, Image.Resampling.BICUBIC)
            return self._display_image(cropped)

        operation = f"region:{x}:{y}:{width}:{height}:{scale:.4f}"
        return self._render_cached(path, operation=operation, format_name=format_name, render=render)
