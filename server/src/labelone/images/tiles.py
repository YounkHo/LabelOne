from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import ceil, floor
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Iterator
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from labelone.datasets.repository import DatasetRepository
from labelone.errors import InvalidPathError


@dataclass(frozen=True, slots=True)
class TileMetadata:
    width: int
    height: int
    tile_size: int
    max_level: int
    format: str
    source_etag: str
    backend: str
    source_format: str | None = None


@dataclass(frozen=True, slots=True)
class RenderedTile:
    content: bytes
    media_type: str
    etag: str
    cache_hit: bool
    width: int
    height: int
    backend: str


@dataclass(slots=True)
class _SingleFlight:
    lock: Lock
    users: int = 0


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


class DeepZoomTileService:
    """Deep Zoom tile renderer with a bounded Pillow fallback and durable cache."""

    def __init__(
        self,
        repository: DatasetRepository,
        cache_root: Path,
        *,
        tile_size: int = 256,
        default_format: str = "webp",
        max_output_pixels: int = 4_194_304,
        max_source_pixels: int = 536_870_912,
    ) -> None:
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or not 32 <= tile_size <= 2048:
            raise ValueError("tile_size must be an integer between 32 and 2048")
        if max_output_pixels < tile_size * tile_size:
            raise ValueError("max_output_pixels must allow at least one full tile")
        if max_source_pixels <= 0:
            raise ValueError("max_source_pixels must be positive")
        self.repository = repository
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.tile_size = tile_size
        self.default_format = self._encoding(default_format)[0]
        self.max_output_pixels = max_output_pixels
        self.max_source_pixels = max_source_pixels
        try:
            import pyvips
            self._pyvips: Any | None = pyvips
        except (ImportError, OSError):
            self._pyvips = None
        self.backend = "pyvips" if self._pyvips is not None else "pillow"
        self._lock = RLock()
        self._single_flights: dict[str, _SingleFlight] = {}

    def _image_path(self, dataset_id: str, asset_id: str) -> Path:
        asset = self.repository.get_asset(dataset_id, asset_id, require_selectable=True)
        if asset.image_path is None or not asset.image_path.is_file():
            raise InvalidPathError(
                "Tile source image is missing",
                details={"dataset_id": dataset_id, "asset_id": asset_id},
            )
        return asset.image_path.resolve()

    @staticmethod
    def _source_etag(path: Path) -> str:
        stat = path.stat()
        return sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()

    @staticmethod
    def _encoding(format_name: str) -> tuple[str, str, str]:
        normalized = format_name.strip().casefold()
        if normalized == "png":
            return "png", "PNG", "image/png"
        if normalized in {"jpg", "jpeg"}:
            return "jpeg", "JPEG", "image/jpeg"
        if normalized == "webp":
            return "webp", "WEBP", "image/webp"
        raise InvalidPathError(
            "Unsupported tile format",
            details={"format": format_name, "supported": ["webp", "png", "jpeg"]},
        )

    @staticmethod
    def _max_level(width: int, height: int) -> int:
        return (max(width, height) - 1).bit_length()

    def _inspect(self, path: Path, *, format_name: str) -> TileMetadata:
        normalized_format, _, _ = self._encoding(format_name)
        for _ in range(2):
            before = self._source_etag(path)
            try:
                if self._pyvips is not None:
                    source = self._pyvips.Image.new_from_file(str(path), access="sequential")
                    width, height = int(source.width), int(source.height)
                    source_format = str(source.get("vips-loader")) if source.get_typeof("vips-loader") else path.suffix.lstrip(".").upper()
                else:
                    with Image.open(path) as source:
                        width, height = source.size
                        source_format = source.format
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError, RuntimeError) as exc:
                raise InvalidPathError(
                    "Could not inspect tile source image",
                    details={"path": str(path), "error": str(exc), "backend": self.backend},
                ) from exc
            after = self._source_etag(path)
            if before == after:
                if width <= 0 or height <= 0:
                    raise InvalidPathError(
                        "Tile source image has invalid dimensions",
                        details={"path": str(path), "width": width, "height": height},
                    )
                if self._pyvips is None and width * height > self.max_source_pixels:
                    raise InvalidPathError(
                        "Tile source exceeds the Pillow decode budget",
                        details={
                            "path": str(path),
                            "pixels": width * height,
                            "maximum": self.max_source_pixels,
                            "backend": self.backend,
                        },
                    )
                return TileMetadata(
                    width=width,
                    height=height,
                    tile_size=self.tile_size,
                    max_level=self._max_level(width, height),
                    format=normalized_format,
                    source_etag=after,
                    backend=self.backend,
                    source_format=source_format,
                )
        raise InvalidPathError(
            "Tile source changed while reading metadata; retry",
            details={"path": str(path)},
        )

    def metadata(
        self,
        dataset_id: str,
        asset_id: str,
        *,
        format_name: str | None = None,
    ) -> TileMetadata:
        path = self._image_path(dataset_id, asset_id)
        return self._inspect(path, format_name=format_name or self.default_format)

    @contextmanager
    def _single_flight(self, key: str) -> Iterator[None]:
        with self._lock:
            entry = self._single_flights.get(key)
            if entry is None:
                entry = _SingleFlight(lock=Lock())
                self._single_flights[key] = entry
            entry.users += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._lock:
                entry.users -= 1
                if entry.users == 0 and self._single_flights.get(key) is entry:
                    self._single_flights.pop(key, None)

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
            return Image.fromarray(array)
        if image.mode not in {"RGB", "RGBA", "L"}:
            return image.convert("RGB")
        return image.copy()

    @staticmethod
    def _validate_coordinate(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidPathError(
                "Tile coordinates must be non-negative integers",
                details={"coordinate": name, "value": value},
            )
        return value

    def _geometry(
        self,
        metadata: TileMetadata,
        *,
        level: int,
        x: int,
        y: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int], int]:
        level = self._validate_coordinate("level", level)
        x = self._validate_coordinate("x", x)
        y = self._validate_coordinate("y", y)
        if level > metadata.max_level:
            raise InvalidPathError(
                "Tile level is outside the pyramid",
                details={"level": level, "max_level": metadata.max_level},
            )
        downsample = 1 << (metadata.max_level - level)
        level_width = _ceil_div(metadata.width, downsample)
        level_height = _ceil_div(metadata.height, downsample)
        columns = _ceil_div(level_width, metadata.tile_size)
        rows = _ceil_div(level_height, metadata.tile_size)
        if x >= columns or y >= rows:
            raise InvalidPathError(
                "Tile coordinate is outside the level",
                details={
                    "level": level,
                    "x": x,
                    "y": y,
                    "columns": columns,
                    "rows": rows,
                    "level_size": [level_width, level_height],
                },
            )
        target_left = x * metadata.tile_size
        target_top = y * metadata.tile_size
        target_right = min(level_width, target_left + metadata.tile_size)
        target_bottom = min(level_height, target_top + metadata.tile_size)
        output_width = target_right - target_left
        output_height = target_bottom - target_top
        if output_width * output_height > self.max_output_pixels:
            raise InvalidPathError(
                "Tile output exceeds the pixel budget",
                details={
                    "output_size": [output_width, output_height],
                    "pixels": output_width * output_height,
                    "maximum": self.max_output_pixels,
                },
            )
        source_box = (
            target_left * downsample,
            target_top * downsample,
            min(metadata.width, target_right * downsample),
            min(metadata.height, target_bottom * downsample),
        )
        return source_box, (output_width, output_height), downsample

    def _render_pillow(
        self,
        path: Path,
        *,
        source_box: tuple[int, int, int, int],
        output_size: tuple[int, int],
        downsample: int,
        encoder: str,
    ) -> bytes:
        try:
            with Image.open(path) as source:
                original_width, original_height = source.size
                if downsample > 1:
                    try:
                        source.draft(
                            "RGB",
                            (_ceil_div(original_width, downsample), _ceil_div(original_height, downsample)),
                        )
                    except (AttributeError, ValueError):
                        pass
                decoded_width, decoded_height = source.size
                left, top, right, bottom = source_box
                decoded_box = (
                    max(0, floor(left * decoded_width / original_width)),
                    max(0, floor(top * decoded_height / original_height)),
                    min(decoded_width, ceil(right * decoded_width / original_width)),
                    min(decoded_height, ceil(bottom * decoded_height / original_height)),
                )
                output = source.crop(decoded_box)
                if output.size != output_size:
                    output = output.resize(output_size, Image.Resampling.LANCZOS)
                output = self._display_image(output)
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise InvalidPathError(
                "Could not render image tile",
                details={"path": str(path), "error": str(exc), "backend": self.backend},
            ) from exc
        if encoder == "JPEG" and output.mode not in {"RGB", "L"}:
            output = output.convert("RGB")
        buffer = BytesIO()
        options = (
            {"quality": 86, "method": 4}
            if encoder == "WEBP"
            else {"quality": 90}
            if encoder == "JPEG"
            else {"compress_level": 4}
        )
        output.save(buffer, encoder, **options)
        return buffer.getvalue()

    def _render_vips(
        self,
        path: Path,
        *,
        source_box: tuple[int, int, int, int],
        output_size: tuple[int, int],
        encoder: str,
    ) -> bytes:
        if self._pyvips is None:
            raise RuntimeError("pyvips backend is unavailable")
        left, top, right, bottom = source_box
        try:
            image = self._pyvips.Image.new_from_file(str(path), access="random")
            tile = image.crop(left, top, right - left, bottom - top)
            horizontal_scale = output_size[0] / max(1, tile.width)
            vertical_scale = output_size[1] / max(1, tile.height)
            if horizontal_scale != 1 or vertical_scale != 1:
                tile = tile.resize(horizontal_scale, vscale=vertical_scale, kernel="lanczos3")
            if tile.bands > 4:
                tile = tile.extract_band(0, n=3)
            elif tile.bands == 2:
                tile = tile.extract_band(0)
            if tile.format not in {"uchar", "char"}:
                minimum = float(tile.min())
                maximum = float(tile.max())
                tile = ((tile - minimum) * (255.0 / max(maximum - minimum, 1e-12))).clip(0, 255).cast("uchar")
            elif tile.format == "char":
                tile = (tile + 128).cast("uchar")
            if encoder == "WEBP":
                return bytes(tile.webpsave_buffer(Q=86, effort=4, strip=True))
            if encoder == "JPEG":
                if tile.bands == 4:
                    tile = tile.flatten(background=[0, 0, 0])
                return bytes(tile.jpegsave_buffer(Q=90, strip=True))
            return bytes(tile.pngsave_buffer(compression=4, strip=True))
        except Exception as exc:
            raise InvalidPathError(
                "Could not render image tile",
                details={"path": str(path), "error": str(exc), "backend": self.backend},
            ) from exc

    @staticmethod
    def _read_valid_cache(path: Path, *, expected_size: tuple[int, int]) -> bytes | None:
        if not path.is_file():
            return None
        try:
            content = path.read_bytes()
            with Image.open(BytesIO(content)) as cached:
                cached.load()
                if cached.size != expected_size:
                    raise ValueError(f"cached size {cached.size} does not match {expected_size}")
            return content
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None

    @staticmethod
    def _write_atomic(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.parent / f".{path.name}.{uuid4().hex}.part"
        try:
            with partial.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
            if hasattr(os, "O_DIRECTORY"):
                descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if partial.exists():
                partial.unlink()

    def tile(
        self,
        dataset_id: str,
        asset_id: str,
        *,
        level: int,
        x: int,
        y: int,
        format_name: str | None = None,
    ) -> RenderedTile:
        path = self._image_path(dataset_id, asset_id)
        normalized_format, encoder, media_type = self._encoding(format_name or self.default_format)
        metadata = self._inspect(path, format_name=normalized_format)
        source_box, output_size, downsample = self._geometry(metadata, level=level, x=x, y=y)
        key = sha256(
            (
                f"deep-zoom-v1|{metadata.source_etag}|{metadata.tile_size}|{level}|{x}|{y}|"
                f"{normalized_format}|{self.backend}|lanczos"
            ).encode()
        ).hexdigest()
        suffix = ".jpg" if normalized_format == "jpeg" else f".{normalized_format}"
        cache_path = self.cache_root / key[:2] / f"{key}{suffix}"
        with self._single_flight(key):
            cached = self._read_valid_cache(cache_path, expected_size=output_size)
            if cached is not None:
                return RenderedTile(
                    content=cached,
                    media_type=media_type,
                    etag=key,
                    cache_hit=True,
                    width=output_size[0],
                    height=output_size[1],
                    backend=self.backend,
                )
            content = (
                self._render_vips(
                    path,
                    source_box=source_box,
                    output_size=output_size,
                    encoder=encoder,
                )
                if self._pyvips is not None
                else self._render_pillow(
                    path,
                    source_box=source_box,
                    output_size=output_size,
                    downsample=downsample,
                    encoder=encoder,
                )
            )
            if self._source_etag(path) != metadata.source_etag:
                raise InvalidPathError(
                    "Tile source changed while rendering; retry",
                    details={"path": str(path), "level": level, "x": x, "y": y},
                )
            self._write_atomic(cache_path, content)
            return RenderedTile(
                content=content,
                media_type=media_type,
                etag=key,
                cache_hit=False,
                width=output_size[0],
                height=output_size[1],
                backend=self.backend,
            )
