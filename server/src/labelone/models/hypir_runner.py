from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback
from typing import Any, Iterator

import numpy as np
from PIL import Image


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def _read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("HYPIR runner message must be an object")
    return payload


@contextmanager
def _repository_imports(repository_root: Path) -> Iterator[None]:
    value = str(repository_root)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)


def _safe_output_path(runtime_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("HYPIR output_path is required")
    output_path = Path(raw_path).expanduser().resolve()
    if runtime_root not in output_path.parents:
        raise ValueError("HYPIR output path escaped the runtime directory")
    return output_path


def _load_model(init: dict[str, Any]):
    repository_root = Path(str(init["repository_root"])).expanduser().resolve()
    base_model_root = Path(str(init["base_model_root"])).expanduser().resolve()
    weight_path = Path(str(init["weight_path"])).expanduser().resolve()
    runtime_root = Path(str(init["runtime_root"])).expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    with _repository_imports(repository_root), redirect_stdout(sys.stderr):
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("The official HYPIR-SD2 runtime requires an NVIDIA CUDA device")
        enhancer_module = importlib.import_module("HYPIR.enhancer.sd2")
        model = enhancer_module.SD2Enhancer(
            base_model_path=str(base_model_root),
            weight_path=str(weight_path),
            lora_modules=list(init["lora_modules"]),
            lora_rank=int(init.get("lora_rank", 256)),
            model_t=int(init.get("model_t", 200)),
            coeff_t=int(init.get("coeff_t", 200)),
            device="cuda",
        )
        model.init_models()
    return torch, model, runtime_root


def _image_tensor(torch, image: Image.Image):  # noqa: ANN001
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)


def _predict(torch, model, runtime_root: Path, request: dict[str, Any]) -> dict[str, object]:  # noqa: ANN001
    started = perf_counter()
    image_path = Path(str(request["image_path"])).expanduser().resolve()
    output_path = _safe_output_path(runtime_root, request.get("output_path"))
    prompt = str(request.get("prompt", ""))
    upscale = int(request.get("upscale", 4))
    patch_size = int(request.get("patch_size", 512))
    stride = int(request.get("stride", 256))
    seed = int(request.get("seed", 231))
    maximum_pixels = int(request.get("max_output_pixels", 64_000_000))
    format_name = str(request.get("format", "png")).casefold()
    if not 1 <= upscale <= 8 or patch_size <= 0 or stride <= 0 or patch_size < stride:
        raise ValueError("HYPIR upscale, patch_size, or stride is outside the supported range")
    if format_name not in {"png", "webp", "jpeg"}:
        raise ValueError("HYPIR output format is unsupported")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        image.load()
    output_size = (image.width * upscale, image.height * upscale)
    if output_size[0] * output_size[1] > maximum_pixels:
        raise ValueError("HYPIR output exceeds the configured pixel budget")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), redirect_stdout(sys.stderr):
        output = model.enhance(
            lq=_image_tensor(torch, image),
            prompt=prompt,
            scale_by="factor",
            upscale=upscale,
            patch_size=patch_size,
            stride=stride,
            return_type="pil",
        )[0]
    if output.size != output_size:
        raise ValueError(f"HYPIR returned {output.size}, expected {output_size}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.part")
    encodings = {
        "png": ("PNG", {"compress_level": 4}),
        "webp": ("WEBP", {"quality": 95, "method": 4}),
        "jpeg": ("JPEG", {"quality": 95}),
    }
    encoder, options = encodings[format_name]
    try:
        output.convert("RGB").save(partial, encoder, **options)
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "output_path": str(output_path),
        "width": output.width,
        "height": output.height,
        "elapsed_ms": (perf_counter() - started) * 1000.0,
    }


def main() -> int:
    try:
        init = _read()
        if init is None or init.get("type") != "init" or init.get("protocol") != 1:
            raise ValueError("HYPIR runner initialization is invalid")
        torch, model, runtime_root = _load_model(init)
        _write({"type": "ready", "protocol": 1})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _write({"type": "startup_error", "message": str(exc)})
        return 2

    while True:
        request = _read()
        if request is None:
            return 0
        request_id = request.get("id")
        try:
            if request.get("op") == "close":
                _write({"id": request_id, "ok": True, "result": {}})
                return 0
            if request.get("op") != "predict":
                raise ValueError("HYPIR runner operation is not supported")
            _write({"id": request_id, "ok": True, "result": _predict(torch, model, runtime_root, request)})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _write({"id": request_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
