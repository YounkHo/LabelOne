from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from labelone.errors import ModelRuntimeError

from ..hypir_config import resolve_hypir_runtime
from ..types import FeatureLayer, InferenceResult
from .base import ModelAdapter


ProcessFactory = Callable[..., subprocess.Popen[str]]


def _integer_parameter(
    parameters: dict[str, object],
    config: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = parameters.get(name, config.get(name, default))
    if isinstance(raw, bool):
        raise ModelRuntimeError(f"HYPIR {name} must be an integer")
    try:
        value = int(raw)
        exact = float(raw) == value
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"HYPIR {name} must be an integer") from exc
    if not exact or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"HYPIR {name} is outside the supported range",
            details={"value": raw, "minimum": minimum, "maximum": maximum},
        )
    return value


def _output_format(parameters: dict[str, object], config: dict[str, Any]) -> str:
    value = str(parameters.get("output_format", config.get("output_format", "png"))).strip().casefold()
    if value not in {"png", "webp", "jpeg"}:
        raise ModelRuntimeError("HYPIR output_format is unsupported", details={"value": value})
    return value


class HypirSd2SubprocessAdapter(ModelAdapter):
    """Runs the official HYPIR-SD2 package in its pinned CUDA Python environment."""

    def __init__(self, record, artifact_store, *, process_factory: ProcessFactory | None = None) -> None:  # noqa: ANN001
        super().__init__(record, artifact_store)
        self.process_factory = process_factory or subprocess.Popen
        self.process: subprocess.Popen[str] | None = None
        self.messages: Queue[dict[str, object] | None] = Queue()
        self.stderr_tail: deque[str] = deque(maxlen=80)
        self.stdout_thread: Thread | None = None
        self.stderr_thread: Thread | None = None
        self.request_id = 0
        self.runtime_dir = (artifact_store.root / ".hypir-runtime" / record.descriptor.id).resolve()

    def _stderr(self) -> str:
        return "".join(self.stderr_tail)[-8192:]

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self.messages.put(None)
                return
            if not isinstance(payload, dict):
                self.messages.put(None)
                return
            self.messages.put(payload)
        self.messages.put(None)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            self.stderr_tail.append(line)

    def _next(self, timeout: float) -> dict[str, object]:
        try:
            payload = self.messages.get(timeout=timeout)
        except Empty as exc:
            self._terminate()
            raise ModelRuntimeError("HYPIR runtime timed out", details={"timeout": timeout, "stderr": self._stderr()}) from exc
        if payload is None:
            self._terminate()
            raise ModelRuntimeError("HYPIR runtime exited or returned invalid JSON", details={"stderr": self._stderr()})
        return payload

    def _write(self, payload: dict[str, object]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise ModelRuntimeError("HYPIR runtime is not running", details={"stderr": self._stderr()})
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._terminate()
            raise ModelRuntimeError("Could not send a request to the HYPIR runtime", details={"stderr": self._stderr()}) from exc

    def _terminate(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def load(self, providers: list[str]) -> list[FeatureLayer]:
        del providers
        paths, reason = resolve_hypir_runtime(self.record.config)
        if paths is None:
            raise ModelRuntimeError("HYPIR-SD2 runtime is not configured", details={"reason": reason})
        lora_modules = self.record.config.get("lora_modules")
        if not isinstance(lora_modules, list) or not lora_modules or any(not isinstance(item, str) or not item for item in lora_modules):
            raise ModelRuntimeError("HYPIR-SD2 lora_modules must be a non-empty string array")
        self._terminate()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.messages = Queue()
        self.stderr_tail.clear()
        runner = Path(__file__).resolve().parents[1] / "hypir_runner.py"
        try:
            process = self.process_factory(
                [str(paths.python), str(runner)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=os.environ.copy(),
            )
        except OSError as exc:
            raise ModelRuntimeError("Could not start the HYPIR Python runtime", details={"error": str(exc)}) from exc
        self.process = process
        self.stdout_thread = Thread(target=self._read_stdout, args=(process,), name="labelone-hypir-stdout", daemon=True)
        self.stderr_thread = Thread(target=self._read_stderr, args=(process,), name="labelone-hypir-stderr", daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()
        self._write({
            "type": "init",
            "protocol": 1,
            "repository_root": str(paths.repository_root),
            "base_model_root": str(paths.base_model_root),
            "weight_path": str(paths.weight),
            "runtime_root": str(self.runtime_dir),
            "lora_modules": lora_modules,
            "lora_rank": self.record.config.get("lora_rank", 256),
            "model_t": self.record.config.get("model_t", 200),
            "coeff_t": self.record.config.get("coeff_t", 200),
        })
        timeout = _integer_parameter({}, self.record.config, "load_timeout", 600, minimum=1, maximum=3600)
        ready = self._next(float(timeout))
        if ready.get("type") != "ready" or ready.get("protocol") != 1:
            message = str(ready.get("message") or "HYPIR runtime did not complete its handshake")
            details = {"stderr": self._stderr()}
            self._terminate()
            raise ModelRuntimeError(message, details=details)
        self.loaded = True
        return []

    def unload(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.request_id += 1
                self._write({"id": self.request_id, "op": "close"})
                self._next(2.0)
            except ModelRuntimeError:
                pass
        self._terminate()
        self.loaded = False

    def list_layers(self) -> list[FeatureLayer]:
        return []

    def predict(self, image_path: Path, capture_layers: list[str], parameters: dict[str, object]) -> InferenceResult:
        if not self.loaded or self.process is None:
            raise ModelRuntimeError("HYPIR-SD2 model is not loaded")
        if capture_layers:
            raise ModelRuntimeError("HYPIR-SD2 does not expose intermediate feature capture")
        prompt = parameters.get("prompt", self.record.config.get("prompt", ""))
        if not isinstance(prompt, str) or len(prompt) > 512 or any(ord(character) < 32 and character != "\t" for character in prompt):
            raise ModelRuntimeError("HYPIR prompt must be a bounded single-line string")
        upscale = _integer_parameter(parameters, self.record.config, "upscale", 4, minimum=1, maximum=8)
        patch_size = _integer_parameter(parameters, self.record.config, "patch_size", 512, minimum=128, maximum=2048)
        stride = _integer_parameter(parameters, self.record.config, "stride", 256, minimum=64, maximum=2048)
        if patch_size < stride or patch_size % 8 or stride % 8:
            raise ModelRuntimeError("HYPIR patch_size and stride must be multiples of 8, with patch_size at least stride")
        seed = _integer_parameter(parameters, self.record.config, "seed", 231, minimum=0, maximum=2**32 - 1)
        maximum_pixels = _integer_parameter({}, self.record.config, "max_output_pixels", 64_000_000, minimum=1, maximum=268_435_456)
        output_format = _output_format(parameters, self.record.config)
        try:
            with Image.open(image_path) as source:
                source_width, source_height = source.size
        except (OSError, ValueError) as exc:
            raise ModelRuntimeError("Could not read the HYPIR input image") from exc
        output_width, output_height = source_width * upscale, source_height * upscale
        if output_width * output_height > maximum_pixels:
            raise ModelRuntimeError(
                "HYPIR output exceeds the configured pixel budget",
                details={"output_size": [output_width, output_height], "maximum_pixels": maximum_pixels},
            )
        suffix = {"png": ".png", "webp": ".webp", "jpeg": ".jpg"}[output_format]
        temporary = self.runtime_dir / f"{uuid4().hex}{suffix}"
        started = perf_counter()
        self.request_id += 1
        request_id = self.request_id
        try:
            self._write({
                "id": request_id,
                "op": "predict",
                "image_path": str(image_path.expanduser().resolve()),
                "output_path": str(temporary),
                "prompt": prompt,
                "upscale": upscale,
                "patch_size": patch_size,
                "stride": stride,
                "seed": seed,
                "format": output_format,
                "max_output_pixels": maximum_pixels,
            })
            timeout = _integer_parameter({}, self.record.config, "inference_timeout", 600, minimum=1, maximum=3600)
            response = self._next(float(timeout))
            if response.get("id") != request_id or response.get("ok") is not True:
                raise ModelRuntimeError(str(response.get("error") or "HYPIR inference failed"), details={"stderr": self._stderr()})
            result = response.get("result")
            if not isinstance(result, dict) or Path(str(result.get("output_path", ""))).resolve() != temporary.resolve():
                raise ModelRuntimeError("HYPIR runtime returned an invalid output path")
            if not temporary.is_file():
                raise ModelRuntimeError("HYPIR runtime did not create its output image")
            with Image.open(temporary) as handle:
                output = handle.convert("RGB")
                output.load()
            if output.size != (output_width, output_height):
                raise ModelRuntimeError(
                    "HYPIR output dimensions do not match the requested scale",
                    details={"expected": [output_width, output_height], "actual": list(output.size)},
                )
            artifact = self.artifact_store.put_raster(
                model_id=self.record.descriptor.id,
                image_path=image_path,
                role="super-resolution",
                image=output,
                format_name=output_format,
                metadata={
                    "kind": "super_resolution",
                    "model": "HYPIR-SD2",
                    "scale": upscale,
                    "source_size": [source_width, source_height],
                    "output_size": [output_width, output_height],
                    "coordinate_mapping": {
                        "kind": "affine",
                        "forward": [upscale, 0, 0, upscale, 0, 0],
                        "inverse": [1 / upscale, 0, 0, 1 / upscale, 0, 0],
                        "topology_safe": True,
                    },
                    "prompt": prompt,
                    "seed": seed,
                    "patch_size": patch_size,
                    "stride": stride,
                },
            )
            timings = {"total": (perf_counter() - started) * 1000.0}
            runtime_elapsed = result.get("elapsed_ms")
            if isinstance(runtime_elapsed, (int, float)):
                timings["runtime"] = float(runtime_elapsed)
            return InferenceResult(model_id=self.record.descriptor.id, image_path=image_path, rasters=[artifact], timings_ms=timings)
        finally:
            temporary.unlink(missing_ok=True)
