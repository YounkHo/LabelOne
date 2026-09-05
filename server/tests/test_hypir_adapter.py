from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shlex
import sys

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from labelone.config import Settings
from labelone.errors import ModelRuntimeError
from labelone.main import create_app
from labelone.models.adapters.hypir import HypirSd2SubprocessAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


LORA_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut",
    "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
]


def _runtime_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path, Path]:
    repository = tmp_path / "HYPIR-source"
    (repository / "HYPIR" / "enhancer").mkdir(parents=True)
    (repository / "HYPIR" / "enhancer" / "sd2.py").write_text("# fixture\n", encoding="utf-8")
    base_model = tmp_path / "stable-diffusion-2-1-base"
    base_model.mkdir()
    for entry in ("model_index.json", "scheduler", "text_encoder", "tokenizer", "unet", "vae"):
        path = base_model / entry
        if "." in entry:
            path.write_text("{}", encoding="utf-8")
        else:
            path.mkdir()
    weight = tmp_path / "HYPIR_sd2.pth"
    weight.write_bytes(b"fixture")
    fake_runner = tmp_path / "fake_hypir_runner.py"
    fake_runner.write_text(
        """from __future__ import annotations
import json
import sys
from pathlib import Path
from PIL import Image

def send(value):
    print(json.dumps(value), flush=True)

init = json.loads(sys.stdin.readline())
send({"type": "ready", "protocol": init["protocol"]})
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "close":
        send({"id": request["id"], "ok": True, "result": {}})
        break
    source = Image.open(request["image_path"]).convert("RGB")
    scale = request["upscale"]
    output = source.resize((source.width * scale, source.height * scale), Image.Resampling.NEAREST)
    output_path = Path(request["output_path"])
    output.save(output_path)
    send({"id": request["id"], "ok": True, "result": {
        "output_path": str(output_path), "width": output.width, "height": output.height, "elapsed_ms": 4.5,
    }})
""",
        encoding="utf-8",
    )
    runtime_python = tmp_path / "hypir-python"
    runtime_python.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake_runner))} \"$@\"\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o700)
    config: dict[str, object] = {
        "allow_external_code": True,
        "runtime_python": str(runtime_python),
        "repository_root": str(repository),
        "base_model_root": str(base_model),
        "weight_path": str(weight),
        "lora_modules": LORA_MODULES,
        "load_timeout": 10,
        "inference_timeout": 10,
        "max_output_pixels": 1024,
    }
    return config, runtime_python, repository, base_model, weight


def _adapter(tmp_path: Path, config: dict[str, object]) -> HypirSd2SubprocessAdapter:
    descriptor = ModelDescriptor(
        id="hypir-test",
        name="hypir-test",
        display_name="HYPIR Test",
        model_type="hypir_sd2",
        provider="fixture",
        task="super_resolution",
        family="hypir",
        adapter="hypir_sd2_pytorch",
        runtime=["HYPIR PyTorch / CUDA"],
        config_path=tmp_path / "hypir.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True, result_kinds=["rasters"]),
    )
    return HypirSd2SubprocessAdapter(ModelRecord(descriptor=descriptor, config=config), ArtifactStore(tmp_path / "artifacts"))


def test_hypir_adapter_runs_external_runtime_and_writes_scaled_raster(tmp_path: Path) -> None:
    config, *_ = _runtime_fixture(tmp_path)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (3, 2), (24, 80, 160)).save(image_path)
    adapter = _adapter(tmp_path, config)

    layers = adapter.load(["CPUExecutionProvider"])
    result = adapter.predict(image_path, [], {
        "prompt": "natural fine detail",
        "upscale": 3,
        "patch_size": 256,
        "stride": 128,
        "seed": 7,
        "output_format": "png",
    })
    adapter.unload()

    assert layers == []
    assert adapter.loaded is False
    assert result.annotations == [] and result.classifications == [] and result.artifacts == []
    assert result.timings_ms["runtime"] == 4.5
    artifact = result.rasters[0]
    assert artifact.role == "super-resolution"
    assert (artifact.width, artifact.height) == (9, 6)
    assert artifact.metadata["kind"] == "super_resolution"
    assert artifact.metadata["coordinate_mapping"]["forward"] == [3, 0, 0, 3, 0, 0]
    with Image.open(artifact.path) as output:
        assert output.size == (9, 6)
        assert output.getpixel((0, 0)) == (24, 80, 160)


def test_hypir_adapter_rejects_invalid_tiles_and_output_budget_before_dispatch(tmp_path: Path) -> None:
    config, *_ = _runtime_fixture(tmp_path)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    adapter = _adapter(tmp_path, config)
    adapter.load([])

    with pytest.raises(ModelRuntimeError, match="multiples of 8"):
        adapter.predict(image_path, [], {"upscale": 2, "patch_size": 255, "stride": 128})
    with pytest.raises(ModelRuntimeError, match="pixel budget"):
        adapter.predict(image_path, [], {"upscale": 8, "patch_size": 256, "stride": 128})
    adapter.unload()


def test_builtin_hypir_loads_through_api_worker_and_serves_content(tmp_path: Path, monkeypatch) -> None:
    _, runtime_python, repository, base_model, weight = _runtime_fixture(tmp_path)
    monkeypatch.setenv("LABELONE_HYPIR_PYTHON", str(runtime_python))
    monkeypatch.setenv("LABELONE_HYPIR_ROOT", str(repository))
    monkeypatch.setenv("LABELONE_HYPIR_SD21_BASE", str(base_model))
    monkeypatch.setenv("LABELONE_HYPIR_SD2_WEIGHT", str(weight))
    image_path = tmp_path / "api-input.png"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(image_path)
    app = create_app(Settings(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        catalog = client.get("/api/v1/models")
        loaded = client.post("/api/v1/models/hypir-sd2/load", json={"providers": ["CPUExecutionProvider"]})
        inferred = client.post("/api/v1/inference-runs", json={
            "model_id": "hypir-sd2",
            "image_path": str(image_path),
            "parameters": {"upscale": 2, "patch_size": 256, "stride": 128},
        })
        content = client.get(f"/api/v1/artifacts/{inferred.json()['rasters'][0]['id']}/content")

    model = next(item for item in catalog.json()["models"] if item["id"] == "hypir-sd2")
    assert model["availability"]["state"] == "available"
    assert model["license_name"] == "HYPIR Non-Commercial License"
    assert loaded.status_code == 200 and loaded.json()["state"] == "loaded"
    assert inferred.status_code == 200
    assert inferred.json()["rasters"][0]["width"] == 8
    with Image.open(BytesIO(content.content)) as output:
        assert output.size == (8, 6)
