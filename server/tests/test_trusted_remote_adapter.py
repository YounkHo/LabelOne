from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request

from PIL import Image
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.adapters.trusted_remote import TrustedRemoteHttpAdapter
from labelone.models.artifacts import ArtifactStore
from labelone.models.catalog import ModelRecord
from labelone.models.types import Availability, AvailabilityState, ModelCapabilities, ModelDescriptor


class _Response:
    def __init__(self, url: str, payload: dict[str, object], status: int = 200) -> None:
        self.url = url
        self.body = json.dumps(payload).encode()
        self.status = status
        self.offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        end = len(self.body) if size < 0 else min(len(self.body), self.offset + size)
        chunk = self.body[self.offset:end]
        self.offset = end
        return chunk

    def close(self) -> None:
        self.closed = True

    def geturl(self) -> str:
        return self.url


class _Opener:
    def __init__(self, responses: dict[str, list[_Response] | _Response]) -> None:
        self.responses = {
            url: list(value) if isinstance(value, list) else [value]
            for url, value in responses.items()
        }
        self.requests: list[Request] = []

    def __call__(self, request: Request, timeout: float) -> _Response:
        del timeout
        self.requests.append(request)
        queue = self.responses.get(request.full_url)
        if not queue:
            raise AssertionError(f"Unexpected remote request: {request.full_url}")
        return queue.pop(0)


def _record(tmp_path: Path, protocol: str, endpoint: str, **config) -> ModelRecord:
    descriptor = ModelDescriptor(
        id="remote",
        name="remote",
        display_name="Trusted Remote",
        model_type="remote_server" if protocol == "labelone_v1" else "grounding_dino_api",
        task="grounding",
        family="remote",
        adapter="trusted_remote_http",
        runtime=["Trusted Remote HTTPS"],
        config_path=tmp_path / "remote.yaml",
        availability=Availability(state=AvailabilityState.AVAILABLE),
        capabilities=ModelCapabilities(predict=True, result_kinds=["annotations"]),
    )
    values = {
        "remote_endpoint": endpoint,
        "trusted_hosts": [endpoint.split("/")[2]],
        "credential_env": "LABELONE_TEST_REMOTE_TOKEN",
        "credential_header": "Token",
        "remote_protocol": protocol,
        **config,
    }
    return ModelRecord(descriptor=descriptor, config=values)


def test_labelone_remote_load_predict_has_annotations_and_no_layers(tmp_path: Path, monkeypatch) -> None:
    endpoint = "https://trusted.example/v1/predict"
    response = _Response(endpoint, {
        "data": {
            "shapes": [{
                "label": "scratch",
                "score": 0.91,
                "shape_type": "rectangle",
                "points": [[2, 3], [30, 20]],
            }]
        }
    })
    opener = _Opener({endpoint: response})
    adapter = TrustedRemoteHttpAdapter(
        _record(tmp_path, "labelone_v1", endpoint, remote_model_id="detector"),
        ArtifactStore(tmp_path / "artifacts"),
        opener=opener,
    )
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    monkeypatch.setenv("LABELONE_TEST_REMOTE_TOKEN", "secret")

    layers = adapter.load(["CPUExecutionProvider"])
    result = adapter.predict(image_path, [], {"conf_threshold": 0.4})

    assert layers == adapter.list_layers() == []
    assert result.annotations[0].label == "scratch"
    assert result.annotations[0].points == [[2.0, 3.0], [30.0, 20.0]]
    assert opener.requests[0].get_header("Token") == "secret"
    request_payload = json.loads(opener.requests[0].data)
    assert request_payload["model"] == "detector"
    assert request_payload["image"].startswith("data:image/png;base64,")
    assert response.closed
    with pytest.raises(ModelRuntimeError, match="do not expose intermediate layers"):
        adapter.predict(image_path, ["hidden.layer"], {})


def test_grounding_remote_runs_explicit_async_protocol_fixture(tmp_path: Path, monkeypatch) -> None:
    endpoint = "https://api.deepdataspace.com"
    submit_url = f"{endpoint}/v2/task/grounding_dino/detection"
    status_url = f"{endpoint}/v2/task_status/task_123"
    submit = _Response(submit_url, {"code": 0, "data": {"task_uuid": "task_123"}})
    running = _Response(status_url, {"code": 0, "data": {"status": "running"}})
    success = _Response(status_url, {
        "code": 0,
        "data": {
            "status": "success",
            "result": {"objects": [{"bbox": [1, 2, 12, 14], "category": "cat", "score": 0.88}]},
        },
    })
    opener = _Opener({submit_url: submit, status_url: [running, success]})
    adapter = TrustedRemoteHttpAdapter(
        _record(tmp_path, "grounding_dino_v2", endpoint, poll_interval=0, max_poll_attempts=3),
        ArtifactStore(tmp_path / "artifacts"),
        opener=opener,
    )
    image_path = tmp_path / "image.png"
    Image.new("RGB", (20, 16), "white").save(image_path)
    monkeypatch.setenv("LABELONE_TEST_REMOTE_TOKEN", "secret")

    adapter.load([])
    result = adapter.predict(image_path, [], {"text_prompt": "cat"})

    assert result.annotations[0].label == "cat"
    assert result.annotations[0].score == pytest.approx(0.88)
    assert result.annotations[0].points == [[1.0, 2.0], [12.0, 14.0]]
    assert [request.full_url for request in opener.requests] == [submit_url, status_url, status_url]


def test_trusted_remote_enforces_encoded_request_budget_before_network(tmp_path: Path, monkeypatch) -> None:
    endpoint = "https://trusted.example/v1/predict"
    opener = _Opener({})
    adapter = TrustedRemoteHttpAdapter(
        _record(
            tmp_path,
            "labelone_v1",
            endpoint,
            remote_model_id="detector",
            max_request_bytes=1024,
        ),
        ArtifactStore(tmp_path / "artifacts"),
        opener=opener,
    )
    image_path = tmp_path / "image.png"
    pixels = bytes((index * 73 + index // 11) % 256 for index in range(128 * 128 * 3))
    Image.frombytes("RGB", (128, 128), pixels).save(image_path)
    monkeypatch.setenv("LABELONE_TEST_REMOTE_TOKEN", "secret")

    adapter.load([])
    with pytest.raises(ModelRuntimeError, match="request byte budget"):
        adapter.predict(image_path, [], {})

    assert opener.requests == []


@pytest.mark.parametrize(
    "config, message",
    [
        ({"remote_endpoint": "http://trusted.example/v1/predict"}, "credential-free HTTPS"),
        ({"trusted_hosts": ["other.example"]}, "not explicitly trusted"),
        ({"credential_env": "bad-name"}, "environment variable"),
        ({"credential_header": "Cookie"}, "header"),
    ],
)
def test_trusted_remote_rejects_insecure_or_implicit_configuration(
    tmp_path: Path,
    monkeypatch,
    config: dict[str, object],
    message: str,
) -> None:
    endpoint = "https://trusted.example/v1/predict"
    record = _record(tmp_path, "labelone_v1", endpoint, remote_model_id="detector", **config)
    adapter = TrustedRemoteHttpAdapter(record, ArtifactStore(tmp_path / "artifacts"), opener=_Opener({}))
    monkeypatch.setenv("LABELONE_TEST_REMOTE_TOKEN", "secret")

    with pytest.raises(ModelRuntimeError) as caught:
        adapter.load([])

    assert message in str(caught.value.details.get("reason"))
