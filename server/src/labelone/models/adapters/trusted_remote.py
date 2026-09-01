from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import re
from time import perf_counter, sleep
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult, FeatureLayer, InferenceResult
from .base import ModelAdapter


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
_PROTOCOLS = {"labelone_v1", "grounding_dino_v2"}
_HEADERS = {"Authorization", "Token", "X-API-Key"}


class RemoteResponse(Protocol):
    status: int

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def geturl(self) -> str: ...


RemoteOpener = Callable[[Request, float], RemoteResponse]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):  # noqa: ANN001
        del request, file_pointer, code, message, headers, new_url
        return None


def _default_open(request: Request, timeout: float) -> RemoteResponse:
    opener = build_opener(_NoRedirect())
    try:
        return opener.open(request, timeout=timeout)
    except HTTPError as exc:
        return exc


def trusted_remote_configuration(config: dict[str, Any], *, require_secret: bool) -> tuple[bool, str | None]:
    endpoint = config.get("remote_endpoint")
    trusted_hosts = config.get("trusted_hosts")
    credential_env = config.get("credential_env")
    protocol = config.get("remote_protocol")
    if not isinstance(endpoint, str) or not endpoint:
        return False, "Remote endpoint is not explicitly configured"
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
        or parsed.query
    ):
        return False, "Remote endpoint must be credential-free HTTPS on the default port"
    hostname = (parsed.hostname or "").casefold().strip(".")
    if not isinstance(trusted_hosts, list) or not trusted_hosts or any(not isinstance(host, str) for host in trusted_hosts):
        return False, "Remote trusted_hosts must be an explicit non-empty list"
    normalized_hosts = {host.casefold().strip(".") for host in trusted_hosts}
    if hostname not in normalized_hosts:
        return False, "Remote endpoint hostname is not explicitly trusted"
    if protocol not in _PROTOCOLS:
        return False, "Remote protocol is not supported"
    if not isinstance(credential_env, str) or not _ENV_NAME.fullmatch(credential_env):
        return False, "Remote credential_env must name an environment variable"
    if require_secret and not os.getenv(credential_env):
        return False, "Remote credential environment variable is not set"
    header = config.get("credential_header", "Token")
    if header not in _HEADERS:
        return False, "Remote credential header is not allowlisted"
    return True, None


class TrustedRemoteHttpAdapter(ModelAdapter):
    """Explicitly trusted HTTPS adapter with no hidden-layer capability."""

    def __init__(self, record, artifact_store, *, opener: RemoteOpener | None = None) -> None:
        super().__init__(record, artifact_store)
        self.opener = opener or _default_open
        self.endpoint = ""
        self.protocol = ""
        self.headers: dict[str, str] = {}
        self.timeout = 30.0
        self.max_request_bytes = 32 * 1024 * 1024
        self.max_response_bytes = 16 * 1024 * 1024

    def load(self, providers: list[str]) -> list[FeatureLayer]:
        del providers
        valid, reason = trusted_remote_configuration(self.record.config, require_secret=True)
        if not valid:
            raise ModelRuntimeError("Trusted remote model configuration is incomplete", details={"reason": reason})
        self.endpoint = str(self.record.config["remote_endpoint"]).rstrip("/")
        self.protocol = str(self.record.config["remote_protocol"])
        credential_env = str(self.record.config["credential_env"])
        credential = os.environ[credential_env]
        header = str(self.record.config.get("credential_header", "Token"))
        self.headers = {"Content-Type": "application/json"}
        self.headers[header] = f"Bearer {credential}" if header == "Authorization" else credential
        try:
            self.timeout = float(self.record.config.get("timeout", 30))
            self.max_request_bytes = int(self.record.config.get("max_request_bytes", 32 * 1024 * 1024))
            self.max_response_bytes = int(self.record.config.get("max_response_bytes", 16 * 1024 * 1024))
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError("Trusted remote timeout and response budget must be numeric") from exc
        if (
            not 0 < self.timeout <= 300
            or not 1024 <= self.max_request_bytes <= 256 * 1024 * 1024
            or not 1024 <= self.max_response_bytes <= 64 * 1024 * 1024
        ):
            raise ModelRuntimeError("Trusted remote timeout or request/response budget is outside the supported range")
        self.loaded = True
        return []

    def unload(self) -> None:
        self.endpoint = ""
        self.protocol = ""
        self.headers = {}
        self.loaded = False

    def list_layers(self) -> list[FeatureLayer]:
        return []

    def _validate_request_url(self, url: str) -> str:
        parsed = urlsplit(url)
        endpoint = urlsplit(self.endpoint)
        if (
            parsed.scheme.casefold() != "https"
            or parsed.hostname is None
            or parsed.hostname.casefold().strip(".") != endpoint.hostname.casefold().strip(".")
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ModelRuntimeError("Trusted remote request URL escaped the configured endpoint")
        return url

    def _json_request(self, url: str, *, method: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() if payload is not None else None
        if encoded is not None and len(encoded) > self.max_request_bytes:
            raise ModelRuntimeError("Trusted remote request exceeded the byte budget")
        request = Request(self._validate_request_url(url), data=encoded, headers=self.headers, method=method)
        response: RemoteResponse | None = None
        try:
            response = self.opener(request, self.timeout)
            final_url = response.geturl() or url
            self._validate_request_url(final_url)
            if final_url != url:
                raise ModelRuntimeError("Trusted remote redirects are forbidden")
            status = int(response.status)
            if status < 200 or status >= 300:
                raise ModelRuntimeError("Trusted remote returned an unsuccessful status", details={"status": status})
            chunks: list[bytes] = []
            size = 0
            while chunk := response.read(min(1024 * 1024, self.max_response_bytes + 1 - size)):
                size += len(chunk)
                if size > self.max_response_bytes:
                    raise ModelRuntimeError("Trusted remote response exceeded the byte budget")
                chunks.append(chunk)
            result = json.loads(b"".join(chunks).decode("utf-8"))
            if not isinstance(result, dict):
                raise ModelRuntimeError("Trusted remote response must be a JSON object")
            return result
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise ModelRuntimeError("Trusted remote request failed", details={"error": str(exc)}) from exc
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _image_data(image_path: Path, maximum_pixels: int, maximum_bytes: int) -> tuple[str, int, int]:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                width, height = image.size
                if width * height > maximum_pixels:
                    raise ModelRuntimeError("Trusted remote image exceeds the pixel budget")
                buffer = BytesIO()
                image.save(buffer, "PNG", compress_level=4)
        except ModelRuntimeError:
            raise
        except (OSError, ValueError) as exc:
            raise ModelRuntimeError("Could not encode image for trusted remote inference") from exc
        content = buffer.getvalue()
        encoded_size = ((len(content) + 2) // 3) * 4
        if encoded_size > maximum_bytes:
            raise ModelRuntimeError("Trusted remote encoded image exceeds the request byte budget")
        return f"data:image/png;base64,{base64.b64encode(content).decode('ascii')}", width, height

    @staticmethod
    def _annotations(shapes: object, *, width: int, height: int) -> list[AnnotationResult]:
        if not isinstance(shapes, list) or len(shapes) > 10_000:
            raise ModelRuntimeError("Trusted remote shapes must be a bounded array")
        annotations: list[AnnotationResult] = []
        for index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                raise ModelRuntimeError("Trusted remote shape must be an object", details={"index": index})
            label = shape.get("label", shape.get("category"))
            score = shape.get("score")
            shape_type = shape.get("shape_type", "rectangle")
            points = shape.get("points")
            if points is None and isinstance(shape.get("bbox"), list) and len(shape["bbox"]) == 4:
                x1, y1, x2, y2 = shape["bbox"]
                points = [[x1, y1], [x2, y2]]
            if not isinstance(label, str) or not label or shape_type not in {"rectangle", "rotation", "polygon", "point"}:
                raise ModelRuntimeError("Trusted remote shape label or type is invalid", details={"index": index})
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ModelRuntimeError("Trusted remote shape score is invalid", details={"index": index})
            if not isinstance(points, list) or not points or len(points) > 100_000:
                raise ModelRuntimeError("Trusted remote shape points are invalid", details={"index": index})
            normalized: list[list[float]] = []
            for point in points:
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in point)
                ):
                    raise ModelRuntimeError("Trusted remote shape point is invalid", details={"index": index})
                normalized.append([
                    float(np.clip(point[0], 0, width)),
                    float(np.clip(point[1], 0, height)),
                ])
            annotations.append(AnnotationResult(
                label=label,
                score=float(score),
                shape_type=shape_type,
                points=normalized,
            ))
        return annotations

    def _labelone_predict(self, image_data: str, parameters: dict[str, object]) -> object:
        model_id = self.record.config.get("remote_model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ModelRuntimeError("Trusted LabelOne remote requires remote_model_id")
        result = self._json_request(self.endpoint, method="POST", payload={
            "model": model_id,
            "image": image_data,
            "params": parameters,
        })
        data = result.get("data")
        if not isinstance(data, dict):
            raise ModelRuntimeError("Trusted LabelOne remote response has no data object")
        return data.get("shapes")

    def _grounding_predict(self, image_data: str, parameters: dict[str, object]) -> object:
        prompt = parameters.get("text_prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512 or any(ord(char) < 32 for char in prompt):
            raise ModelRuntimeError("Trusted Grounding DINO remote requires a bounded text_prompt")
        model_id = self.record.config.get("remote_model_id", "GroundingDino-1.6-Pro")
        if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", model_id):
            raise ModelRuntimeError("Trusted Grounding DINO remote_model_id is invalid")
        try:
            confidence = float(parameters.get("conf_threshold", self.record.config.get("conf_threshold", 0.25)))
            iou = float(parameters.get("iou_threshold", self.record.config.get("iou_threshold", 0.8)))
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError("Trusted Grounding DINO thresholds must be numeric") from exc
        if not np.isfinite(confidence) or not np.isfinite(iou) or not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ModelRuntimeError("Trusted Grounding DINO thresholds must be between zero and one")
        result = self._json_request(
            f"{self.endpoint}/v2/task/grounding_dino/detection",
            method="POST",
            payload={
                "model": model_id,
                "image": image_data,
                "prompt": {"type": "text", "text": prompt.rstrip(".")},
                "targets": ["bbox"],
                "bbox_threshold": confidence,
                "iou_threshold": iou,
            },
        )
        data = result.get("data")
        task_uuid = data.get("task_uuid") if isinstance(data, dict) else None
        if result.get("code") != 0 or not isinstance(task_uuid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_uuid):
            raise ModelRuntimeError("Trusted Grounding DINO remote did not return a safe task id")
        attempts = int(self.record.config.get("max_poll_attempts", 30))
        interval = float(self.record.config.get("poll_interval", 1.0))
        if not 1 <= attempts <= 300 or not 0 <= interval <= 10:
            raise ModelRuntimeError("Trusted Grounding DINO polling configuration is invalid")
        for _ in range(attempts):
            if interval:
                sleep(interval)
            status = self._json_request(
                f"{self.endpoint}/v2/task_status/{quote(task_uuid, safe='')}",
                method="GET",
            )
            status_data = status.get("data")
            if status.get("code") != 0 or not isinstance(status_data, dict):
                raise ModelRuntimeError("Trusted Grounding DINO status response is invalid")
            state = status_data.get("status")
            if state in {"waiting", "running"}:
                continue
            if state != "success":
                raise ModelRuntimeError("Trusted Grounding DINO task failed", details={"state": state})
            output = status_data.get("result")
            if not isinstance(output, dict):
                raise ModelRuntimeError("Trusted Grounding DINO result is invalid")
            objects = output.get("objects")
            if not isinstance(objects, list):
                raise ModelRuntimeError("Trusted Grounding DINO objects are invalid")
            return [
                {"category": item.get("category"), "score": item.get("score"), "bbox": item.get("bbox")}
                if isinstance(item, dict) else item
                for item in objects
            ]
        raise ModelRuntimeError("Trusted Grounding DINO task timed out")

    def predict(
        self,
        image_path: Path,
        capture_layers: list[str],
        parameters: dict[str, object],
    ) -> InferenceResult:
        if not self.loaded:
            raise ModelRuntimeError("Trusted remote model is not loaded")
        if capture_layers:
            raise ModelRuntimeError("Remote black-box models do not expose intermediate layers")
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            raise ModelRuntimeError("Inference image does not exist")
        started = perf_counter()
        maximum_pixels = int(parameters.get("max_image_pixels", self.record.config.get("max_image_pixels", 64_000_000)))
        if not 1 <= maximum_pixels <= 268_435_456:
            raise ModelRuntimeError("Trusted remote image pixel budget is invalid")
        image_data, width, height = self._image_data(image_path, maximum_pixels, self.max_request_bytes)
        preprocessed = perf_counter()
        shapes = (
            self._labelone_predict(image_data, parameters)
            if self.protocol == "labelone_v1"
            else self._grounding_predict(image_data, parameters)
        )
        inferred = perf_counter()
        annotations = self._annotations(shapes, width=width, height=height)
        finished = perf_counter()
        return InferenceResult(
            model_id=self.record.descriptor.id,
            image_path=image_path,
            annotations=annotations,
            timings_ms={
                "preprocess": (preprocessed - started) * 1000,
                "inference": (inferred - preprocessed) * 1000,
                "postprocess": (finished - inferred) * 1000,
                "total": (finished - started) * 1000,
            },
        )
