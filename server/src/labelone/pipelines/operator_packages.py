from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
from typing import Mapping
import zipfile

import numpy as np
from PIL import Image
import yaml

from .registry import OperatorContract, PipelineValidationError, validate_operator_contract_schema


_PACKAGE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){1,7}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
_ENTRYPOINT = re.compile(r"^[A-Za-z0-9_./-]+\.py:[A-Za-z_][A-Za-z0-9_]*$")
_PACKAGE_SCHEMA_FIELDS = {"$schema", "type", "default", "properties", "additionalProperties"}


@dataclass(frozen=True, slots=True)
class InstalledOperatorPackage:
    contract: OperatorContract
    package_dir: Path
    entrypoint: str
    digest: str
    annotation_mode: str
    annotation_entrypoint: str | None


@dataclass(frozen=True, slots=True)
class InspectedOperatorPackage:
    contract: OperatorContract
    entrypoint: str
    digest: str
    annotation_mode: str
    annotation_entrypoint: str | None
    filename: str


class OperatorPackageManager:
    """Installs trusted local operator bundles and executes them out of process.

    The child process is a crash/timeout boundary, not a complete OS security
    sandbox. Imported packages run with the current user's filesystem rights.
    """

    def __init__(
        self,
        root: Path,
        *,
        maximum_archive_bytes: int = 64 * 1024 * 1024,
        maximum_expanded_bytes: int = 256 * 1024 * 1024,
        maximum_files: int = 128,
        execution_timeout: float = 120.0,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.package_root = self.root / "packages"
        self.staging_root = self.root / "staging"
        self.runtime_root = self.root / "runtime"
        for directory in (self.package_root, self.staging_root, self.runtime_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.maximum_archive_bytes = maximum_archive_bytes
        self.maximum_expanded_bytes = maximum_expanded_bytes
        self.maximum_files = maximum_files
        self.execution_timeout = execution_timeout
        self._installed: dict[str, InstalledOperatorPackage] = {}

    @staticmethod
    def _manifest_contract(manifest: Mapping[str, object]) -> tuple[OperatorContract, str, str, str | None]:
        allowed = {
            "api_version", "id", "name", "version", "description", "entrypoint",
            "parameters_schema", "size_behavior", "spatial_behavior", "annotation_policy", "annotation_entrypoint",
        }
        extras = sorted(str(key) for key in manifest if key not in allowed)
        if extras:
            raise PipelineValidationError("Operator manifest contains unknown fields", details={"fields": extras})
        if manifest.get("api_version") != "labelone.operator/v1":
            raise PipelineValidationError("Unsupported operator api_version")
        operator_id = manifest.get("id")
        version = manifest.get("version")
        name = manifest.get("name")
        description = manifest.get("description")
        entrypoint = manifest.get("entrypoint")
        if not isinstance(operator_id, str) or not _PACKAGE_ID.fullmatch(operator_id) or operator_id.startswith(("opencv.", "labelone.")):
            raise PipelineValidationError("Custom operator id must be a namespaced lowercase id")
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            raise PipelineValidationError("Operator version must be semantic version x.y.z")
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise PipelineValidationError("Operator name is invalid")
        if not isinstance(description, str) or not description.strip() or len(description) > 500:
            raise PipelineValidationError("Operator description is invalid")
        if not isinstance(entrypoint, str) or not _ENTRYPOINT.fullmatch(entrypoint) or ".." in PurePosixPath(entrypoint.partition(":")[0]).parts:
            raise PipelineValidationError("Operator entrypoint is invalid")
        size_behavior = manifest.get("size_behavior", "dynamic")
        if size_behavior not in {"preserve", "deterministic", "dynamic"}:
            raise PipelineValidationError("Operator size_behavior is invalid")
        spatial_behavior = manifest.get("spatial_behavior")
        if spatial_behavior not in {"none", "scale_xy", "custom"}:
            raise PipelineValidationError(
                "Operator spatial_behavior must explicitly be none, scale_xy, or custom"
            )
        annotation_mode = manifest.get("annotation_policy")
        if annotation_mode not in {"preserve", "scale", "transform"}:
            raise PipelineValidationError("Operator annotation_policy is invalid")
        annotation_entrypoint = manifest.get("annotation_entrypoint")
        if annotation_entrypoint is not None and (
            not isinstance(annotation_entrypoint, str)
            or not _ENTRYPOINT.fullmatch(annotation_entrypoint)
            or ".." in PurePosixPath(annotation_entrypoint.partition(":")[0]).parts
        ):
            raise PipelineValidationError("Operator annotation_entrypoint is invalid")
        if spatial_behavior == "none" and (size_behavior != "preserve" or annotation_mode != "preserve"):
            raise PipelineValidationError(
                "Non-spatial operators must preserve both image size and annotations"
            )
        if spatial_behavior == "scale_xy" and (size_behavior == "preserve" or annotation_mode != "scale"):
            raise PipelineValidationError(
                "scale_xy operators must declare deterministic/dynamic size and scale annotations"
            )
        if spatial_behavior == "custom" and (annotation_mode != "transform" or not annotation_entrypoint):
            raise PipelineValidationError(
                "Custom spatial operators must declare annotation_policy transform and annotation_entrypoint"
            )
        if spatial_behavior != "custom" and annotation_entrypoint is not None:
            raise PipelineValidationError("annotation_entrypoint is only valid for custom spatial operators")
        raw_schema = manifest.get("parameters_schema", {})
        if not isinstance(raw_schema, Mapping):
            raise PipelineValidationError("parameters_schema must be an object")
        schema_extras = sorted(str(key) for key in raw_schema if key not in _PACKAGE_SCHEMA_FIELDS)
        if schema_extras:
            raise PipelineValidationError(
                "parameters_schema contains unsupported fields",
                details={"fields": schema_extras},
            )
        properties = raw_schema.get("properties", {})
        if not isinstance(properties, Mapping) or len(properties) > 64:
            raise PipelineValidationError("parameters_schema properties are invalid")
        if any(not isinstance(value, Mapping) for value in properties.values()):
            raise PipelineValidationError("Operator parameter schema must be an object")
        schema = {
            "$schema": raw_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
            "type": raw_schema.get("type", "object"),
            "default": raw_schema.get("default", {}),
            "properties": {str(name): dict(value) for name, value in properties.items()},
            "additionalProperties": raw_schema.get("additionalProperties", False),
        }
        contract = OperatorContract(
            kind=operator_id,
            title=name.strip(),
            description=description.strip(),
            version=version,
            input_type="image",
            output_type="image",
            annotation_policy={"mode": annotation_mode, "spatial_behavior": spatial_behavior, "synchronized": True},
            parameters_schema=schema,
            size_behavior=str(size_behavior),
            node_role="transform",
        )
        validate_operator_contract_schema(contract)
        return contract, entrypoint, str(annotation_mode), annotation_entrypoint

    def _inspect_archive(self, content: bytes) -> tuple[zipfile.ZipFile, dict[str, object]]:
        if not content or len(content) > self.maximum_archive_bytes:
            raise PipelineValidationError("Operator ZIP exceeds the archive budget")
        try:
            archive = zipfile.ZipFile(BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise PipelineValidationError("Operator package is not a valid ZIP") from exc
        entries = archive.infolist()
        if not entries or len(entries) > self.maximum_files:
            archive.close()
            raise PipelineValidationError("Operator ZIP file count is outside the package budget")
        total = 0
        names: set[str] = set()
        manifests: list[zipfile.ZipInfo] = []
        for entry in entries:
            raw_name = entry.filename
            path = PurePosixPath(raw_name)
            if not raw_name or raw_name.startswith(("/", "\\")) or "\\" in raw_name or any(part in {"", ".", ".."} for part in path.parts):
                archive.close()
                raise PipelineValidationError("Operator ZIP contains an unsafe path", details={"path": raw_name})
            normalized = path.as_posix()
            if normalized in names:
                archive.close()
                raise PipelineValidationError("Operator ZIP contains duplicate paths", details={"path": normalized})
            names.add(normalized)
            mode = (entry.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                archive.close()
                raise PipelineValidationError("Operator ZIP cannot contain symlinks", details={"path": normalized})
            total += entry.file_size
            if total > self.maximum_expanded_bytes:
                archive.close()
                raise PipelineValidationError("Operator ZIP exceeds the expanded size budget")
            if entry.compress_size == 0 and entry.file_size > 0 or entry.compress_size and entry.file_size / entry.compress_size > 1000:
                archive.close()
                raise PipelineValidationError("Operator ZIP has an unsafe compression ratio", details={"path": normalized})
            if normalized in {"operator.yaml", "operator.yml", "operator.json"}:
                manifests.append(entry)
        if len(manifests) != 1:
            archive.close()
            raise PipelineValidationError("Operator ZIP must contain exactly one root operator.yaml/operator.json")
        raw_manifest = archive.read(manifests[0])
        if len(raw_manifest) > 256 * 1024:
            archive.close()
            raise PipelineValidationError("Operator manifest is too large")
        try:
            if manifests[0].filename.endswith(".json"):
                manifest = json.loads(raw_manifest)
            else:
                manifest = yaml.safe_load(raw_manifest)
        except Exception as exc:
            archive.close()
            raise PipelineValidationError("Operator manifest cannot be parsed") from exc
        if not isinstance(manifest, dict):
            archive.close()
            raise PipelineValidationError("Operator manifest must be an object")
        return archive, manifest

    @staticmethod
    def _safe_component(value: str) -> str:
        return value.replace(".", "_").replace("-", "_")

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        allowed = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
        # The source checkout is used directly in development/test, while a
        # wheel installation resolves the same expression to site-packages.
        allowed["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        allowed["PYTHONNOUSERSITE"] = "1"
        allowed["PYTHONDONTWRITEBYTECODE"] = "1"
        return allowed

    def _probe(self, package_dir: Path, entrypoint: str) -> None:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "labelone.pipelines.operator_worker", "--package", str(package_dir), "--entrypoint", entrypoint, "--probe"],
                cwd=self.runtime_root,
                env=self._worker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                # Importing numpy/OpenCV can exceed a deliberately tiny
                # per-image execution timeout on a busy machine. Package
                # probing has its own bounded startup allowance.
                timeout=15.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PipelineValidationError("Operator entrypoint import timed out") from exc
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace")[-4000:].strip()
            raise PipelineValidationError("Operator entrypoint import failed", details={"worker_error": message})

    def inspect_zip(self, content: bytes, *, filename: str = "operator.zip") -> InspectedOperatorPackage:
        if not filename.casefold().endswith(".zip"):
            raise PipelineValidationError("Operator package filename must end in .zip")
        digest = sha256(content).hexdigest()
        archive, manifest = self._inspect_archive(content)
        try:
            contract, entrypoint, annotation_mode, annotation_entrypoint = self._manifest_contract(manifest)
        finally:
            archive.close()
        return InspectedOperatorPackage(contract, entrypoint, digest, annotation_mode, annotation_entrypoint, filename)

    def install_zip(self, content: bytes, *, filename: str = "operator.zip") -> InstalledOperatorPackage:
        if not filename.casefold().endswith(".zip"):
            raise PipelineValidationError("Operator package filename must end in .zip")
        digest = sha256(content).hexdigest()
        archive, manifest = self._inspect_archive(content)
        try:
            contract, entrypoint, annotation_mode, annotation_entrypoint = self._manifest_contract(manifest)
        except Exception:
            archive.close()
            raise

        staging = Path(mkdtemp(prefix="operator-", dir=self.staging_root))
        try:
            existing = self._installed.get(contract.kind)
            if existing is not None and existing.digest != digest:
                raise PipelineValidationError("Operator id already has an active package", details={"id": contract.kind})
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                target = staging.joinpath(*PurePosixPath(entry.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            archive.close()
            entrypoint_path = staging / entrypoint.partition(":")[0]
            if not entrypoint_path.is_file():
                raise PipelineValidationError("Operator entrypoint file is missing")
            self._probe(staging, entrypoint)
            if annotation_entrypoint:
                annotation_entrypoint_path = staging / annotation_entrypoint.partition(":")[0]
                if not annotation_entrypoint_path.is_file():
                    raise PipelineValidationError("Operator annotation entrypoint file is missing")
                self._probe(staging, annotation_entrypoint)
            installed_record = {
                "manifest": manifest,
                "digest": digest,
                "filename": filename,
            }
            (staging / ".labelone-package.json").write_text(
                json.dumps(installed_record, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            final_dir = self.package_root / self._safe_component(contract.kind) / contract.version / digest
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                shutil.rmtree(staging)
            else:
                os.replace(staging, final_dir)
            package = InstalledOperatorPackage(contract, final_dir, entrypoint, digest, annotation_mode, annotation_entrypoint)
            self._installed[contract.kind] = package
            return package
        except Exception:
            archive.close()
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def load_installed(self) -> list[str]:
        warnings: list[str] = []
        for record_path in sorted(self.package_root.glob("*/*/*/.labelone-package.json")):
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                manifest = payload["manifest"]
                digest = payload["digest"]
                if not isinstance(manifest, dict) or not isinstance(digest, str) or record_path.parent.name != digest:
                    raise PipelineValidationError("Installed operator record is invalid")
                contract, entrypoint, annotation_mode, annotation_entrypoint = self._manifest_contract(manifest)
                self._probe(record_path.parent, entrypoint)
                if annotation_entrypoint:
                    self._probe(record_path.parent, annotation_entrypoint)
                package = InstalledOperatorPackage(contract, record_path.parent, entrypoint, digest, annotation_mode, annotation_entrypoint)
                existing = self._installed.get(contract.kind)
                if existing is None or existing.contract.version < contract.version:
                    self._installed[contract.kind] = package
            except Exception as exc:
                warnings.append(f"{record_path.parent}: {exc}")
        return warnings

    def list(self) -> list[InstalledOperatorPackage]:
        return sorted(self._installed.values(), key=lambda package: package.contract.kind)

    def get(self, kind: str) -> InstalledOperatorPackage:
        try:
            return self._installed[kind]
        except KeyError as exc:
            raise PipelineValidationError("Custom operator package is not installed", details={"kind": kind}) from exc

    def has(self, kind: str) -> bool:
        return kind in self._installed

    def execute(self, kind: str, image: Image.Image, parameters: Mapping[str, object]) -> Image.Image:
        package = self.get(kind)
        rgba = image.mode == "RGBA"
        array = np.asarray(image.convert("RGBA" if rgba else "RGB"), dtype=np.uint8)
        with TemporaryDirectory(prefix="operator-run-", dir=self.runtime_root) as temporary:
            runtime = Path(temporary)
            input_path = runtime / "input.npy"
            output_path = runtime / "output.npy"
            np.save(input_path, array, allow_pickle=False)
            try:
                completed = subprocess.run(
                    [
                        sys.executable, "-m", "labelone.pipelines.operator_worker",
                        "--package", str(package.package_dir),
                        "--entrypoint", package.entrypoint,
                        "--input", str(input_path),
                        "--output", str(output_path),
                        "--parameters", json.dumps(dict(parameters), ensure_ascii=False, separators=(",", ":")),
                    ],
                    cwd=runtime,
                    env=self._worker_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=self.execution_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise PipelineValidationError(
                    "Custom operator execution timed out",
                    details={"kind": kind, "timeout_seconds": self.execution_timeout},
                ) from exc
            if completed.returncode != 0 or not output_path.is_file():
                message = completed.stderr.decode("utf-8", errors="replace")[-4000:].strip()
                raise PipelineValidationError("Custom operator execution failed", details={"kind": kind, "worker_error": message})
            result = np.load(output_path, allow_pickle=False)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        mode = "L" if result.ndim == 2 else "RGBA" if result.shape[2] == 4 else "RGB"
        return Image.fromarray(np.ascontiguousarray(result), mode=mode)

    def transform_annotations(
        self,
        kind: str,
        document: Mapping[str, object],
        parameters: Mapping[str, object],
        *,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> dict[str, object]:
        package = self.get(kind)
        if package.annotation_mode != "transform" or not package.annotation_entrypoint:
            raise PipelineValidationError(
                "Custom spatial operator has no annotation transform entrypoint",
                details={"kind": kind},
            )
        with TemporaryDirectory(prefix="operator-annotation-", dir=self.runtime_root) as temporary:
            runtime = Path(temporary)
            input_path = runtime / "annotation-input.json"
            output_path = runtime / "annotation-output.json"
            input_path.write_text(json.dumps(dict(document), ensure_ascii=False), encoding="utf-8")
            context = {
                "input_width": input_size[0],
                "input_height": input_size[1],
                "output_width": output_size[0],
                "output_height": output_size[1],
            }
            try:
                completed = subprocess.run(
                    [
                        sys.executable, "-m", "labelone.pipelines.operator_worker",
                        "--package", str(package.package_dir),
                        "--entrypoint", package.annotation_entrypoint,
                        "--annotation-only",
                        "--annotation-input", str(input_path),
                        "--annotation-output", str(output_path),
                        "--parameters", json.dumps(dict(parameters), ensure_ascii=False, separators=(",", ":")),
                        "--context", json.dumps(context, separators=(",", ":")),
                    ],
                    cwd=runtime,
                    env=self._worker_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=self.execution_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise PipelineValidationError(
                    "Custom annotation transform timed out",
                    details={"kind": kind, "timeout_seconds": self.execution_timeout},
                ) from exc
            if completed.returncode != 0 or not output_path.is_file():
                message = completed.stderr.decode("utf-8", errors="replace")[-4000:].strip()
                raise PipelineValidationError(
                    "Custom annotation transform failed",
                    details={"kind": kind, "worker_error": message},
                )
            try:
                transformed = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise PipelineValidationError("Custom annotation transform returned invalid JSON", details={"kind": kind}) from exc
        if not isinstance(transformed, dict):
            raise PipelineValidationError("Custom annotation transform must return an object", details={"kind": kind})
        return transformed
