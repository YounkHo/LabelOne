from __future__ import annotations

from io import BytesIO
from pathlib import Path
from textwrap import indent
import zipfile

import numpy as np
from PIL import Image
import pytest

from labelone.pipelines.operator_packages import OperatorPackageManager
from labelone.pipelines.engine import PipelineEngine
from labelone.pipelines.registry import (
    PipelineValidationError,
    normalize_parameters,
    register_operator_contracts,
    unregister_operator_contracts,
)


def _package(
    source: str,
    *,
    operator_id: str = "acme.invert",
    extra: dict[str, bytes] | None = None,
    parameters_schema: str | None = None,
    description: str | None = "Test operator package",
    size_behavior: str = "preserve",
    spatial_behavior: str | None = "none",
    annotation_policy: str = "preserve",
    annotation_entrypoint: str | None = None,
) -> bytes:
    schema = parameters_schema or """\
type: object
properties:
  amount:
    title: Amount
    description: Controls the fixture effect strength.
    type: number
    minimum: 0
    maximum: 1
    default: 1
"""
    manifest = f"""\
api_version: labelone.operator/v1
id: {operator_id}
name: ACME operator
{f"description: {description}" if description is not None else ""}
version: 1.0.0
entrypoint: operator.py:process
size_behavior: {size_behavior}
{f"spatial_behavior: {spatial_behavior}" if spatial_behavior is not None else ""}
annotation_policy: {annotation_policy}
{f"annotation_entrypoint: {annotation_entrypoint}" if annotation_entrypoint is not None else ""}
parameters_schema:
{indent(schema.strip(), "  ")}
"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("operator.yaml", manifest)
        archive.writestr("operator.py", source)
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_operator_zip_requires_verified_annotation_spatial_contract(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators")
    with pytest.raises(PipelineValidationError, match="spatial_behavior"):
        manager.inspect_zip(_package("def process(image, parameters): return image\n", spatial_behavior=None))
    with pytest.raises(PipelineValidationError, match="must preserve"):
        manager.inspect_zip(_package(
            "def process(image, parameters): return image\n",
            size_behavior="dynamic",
            spatial_behavior="none",
            annotation_policy="preserve",
        ))
    with pytest.raises(PipelineValidationError, match="scale annotations"):
        manager.inspect_zip(_package(
            "def process(image, parameters): return image\n",
            size_behavior="dynamic",
            spatial_behavior="scale_xy",
            annotation_policy="preserve",
        ))


def test_scale_xy_operator_declares_synchronized_annotation_contract(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators")
    inspected = manager.inspect_zip(_package(
        "def process(image, parameters): return image\n",
        size_behavior="dynamic",
        spatial_behavior="scale_xy",
        annotation_policy="scale",
    ))
    assert inspected.contract.annotation_policy == {
        "mode": "scale",
        "spatial_behavior": "scale_xy",
        "synchronized": True,
    }


def test_scale_xy_operator_scales_annotation_points_with_runtime_output(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)
    installed = manager.install_zip(_package(
        "import numpy as np\n"
        "def process(image, parameters):\n"
        "    return np.repeat(image, 2, axis=1)\n",
        operator_id="acme.resize_x2",
        size_behavior="dynamic",
        spatial_behavior="scale_xy",
        annotation_policy="scale",
    ))
    engine = object.__new__(PipelineEngine)
    engine.operator_packages = manager
    document = {"shapes": [{"label": "box", "shape_type": "rectangle", "points": [[2, 3], [6, 7]]}]}
    output = engine._custom_operator(installed.contract.kind, Image.new("RGB", (10, 10), "black"), document, {})
    assert output.size == (20, 10)
    assert document["shapes"][0]["points"] == [[4.0, 3.0], [12.0, 7.0]]


def test_custom_spatial_operator_executes_annotation_entrypoint(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)
    installed = manager.install_zip(_package(
        "import numpy as np\n"
        "def process(image, parameters):\n"
        "    return np.ascontiguousarray(image[:, ::-1])\n"
        "def transform_annotations(document, parameters, context):\n"
        "    for shape in document.get('shapes', []):\n"
        "        for point in shape.get('points', []):\n"
        "            point[0] = context['input_width'] - point[0]\n"
        "    return document\n",
        operator_id="acme.flip",
        size_behavior="preserve",
        spatial_behavior="custom",
        annotation_policy="transform",
        annotation_entrypoint="operator.py:transform_annotations",
    ))
    engine = object.__new__(PipelineEngine)
    engine.operator_packages = manager
    document = {"shapes": [{"label": "line", "shape_type": "line", "points": [[2, 3], [6, 7]]}]}
    output = engine._custom_operator(installed.contract.kind, Image.new("RGB", (10, 10), "black"), document, {})
    assert output.size == (10, 10)
    assert document["shapes"][0]["points"] == [[8, 3], [4, 7]]


def test_custom_spatial_operator_requires_valid_annotation_output(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)
    with pytest.raises(PipelineValidationError, match="annotation_entrypoint"):
        manager.inspect_zip(_package(
            "def process(image, parameters): return image\n",
            spatial_behavior="custom",
            annotation_policy="transform",
        ))
    installed = manager.install_zip(_package(
        "def process(image, parameters): return image\n"
        "def transform_annotations(document, parameters, context):\n"
        "    document['shapes'][0]['points'][0][0] = context['output_width'] + 1\n"
        "    return document\n",
        operator_id="acme.bad_transform",
        spatial_behavior="custom",
        annotation_policy="transform",
        annotation_entrypoint="operator.py:transform_annotations",
    ))
    engine = object.__new__(PipelineEngine)
    engine.operator_packages = manager
    document = {"shapes": [{"label": "point", "shape_type": "point", "points": [[2, 3]]}]}
    with pytest.raises(Exception, match="out-of-bounds"):
        engine._custom_operator(installed.contract.kind, Image.new("RGB", (10, 10), "black"), document, {})


def test_operator_zip_installs_executes_and_restores_from_content_addressed_directory(tmp_path: Path) -> None:
    content = _package(
        "import numpy as np\n"
        "def process(image, amount=1.0):\n"
        "    return np.clip(image * (1 - amount) + (255 - image) * amount, 0, 255).astype(np.uint8)\n"
    )
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)

    installed = manager.install_zip(content, filename="acme-invert.zip")
    source = Image.fromarray(np.full((12, 16, 3), 20, dtype=np.uint8), mode="RGB")
    result = manager.execute(installed.contract.kind, source, {"amount": 1.0})

    assert installed.contract.kind == "acme.invert"
    assert installed.contract.input_type == installed.contract.output_type == "image"
    assert installed.contract.size_behavior == "preserve"
    assert installed.package_dir.name == installed.digest
    assert int(np.asarray(result)[0, 0, 0]) == 235

    restored = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)
    assert restored.load_installed() == []
    assert [package.contract.kind for package in restored.list()] == ["acme.invert"]
    assert np.array_equal(np.asarray(restored.execute("acme.invert", source, {"amount": 0.0})), np.asarray(source))


def test_operator_zip_inspection_validates_manifest_without_executing_package(tmp_path: Path, monkeypatch) -> None:
    manager = OperatorPackageManager(tmp_path / "operators")
    monkeypatch.setattr(manager, "_probe", lambda *_args: pytest.fail("inspection must not execute package code"))

    inspected = manager.inspect_zip(
        _package("raise RuntimeError('must not run during inspection')\n"),
        filename="acme-inspect.zip",
    )

    assert inspected.contract.kind == "acme.invert"
    assert inspected.contract.title == "ACME operator"
    assert inspected.contract.description == "Test operator package"
    assert inspected.contract.parameters_schema["properties"]["amount"]["description"]
    assert inspected.entrypoint == "operator.py:process"
    assert inspected.filename == "acme-inspect.zip"
    assert len(inspected.digest) == 64
    assert manager.list() == []


def test_operator_zip_preserves_valid_parameter_ui_hints(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators")
    inspected = manager.inspect_zip(_package(
        "def process(image, parameters): return image\n",
        parameters_schema="""\
type: object
properties:
  width:
    title: Width
    description: Exact output width in pixels.
    type: integer
    minimum: 1
    maximum: 1000000
    multipleOf: 1
    x-ui: {control: number, role: target-width, unit: px}
""",
    ))

    width = inspected.contract.parameters_schema["properties"]["width"]
    assert width["multipleOf"] == 1
    assert width["x-ui"] == {"control": "number", "role": "target-width", "unit": "px"}


def test_operator_zip_requires_operator_and_parameter_descriptions(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators")
    with pytest.raises(PipelineValidationError, match="description"):
        manager.inspect_zip(
            _package("def process(image, parameters): return image\n", description=None),
            filename="missing-description.zip",
        )
    with pytest.raises(PipelineValidationError, match="parameter description"):
        manager.inspect_zip(
            _package(
                "def process(image, parameters): return image\n",
                parameters_schema="""\
type: object
properties:
  amount: {title: Amount, type: number, default: 1}
""",
            ),
            filename="missing-parameter-description.zip",
        )


@pytest.mark.parametrize("unsafe_name", ["../escape.py", "/absolute.py", "folder\\escape.py"])
def test_operator_zip_rejects_path_traversal_and_absolute_paths(tmp_path: Path, unsafe_name: str) -> None:
    manager = OperatorPackageManager(tmp_path / "operators")
    content = _package("def process(image, parameters): return image\n", extra={unsafe_name: b"bad"})

    with pytest.raises(PipelineValidationError, match="unsafe path"):
        manager.install_zip(content)

    assert not (tmp_path / "escape.py").exists()


def test_operator_zip_rejects_non_array_output_without_crashing_service(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)
    installed = manager.install_zip(_package("def process(image, parameters): return 'not an image'\n"))

    with pytest.raises(PipelineValidationError, match="execution failed") as captured:
        manager.execute(installed.contract.kind, Image.new("RGB", (8, 8), "black"), {"amount": 1.0})

    assert "numpy.ndarray" in str(captured.value.details.get("worker_error"))


def test_operator_timeout_terminates_only_the_child_run(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=0.5)
    installed = manager.install_zip(_package(
        "import time\n"
        "def process(image, parameters):\n"
        "    time.sleep(10)\n"
        "    return image\n"
    ))

    with pytest.raises(PipelineValidationError, match="timed out"):
        manager.execute(installed.contract.kind, Image.new("RGB", (8, 8), "black"), {"amount": 1.0})


@pytest.mark.parametrize(
    "schema, message",
    [
        (
            """\
type: object
properties:
  amount: {title: Amount, description: Controls the fixture value., type: number, minimum: 0, maximum: 1, default: 2}
""",
            "maximum",
        ),
        (
            """\
type: object
properties:
  amount: {title: Amount, description: Controls the fixture value., type: number, minimum: 2, maximum: 1, default: 1}
""",
            "minimum cannot exceed maximum",
        ),
        (
            """\
type: object
properties:
  amount: {title: Amount, description: Controls the fixture value., type: integer, enum: [0, 0.5], default: 0}
""",
            "integer",
        ),
        (
            """\
type: object
properties:
  mode: {title: Mode, description: Selects the fixture mode., type: string, enum: [soft, hard], default: invalid}
""",
            "allowed value",
        ),
        (
            """\
type: object
properties:
  enabled: {title: Enabled, description: Enables the fixture behavior., type: boolean, default: 1}
""",
            "boolean",
        ),
        (
            """\
type: object
properties:
  count: {title: Count, description: Sets the fixture count., type: integer, minimum: 0.2, maximum: 0.8}
""",
            "must contain an integer",
        ),
        (
            """\
type: object
properties:
  amount: 3
""",
            "schema must be an object",
        ),
        (
            """\
type: object
required: [amount]
properties:
  amount: {title: Amount, description: Controls the fixture value., type: number, default: 0.5}
""",
            "unsupported fields",
        ),
    ],
)
def test_operator_zip_rejects_invalid_parameter_contract_before_install(
    tmp_path: Path,
    schema: str,
    message: str,
) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)

    with pytest.raises(PipelineValidationError, match=message):
        manager.install_zip(
            _package("def process(image, amount=1): return image\n", parameters_schema=schema),
            filename="invalid.zip",
        )

    assert manager.list() == []
    assert not list(manager.package_root.rglob(".labelone-package.json"))
    assert not list(manager.staging_root.iterdir())


def test_installed_operator_runtime_parameters_still_enforce_declared_range(tmp_path: Path) -> None:
    manager = OperatorPackageManager(tmp_path / "operators", execution_timeout=5.0)
    installed = manager.install_zip(
        _package(
            "def process(image, amount=1): return image\n",
            operator_id="validation.runtime",
        )
    )
    register_operator_contracts([installed.contract])
    try:
        with pytest.raises(PipelineValidationError, match="maximum"):
            normalize_parameters(installed.contract.kind, {"amount": 1.01}, node_id="runtime")
    finally:
        unregister_operator_contracts([installed.contract.kind])
