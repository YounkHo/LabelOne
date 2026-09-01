from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np


def _load_function(package_dir: Path, entrypoint: str):
    raw_path, separator, function_name = entrypoint.partition(":")
    if not separator or not raw_path or not function_name:
        raise ValueError("entrypoint must use relative_file.py:function_name")
    module_path = (package_dir / raw_path).resolve()
    if package_dir.resolve() not in module_path.parents or not module_path.is_file() or module_path.suffix != ".py":
        raise ValueError("entrypoint must reference a Python file inside the package")
    spec = importlib.util.spec_from_file_location(f"labelone_user_operator_{package_dir.name}", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("operator module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"entrypoint function is missing: {function_name}")
    return function


def _invoke(function, image: np.ndarray, parameters: dict[str, object]) -> np.ndarray:
    signature = inspect.signature(function)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
    ]
    has_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if len(positional) >= 2 and positional[1].name in {"params", "parameters", "options"}:
        result = function(image, parameters)
    elif has_kwargs or all(name in signature.parameters for name in parameters):
        result = function(image, **parameters)
    else:
        result = function(image, parameters)
    if not isinstance(result, np.ndarray):
        raise TypeError("operator must return numpy.ndarray")
    if result.dtype != np.uint8:
        raise TypeError("operator output dtype must be uint8")
    if result.ndim not in {2, 3} or (result.ndim == 3 and result.shape[2] not in {1, 3, 4}):
        raise ValueError("operator output shape must be HxW, HxWx1, HxWx3, or HxWx4")
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("operator output dimensions must be positive")
    if result.shape[0] * result.shape[1] > 64_000_000 or result.nbytes > 256 * 1024 * 1024:
        raise ValueError("operator output exceeds the image budget")
    return np.ascontiguousarray(result)


def _invoke_annotations(
    function,
    document: dict[str, object],
    parameters: dict[str, object],
    context: dict[str, object],
) -> dict[str, object]:
    result = function(document, parameters, context)
    if not isinstance(result, dict):
        raise TypeError("annotation entrypoint must return a document object")
    shapes = result.get("shapes", [])
    if not isinstance(shapes, list):
        raise TypeError("annotation document shapes must be a list")
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > 64 * 1024 * 1024:
        raise ValueError("annotation output exceeds the document budget")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parameters", default="{}")
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--annotation-input", type=Path)
    parser.add_argument("--annotation-output", type=Path)
    parser.add_argument("--context", default="{}")
    arguments = parser.parse_args()
    function = _load_function(arguments.package, arguments.entrypoint)
    if arguments.probe:
        return 0
    parameters = json.loads(arguments.parameters)
    if not isinstance(parameters, dict):
        raise ValueError("operator parameters must be an object")
    if arguments.annotation_only:
        if arguments.annotation_input is None or arguments.annotation_output is None:
            raise ValueError("annotation input and output are required")
        document = json.loads(arguments.annotation_input.read_text(encoding="utf-8"))
        context = json.loads(arguments.context)
        if not isinstance(document, dict) or not isinstance(context, dict):
            raise ValueError("annotation document and context must be objects")
        transformed = _invoke_annotations(function, document, parameters, context)
        arguments.annotation_output.write_text(json.dumps(transformed, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return 0
    if arguments.input is None or arguments.output is None:
        raise ValueError("input and output are required outside probe mode")
    image = np.load(arguments.input, allow_pickle=False)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("operator input must be RGB/RGBA uint8 HWC")
    result = _invoke(function, image, parameters)
    np.save(arguments.output, result, allow_pickle=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
