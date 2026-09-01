from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image

from .registry import OperatorContract, PipelineValidationError


_OPENCV_OPERATOR_DESCRIPTIONS = {
    "opencv.gaussian_blur": "使用高斯核平滑图像，降低高频噪声与细节。",
    "opencv.median_blur": "使用邻域中值抑制椒盐噪声并保留主要边缘。",
    "opencv.box_blur": "使用均值窗口快速平滑图像。",
    "opencv.bilateral_filter": "在平滑噪声的同时尽量保留边缘。",
    "opencv.threshold": "把灰度图按阈值转换为二值或截断结果。",
    "opencv.adaptive_threshold": "根据局部邻域亮度计算每个区域的二值阈值。",
    "opencv.canny": "使用 Canny 算法提取图像边缘。",
    "opencv.equalize_hist": "均衡直方图以增强整体对比度。",
    "opencv.clahe": "使用受限局部直方图均衡增强局部对比度。",
    "opencv.morphology": "使用形态学操作处理前景区域、空洞与边缘。",
    "opencv.sobel": "使用 Sobel 导数计算指定方向的梯度。",
    "opencv.scharr": "使用 Scharr 算子计算更精确的局部梯度。",
    "opencv.laplacian": "使用拉普拉斯算子提取二阶边缘变化。",
    "opencv.denoise": "使用非局部均值降低彩色或灰度图像噪声。",
    "opencv.invert": "反转每个颜色通道的像素值。",
    "opencv.normalize": "把像素值线性归一化到指定范围。",
    "opencv.gamma": "通过 Gamma 曲线调整中间调明暗。",
    "opencv.unsharp_mask": "通过反锐化掩模增强边缘与局部细节。",
    "opencv.distance_transform": "计算二值前景像素到最近背景像素的距离。",
    "opencv.fourier_transform": "将空间域图像转换为傅里叶频谱，用于观察周期结构与频率能量。",
    "opencv.haar_wavelet": "生成多尺度 Haar 小波系数图，同时观察低频轮廓与方向细节。",
}

_OPENCV_PARAMETER_METADATA: dict[str, tuple[str, str]] = {
    "kernel_size": ("核大小", "控制滤波或梯度计算使用的邻域尺寸。"),
    "sigma": ("Sigma", "控制高斯分布的标准差；为 0 时由核大小自动推导。"),
    "diameter": ("邻域直径", "控制双边滤波计算时使用的像素邻域直径。"),
    "sigma_color": ("颜色 Sigma", "控制颜色差异多大时仍参与双边滤波。"),
    "sigma_space": ("空间 Sigma", "控制空间距离多远的像素仍参与双边滤波。"),
    "threshold": ("阈值", "设置像素分割、锐化或距离变换使用的判断阈值。"),
    "max_value": ("最大值", "设置阈值处理后前景像素的输出值。"),
    "mode": ("处理模式", "选择该算子的输出模式或通道处理方式。"),
    "method": ("计算方法", "选择局部统计值的计算方法。"),
    "block_size": ("区块大小", "控制自适应阈值计算使用的局部奇数邻域。"),
    "c": ("偏移量 C", "从局部统计阈值中减去的常数。"),
    "lower": ("低阈值", "设置 Canny 边缘连接的低阈值。"),
    "upper": ("高阈值", "设置 Canny 强边缘判定的高阈值。"),
    "aperture_size": ("Sobel 孔径", "设置 Canny 内部 Sobel 导数的核大小。"),
    "l2_gradient": ("L2 梯度", "启用后使用更精确的欧氏范数计算梯度。"),
    "clip_limit": ("裁剪上限", "限制局部直方图的对比度放大量。"),
    "tile_grid_size": ("网格大小", "控制 CLAHE 将图像划分成多少局部区块。"),
    "operation": ("形态学操作", "选择腐蚀、膨胀、开闭运算或形态学梯度。"),
    "kernel_shape": ("核形状", "选择形态学结构元素的几何形状。"),
    "iterations": ("迭代次数", "设置形态学操作重复执行的次数。"),
    "dx": ("X 阶数", "设置水平方向导数的阶数。"),
    "dy": ("Y 阶数", "设置垂直方向导数的阶数。"),
    "scale": ("缩放倍率", "对计算得到的梯度值应用倍率。"),
    "delta": ("结果偏移", "在梯度结果上增加固定偏移值。"),
    "strength": ("亮度去噪强度", "控制亮度分量的非局部均值去噪强度。"),
    "color_strength": ("颜色去噪强度", "控制颜色分量的非局部均值去噪强度。"),
    "template_window_size": ("模板窗口", "设置比较像素邻域时使用的模板窗口大小。"),
    "search_window_size": ("搜索窗口", "设置寻找相似邻域时使用的搜索范围。"),
    "alpha": ("输出下限", "设置归一化后像素范围的较小端点。"),
    "beta": ("输出上限", "设置归一化后像素范围的较大端点。"),
    "gamma": ("Gamma", "小于 1 提亮中间调，大于 1 压暗中间调。"),
    "amount": ("锐化量", "控制原图与模糊图差值的增强倍率。"),
    "mask_size": ("距离掩模", "设置距离变换近似计算使用的掩模大小。"),
    "center": ("频谱中心化", "把零频分量移动到图像中心，便于观察频率分布。"),
    "levels": ("分解层数", "设置 Haar 小波的多尺度分解层数。"),
}


def _schema(properties: dict[str, dict[str, object]]) -> dict[str, object]:
    described_properties: dict[str, dict[str, object]] = {}
    for name, property_schema in properties.items():
        title, description = _OPENCV_PARAMETER_METADATA[name]
        described_properties[name] = {"title": title, "description": description, **property_schema}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "default": {},
        "properties": described_properties,
        "additionalProperties": False,
    }


def _contract(
    kind: str,
    title: str,
    properties: dict[str, dict[str, object]],
    *,
    frequency_domain: bool = False,
) -> OperatorContract:
    return OperatorContract(
        kind=kind,
        title=title,
        description=_OPENCV_OPERATOR_DESCRIPTIONS[kind],
        version="1.0.0",
        input_type="image",
        output_type="image",
        annotation_policy={"mode": "suppress" if frequency_domain else "preserve", "coordinates": "unavailable" if frequency_domain else "unchanged", "spatial_behavior": "domain_transform" if frequency_domain else "none"},
        parameters_schema=_schema(properties),
    )


OPENCV_OPERATORS: dict[str, OperatorContract] = {
    "opencv.gaussian_blur": _contract("opencv.gaussian_blur", "Gaussian blur", {
        "kernel_size": {"type": "integer", "minimum": 1, "maximum": 99, "default": 5},
        "sigma": {"type": "number", "minimum": 0.0, "maximum": 100.0, "default": 0.0},
    }),
    "opencv.median_blur": _contract("opencv.median_blur", "Median blur", {
        "kernel_size": {"type": "integer", "minimum": 3, "maximum": 99, "default": 5},
    }),
    "opencv.box_blur": _contract("opencv.box_blur", "Box blur", {
        "kernel_size": {"type": "integer", "minimum": 1, "maximum": 99, "default": 5},
    }),
    "opencv.bilateral_filter": _contract("opencv.bilateral_filter", "Bilateral filter", {
        "diameter": {"type": "integer", "minimum": 1, "maximum": 99, "default": 9},
        "sigma_color": {"type": "number", "minimum": 0.0, "maximum": 500.0, "default": 75.0},
        "sigma_space": {"type": "number", "minimum": 0.0, "maximum": 500.0, "default": 75.0},
    }),
    "opencv.threshold": _contract("opencv.threshold", "Threshold", {
        "threshold": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 127.0},
        "max_value": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 255.0},
        "mode": {"type": "string", "enum": ["binary", "binary_inv", "trunc", "tozero", "tozero_inv", "otsu"], "default": "binary"},
    }),
    "opencv.adaptive_threshold": _contract("opencv.adaptive_threshold", "Adaptive threshold", {
        "max_value": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 255.0},
        "method": {"type": "string", "enum": ["mean", "gaussian"], "default": "gaussian"},
        "mode": {"type": "string", "enum": ["binary", "binary_inv"], "default": "binary"},
        "block_size": {"type": "integer", "minimum": 3, "maximum": 255, "default": 11},
        "c": {"type": "number", "minimum": -255.0, "maximum": 255.0, "default": 2.0},
    }),
    "opencv.canny": _contract("opencv.canny", "Canny edges", {
        "lower": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 100.0},
        "upper": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 200.0},
        "aperture_size": {"type": "integer", "enum": [3, 5, 7], "default": 3},
        "l2_gradient": {"type": "boolean", "default": False},
    }),
    "opencv.equalize_hist": _contract("opencv.equalize_hist", "Histogram equalization", {
        "mode": {"type": "string", "enum": ["luminance", "per_channel"], "default": "luminance"},
    }),
    "opencv.clahe": _contract("opencv.clahe", "CLAHE", {
        "clip_limit": {"type": "number", "minimum": 0.1, "maximum": 100.0, "default": 2.0},
        "tile_grid_size": {"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
    }),
    "opencv.morphology": _contract("opencv.morphology", "Morphology", {
        "operation": {"type": "string", "enum": ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"], "default": "open"},
        "kernel_shape": {"type": "string", "enum": ["rect", "ellipse", "cross"], "default": "ellipse"},
        "kernel_size": {"type": "integer", "minimum": 1, "maximum": 99, "default": 3},
        "iterations": {"type": "integer", "minimum": 1, "maximum": 32, "default": 1},
    }),
    "opencv.sobel": _contract("opencv.sobel", "Sobel gradient", {
        "dx": {"type": "integer", "minimum": 0, "maximum": 2, "default": 1},
        "dy": {"type": "integer", "minimum": 0, "maximum": 2, "default": 0},
        "kernel_size": {"type": "integer", "enum": [1, 3, 5, 7], "default": 3},
        "scale": {"type": "number", "minimum": 0.001, "maximum": 100.0, "default": 1.0},
        "delta": {"type": "number", "minimum": -255.0, "maximum": 255.0, "default": 0.0},
    }),
    "opencv.scharr": _contract("opencv.scharr", "Scharr gradient", {
        "dx": {"type": "integer", "minimum": 0, "maximum": 1, "default": 1},
        "dy": {"type": "integer", "minimum": 0, "maximum": 1, "default": 0},
        "scale": {"type": "number", "minimum": 0.001, "maximum": 100.0, "default": 1.0},
        "delta": {"type": "number", "minimum": -255.0, "maximum": 255.0, "default": 0.0},
    }),
    "opencv.laplacian": _contract("opencv.laplacian", "Laplacian edges", {
        "kernel_size": {"type": "integer", "enum": [1, 3, 5, 7], "default": 3},
        "scale": {"type": "number", "minimum": 0.001, "maximum": 100.0, "default": 1.0},
        "delta": {"type": "number", "minimum": -255.0, "maximum": 255.0, "default": 0.0},
    }),
    "opencv.denoise": _contract("opencv.denoise", "Non-local means denoise", {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 100.0, "default": 10.0},
        "color_strength": {"type": "number", "minimum": 0.0, "maximum": 100.0, "default": 10.0},
        "template_window_size": {"type": "integer", "minimum": 3, "maximum": 21, "default": 7},
        "search_window_size": {"type": "integer", "minimum": 3, "maximum": 51, "default": 21},
    }),
    "opencv.invert": _contract("opencv.invert", "Invert", {}),
    "opencv.normalize": _contract("opencv.normalize", "Min-max normalize", {
        "alpha": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 0.0},
        "beta": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 255.0},
    }),
    "opencv.gamma": _contract("opencv.gamma", "Gamma correction", {
        "gamma": {"type": "number", "minimum": 0.05, "maximum": 10.0, "default": 1.0},
    }),
    "opencv.unsharp_mask": _contract("opencv.unsharp_mask", "Unsharp mask", {
        "sigma": {"type": "number", "minimum": 0.1, "maximum": 50.0, "default": 1.0},
        "amount": {"type": "number", "minimum": 0.0, "maximum": 5.0, "default": 1.0},
        "threshold": {"type": "integer", "minimum": 0, "maximum": 255, "default": 0},
    }),
    "opencv.distance_transform": _contract("opencv.distance_transform", "Distance transform", {
        "threshold": {"type": "number", "minimum": 0.0, "maximum": 255.0, "default": 127.0},
        "mask_size": {"type": "integer", "enum": [3, 5], "default": 5},
    }),
    "opencv.fourier_transform": _contract("opencv.fourier_transform", "傅里叶变换", {
        "mode": {"type": "string", "enum": ["magnitude", "phase"], "default": "magnitude"},
        "center": {"type": "boolean", "default": True},
    }, frequency_domain=True),
    "opencv.haar_wavelet": _contract("opencv.haar_wavelet", "Haar 小波变换", {
        "levels": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
    }, frequency_domain=True),
}


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - installation error path
        raise PipelineValidationError(
            "OpenCV operators require opencv-python-headless",
            details={"dependency": "opencv-python-headless"},
        ) from exc
    return cv2


def _odd(name: str, value: int, *, minimum: int = 1) -> int:
    if value < minimum or value % 2 == 0:
        raise PipelineValidationError(f"{name} must be an odd integer >= {minimum}")
    return value


def _gray(cv2: Any, rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _image(result: np.ndarray) -> Image.Image:
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(result), mode="L")
    if result.ndim == 3 and result.shape[2] == 3:
        return Image.fromarray(np.ascontiguousarray(result), mode="RGB")
    if result.ndim == 3 and result.shape[2] == 4:
        return Image.fromarray(np.ascontiguousarray(result), mode="RGBA")
    raise PipelineValidationError(
        "OpenCV operator returned an unsupported image shape",
        details={"shape": list(result.shape), "dtype": str(result.dtype)},
    )


def _normalize_visual(value: np.ndarray) -> np.ndarray:
    finite = np.asarray(value, dtype=np.float64)
    finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
    minimum, maximum = float(finite.min()), float(finite.max())
    if maximum <= minimum:
        return np.zeros(finite.shape, dtype=np.uint8)
    return np.clip((finite - minimum) * 255.0 / (maximum - minimum), 0, 255).astype(np.uint8)


def _haar_wavelet(gray: np.ndarray, levels: int) -> np.ndarray:
    source = np.asarray(gray, dtype=np.float32)
    source_height, source_width = source.shape
    multiple = 2 ** levels
    padded_height = ((source_height + multiple - 1) // multiple) * multiple
    padded_width = ((source_width + multiple - 1) // multiple) * multiple
    output = np.pad(source, ((0, padded_height - source_height), (0, padded_width - source_width)), mode="edge")
    height, width = output.shape
    for _ in range(levels):
        even_height, even_width = height - height % 2, width - width % 2
        if even_height < 2 or even_width < 2:
            break
        block = output[:even_height, :even_width]
        low_x = (block[:, 0::2] + block[:, 1::2]) * 0.5
        high_x = (block[:, 0::2] - block[:, 1::2]) * 0.5
        ll = (low_x[0::2] + low_x[1::2]) * 0.5
        lh = (low_x[0::2] - low_x[1::2]) * 0.5
        hl = (high_x[0::2] + high_x[1::2]) * 0.5
        hh = (high_x[0::2] - high_x[1::2]) * 0.5
        half_h, half_w = even_height // 2, even_width // 2
        output[:half_h, :half_w] = ll
        output[:half_h, half_w:even_width] = lh
        output[half_h:even_height, :half_w] = hl
        output[half_h:even_height, half_w:even_width] = hh
        height, width = half_h, half_w
    return _normalize_visual(output)[:source_height, :source_width]


def apply_opencv_operator(kind: str, image: Image.Image, parameters: Mapping[str, object]) -> Image.Image:
    """Run an allowlisted single-image OpenCV operator using an RGB boundary."""
    if kind not in OPENCV_OPERATORS:
        raise PipelineValidationError("Unknown OpenCV operator", details={"kind": kind})
    cv2 = _cv2()
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    p = parameters

    if kind == "opencv.gaussian_blur":
        kernel = _odd("kernel_size", int(p["kernel_size"]))
        result = cv2.GaussianBlur(rgb, (kernel, kernel), float(p["sigma"]))
    elif kind == "opencv.median_blur":
        result = cv2.medianBlur(rgb, _odd("kernel_size", int(p["kernel_size"]), minimum=3))
    elif kind == "opencv.box_blur":
        kernel = int(p["kernel_size"])
        result = cv2.blur(rgb, (kernel, kernel))
    elif kind == "opencv.bilateral_filter":
        result = cv2.bilateralFilter(rgb, int(p["diameter"]), float(p["sigma_color"]), float(p["sigma_space"]))
    elif kind == "opencv.threshold":
        gray = _gray(cv2, rgb)
        modes = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
            "otsu": cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        }
        _, result = cv2.threshold(gray, float(p["threshold"]), float(p["max_value"]), modes[str(p["mode"])])
    elif kind == "opencv.adaptive_threshold":
        block = _odd("block_size", int(p["block_size"]), minimum=3)
        method = cv2.ADAPTIVE_THRESH_MEAN_C if p["method"] == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        mode = cv2.THRESH_BINARY if p["mode"] == "binary" else cv2.THRESH_BINARY_INV
        result = cv2.adaptiveThreshold(_gray(cv2, rgb), float(p["max_value"]), method, mode, block, float(p["c"]))
    elif kind == "opencv.canny":
        if float(p["lower"]) > float(p["upper"]):
            raise PipelineValidationError("Canny lower threshold cannot exceed upper threshold")
        result = cv2.Canny(_gray(cv2, rgb), float(p["lower"]), float(p["upper"]), apertureSize=int(p["aperture_size"]), L2gradient=bool(p["l2_gradient"]))
    elif kind == "opencv.equalize_hist":
        if p["mode"] == "per_channel":
            result = cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(rgb)])
        else:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif kind == "opencv.clahe":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        grid = int(p["tile_grid_size"])
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=float(p["clip_limit"]), tileGridSize=(grid, grid)).apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif kind == "opencv.morphology":
        shapes = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
        operations = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }
        size = _odd("kernel_size", int(p["kernel_size"]))
        kernel = cv2.getStructuringElement(shapes[str(p["kernel_shape"])], (size, size))
        operation = str(p["operation"])
        if operation == "erode":
            result = cv2.erode(rgb, kernel, iterations=int(p["iterations"]))
        elif operation == "dilate":
            result = cv2.dilate(rgb, kernel, iterations=int(p["iterations"]))
        else:
            result = cv2.morphologyEx(rgb, operations[operation], kernel, iterations=int(p["iterations"]))
    elif kind in {"opencv.sobel", "opencv.scharr"}:
        dx, dy = int(p["dx"]), int(p["dy"])
        if dx == 0 and dy == 0:
            raise PipelineValidationError("Gradient dx and dy cannot both be zero")
        gray = _gray(cv2, rgb)
        if kind == "opencv.sobel":
            gradient = cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=int(p["kernel_size"]), scale=float(p["scale"]), delta=float(p["delta"]))
        else:
            if dx + dy != 1:
                raise PipelineValidationError("Scharr requires exactly one of dx or dy to be 1")
            gradient = cv2.Scharr(gray, cv2.CV_32F, dx, dy, scale=float(p["scale"]), delta=float(p["delta"]))
        result = cv2.convertScaleAbs(gradient)
    elif kind == "opencv.laplacian":
        gradient = cv2.Laplacian(_gray(cv2, rgb), cv2.CV_32F, ksize=int(p["kernel_size"]), scale=float(p["scale"]), delta=float(p["delta"]))
        result = cv2.convertScaleAbs(gradient)
    elif kind == "opencv.denoise":
        template = _odd("template_window_size", int(p["template_window_size"]), minimum=3)
        search = _odd("search_window_size", int(p["search_window_size"]), minimum=3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, float(p["strength"]), float(p["color_strength"]), template, search)
        result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    elif kind == "opencv.invert":
        result = cv2.bitwise_not(rgb)
    elif kind == "opencv.normalize":
        alpha, beta = float(p["alpha"]), float(p["beta"])
        if alpha >= beta:
            raise PipelineValidationError("Normalize alpha must be smaller than beta")
        result = cv2.normalize(rgb, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
    elif kind == "opencv.gamma":
        gamma = float(p["gamma"])
        table = np.array([((index / 255.0) ** gamma) * 255.0 for index in range(256)], dtype=np.uint8)
        result = cv2.LUT(rgb, table)
    elif kind == "opencv.unsharp_mask":
        sigma, amount = float(p["sigma"]), float(p["amount"])
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
        sharpened = cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0)
        threshold = int(p["threshold"])
        if threshold:
            low_contrast = np.max(np.abs(rgb.astype(np.int16) - blurred.astype(np.int16)), axis=2) < threshold
            sharpened[low_contrast] = rgb[low_contrast]
        result = sharpened
    elif kind == "opencv.distance_transform":
        _, binary = cv2.threshold(_gray(cv2, rgb), float(p["threshold"]), 255, cv2.THRESH_BINARY)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, int(p["mask_size"]))
        result = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif kind == "opencv.fourier_transform":
        gray = _gray(cv2, rgb)
        if gray.size > 16_000_000:
            raise PipelineValidationError("Fourier transform exceeds the 16-megapixel frequency budget")
        spectrum = np.fft.fft2(gray.astype(np.float32))
        if bool(p["center"]):
            spectrum = np.fft.fftshift(spectrum)
        if p["mode"] == "phase":
            result = np.clip((np.angle(spectrum) + np.pi) * 255.0 / (2 * np.pi), 0, 255).astype(np.uint8)
        else:
            result = _normalize_visual(np.log1p(np.abs(spectrum)))
    elif kind == "opencv.haar_wavelet":
        gray = _gray(cv2, rgb)
        if gray.size > 16_000_000:
            raise PipelineValidationError("Haar wavelet transform exceeds the 16-megapixel frequency budget")
        result = _haar_wavelet(gray, int(p["levels"]))
    else:  # pragma: no cover - the allowlist above is exhaustive
        raise PipelineValidationError("OpenCV operator is not implemented", details={"kind": kind})
    return _image(result)
