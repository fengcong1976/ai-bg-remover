# -*- coding: utf-8 -*-
"""AI 去背景核心模块（直推版）。

内置 u2net 模型 + onnxruntime 直连推理（与视频转图片/表情包生成器同一方案），
仅依赖 numpy + onnxruntime + Pillow，不使用 rembg 库、无需联网下载模型。
模型文件放在项目目录下的 models/ 子目录。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter

from chroma_key import chroma_from_params

# ── 模型目录：优先找 exe 同级/项目内 models，避免联网下载（约168MB）──
if getattr(sys, "frozen", False):
    # 打包后优先找 exe 同级 models/；PyInstaller 6 onedir 把 datas 放进 _internal/（sys._MEIPASS）
    _exe_models = os.path.join(os.path.dirname(sys.executable), "models")
    _bundled = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "models")
    _MODELS_DIR = _exe_models if os.path.isfile(os.path.join(_exe_models, "u2net.onnx")) else _bundled
else:
    _MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["U2NET_HOME"] = _MODELS_DIR

_session_lock = threading.Lock()
_session_cache: dict[str, object] = {}


def _get_session(model: str = "u2net"):
    """加载 onnx 模型为 onnxruntime 会话（带缓存，仅初始化一次）。"""
    if model not in _session_cache:
        with _session_lock:
            if model not in _session_cache:
                import onnxruntime
                path = os.path.join(_MODELS_DIR, model + ".onnx")
                if not os.path.isfile(path):
                    raise RuntimeError("未找到内置 AI 模型：" + path)
                _session_cache[model] = onnxruntime.InferenceSession(
                    path, providers=["CPUExecutionProvider"])
    return _session_cache[model]


def _run_u2net(im: Image.Image, model: str = "u2net") -> Image.Image:
    """u2net 直推抠图：320×320 预处理→均值方差归一→推理→边缘轻微平滑，返回 RGBA。"""
    import numpy as np
    session = _get_session(model)
    w, h = im.size
    rgb = im.convert("RGB").resize((320, 320), Image.LANCZOS)
    ary = np.asarray(rgb, dtype=np.float32)
    ary /= max(float(ary.max()), 1e-6)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((ary - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    pred = session.run(None, {session.get_inputs()[0].name: x})[0][:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = np.squeeze(pred).clip(0, 1)
    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    mask = mask.resize((w, h), Image.LANCZOS)
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))  # 轻微平滑边缘
    rgba = im.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def _load_bg_image(bg_path: str, size: tuple[int, int]) -> Image.Image:
    """加载背景图片并缩放到目标尺寸 (w, h)，返回 RGB 图。"""
    with Image.open(bg_path) as bg:
        return bg.convert("RGB").resize(size, Image.LANCZOS)


def _composite_over(rgba: Image.Image, bg_image: Image.Image) -> Image.Image:
    """把 RGBA 前景合成到同尺寸 RGB 背景图上，返回 RGB。"""
    return Image.alpha_composite(bg_image.convert("RGBA"), rgba).convert("RGB")


def remove_background(input_path: str, output_path: str,
                      model: str = "u2net",
                      progress_cb: Optional[Callable[[int, int, str], None]] = None,
                      bg_image: Optional[str] = None,
                      bg_color: Optional[tuple] = None,
                      remove_method: str = "ai",
                      chroma: Optional[dict] = None) -> bool:
    """移除单张图片的背景。

    Args:
        input_path:  输入图片路径（jpg/png/webp 等）
        output_path: 输出 PNG 路径
        model:       模型名（内置 u2net）
        progress_cb: 进度回调 (current, total, filename)
        bg_image:    可选背景图片路径；自动缩放到原图尺寸并合成
        bg_color:    可选纯色背景 RGB 元组；与 bg_image 二选一，
                     都不提供时输出透明 PNG
        remove_method: 去背方式："ai"（U²-Net 抠图）或
                       "chroma"（纯色幕布去背，参数见 chroma）
        chroma:      幕布去背参数 {key, tol, keep_same, despill, edge}；
                     key=None 时自动取画面四角取样
    Returns:
        True 成功，False 失败
    """
    try:
        with Image.open(input_path) as im:
            if remove_method == "chroma":
                rgba_in = im.convert("RGBA")
                arr = np.asarray(rgba_in.convert("RGB"))
                src_a = np.asarray(rgba_in.getchannel("A"))
                has_alpha = int(src_a.min()) < 255
                # chroma_from_params 内部已按原图透明度裁剪 alpha
                out = chroma_from_params(
                    arr, chroma, src_alpha=src_a if has_alpha else None)
                result = Image.fromarray(out)
            else:
                result = _run_u2net(im, model)
        if bg_image:
            result = _composite_over(result, _load_bg_image(bg_image, result.size))
        elif bg_color:
            result = _composite_over(
                result, Image.new("RGB", result.size, tuple(bg_color)))
        result.save(output_path)
        return True
    except Exception as e:
        print(f"[去背景失败] {input_path}: {e}")
        return False


def is_rembg_available() -> bool:
    """检查 AI 推理引擎（onnxruntime）是否可用。"""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def rembg_import_error() -> Optional[str]:
    """尝试导入推理引擎，成功返回 None，失败返回完整错误堆栈。"""
    try:
        import onnxruntime  # noqa: F401
        return None
    except Exception:
        import traceback
        return traceback.format_exc()
