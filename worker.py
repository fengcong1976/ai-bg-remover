# -*- coding: utf-8 -*-
"""后台工作线程。

DepCheckWorker：启动时后台预加载 AI 引擎，避免阻塞界面显示。
ProcessWorker：统一处理混合文件队列（图片/视频按后缀自动识别），
逐个文件去背景，通过信号回传进度与结果。
"""
from __future__ import annotations

import os

from PySide6.QtCore import QThread, Signal

from bg_remover import remove_background, rembg_import_error
from video_remover import (
    VideoCancelled, composite_image_over_video, remove_video_bg)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts"}


class DepCheckWorker(QThread):
    """后台预加载 AI 引擎（import onnxruntime），避免阻塞界面显示。"""
    ok = Signal()
    fail = Signal(str)

    def run(self) -> None:
        err = rembg_import_error()
        if err is None:
            self.ok.emit()
        else:
            self.fail.emit(err)


class ProcessWorker(QThread):
    """统一处理队列：图片输出透明/合成 PNG，视频输出 mp4。

    bg_mode: "transparent"（透明，仅图片；视频回退白色）
             "color"（bg_color 为 RGB 元组）
             "image"（bg_image 为背景图片路径，自动缩放到原尺寸）
    """
    progress = Signal(int, int, str, str)  # 当前文件, 总数, 文件名, 帧详情
    log = Signal(str)
    finished_ok = Signal(dict)             # {success, failed, total}
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, files: list[str], output_dir: str, model: str,
                 bg_mode: str = "transparent",
                 bg_color: tuple | None = None,
                 bg_image: str | None = None,
                 remove_method: str = "ai",
                 chroma: dict | None = None):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.model = model
        self.bg_mode = bg_mode
        self.bg_color = bg_color or (255, 255, 255)
        self.bg_image = bg_image
        self.remove_method = remove_method
        self.chroma = chroma or {}
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        total = len(self.files)
        success = 0
        failed = 0
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            for idx, path in enumerate(self.files, 1):
                if self._cancel:
                    self.cancelled.emit()
                    return
                fname = os.path.basename(path)
                ext = os.path.splitext(path)[1].lower()
                try:
                    if ext in IMG_EXTS:
                        self._process_image(path, fname, idx, total)
                    elif ext in VID_EXTS:
                        self._process_video(path, fname, idx, total)
                    else:
                        failed += 1
                        self.log.emit(f"⚠️ 跳过不支持的文件：{fname}")
                        self.progress.emit(idx, total, fname, "")
                        continue
                    success += 1
                    self.progress.emit(idx, total, fname, "")
                except VideoCancelled:
                    self.cancelled.emit()
                    return
                except Exception as e:
                    failed += 1
                    self.log.emit(f"❌ {fname}：{e}")
                    self.progress.emit(idx, total, fname, "")
            self.finished_ok.emit(
                {"success": success, "failed": failed, "total": total})
        except Exception as e:
            self.error.emit(str(e))

    def _bg_is_video(self) -> bool:
        """背景素材是否为视频文件（bg_mode == "image" 时按扩展名识别）。"""
        return (self.bg_mode == "image" and self.bg_image
                and os.path.splitext(self.bg_image)[1].lower() in VID_EXTS)

    def _process_image(self, path: str, fname: str,
                       idx: int, total: int) -> None:
        base = os.path.splitext(fname)[0]
        if self._bg_is_video():
            # 图片 + 背景视频 → 输出 mp4（时长以背景视频为准）
            out = os.path.join(self.output_dir, base + "_合成.mp4")
            composite_image_over_video(
                path, self.bg_image, out, self.model,
                progress_cb=lambda c, t, f: self.progress.emit(
                    idx, total, fname, f"{c}/{t} 帧"),
                cancel_flag=lambda: self._cancel,
                remove_method=self.remove_method, chroma=self.chroma)
            return
        out = os.path.join(self.output_dir, base + ".png")
        bg_image = self.bg_image if self.bg_mode == "image" else None
        bg_color = self.bg_color if self.bg_mode == "color" else None
        if not remove_background(path, out, self.model,
                                 bg_image=bg_image, bg_color=bg_color,
                                 remove_method=self.remove_method,
                                 chroma=self.chroma):
            raise RuntimeError("处理失败")

    def _process_video(self, path: str, fname: str,
                       idx: int, total: int) -> None:
        out = os.path.join(
            self.output_dir, os.path.splitext(fname)[0] + "_去背景.mp4")
        bg_video = None
        if self.bg_mode == "image":
            color, img = None, None
            if self._bg_is_video():
                bg_video = self.bg_image
            else:
                img = self.bg_image
        elif self.bg_mode == "color":
            color, img = self.bg_color, None
        else:  # 透明 → 视频不支持透明通道，回退白色
            color, img = (255, 255, 255), None
            self.log.emit(f"ℹ️ {fname}：视频不支持透明背景，使用白色背景")
        remove_video_bg(
            path, out, self.model, color,
            progress_cb=lambda c, t, f: self.progress.emit(
                idx, total, fname, f"{c}/{t} 帧"),
            cancel_flag=lambda: self._cancel,
            bg_image=img, bg_video=bg_video,
            remove_method=self.remove_method, chroma=self.chroma)
