# -*- coding: utf-8 -*-
"""视频去背景核心模块。

处理流程：
  1. cv2 逐帧读取视频
  2. 每帧用 u2net 抠图（复用 bg_remover 的直推方案）
  3. 按透明度与所选背景色合成，cv2.VideoWriter 写入临时文件
  4. 用内置 ffmpeg 转码为 H.264（兼容性最好），若原视频有音轨则一并混回
取消时自动清理临时文件。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from bg_remover import _load_bg_image, _run_u2net
from chroma_key import DEFAULT_TOL, chroma_auto_color, chroma_key_rgba

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class VideoCancelled(Exception):
    """用户取消处理时抛出。"""


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str) -> dict:
    """读取视频基本信息：fps、总帧数、宽高、时长（秒）。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件：" + path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps <= 0:
        fps = 30.0
    return {
        "fps": fps,
        "frames": frames,
        "width": w,
        "height": h,
        "duration": frames / fps if frames > 0 else 0.0,
    }


def _has_audio(path: str) -> bool:
    """用 ffmpeg 探测视频是否包含音轨。"""
    try:
        proc = subprocess.run(
            [_ffmpeg_exe(), "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=_NO_WINDOW, timeout=30)
        return b"Audio:" in proc.stderr
    except Exception:
        return False


def _encode(tmp_video: str, output_path: str,
            audio_sources: tuple = ()) -> None:
    """将临时视频转码为 H.264 mp4。

    audio_sources 为可选用作音轨的媒体路径；提供多个且都含音轨时
    用 amix 混音，只取一个则直接沿用；都没有则输出无声视频。
    """
    ffmpeg = _ffmpeg_exe()
    base = [ffmpeg, "-y", "-i", tmp_video]
    srcs = [p for p in audio_sources if p and _has_audio(p)]
    for p in srcs:
        base += ["-i", p]
    if not srcs:
        base += ["-map", "0:v:0"]
    elif len(srcs) == 1:
        base += ["-map", "0:v:0", "-map", "1:a:0",
                 "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        fc = ("[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
              "[2:a]aformat=sample_rates=44100:channel_layouts=stereo[a2];"
              "[a1][a2]amix=inputs=2:duration=longest:normalize=0[aout]")
        base += ["-filter_complex", fc,
                 "-map", "0:v:0", "-map", "[aout]",
                 "-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd = base + ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  output_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        # 环境缺少 libx264 时回退为流复制（体积稍大但通用，仍保留音轨）
        cmd2 = base + ["-c:v", "copy", output_path]
        proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               creationflags=_NO_WINDOW)
        if proc2.returncode != 0:
            err = (proc2.stderr or proc.stderr or b"").decode("utf-8", "ignore")[-800:]
            raise RuntimeError("ffmpeg 编码输出失败：" + err)


def remove_video_bg(input_path: str, output_path: str,
                    model: str = "u2net",
                    bg_color: tuple = (255, 255, 255),
                    progress_cb: Optional[Callable[[int, int, str], None]] = None,
                    cancel_flag=None,
                    bg_image: Optional[str] = None,
                    bg_video: Optional[str] = None,
                    remove_method: str = "ai",
                    chroma: Optional[dict] = None) -> None:
    """移除视频背景并输出为 mp4。

    Args:
        input_path:  输入视频路径（mp4/avi/mov/mkv 等）
        output_path: 输出 mp4 路径
        model:       模型名（内置 u2net）
        bg_color:    纯色背景 RGB 元组；bg_image / bg_video 提供时忽略
        progress_cb: 进度回调 (current_frame, total_frames, filename)
        cancel_flag: 可调用对象，返回 True 则取消（抛出 VideoCancelled）
        bg_image:    可选背景图片路径，自动缩放到视频尺寸后逐帧合成
        bg_video:    可选背景视频路径。输出时长以背景视频为准（前景较短
                     则定格最后一帧，较长则截断），输出尺寸以前景为准，
                     背景逐帧缩放到该尺寸
        remove_method: 去背方式："ai" 或 "chroma"（纯色幕布）。
                       幕布方式首帧取样后全片复用，速度远快于 AI
        chroma:      幕布去背参数 {key, tol, keep_same, despill, edge}
    """
    info = probe_video(input_path)
    w, h = info["width"], info["height"]
    fg_fps = max(info["fps"], 1e-6)
    fname = os.path.basename(input_path)
    color = tuple(bg_color) if bg_color else (255, 255, 255)

    # 幕布去背参数（首帧取样后复用，保持全片幕布色一致）
    use_chroma = remove_method == "chroma"
    ch = chroma or {}
    ch_tol = int(ch.get("tol", DEFAULT_TOL))
    ch_keep = bool(ch.get("keep_same", True))
    ch_despill = str(ch.get("despill", "标准"))
    ch_edge = str(ch.get("edge", "标准"))
    ckey = ch.get("key")            # 单个颜色或多次取样的颜色列表

    # 背景来源三选一：背景视频 > 背景图片 > 纯色
    bg_cap: Optional[cv2.VideoCapture] = None
    bg_src = None
    bg_img_ary: Optional[np.ndarray] = None
    total = info["frames"]          # 输出总帧数
    fps_out = info["fps"]           # 输出帧率
    if bg_video:
        bg_cap = cv2.VideoCapture(bg_video)
        if not bg_cap.isOpened():
            raise RuntimeError("无法打开背景视频：" + bg_video)
        fps_out = bg_cap.get(cv2.CAP_PROP_FPS) or fps_out
        if fps_out <= 0:
            fps_out = info["fps"]
        total = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or total

        def _bg_frames():
            while True:
                ok, frame = bg_cap.read()
                if not ok:
                    return
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield cv2.resize(
                    rgb, (w, h), interpolation=cv2.INTER_LANCZOS4
                ).astype(np.float32)
        bg_src = _bg_frames()
    elif bg_image:
        bg_img_ary = np.asarray(
            _load_bg_image(bg_image, (w, h)), dtype=np.float32)

    tmp_dir = tempfile.mkdtemp(prefix="aibg_video_")
    tmp_video = os.path.join(tmp_dir, "nosound.mp4")
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_out, (w, h))
    if not writer.isOpened():
        if bg_cap is not None:
            bg_cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("无法创建临时视频写入器")

    cap = cv2.VideoCapture(input_path)
    idx = 0
    fg_frame = None
    fg_idx = -1          # 已读取到的前景帧号（0-based）
    try:
        while True:
            if cancel_flag is not None and callable(cancel_flag) and cancel_flag():
                cap.release()
                raise VideoCancelled()

            # 1) 取前景帧：静态背景 1:1 顺序读；背景视频按时间戳映射读，
            #    前景读完后定格最后一帧（输出时长以背景视频为准）
            if bg_cap is not None:
                target = int(idx * fg_fps / fps_out)
                while fg_idx < target:
                    ok, f = cap.read()
                    if not ok:
                        break
                    fg_frame, fg_idx = f, fg_idx + 1
                if fg_frame is None:
                    break
            else:
                ok, f = cap.read()
                if not ok:
                    break
                fg_frame, fg_idx = f, fg_idx + 1

            # 2) 前景抠图（BGR → RGB）；幕布方式首帧取样后全片复用
            rgb = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB)
            if use_chroma:
                if ckey is None:
                    ckey = chroma_auto_color(rgb)
                rgba = chroma_key_rgba(rgb, ckey, ch_tol, ch_keep,
                                       ch_despill, ch_edge).astype(np.float32)
            else:
                rgba = np.asarray(_run_u2net(Image.fromarray(rgb), model),
                                  dtype=np.float32)
            alpha = rgba[:, :, 3:4] / 255.0

            # 3) 取背景帧（RGB float32）并与前景合成
            if bg_cap is not None:
                try:
                    bg = next(bg_src)
                except StopIteration:
                    break
            elif bg_img_ary is not None:
                bg = bg_img_ary
            else:
                bg = np.empty_like(rgba[:, :, :3])
                bg[:, :] = color  # comp 为 RGB 顺序，写文件前统一转 BGR
            comp = (rgba[:, :, :3] * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            idx += 1
            if progress_cb:
                progress_cb(idx, total, fname)
        cap.release()
        writer.release()

        if idx == 0:
            raise RuntimeError("未能从视频中读取到任何画面")

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        # 前景音轨 + 背景视频音轨（若有）一起混入
        audio_srcs = [input_path]
        if bg_video:
            audio_srcs.append(bg_video)
        _encode(tmp_video, output_path, audio_srcs)
    finally:
        if bg_cap is not None:
            bg_cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def composite_image_over_video(img_path: str, bg_video: str, output_path: str,
                               model: str = "u2net",
                               progress_cb: Optional[Callable[[int, int, str], None]] = None,
                               cancel_flag=None,
                               remove_method: str = "ai",
                               chroma: Optional[dict] = None) -> None:
    """静态前景图片合成到背景视频，输出 mp4。

    输出时长/帧数/帧率以背景视频为准，输出尺寸以图片为准，
    背景逐帧缩放到图片尺寸；音轨沿用背景视频（若有）。
    remove_method: "ai" 抠图或 "chroma" 幕布去背（参数见 chroma）。
    """
    if remove_method == "chroma":
        ch = chroma or {}
        with Image.open(img_path) as im:
            rgba_in = im.convert("RGBA")
            arr = np.asarray(rgba_in.convert("RGB"))
            src_a = np.asarray(rgba_in.getchannel("A"))
        has_alpha = int(src_a.min()) < 255
        key = ch.get("key") or chroma_auto_color(arr)
        fg = chroma_key_rgba(
            arr, key, int(ch.get("tol", DEFAULT_TOL)),
            bool(ch.get("keep_same", True)),
            str(ch.get("despill", "标准")),
            str(ch.get("edge", "标准")),
            src_alpha=src_a if has_alpha else None).astype(np.float32)
    else:
        with Image.open(img_path) as im:
            fg = np.asarray(_run_u2net(im, model), dtype=np.float32)
    h, w = fg.shape[:2]
    alpha = fg[:, :, 3:4] / 255.0
    fg_rgb = fg[:, :, :3]
    fname = os.path.basename(img_path)

    cap = cv2.VideoCapture(bg_video)
    if not cap.isOpened():
        raise RuntimeError("无法打开背景视频：" + bg_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        fps = 30.0

    tmp_dir = tempfile.mkdtemp(prefix="aibg_imgvideo_")
    tmp_video = os.path.join(tmp_dir, "nosound.mp4")
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("无法创建临时视频写入器")
    try:
        idx = 0
        while True:
            if cancel_flag is not None and callable(cancel_flag) and cancel_flag():
                cap.release()
                raise VideoCancelled()
            ok, frame = cap.read()
            if not ok:
                break
            bg = cv2.cvtColor(
                cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4),
                cv2.COLOR_BGR2RGB).astype(np.float32)
            comp = (fg_rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            idx += 1
            if progress_cb:
                progress_cb(idx, total, fname)
        cap.release()
        writer.release()

        if idx == 0:
            raise RuntimeError("未能从背景视频中读取到任何画面")

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        _encode(tmp_video, output_path, [bg_video])  # 沿用背景视频音轨
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
