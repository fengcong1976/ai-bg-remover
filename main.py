# -*- coding: utf-8 -*-
"""AI 去背景工具 v2.1 - PySide6 图形界面（单文件工作流）。

布局参考表情包制作工具：
  - 左侧大预览框：显示当前图片 / 视频首帧；幕布方式下直接在画面上点击取色
  - 右侧参数区：去背方式、幕布参数（取样/容差/范围/去残留/边缘/实时预览）、
    背景设置、输出目录
  - 两种去背方式：AI 自动抠图（U²-Net）/ 纯色幕布去背（绿幕/蓝幕/纯色底）
  - 背景模式：透明 / 指定颜色 / 背景图片 / 背景视频（时长以背景视频为准）
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSlider, QTextEdit, QVBoxLayout, QWidget,
)

from about_dialog import show_about
from bg_remover import rembg_import_error
from chroma_key import (DEFAULT_TOL, chroma_auto_color, chroma_key_rgba,
                        chroma_tol_hint)
from version import VERSION as APP_VERSION
from worker import DepCheckWorker, IMG_EXTS, ProcessWorker, VID_EXTS


def _base_dir() -> str:
    """程序所在目录（兼容 PyInstaller 打包后的运行环境）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _resource_path(name: str) -> str:
    """定位资源文件（兼容 PyInstaller 打包后的运行环境）。"""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, name)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


# 模型固定使用 u2net（已内置到 models/ 目录）
MODEL_NAME = "u2net"

FILE_FILTER = ";;".join([
    "媒体文件 (*.png *.jpg *.jpeg *.webp *.bmp *.gif "
    "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts)",
    "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp *.gif)",
    "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts)",
    "所有文件 (*.*)",
])

BG_FILTER = ";;".join([
    "背景素材 (*.png *.jpg *.jpeg *.webp *.bmp "
    "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts)",
    "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp)",
    "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts)",
])

PREVIEW_MAX = 900    # 预览图最长边（像素）


def _kind(path: str) -> str | None:
    """按后缀识别类型：图片 / 视频 / 其他(None)。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXTS:
        return "image"
    if ext in VID_EXTS:
        return "video"
    return None


def _to_qimage(arr: np.ndarray) -> QImage:
    """numpy H×W×3(RGB) / H×W×4(RGBA) → QImage（复制脱离原缓冲）。"""
    h, w = arr.shape[:2]
    if arr.shape[2] == 4:
        img = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    else:
        img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()


def _checkerboard(h: int, w: int, cell: int = 12) -> np.ndarray:
    """透明区域底纹棋盘格。"""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx // cell + yy // cell) % 2).astype(bool)
    board = np.full((h, w, 3), 255, np.uint8)
    board[mask] = (204, 204, 204)
    return board


def _load_preview_arr(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """加载图片 / 视频首帧为 RGB 数组，最长边缩到 PREVIEW_MAX。

    返回 (rgb, alpha)：图片自带透明通道时 alpha 为 H×W 数组，否则为 None
    （视频首帧没有透明通道）。
    """
    if _kind(path) == "video":
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("无法从视频中读取画面：" + path)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        alpha = None
    else:
        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            rgb = np.asarray(rgba.convert("RGB"))
            a = np.asarray(rgba.getchannel("A"))
            alpha = None if int(a.min()) == 255 else a
    h, w = rgb.shape[:2]
    scale = max(h, w) / PREVIEW_MAX
    if scale > 1.0:
        size = (int(w / scale), int(h / scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    return rgb, alpha


def _over_checker(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """把带透明通道的画面铺到棋盘格上（预览用，透明处显示棋盘格）。"""
    if alpha is None:
        return rgb
    h, w = rgb.shape[:2]
    board = _checkerboard(h, w)
    a = alpha[..., None].astype(np.float32) / 255.0
    return (rgb.astype(np.float32) * a
            + board.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def _auto_key(arr: np.ndarray, alpha: np.ndarray | None) -> tuple:
    """自动取样幕布色：四角小块的中位数，跳过全透明像素。"""
    if alpha is None:
        return chroma_auto_color(arr)
    h, w = arr.shape[:2]
    k = max(4, int(min(w, h) * 0.05))
    pix, mask = [], []
    for ys, xs in ((slice(0, k), slice(0, k)), (slice(0, k), slice(-k, None)),
                   (slice(-k, None), slice(0, k)),
                   (slice(-k, None), slice(-k, None))):
        pix.append(arr[ys, xs].reshape(-1, 3))
        mask.append((alpha[ys, xs].reshape(-1) > 32))
    pix = np.concatenate(pix)
    mask = np.concatenate(mask)
    if mask.sum() >= 16:
        return tuple(int(v) for v in np.median(pix[mask], axis=0))
    # 四角全透明（如圆角图标）：改用整圈不透明像素的中位数
    opaque = alpha > 32
    b = max(2, int(min(h, w) * 0.06))
    ring = np.zeros_like(opaque)
    ring[:b, :] = ring[-b:, :] = True
    ring[:, :b] = ring[:, -b:] = True
    sel = opaque & ring
    if sel.sum() < 16:
        sel = opaque
    if sel.sum() < 16:
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(arr[sel], axis=0))


class PreviewCanvas(QWidget):
    """预览画布：等比缩放显示 + 幕布取样点标记 + 取色。"""

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.qimg: QImage | None = None
        self._dst = None
        self.setMinimumSize(430, 380)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_view(self, qimg: QImage) -> None:
        self.qimg = qimg
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(58, 58, 58))
        if self.qimg is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(),
                       Qt.AlignmentFlag.AlignCenter,
                       "点击「添加图片/视频」\n或直接把文件拖到窗口")
            return
        pm = QPixmap.fromImage(self.qimg)
        scaled = pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._dst = scaled.rect().translated(x, y)
        p.drawPixmap(x, y, scaled)
        # 幕布取样点标记：黑边白圈
        if self.win.points:
            h, w = self.win.arr.shape[:2]
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for (ax, ay) in self.win.points:
                mx = int(self._dst.x() + ax * self._dst.width() / w)
                my = int(self._dst.y() + ay * self._dst.height() / h)
                p.setPen(QPen(QColor(0, 0, 0), 2))
                p.drawEllipse(QPoint(mx, my), 5, 5)
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawEllipse(QPoint(mx, my), 5, 5)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.win.pick_at(e.position().toPoint())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.win.pick_at(e.position().toPoint())

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.win.end_pick()


class AIPreviewWorker(QThread):
    """后台跑一次 U²-Net，用于 AI 模式的实时预览（不阻塞界面）。"""

    done = Signal(object)

    def __init__(self, rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self._rgb = rgb

    def run(self) -> None:
        try:
            from bg_remover import _run_u2net
            rgba = np.asarray(_run_u2net(Image.fromarray(self._rgb)))
            alpha = rgba[:, :, 3].copy()
        except Exception:
            alpha = None
        self.done.emit(alpha)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AI 去背景工具 v{APP_VERSION}")
        self.setMinimumSize(1000, 700)
        self.resize(1140, 780)
        self.setAcceptDrops(True)
        self.worker: ProcessWorker | None = None
        self.dep_worker: DepCheckWorker | None = None
        self.current_file: str | None = None   # 当前加载的单个文件
        self.is_video = False
        self.arr: np.ndarray | None = None     # 预览用 RGB 数组（已缩放）
        self.alpha: np.ndarray | None = None   # 原图透明度（无透明通道时为 None）
        self._base_qimg: QImage | None = None
        self.samples: list[tuple] = []         # 幕布取样颜色（近 8 次）
        self.points: list[tuple] = []          # 取样点坐标（预览图坐标）
        self.bg_mode = "transparent"       # transparent / color / image
        self.bg_color = (255, 255, 255)    # 自定义颜色 RGB
        self.bg_image: str | None = None
        self.output_dir: str | None = None
        self._output_manual = False        # 用户是否手动选过输出目录
        self._chroma_state: dict | None = None  # 幕布参数（会话内记忆）
        self._picking = False                  # 正在画面取色（此时预览显示原图）
        self._bg_cache: tuple | None = None     # 预览底板缓存 (path, (h,w), arr)
        self._ai_workers: list[AIPreviewWorker] = []
        self._ai_running = False
        self._ai_pending = False
        self._ai_alpha: np.ndarray | None = None  # AI 预览抠图结果（与 arr 同尺寸）
        self._ai_token: str | None = None         # 结果对应的文件，切换文件后丢弃
        # 预览刷新防抖定时器（先创建，参数联动在 _init_ui 中就会触发刷新）
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._render)
        self._init_ui()
        # 界面先显示，AI 引擎在后台线程加载（首次启动较慢）
        self.btn_start.setEnabled(False)
        self.lbl_status.setText("正在初始化 AI 引擎…（首次启动需加载组件，请稍候）")
        self.lbl_status.setStyleSheet("color: #c47f00;")
        QTimer.singleShot(100, self._start_dep_check)

    # ── 界面构建 ──────────────────────────────────────
    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # ── 菜单栏 ──────────────────────────────────────
        help_menu = self.menuBar().addMenu("帮助(&H)")
        act_about = help_menu.addAction("关于(&A)...")
        act_about.triggered.connect(self._show_about)

        # ── 顶部按钮行 ──────────────────────────────────
        top_row = QHBoxLayout()
        self.btn_add = QPushButton("添加图片/视频")
        self.btn_add.setToolTip("选择要处理的图片或视频（单个文件）")
        self.btn_add.clicked.connect(self._pick_file)
        self.btn_remove = QPushButton("移除当前")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._remove_current)
        top_row.addWidget(self.btn_add)
        top_row.addWidget(self.btn_remove)
        top_row.addStretch()
        layout.addLayout(top_row)

        # ── 中部：左预览 + 右参数 ───────────────────────
        body = QHBoxLayout()

        preview_group = QGroupBox("预览（幕布方式：在画面上点击或拖动取色）")
        pv_layout = QVBoxLayout(preview_group)
        self.canvas = PreviewCanvas(self)
        pv_layout.addWidget(self.canvas)
        body.addWidget(preview_group, 1)

        # 右栏：参数区放进滚动容器，窗口变矮时也不会把控件压扁/截断
        right = QWidget()
        right.setFixedWidth(372)
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_col = QVBoxLayout(inner)
        inner_col.setContentsMargins(0, 0, 4, 0)
        inner_col.setSpacing(8)

        # ① 去背设置（方式 + 幕布参数，同一块内切换）
        method_group = QGroupBox("① 去背设置")
        mg = QVBoxLayout(method_group)
        mg.setSpacing(6)

        self.cb_method = QComboBox()
        self.cb_method.addItem("AI 自动抠图（复杂背景 / 人物）", "ai")
        self.cb_method.addItem(
            "纯色幕布去背（绿幕 / 蓝幕 / 纯色底）", "chroma")
        self.cb_method.setToolTip(
            "AI 自动抠图：U²-Net 深度学习模型，适合人物与复杂背景；\n"
            "纯色幕布去背：在左侧画面点击取色，速度快")
        self.cb_method.currentIndexChanged.connect(self._on_method_changed)
        mg.addWidget(self.cb_method)

        tip = QLabel("AI 抠图适合人物与复杂背景；纯色幕布请在左侧画面上点击取色")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #777;")
        mg.addWidget(tip)

        # 幕布参数（仅纯色幕布方式显示）
        self.chroma_group = QWidget()
        cg = QVBoxLayout(self.chroma_group)
        cg.setContentsMargins(0, 4, 0, 0)
        cg.setSpacing(6)

        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        self.swatch = QLabel()
        self.swatch.setFixedSize(44, 24)
        self.swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hex = QLabel("#000000")
        self.lbl_hex.setMinimumWidth(66)
        self.btn_clear = QPushButton("清除取样")
        self.btn_clear.setToolTip("清除后自动取画面四角的中位数颜色作为幕布色")
        self.btn_clear.clicked.connect(self._clear_samples)
        key_row.addWidget(self.swatch)
        key_row.addWidget(self.lbl_hex, 1)
        key_row.addWidget(self.btn_clear)
        cg.addLayout(key_row)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        tol_row = QHBoxLayout()
        tol_row.setSpacing(6)
        self.sl_tol = QSlider(Qt.Orientation.Horizontal)
        self.sl_tol.setRange(0, 100)
        self.sl_tol.setValue(DEFAULT_TOL)
        self.sl_tol.setToolTip(
            "容差只扩张明暗/彩度范围，色相窗口固定 ±14°（拉满也不会吃到别的颜色）；\n"
            "纯色 PNG 可用 0~10，渐变 / 噪点 / JPEG 背景建议 30~50")
        self.lbl_tol = QLabel(str(DEFAULT_TOL))
        self.lbl_tol.setFixedWidth(26)
        self.lbl_tol.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sl_tol.valueChanged.connect(self._on_tol_changed)
        tol_row.addWidget(self.sl_tol, 1)
        tol_row.addWidget(self.lbl_tol)
        form.addRow("容差", tol_row)
        self.lbl_tol.setToolTip(chroma_tol_hint(DEFAULT_TOL))

        self.cb_scope = QComboBox()
        self.cb_scope.addItems(["全图同色（推荐）", "仅外部连通"])
        self.cb_scope.setToolTip(
            "全图同色：主体内部封闭的同色区域也去掉；\n"
            "仅外部连通：保留主体内部与背景同色的部分")
        self.cb_scope.currentIndexChanged.connect(lambda _: self._schedule())
        form.addRow("范围", self.cb_scope)

        self.cb_despill = QComboBox()
        self.cb_despill.addItems(["关", "轻微", "标准", "加强"])
        self.cb_despill.setCurrentIndex(2)
        self.cb_despill.setToolTip("压掉边缘残留的幕布色反光（绿边 / 蓝边）")
        self.cb_despill.currentIndexChanged.connect(lambda _: self._schedule())
        form.addRow("去残留", self.cb_despill)

        self.cb_edge = QComboBox()
        self.cb_edge.addItems(["无", "轻微", "标准", "加强"])
        self.cb_edge.setCurrentIndex(2)
        self.cb_edge.setToolTip("边缘收缩 + 羽化强度")
        self.cb_edge.currentIndexChanged.connect(lambda _: self._schedule())
        form.addRow("边缘", self.cb_edge)
        cg.addLayout(form)

        self.chk_preview = QCheckBox("实时预览（棋盘格表示透明）")
        self.chk_preview.setChecked(True)
        self.chk_preview.toggled.connect(lambda _: self._schedule())
        cg.addWidget(self.chk_preview)

        mg.addWidget(self.chroma_group)
        inner_col.addWidget(method_group)

        # ② 背景设置
        bg_group = QGroupBox("② 背景设置")
        bg_col = QVBoxLayout(bg_group)
        bg_col.setSpacing(6)

        self.rb_transparent = QRadioButton("透明背景（视频将自动改用白色）")
        self.rb_transparent.setChecked(True)
        self.rb_transparent.toggled.connect(self._on_bg_mode)
        bg_col.addWidget(self.rb_transparent)

        self.rb_color = QRadioButton("指定颜色")
        self.rb_color.toggled.connect(self._on_bg_mode)
        color_opt_row = QHBoxLayout()
        color_opt_row.setContentsMargins(22, 0, 0, 0)
        color_opt_row.setSpacing(8)
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(48, 24)
        self.btn_color.setToolTip("点击选择背景颜色")
        self.btn_color.clicked.connect(self._choose_bg_color)
        self._update_color_btn()
        color_opt_row.addWidget(self.rb_color)
        color_opt_row.addWidget(self.btn_color)
        color_opt_row.addStretch()
        bg_col.addLayout(color_opt_row)

        self.rb_image = QRadioButton("背景图片 / 视频")
        self.rb_image.setToolTip(
            "选择背景视频时：输出时长以背景视频为准，前景较短则定格最后一帧，"
            "较长则截断；输出尺寸以前景（原图 / 原视频）为准")
        self.rb_image.toggled.connect(self._on_bg_mode)
        bg_col.addWidget(self.rb_image)

        img_box = QVBoxLayout()
        img_box.setContentsMargins(22, 0, 0, 0)
        img_box.setSpacing(6)
        self.btn_bgimg = QPushButton("导入背景图片 / 视频")
        self.btn_bgimg.clicked.connect(self._pick_bg_image)
        self.lbl_bgimg = QLineEdit()
        self.lbl_bgimg.setReadOnly(True)
        self.lbl_bgimg.setPlaceholderText("（未选择）")
        img_box.addWidget(self.btn_bgimg)
        img_box.addWidget(self.lbl_bgimg)
        bg_col.addLayout(img_box)
        inner_col.addWidget(bg_group)

        # ③ 输出目录
        out_group = QGroupBox("③ 输出目录")
        out_col = QVBoxLayout(out_group)
        out_col.setSpacing(6)
        self.lbl_output = QLineEdit()
        self.lbl_output.setReadOnly(True)
        self.lbl_output.setPlaceholderText("（加载文件后自动生成）")
        self.btn_output = QPushButton("选择输出目录")
        self.btn_output.clicked.connect(self._pick_output_dir)
        out_col.addWidget(self.lbl_output)
        out_col.addWidget(self.btn_output)
        inner_col.addWidget(out_group)

        inner_col.addStretch()
        scroll.setWidget(inner)
        right_col.addWidget(scroll)

        body.addWidget(right)
        layout.addLayout(body, 1)

        # ── 进度 / 状态 / 日志 ──────────────────────────
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #555;")
        layout.addWidget(self.lbl_status)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(96)
        layout.addWidget(self.log)

        # ── 按钮 ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 15px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #aaa; }"
        )
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_open_out.clicked.connect(self._open_output)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_open_out)
        layout.addLayout(btn_row)

        # 初始状态：仅 AI 方式，隐藏幕布参数
        self._on_method_changed()

    # ── 拖拽 ──────────────────────────────────────────
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile()]
        self._load_first_available(paths)

    # ── 文件加载 ──────────────────────────────────────
    def _pick_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "选择图片/视频", "",
                                           FILE_FILTER)
        if f:
            self._load_file(f)

    def _load_first_available(self, paths: list[str]) -> None:
        """从拖入的文件/文件夹中取第一个可处理的文件。"""
        cand: list[str] = []
        for p in paths:
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp) and _kind(fp):
                        cand.append(fp)
                        break
            elif os.path.isfile(p) and _kind(p):
                cand.append(p)
        if not cand:
            self._log("没有可识别的图片/视频文件")
            return
        if len(cand) > 1:
            self._log(f"一次只处理一个文件，已忽略其余 {len(cand) - 1} 个")
        self._load_file(cand[0])

    def _load_file(self, path: str) -> None:
        try:
            arr, alpha = _load_preview_arr(path)
        except Exception as e:
            QMessageBox.warning(self, "提示", f"无法读取文件：\n{e}")
            return
        self.current_file = path
        self.is_video = _kind(path) == "video"
        self.arr = arr
        self.alpha = alpha
        self.samples.clear()
        self.points.clear()
        self._base_qimg = _to_qimage(_over_checker(arr, alpha))
        self.btn_remove.setEnabled(True)
        if not self._output_manual:
            self.output_dir = os.path.join(
                os.path.dirname(path), "output")
            self.lbl_output.setText(self.output_dir + "（自动）")
            self.lbl_output.setToolTip(self.output_dir)
        h, w = arr.shape[:2]
        self.lbl_status.setText(
            f"已加载：{os.path.basename(path)}"
            f"（{w}×{h}{'，视频显示首帧' if self.is_video else ''}）")
        self._update_swatch()
        self._schedule()
        # AI 模式：文件一加载就在后台生成抠图预览
        self._ai_alpha = None
        self._picking = False
        if self._remove_method() == "ai":
            self._start_ai_preview()

    def _remove_current(self) -> None:
        self.current_file = None
        self.arr = None
        self.alpha = None
        self.is_video = False
        self._ai_alpha = None
        self.samples.clear()
        self.points.clear()
        self._base_qimg = None
        self.btn_remove.setEnabled(False)
        self.canvas.set_view(None)  # type: ignore[arg-type]
        self.lbl_status.setText("就绪")
        if not self._output_manual:
            self.output_dir = None
            self.lbl_output.setText("")
            self.lbl_output.setToolTip("")

    # ── 幕布取色 / 预览 ───────────────────────────────
    def pick_at(self, pos: QPoint) -> None:
        """画布点击：吸取幕布色（单点取样，再次点击即替换）。"""
        if self.arr is None:
            self.lbl_status.setText("请先添加图片或视频，再在画面上取色")
            return
        if self.canvas._dst is None:
            return
        if self._remove_method() != "chroma":
            # 在 AI 模式下点击画面，视为想吸取背景色：自动切到幕布方式
            self.cb_method.setCurrentIndex(1)
        dst = self.canvas._dst
        if not dst.contains(pos):
            return
        h, w = self.arr.shape[:2]
        ax = int((pos.x() - dst.x()) * w / dst.width())
        ay = int((pos.y() - dst.y()) * h / dst.height())
        ax = min(max(ax, 0), w - 1)
        ay = min(max(ay, 0), h - 1)
        if self.alpha is not None and self.alpha[ay, ax] <= 8:
            # 点在原图透明区域：那里没有真实颜色，不写入取样
            self.lbl_status.setText("该位置是透明区域，请点击需要去掉的背景色")
            return
        y0, y1 = max(ay - 2, 0), ay + 3
        x0, x1 = max(ax - 2, 0), ax + 3
        pix = self.arr[y0:y1, x0:x1].reshape(-1, 3)
        if self.alpha is not None:
            # 只取不透明像素，避免吸到透明区域的黑色
            keep = self.alpha[y0:y1, x0:x1].reshape(-1) > 32
            if keep.any():
                pix = pix[keep]
            else:
                return
        # 取色过程中先切回原图：避免在去背后画面上取色，取到的是合成后的颜色
        self._picking = True
        if self._base_qimg is not None:
            self.canvas.set_view(self._base_qimg)
        med = np.median(pix, axis=0)
        # 单点取样：再次点击直接替换旧取样点
        self.samples = [tuple(int(v) for v in med)]
        self.points = [(ax, ay)]
        self._update_swatch()
        self._schedule()

    def end_pick(self) -> None:
        """松开鼠标：取色结束，恢复去背结果预览。"""
        if self._picking:
            self._picking = False
            self._schedule()

    def _on_tol_changed(self, v: int) -> None:
        self.lbl_tol.setText(str(v))
        self.lbl_tol.setToolTip(chroma_tol_hint(v))
        self._schedule()

    def _clear_samples(self) -> None:
        self.samples.clear()
        self.points.clear()
        self._update_swatch()
        self._schedule()

    def key_color(self) -> tuple:
        """当前幕布色（取样点颜色，未取样时自动取画面四角）。"""
        if self.arr is None:
            return (60, 176, 67)
        if not self.samples:
            return _auto_key(self.arr, self.alpha)
        return self.samples[0]

    def key_colors(self) -> list:
        """去背用的幕布色列表（单点取样，单元素）。"""
        return [self.key_color()]

    def _start_ai_preview(self) -> None:
        """AI 模式预览：后台对当前画面跑一次 U²-Net，完成后叠加显示。"""
        if self.arr is None:
            return
        if self._ai_alpha is not None and self._ai_token == self.current_file:
            self._render()
            return
        if self._ai_running:
            self._ai_pending = True   # 上一个还在算，算完自动补算当前文件
            return
        self._ai_running = True
        token = self.current_file
        self.lbl_status.setText("AI 预览生成中…（后台推理，约 1~2 秒）")
        self.lbl_status.setStyleSheet("color: #c47f00;")
        worker = AIPreviewWorker(self.arr.copy(), self)
        worker.done.connect(lambda a, t=token: self._on_ai_preview(a, t))
        worker.finished.connect(self._on_ai_worker_done)
        self._ai_workers.append(worker)
        worker.start()

    def _on_ai_preview(self, alpha, token: str | None) -> None:
        if token != self.current_file:
            return      # 已切换到别的文件，丢弃旧结果
        if alpha is None:
            self.lbl_status.setText("AI 预览生成失败（不影响正式处理）")
            self.lbl_status.setStyleSheet("color: #b03030;")
            return
        self._ai_alpha = alpha
        self._ai_token = token
        self._status_ready()
        if self._remove_method() == "ai":
            self._render()

    def _on_ai_worker_done(self) -> None:
        self._ai_running = False
        self._ai_workers = [w for w in self._ai_workers if w.isRunning()]
        if self._ai_pending:
            self._ai_pending = False
            if self._remove_method() == "ai" and self.arr is not None:
                self._start_ai_preview()

    def _status_ready(self) -> None:
        """恢复"已加载 xxx"状态文字。"""
        self.lbl_status.setStyleSheet("color: #555;")
        if not self.current_file:
            self.lbl_status.setText("就绪")
            return
        h, w = self.arr.shape[:2]
        self.lbl_status.setText(
            f"已加载：{os.path.basename(self.current_file)}"
            f"（{w}×{h}{'，视频显示首帧' if self.is_video else ''}）")

    def _update_swatch(self) -> None:
        r, g, b = self.key_color()
        self.swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            "border: 1px solid #888; border-radius: 4px;")
        self.lbl_hex.setText(f"#{r:02X}{g:02X}{b:02X}")

    def _schedule(self) -> None:
        self._timer.start()

    def _render(self) -> None:
        """按当前参数刷新预览画面（合成底板与最终输出一致）。"""
        if self.arr is None:
            return
        if self._remove_method() != "chroma":
            self._render_ai()
            return
        if self._picking:
            # 取色过程中先显示原图，取到的是画面真实颜色而不是合成后的颜色
            self.canvas.set_view(self._base_qimg)
            return
        if not self.chk_preview.isChecked():
            self.canvas.set_view(self._base_qimg)
            return
        p = self._chroma_params()
        rgba = chroma_key_rgba(self.arr, p["key"], p["tol"],
                               p["keep_same"], p["despill"], p["edge"],
                               src_alpha=self.alpha)
        h, w = rgba.shape[:2]
        board = self._preview_bg(h, w)
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        comp = (rgba[:, :, :3].astype(np.float32) * a
                + board.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        self.canvas.set_view(_to_qimage(comp))

    def _render_ai(self) -> None:
        """AI 模式预览：用后台抠图结果叠加当前背景底板。"""
        if self._ai_alpha is None:
            self.canvas.set_view(self._base_qimg)
            return
        alpha = self._ai_alpha.astype(np.float32) / 255.0
        if self.alpha is not None:
            alpha = alpha * (self.alpha.astype(np.float32) / 255.0)
        h, w = self.arr.shape[:2]
        board = self._preview_bg(h, w)
        a = alpha[:, :, None]
        comp = (self.arr.astype(np.float32) * a
                + board.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        self.canvas.set_view(_to_qimage(comp))

    def _preview_bg(self, h: int, w: int) -> np.ndarray:
        """预览底板：透明→棋盘格，指定颜色→该色，背景图/视频→首帧铺满。"""
        if self.bg_mode == "color":
            return np.broadcast_to(
                np.array(self.bg_color, np.uint8), (h, w, 3)).copy()
        if self.bg_mode == "image" and self.bg_image:
            cache = self._bg_cache
            if cache and cache[0] == self.bg_image and cache[1] == (h, w):
                return cache[2]
            try:
                if _kind(self.bg_image) == "video":
                    cap = cv2.VideoCapture(self.bg_image)
                    ok, frame = cap.read()
                    cap.release()
                    if not ok:
                        raise RuntimeError("无法读取背景视频")
                    bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    with Image.open(self.bg_image) as im:
                        bg = np.asarray(im.convert("RGB"))
                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
            except Exception:
                return _checkerboard(h, w)
            self._bg_cache = (self.bg_image, (h, w), bg)
            return bg
        return _checkerboard(h, w)

    # ── 去背方式 ──────────────────────────────────────
    def _remove_method(self) -> str:
        return self.cb_method.currentData() or "ai"

    def _on_method_changed(self) -> None:
        chroma = self._remove_method() == "chroma"
        self.chroma_group.setVisible(chroma)
        self._schedule()
        if not chroma:
            self._start_ai_preview()

    def _chroma_params(self) -> dict:
        return {"key": self.key_colors(), "tol": self.sl_tol.value(),
                # 全图同色 = 不做连通域限制（keep_same=False）；
                # 仅外部连通 = 只删与画面边缘连通的部分（keep_same=True）
                "keep_same": self.cb_scope.currentText() == "仅外部连通",
                "despill": self.cb_despill.currentText(),
                "edge": self.cb_edge.currentText()}

    # ── 背景设置 ──────────────────────────────────────
    def _on_bg_mode(self) -> None:
        if self.rb_transparent.isChecked():
            self.bg_mode = "transparent"
        elif self.rb_color.isChecked():
            self.bg_mode = "color"
        else:
            self.bg_mode = "image"
        self._schedule()

    def _choose_bg_color(self) -> None:
        c = QColor(*self.bg_color)
        picked = QColorDialog.getColor(c, self, "选择背景颜色")
        if picked.isValid():
            self.bg_color = (picked.red(), picked.green(), picked.blue())
            self._update_color_btn()
            if not self.rb_color.isChecked():
                self.rb_color.setChecked(True)
            self._schedule()

    def _update_color_btn(self) -> None:
        r, g, b = self.bg_color
        self.btn_color.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            "border: 1px solid #888; border-radius: 4px;")

    def _pick_bg_image(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "选择背景图片/视频", "",
                                           BG_FILTER)
        if f:
            self.bg_image = f
            self.lbl_bgimg.setText(f)
            self.lbl_bgimg.setToolTip(f)
            if not self.rb_image.isChecked():
                self.rb_image.setChecked(True)
            self._bg_cache = None
            self._schedule()

    # ── 输出目录 ──────────────────────────────────────
    def _pick_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.output_dir = d
            self._output_manual = True
            self.lbl_output.setText(d)
            self.lbl_output.setToolTip(d)

    # ── 引擎预加载 ────────────────────────────────────
    def _start_dep_check(self) -> None:
        self.dep_worker = DepCheckWorker()
        self.dep_worker.ok.connect(self._on_deps_ok)
        self.dep_worker.fail.connect(self._on_deps_fail)
        self.dep_worker.start()

    def _on_deps_ok(self) -> None:
        self.btn_start.setEnabled(True)
        self.lbl_status.setStyleSheet("color: #555;")
        if not self.current_file:
            self.lbl_status.setText("就绪")

    def _on_deps_fail(self, err: str) -> None:
        try:
            log_path = os.path.join(_base_dir(), "启动错误.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(err)
        except Exception:
            log_path = None
        self.lbl_status.setText("AI 引擎初始化失败")
        self.lbl_status.setStyleSheet("color: #c00000;")
        msg = "初始化 AI 引擎失败：\n\n" + err[:800]
        if log_path:
            msg += "\n\n完整错误已保存到：" + log_path
        QMessageBox.warning(self, "依赖缺失", msg)

    # ── 处理 ──────────────────────────────────────────
    def _start(self) -> None:
        if not self.current_file or self.arr is None:
            QMessageBox.warning(self, "提示", "请先添加要处理的图片或视频")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "提示", "请选择输出目录")
            return
        # 透明模式 + 视频 → 确认
        if self.bg_mode == "transparent" and self.is_video:
            ret = QMessageBox.question(
                self, "提示",
                "视频不支持透明背景，将使用白色背景处理。\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
        # 背景图片/视频校验
        if self.bg_mode == "image":
            if not self.bg_image or not os.path.isfile(self.bg_image):
                QMessageBox.warning(
                    self, "提示", "请先导入有效的背景图片或背景视频")
                return

        method = self._remove_method()
        chroma_params = self._chroma_params() if method == "chroma" else None

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setMaximum(1)
        self.log.clear()
        mode_desc = {"transparent": "透明背景", "color": "自定义颜色",
                     "image": "背景图片/视频"}[self.bg_mode]
        method_desc = {"ai": "AI 自动抠图", "chroma": "纯色幕布去背"}[method]
        self._log(f"开始处理：{os.path.basename(self.current_file)}，"
                  f"去背：{method_desc}，模式：{mode_desc}")

        self.worker = ProcessWorker(
            [self.current_file], self.output_dir, MODEL_NAME,
            bg_mode=self.bg_mode, bg_color=self.bg_color,
            bg_image=self.bg_image,
            remove_method=method, chroma=chroma_params)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._log("正在取消...")

    def _on_progress(self, current: int, total: int,
                     fname: str, detail: str) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(current - 1 if detail else current)
        if detail:
            self.lbl_status.setText(
                f"处理中 [{current}/{total}] {fname} — {detail}")
        else:
            self.lbl_status.setText(f"处理中 [{current}/{total}] {fname}")

    def _on_finished(self, result: dict) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        s, f = result["success"], result["failed"]
        self.lbl_status.setText(f"完成：成功 {s} 个，失败 {f} 个")
        self._log(f"完成：成功 {s} 个，失败 {f} 个")
        self._log(f"输出目录：{self.output_dir}")

    def _on_cancelled(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("已取消")
        self._log("已取消")

    def _on_error(self, msg: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("出错了")
        self._log(f"错误：{msg}")
        QMessageBox.critical(self, "错误", msg)

    def _open_output(self) -> None:
        if self.output_dir and os.path.isdir(self.output_dir):
            os.startfile(self.output_dir)
        else:
            QMessageBox.information(self, "提示", "输出目录不存在")

    def _log(self, msg: str) -> None:
        self.log.append(msg)

    def _show_about(self) -> None:
        show_about("AI 去背景工具", self, version=APP_VERSION)

    def closeEvent(self, event) -> None:
        for w in self._ai_workers:
            if w.isRunning():
                w.wait(20000)
        if self.dep_worker and self.dep_worker.isRunning():
            self.dep_worker.wait(10000)
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(_resource_path("图标.png")))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
