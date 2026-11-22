# -*- coding: utf-8 -*-
"""纯色幕布去背模块（与表情包生成器 v2.4 的算法保持同步）。

适用于绿幕 / 蓝幕 / 纯白底等背景颜色统一的素材：
  - 自动取样：取画面四角小块的中位数颜色（抗噪点、抗角落字幕）
  - 手动指定：直接给定 RGB 幕布色
  - 可调参数：容差 0~100、范围（全图同色 / 仅外部连通）、
    去残留（关/轻微/标准/加强）、边缘（无/轻微/标准/加强）
  - 判定结构：**色相是硬墙（固定 ±14°，不随容差变化）**，
    容差只扩张同一幕布的明度 / 彩度跨度
纯 numpy + opencv 实现，不走 AI 推理，速度远快于 AI 自动抠图。
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# 边缘档位 → (收缩像素, 羽化 sigma)；收缩支持 0.5px 级（2 倍分辨率腐蚀后缩回）
EDGE_CFG = {"无": (0.0, 0.0), "轻微": (0.5, 0.5),
            "标准": (1.0, 0.8), "加强": (1.5, 1.2)}
# 残留档位 → (k=允许主色通道超出量, band=作用边界像素宽)
DESPILL_CFG = {"关": (0.0, 0.0), "轻微": (8.0, 5.0),
               "标准": (4.0, 11.0), "加强": (2.0, 13.0)}
# 浅色/白幕布亮度补偿强度
DESPILL_TINT = {"关": 0.0, "轻微": 0.04, "标准": 0.08, "加强": 0.14}

# 判定结构：色相硬墙固定 ±14°（12°~14° 的 2° 过渡只为吸收 Lab 量化误差），
# 容差只控制明度跨度 ΔL（2→32）与彩度跨度 ΔC（2→16）：
#   10 ≈ 明度 ±10 / 彩度 ±6   35 ≈ ±19 / ±10   50 ≈ ±23 / ±12   100 ≈ ±32 / ±16
HUE_WALL = 14.0
HUE_FADE = 2.0
L_SPAN = 30.0
C_SPAN = 14.0
SPAN_POW = 0.55
# 容差默认值（旧版默认 20 只认窄色差，渐变底去不净）
DEFAULT_TOL = 35
_CH_SUM = np.ones((1, 3), np.float32)  # cv2.transform 用：3 通道求和成 1 通道


def chroma_auto_color(arr: np.ndarray) -> tuple:
    """自动取样：取画面四角小块的中位数颜色（抗噪点、抗角落字幕）。"""
    h, w = arr.shape[:2]
    k = max(4, int(min(w, h) * 0.05))
    patches = [arr[:k, :k].reshape(-1, 3), arr[:k, -k:].reshape(-1, 3),
               arr[-k:, :k].reshape(-1, 3), arr[-k:, -k:].reshape(-1, 3)]
    med = np.median(np.concatenate(patches, axis=0), axis=0)
    return tuple(int(v) for v in med)


def _chroma_kind(key) -> str:
    """按取样颜色判断幕布类型：绿幕 / 蓝幕 / 浅色底（白、灰、青等）。

    只看色度上的「主通道优势」：绿比另外两个通道高多少（G - max(R,B)）、
    蓝比另外两个通道高多少（B - max(R,G)）。优势都不明显时（青、灰、浅黄）
    一律归为浅色底走通用分支，不能默认按蓝幕处理，否则青色/浅蓝主体会被压掉。
    """
    r, g, b = [float(v) for v in key]
    gm = g - max(r, b)
    bm = b - max(r, g)
    if gm >= 25 and gm >= bm:
        return "green"
    if bm >= 25:
        return "blue"
    return "plain"


def chroma_tol_span(tol: float) -> tuple[float, float]:
    """容差滑杆值（0~100）-> (明度跨度 ΔL, 彩度跨度 ΔC)。

    容差只扩张「同一块幕布的明暗/浓淡不均匀」：0 档只认几乎一模一样的颜色
    （ΔL=ΔC=2），拉满时明度跨度约 32、彩度跨度约 16。
    色相窗口是固定的硬墙（见 HUE_WALL），不随容差变化。
    """
    t = max(0.0, min(100.0, float(tol))) / 100.0
    f = t ** SPAN_POW
    return L_SPAN * f + 2.0, C_SPAN * f + 2.0


def chroma_tol_hint(tol: float) -> str:
    """UI 提示：容差当前覆盖的明度/彩度跨度（色相窗口固定不变）。"""
    dl, dc = chroma_tol_span(tol)
    return "明度 ±%.0f、彩度 ±%.0f（色相固定 ±%.0f°）" % (dl, dc, HUE_WALL)


def _smoothstep(e0, e1, x):
    """smoothstep（原地运算，减少大图上的临时数组分配）。"""
    t = np.subtract(x, e0, dtype=np.float32)
    t *= 1.0 / max(e1 - e0, 1e-6)
    np.clip(t, 0.0, 1.0, out=t)
    u = t * -2.0
    u += 3.0
    t *= t
    t *= u
    return t


def _shrink_alpha(alpha, px):
    """alpha 收缩：支持 0.5px 级（先 2 倍放大腐蚀，再缩回，避免只能整像素）。"""
    if px <= 0:
        return alpha
    h, w = alpha.shape[:2]
    up = cv2.resize(alpha, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    k = 1 + 2 * int(round(px * 2))   # 0.5px -> 3x3；1.0px -> 5x5
    up = cv2.erode(up, np.ones((k, k), np.uint8))
    return cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)


def _chroma_bg_amount(arr: np.ndarray, key, tol: float) -> np.ndarray:
    """每个像素含多少「幕布成分」：1=纯幕布，0=纯前景（整幅只转一次 LAB）。

    判定结构：**色相是硬墙**（固定窗口，容差再大也不越过），
    **容差只扩张明度 L 与彩度 C 的范围**——对应同一块幕布的明暗不均、
    渐变、阴影与反光。低饱和幕布（白/灰/米白）色相无意义，只按彩度+明度判。
    抗锯齿边缘的色相落在幕布与主体之间，这里会判成前景，
    由 _chroma_mixed_alpha() 用混合线投影找回平滑 alpha。
    """
    lab = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.float32)
    L = lab[..., 0]
    L *= (100.0 / 255.0)
    a = lab[..., 1]
    a -= 128.0
    b = lab[..., 2]
    b -= 128.0
    kl = cv2.cvtColor(np.array([[list(key)]], dtype=np.uint8),
                      cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    Lk = float(kl[0]) * 100.0 / 255.0
    ak, bk = float(kl[1]) - 128.0, float(kl[2]) - 128.0
    Ck = math.hypot(ak, bk)
    hk = math.degrees(math.atan2(bk, ak))

    C, H = cv2.cartToPolar(a, b, angleInDegrees=True)
    Lmax, Cmax = chroma_tol_span(tol)
    dL = np.subtract(L, Lk)
    # 阴影通常比幕布更暗：暗侧给 1.25 倍余量，亮侧按标称跨度
    dl_n = np.multiply(dL, 1.0 / Lmax)
    np.maximum(dl_n, np.multiply(dL, -1.0 / (Lmax * 1.25)), out=dl_n)
    dc_n = np.abs(np.subtract(C, Ck))
    dc_n *= 1.0 / Cmax
    m = np.maximum(dl_n, dc_n, out=dl_n)
    amount = 1.0 - _smoothstep(0.82, 1.18, m)
    if Ck >= 12.0:  # 有明确色相才设色相墙；近中性色（白/灰）只看彩度+明度
        dh = np.abs(np.subtract(H, hk))
        np.minimum(dh, np.subtract(360.0, dh), out=dh)
        # 12°~14° 之间快速收到 0：14° 外无论容差多大都判前景（色相硬墙）
        wall = 1.0 - _smoothstep(HUE_WALL - HUE_FADE, HUE_WALL, dh)
        # 彩度极低时色相不可靠（几乎是灰色），不参与色相墙判断
        np.copyto(wall, 1.0, where=C < 6.0)
        amount *= wall
    np.clip(amount, 0.0, 1.0, out=amount)
    return amount.astype(np.float32)


def _chroma_mixed_alpha(arr: np.ndarray, key, amount: np.ndarray) -> np.ndarray:
    """把「幕布成分」细化成平滑 alpha（0~255，255=保留主体）。

    抗锯齿边缘是「幕布色 + 主体色」的线性混合，色相必然落在两者之间，
    只靠色相墙会把整条边缘判成前景、形成硬边。这里取附近「确定是前景」的
    像素平均色作为本地前景色 F，把像素投影到幕布色 K → F 的连线上：
        t = (P-F)·(K-F) / |K-F|²   （P 里幕布成分的比例）
    纯幕布得 t=1（全透明），纯前景得 t=0（全保留），边缘混合色得到中间值。
    邻域里没有可靠前景色（幕布内部）时退回粗判定 amount。
    """
    f = arr.astype(np.float32)
    fg = (amount < 0.02).astype(np.float32)
    fg3 = cv2.merge([fg, fg, fg])
    num = cv2.boxFilter(cv2.multiply(f, fg3), -1, (13, 13), normalize=False)
    den = cv2.boxFilter(fg, -1, (13, 13), normalize=False)
    F = cv2.divide(num, cv2.merge([den, den, den]) + 1e-6)
    K = np.array(key, dtype=np.float32)
    V = cv2.subtract(F, K)
    # 通道求和用 cv2.transform（1×3 全 1 矩阵），比 numpy 的 sum(axis=2) 快数倍
    vv = cv2.transform(cv2.multiply(V, V), _CH_SUM)
    ok = (den > 2.5) & (vv > 900.0)   # 附近有前景，且前景与幕布色差够大
    dot = np.einsum("ijk,ijk->ij", cv2.subtract(F, f), V)
    t = np.clip(dot / np.maximum(vv, 1e-6), 0.0, 1.0)
    t = np.where(ok, t, amount)
    return np.clip((1.0 - t) * 255.0, 0.0, 255.0).astype(np.uint8)


def _despill_all(out, alpha, kind, key, level):
    """全画幅去溢色：只在「贴着主体边界的溢色带」内压超量通道，主体中心本色不动。

    作用范围：alpha 半透明的像素（必然混了幕布色）权重=1；alpha 不透明的像素
    只压离轮廓 band 像素以内的部分，越往里权重越低，所以绿蛇/绿衣服等本体中心
    不会被压灰。绿/蓝幕只削减「超出另一通道 + k」的部分，不改变色相方向。
    """
    k, band = DESPILL_CFG.get(level, (0.0, 0.0))
    if k <= 0:
        return out
    tint = DESPILL_TINT.get(level, 0.0)
    f = out.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    a = alpha.astype(np.float32) / 255.0
    core = (a >= 0.92).astype(np.uint8)
    d_in = (cv2.distanceTransform(core, cv2.DIST_L2, 5) if core.any()
            else np.zeros(a.shape, np.float32))
    w_edge = 1.0 - _smoothstep(1.0, max(band, 2.0), d_in)
    spill = np.where(a < 0.92, 1.0, w_edge)
    spill = np.where(alpha > 0, spill, 0.0)
    if kind == "green":
        # 上限取两者较小值：均衡色用 max(r,b)+k；R≈0 的典型绿溢色用 (r+b)/2+k/2，
        # 否则 max(r,b) 仍偏高，白底上会看出淡绿
        lim = np.minimum(np.maximum(r, b) + k, (r + b) * 0.5 + k * 0.5)
        # 只处理 G 本来就压过 R、B 的像素：G 低于别的通道说明它不是绿溢色
        # （蓝色/青色主体），绝不能被抬到 lim，否则蓝色会被压灰压淡
        over = np.maximum(g - lim, 0.0) * spill * (g > np.maximum(r, b))
        g = g - over
    elif kind == "blue":
        lim = np.minimum(np.maximum(r, g) + k, (r + g) * 0.5 + k * 0.5)
        over = np.maximum(b - lim, 0.0) * spill * (b > np.maximum(r, g))
        b = b - over
    else:
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        kl = 0.299 * key[0] + 0.587 * key[1] + 0.114 * key[2]
        over = np.maximum(luma - kl, 0.0) * spill * tint
        r, g, b = r - over, g - over, b - over
    return np.dstack([np.clip(r, 0, 255), np.clip(g, 0, 255),
                      np.clip(b, 0, 255)]).astype(np.uint8)


def _first_key(key) -> tuple:
    """把幕布色统一成单个 (r, g, b)：兼容单色与旧的取样点列表（取第一个）。"""
    if key is None:
        return (0, 0, 0)
    k = list(key)
    if not k:
        return (0, 0, 0)
    if isinstance(k[0], (int, float, np.integer, np.floating)):
        return tuple(int(v) for v in k[:3])
    return tuple(int(v) for v in list(k[0])[:3])


def chroma_key_rgba(arr: np.ndarray, key: Optional[tuple | list],
                    tol: float = DEFAULT_TOL, keep_same: bool = True,
                    despill: str = "标准", edge: str = "标准",
                    src_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """纯色幕布去背（绿幕 / 蓝幕 / 纯白底），返回 H×W×4 RGBA。

    arr: H×W×3 uint8 RGB 画面
    key: 幕布颜色 (r, g, b)；None 时自动取画面四角中位数
    tol: 容差 0~100；只扩张明度/彩度范围（ΔL 2→32、ΔC 2→16），
         色相窗口固定 ±14°，换算见 chroma_tol_span()；越大移除越多
    keep_same: True=仅删与画面边缘连通的背景（主体内部同色块保留）；
               False=全图同色删除，主体内部封闭区域的背景色也一并删掉（推荐）
    despill: 去背景色残留档位（关/轻微/标准/加强），兼容旧的 True/False
    edge: 边缘档位（无/轻微/标准/加强），0.5px 收缩 + 羽化
    src_alpha: 原图透明度（H×W uint8）；透明区域按「画面外」参与连通判定，
               并在结果中保持透明

    核心：色相是硬墙，容差只在墙内扩张明度/彩度——取一个幕布色后，容差再大
    也只认「这一片色相的明暗变化」，不会把青蓝、别的颜色一起算成背景。
    再从画面边框向内做连通域。幕布类型只用来决定去溢色压哪个通道。
    keep_same=True 时只有与边框连通的候选才算背景，主体内部同色块（绿衣服）
    不连通，因此不会被吃掉；keep_same=False 时所有颜色命中的像素都按软 alpha
    处理，封闭区域也会一起变透明。
    抗锯齿边缘用「幕布色 → 本地前景色」的混合线投影估出幕布成分比例，
    因此不会被色相墙切成硬边；最后半像素收缩 + 羽化。
    """
    h, w = arr.shape[:2]
    key = _first_key(key if key is not None else chroma_auto_color(arr))
    if isinstance(despill, bool):
        despill = "标准" if despill else "关"
    kind = _chroma_kind(key)

    # 每个像素的「幕布成分」：色相硬墙 + 容差控制的明度/彩度跨度
    amount = _chroma_bg_amount(arr, key, tol)

    # 1) 颜色上像幕布的候选（把过渡带也纳入，交给软 alpha 处理）
    near = amount > 0.06
    if src_alpha is not None:
        # 原图透明区域等于「画面外」：一并纳入候选，让背景能连通到边框
        # （否则圆角图标这类四周透明、背景不贴边的素材会一个像素都去不掉）
        near = near | (src_alpha <= 8)
    # 2) 从边框向内连通：只有和边框连通的候选才是背景，主体内部同色区域保留
    n, labels = cv2.connectedComponents(near.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :],
                                       labels[:, 0], labels[:, -1]]))
    border = border[border != 0]
    bg_conn = np.isin(labels, border) if border.size else np.zeros_like(near)

    # 3) 软 alpha：混合线投影，边缘得到 0~255 的平滑过渡
    alpha = _chroma_mixed_alpha(arr, key, amount)
    # 投影只在「疑似背景 + 2px」范围内生效：主体内部（尤其全图同色模式下，
    # 主体色也可能落在幕布色→局部前景色的连线上）必须保持不透明，
    # 否则会被投影误算成半透明而掉色。
    scope = bg_conn if keep_same else near
    region = cv2.dilate(scope.astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    alpha = np.where(region, alpha, 255).astype(np.uint8)

    # 4) 全画幅去溢色（在收缩前做，边缘反光能被一起压掉）
    out = _despill_all(arr.copy(), alpha, kind, key, despill)

    # 5) 半像素收缩 + 羽化：把淡边整体推进透明区
    shrink, sig = EDGE_CFG.get(edge, (1.0, 0.8))
    alpha = _shrink_alpha(alpha, shrink)
    if sig:
        ks = 3 if sig <= 0.9 else 5
        alpha = cv2.GaussianBlur(alpha, (ks, ks), sigmaX=sig)
        # 羽化尾巴归零：alpha<6 只有 2% 不透明度，肉眼不可见，
        # 留着反而在暗角形成半透明雾
        alpha = np.where(alpha < 6, 0, alpha).astype(np.uint8)

    if src_alpha is not None:
        # 原图本来就透明的像素继续保持透明
        alpha = np.where(src_alpha <= 8, 0, alpha).astype(np.uint8)

    return np.dstack([out, alpha])


def chroma_from_params(arr: np.ndarray, params: Optional[dict],
                       src_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """按 UI 参数字典执行幕布去背，返回 H×W×4 RGBA。

    params: {key, tol, keep_same, despill, edge}；key=None 自动取样。
    src_alpha: 原图透明度，用于「仅外部连通」判定与透明区域保持。
    """
    p = params or {}
    return chroma_key_rgba(
        arr, p.get("key"), tol=float(p.get("tol", DEFAULT_TOL)),
        keep_same=bool(p.get("keep_same", True)),
        despill=str(p.get("despill", "标准")),
        edge=str(p.get("edge", "标准")),
        src_alpha=src_alpha)
