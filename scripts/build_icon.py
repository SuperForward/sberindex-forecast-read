"""Rebuild the Windows .ico from the same drawing that make_icon() uses.

Windows picks a size from the .ico depending on where the icon appears:
16 px for taskbar labels, 32 px for the classic tray, 48 px for large
tiles, 256 px for the modern taskbar and jump list. All four go in one
file - not doing that leaves Explorer showing a blurry upscale.

Pillow is not a dependency, so the .ico is written by hand: an ICONDIR
header, one ICONDIRENTRY per size, then the PNG bytes for each image
back-to-back. Modern Windows accepts PNG payloads inside .ico for any
size, and QPixmap already knows how to save PNG.

Run: python scripts/build_icon.py
"""

import math
import struct
import sys
from io import BytesIO
from pathlib import Path

from PySide6.QtCore import Qt, QBuffer, QByteArray, QRectF, QRect
import numpy as np
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap)


SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
OUT = Path(__file__).resolve().parent.parent / "app" / "assets" / "app.ico"
OUT_SHORTCUT = Path(__file__).resolve().parent.parent / "app" / "assets" / "app_shortcut.ico"


# --- геометрия и цвет -------------------------------------------------------

KEYS = ("#0d9fe6", "#1fb07a", "#21a038", "#8cc63f")    # фирменные цвета Сбера
KEY_POS = (0.0, 0.35, 0.65, 1.0)
BARS = ((12, 40), (26.5, 31), (41, 22))                # x, верх; низ – 54, ширина 11
LINE = ((9.71, 32.19), (22.65, 22.79), (37.69, 17.31), (49, 6))  # 36°, 20°, 45°; отрезки по 16, равные зазоры до столбцов

HEAD = ((43.5, 6), (49, 6), (49, 11.5))                # прямой угол: на 45° линии симметричен
SHIFT = (1.8, 2.2)                                     # оптический центр: чуть выше геометрического


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _to_oklch(hex_: str) -> tuple[float, float, float]:
    r, g, b = (_lin(int(hex_[i:i + 2], 16) / 255) for i in (1, 3, 5))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, math.hypot(A, B), math.atan2(B, A)


def _from_oklch(L: float, C: float, H: float) -> QColor:
    A, B = C * math.cos(H), C * math.sin(H)
    l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3
    s = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return QColor.fromRgbF(_srgb(r), _srgb(g), _srgb(b))


def gradient_stops(n: int = 33) -> list[tuple[float, QColor]]:
    """Переход между фирменными цветами в OKLCH: середина остаётся сочной,
    без серого провала, который даёт смешение в RGB."""
    keys = [_to_oklch(k) for k in KEYS]
    out = []
    for i in range(n):
        t = i / (n - 1)
        j = max(k for k in range(len(KEY_POS) - 1) if KEY_POS[k] <= t) if t < 1 else len(KEY_POS) - 2
        f = (t - KEY_POS[j]) / (KEY_POS[j + 1] - KEY_POS[j])
        (L0, C0, H0), (L1, C1, H1) = keys[j], keys[j + 1]
        dh = (H1 - H0 + math.pi) % (2 * math.pi) - math.pi
        out.append((t, _from_oklch(L0 + (L1 - L0) * f, C0 + (C1 - C0) * f, H0 + dh * f)))
    return out


def squircle(size: float = 64, n: float = 5.0, steps: int = 240) -> QPainterPath:
    """Суперэллипс |x|^n + |y|^n = 1: скругление плавно переходит в сторону,
    без излома, как у значков iOS/macOS."""
    path = QPainterPath()
    r = size / 2
    for i in range(steps + 1):
        t = 2 * math.pi * i / steps
        c, s_ = math.cos(t), math.sin(t)
        x = r + r * math.copysign(abs(c) ** (2 / n), c)
        y = r + r * math.copysign(abs(s_) ** (2 / n), s_)
        path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
    path.closeSubpath()
    return path


def _grain(size: int, alpha: int = 5) -> QImage:
    """Зерно ±alpha: убирает полосы в плавном переходе."""
    rng = np.random.default_rng(7)
    v = rng.integers(0, 2, (size, size), dtype=np.uint8) * 255
    a = rng.integers(0, alpha + 1, (size, size), dtype=np.uint8)
    arr = np.dstack([v, v, v, a]).astype(np.uint8)       # RGBA
    img = QImage(arr.tobytes(), size, size, 4 * size, QImage.Format_RGBA8888)
    return img.copy()


def render(size: int, inset: int = 0) -> bytes:
    """Значок SberIndex на сетке 64×64.

    Суперэллипс с диагональным градиентом фирменных цветов Сбера в OKLCH,
    едва заметный блик сверху, внутренняя грань (светлее сверху, темнее
    снизу), зерно на крупных размерах. Белые растущие столбцы и линия
    прогноза со стрелкой, рисунок оптически по центру. На 16–24 px – только
    столбцы, выровненные по пиксельной сетке."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    u = (size - 2 * inset) / 64.0
    p.translate(inset, inset)
    p.scale(u, u)                           # дальше рисуем в единицах 64×64
    small = size <= 24
    p.setPen(Qt.NoPen)

    if size >= 96:
        shape = squircle()
    else:
        # мелкие размеры: прямые стороны точно по пиксельной сетке и целый
        # радиус – у суперэллипса выпуклые стороны ложатся между пикселями
        # и весь край выходит полупрозрачным (размытым)
        r = round(size * 0.22) / u
        shape = QPainterPath()
        shape.addRoundedRect(QRectF(0, 0, 64, 64), r, r)
    g = QLinearGradient(0, 0, 64, 64)
    for pos, c in gradient_stops():
        g.setColorAt(pos, c)
    p.fillPath(shape, QBrush(g))
    p.setClipPath(shape)
    if size >= 40:
        hl = QLinearGradient(0, 0, 0, 64)                # блик в верхней трети
        hl.setColorAt(0.0, QColor(255, 255, 255, 26))
        hl.setColorAt(0.4, QColor(255, 255, 255, 0))
        p.fillPath(shape, QBrush(hl))
        rim = QLinearGradient(0, 0, 0, 64)               # внутренняя грань
        rim.setColorAt(0.0, QColor(255, 255, 255, 38))
        rim.setColorAt(0.5, QColor(255, 255, 255, 0))
        rim.setColorAt(1.0, QColor(0, 40, 20, 22))
        p.setPen(QPen(QBrush(rim), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPath(shape)                                # половина обводки снаружи срезана клипом
        p.setPen(Qt.NoPen)
    if size >= 128:
        p.save()
        p.resetTransform()
        p.translate(inset, inset)
        p.drawImage(0, 0, _grain(size - 2 * inset))
        p.restore()

    white, soft = QColor(255, 255, 255), QColor(255, 255, 255, 165)
    if small:
        # 16–24 px: раскладка в целых пикселях с гарантированным зазором
        # (на 16 px зазор в 4 единицы сетки – ровно 1 px и после сглаживания
        # пропадал, столбцы сливались); без сглаживания краёв
        p.save()
        p.resetTransform()
        p.translate(inset, inset)
        p.setRenderHint(QPainter.Antialiasing, False)
        s_ = size - 2 * inset
        m = round(s_ * 0.19)
        gap = 1 if s_ < 20 else 2
        w = (s_ - 2 * m - 2 * gap) // 3
        x = m + (s_ - 2 * m - 2 * gap - 3 * w) // 2
        bottom = s_ - m
        span = s_ - 2 * m
        p.setBrush(QBrush(white))
        for frac in (0.4, 0.65, 0.9):
            h = max(2, round(span * frac))
            p.drawRect(QRect(x, bottom - h, w, h))
            x += w + gap
        p.restore()
    elif size < 96:
        # 32–64 px (ярлык на рабочем столе – 48): столбцы по целым пикселям – края резкие
        p.save()
        p.resetTransform()
        p.translate(inset, inset)
        # одна ширина и один зазор на все столбцы – если округлять каждый край
        # отдельно, зазоры и ширины расходятся на пиксель
        w = round(11 * u)
        gap = max(1, round(3.5 * u))
        x0 = round((BARS[0][0] + SHIFT[0]) * u)
        y1 = round((54 + SHIFT[1]) * u)
        for i, (bx, top) in enumerate(BARS):
            y0 = round((top + SHIFT[1]) * u)
            p.setBrush(QBrush(white if i < 2 else soft))
            p.drawRoundedRect(QRectF(x0, y0, w, y1 - y0), 0.8, 0.8)
            x0 += w + gap
        p.restore()
    p.translate(*SHIFT)
    if size >= 96:
        for i, (x, top) in enumerate(BARS):
            p.setBrush(QBrush(white if i < 2 else soft))
            p.drawRoundedRect(QRectF(x, top, 11, 54 - top), 2.5, 2.5)
    if not small:
        width = 3.4
        if size < 96:
            width = max(2, round(3.4 * u)) / u              # целая толщина в пикселях
        p.setPen(QPen(white, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        for pts in (LINE, HEAD):
            path = QPainterPath()
            path.moveTo(*pts[0])
            for x, y in pts[1:]:
                path.lineTo(x, y)
            p.drawPath(path)
    p.end()

    # QByteArray must be kept alive as its own name: passing it straight into
    # QBuffer(...) lets Python garbage-collect the temporary between lines,
    # and the buffer writes into freed memory - Windows notices it as an
    # access violation the moment QPixmap.save touches the underlying storage.
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.WriteOnly)
    pm.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def write_svg(out: Path) -> None:
    """SVG для вкладки браузера – из той же геометрии и цветов, что .ico (без зерна)."""
    pts = []
    r, n, steps = 32, 5.0, 120
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s_ = math.cos(t), math.sin(t)
        pts.append(f"{r + r * math.copysign(abs(c) ** (2 / n), c):.2f} {r + r * math.copysign(abs(s_) ** (2 / n), s_):.2f}")
    d = "M" + " L".join(pts) + " Z"
    stops = "\n".join(f'      <stop offset="{t:.3f}" stop-color="{c.name()}"/>' for t, c in gradient_stops(17))
    rect = lambda i, x, top: (f'    <rect x="{x}" y="{top}" width="11" height="{54 - top}" rx="2.5"'
                              + (' fill-opacity=".65"' if i == 2 else "") + "/>")
    bars = "\n".join(rect(i, x, top) for i, (x, top) in enumerate(BARS))
    poly = lambda pts_: "M" + " L".join(f"{x} {y}" for x, y in pts_)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <!-- Значок SberIndex: генерируется scripts/build_icon.py, руками не править -->
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
{stops}
    </linearGradient>
    <linearGradient id="hl" x1="0" y1="0" x2="0" y2="64" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#fff" stop-opacity=".10"/>
      <stop offset=".4" stop-color="#fff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="rim" x1="0" y1="0" x2="0" y2="64" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#fff" stop-opacity=".15"/>
      <stop offset=".5" stop-color="#fff" stop-opacity="0"/>
      <stop offset="1" stop-color="#002814" stop-opacity=".09"/>
    </linearGradient>
    <clipPath id="shape"><path d="{d}"/></clipPath>
  </defs>
  <path d="{d}" fill="url(#bg)"/>
  <g clip-path="url(#shape)">
    <path d="{d}" fill="url(#hl)"/>
    <path d="{d}" fill="none" stroke="url(#rim)" stroke-width="1.6"/>
  </g>
  <g fill="#fff" transform="translate({SHIFT[0]} {SHIFT[1]})">
{bars}
    <g fill="none" stroke="#fff" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round">
      <path d="{poly(LINE)}"/>
      <path d="{poly(HEAD)}"/>
    </g>
  </g>
</svg>
"""
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


def write_ico(sizes: list[int], out: Path, inset: int = 0) -> None:
    images = [(sz, render(sz, inset=inset)) for sz in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    dir_size = 6 + 16 * len(images)
    entries = BytesIO()
    payload = BytesIO()
    offset = dir_size
    for sz, png in images:
        # Width/height 0 means 256 - Explorer expects that quirk.
        w = 0 if sz >= 256 else sz
        h = 0 if sz >= 256 else sz
        entries.write(struct.pack(
            "<BBBBHHII",
            w, h, 0, 0, 1, 32, len(png), offset,
        ))
        payload.write(png)
        offset += len(png)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + entries.getvalue() + payload.getvalue())
    print(f"wrote {out}  ({out.stat().st_size:,} bytes, {len(images)} sizes)")


def write_logo(icon_svg: Path, out: Path, ink: str = "#15191e", sub: str = "#6b7480") -> None:
    """Логотип со словом: значок (вложенный app.svg) + «SberIndex: Forecast» и подпись.
    Для шапки README и титульного слайда. Текст – системным шрифтом;
    ink/sub – цвета названия и подписи (для тёмного фона светлые)."""
    body = icon_svg.read_text(encoding="utf-8")
    body = body.replace('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">',
                        '<svg x="0" y="0" width="96" height="96" viewBox="0 0 64 64">', 1)
    body = body.replace('id="bg"', 'id="logo-bg"').replace("url(#bg)", "url(#logo-bg)")
    body = body.replace('id="hl"', 'id="logo-hl"').replace("url(#hl)", "url(#logo-hl)")
    body = body.replace('id="rim"', 'id="logo-rim"').replace("url(#rim)", "url(#logo-rim)")
    body = body.replace('id="shape"', 'id="logo-shape"').replace("url(#shape)", "url(#logo-shape)")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 96" width="520" height="96">
  <!-- Логотип SberIndex: генерируется scripts/build_icon.py, руками не править -->
{body.strip()}
  <text x="116" y="54" font-family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif" font-size="42"
        font-weight="700" letter-spacing="-0.5" fill="{ink}">SberIndex: Forecast</text>
  <text x="118" y="80" font-family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif" font-size="17"
        fill="{sub}">прогноз расходов · сдвиги в муниципалитетах</text>
</svg>
"""
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    # QPainter / QFont need a QGuiApplication even for offscreen work; full
    # QApplication would pull in QtWidgets and try to talk to the display
    # subsystem, which segfaults in headless shells.
    _ = QGuiApplication.instance() or QGuiApplication(sys.argv)
    write_ico(SIZES, OUT, inset=0)
    write_ico(SIZES, OUT_SHORTCUT, inset=2)
    # превью для README и проверки глазами
    (OUT.parent / "app_preview.png").write_bytes(render(256))
    (OUT.parent / "tray_preview.png").write_bytes(render(32))
    write_svg(OUT.parent / "app.svg")
    write_logo(OUT.parent / "app.svg", OUT.parent / "logo.svg")
    write_logo(OUT.parent / "app.svg", OUT.parent / "logo_dark.svg", ink="#e6eaef", sub="#8a93a0")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_icon", "ui", main)
