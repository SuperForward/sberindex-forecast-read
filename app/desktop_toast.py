"""Custom frameless Qt toast, stacked bottom-right on the primary screen.

Замена системным Windows-toast'ам, которые молча теряются в Focus Assist /
Action Center. Работает всегда, пока приложение живо – даже когда главное
окно свёрнуто в трей. Звук проигрывается через winsound, чтобы не зависеть
от WebAudio в скрытом QWebEngineView.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_log = logging.getLogger("pc_notify")

_MARGIN = 18
_GAP = 10
_WIDTH = 380
_LIFETIME_MS = 8000
_FADE_MS = 260
_MAX_STACK = 5

# Кастомные WAV из app/assets/sounds/ – синтезированы под те же частоты и
# огибающие, что WebAudio-мелодии в терминале. Играются напрямую через
# winsound(SND_FILENAME), независимо от системных звуковых схем и Focus Assist.
_SOUND_DIR = Path(__file__).resolve().parent / "assets" / "sounds"
_MELODY_FILE = {
    "single": "1.wav",
    "double": "2.wav",
    "soft":   "3.wav",
    "rise":   "4.wav",
    "sharp":  "5.wav",
}

_KIND_COLOR = {
    "listing": "#ffb020",
    "setup":   "#7cd992",
    "pattern": "#6bb0ff",
    "setup_alerts": "#7cd992",
}

_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app.ico"

_stack: list["ToastWidget"] = []


def _screen_geometry() -> QRect:
    app = QApplication.instance()
    if app is None:
        return QRect(0, 0, 1920, 1080)
    scr = QGuiApplication.primaryScreen()
    return scr.availableGeometry() if scr else QRect(0, 0, 1920, 1080)


def _relayout() -> None:
    """Столбик тостов у правого нижнего угла, снизу вверх."""
    geo = _screen_geometry()
    x = geo.right() - _WIDTH - _MARGIN
    y = geo.bottom() - _MARGIN
    for w in reversed(_stack):
        h = w.sizeHint().height()
        y -= h
        w.move_smooth(QPoint(x, y))
        y -= _GAP


def _play_sound(melody: str, volume: float) -> None:
    """Non-blocking playback кастомного WAV. Громкость управляется системным
    микшером (winsound не даёт per-call регулятор); флаг `sound=false` в
    настройках приходит сюда как volume<=0 и глушит звук."""
    if sys.platform != "win32":
        return
    if volume <= 0.01:
        return
    fname = _MELODY_FILE.get(melody, "1.wav")
    path = _SOUND_DIR / fname
    if not path.exists():
        _log.warning("toast_sound_missing  %s", path)
        return

    def _fire():
        try:
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception as exc:  # pragma: no cover
            _log.warning("toast_sound_fail  %s", exc)

    threading.Thread(target=_fire, daemon=True).start()


class ToastWidget(QWidget):
    def __init__(self, title: str, body: str, kind: str) -> None:
        super().__init__(
            None,
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.BypassWindowManagerHint,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedWidth(_WIDTH)

        if _ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_ICON_PATH)))

        accent = _KIND_COLOR.get(kind, "#ffb020")

        card = QFrame(self)
        card.setObjectName("card")
        card.setStyleSheet(f"""
            QFrame#card {{
              background:#181c22;
              border:1px solid #2a323e;
              border-left:3px solid {accent};
              border-radius:8px;
            }}
            QLabel#title {{
              color:#f0f2f5;
              font: 600 12.5px "Segoe UI";
            }}
            QLabel#body {{
              color:#a0a8b4;
              font: 11.5px "Segoe UI";
            }}
            QLabel#tag {{
              color:{accent};
              font: 600 9.5px "Segoe UI";
              letter-spacing:1px;
            }}
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 180))
        card.setGraphicsEffect(shadow)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(2)

        tag_row = QHBoxLayout()
        tag_row.setSpacing(0)
        tag = QLabel((kind or "notice").upper())
        tag.setObjectName("tag")
        tag_row.addWidget(tag)
        tag_row.addStretch(1)
        lay.addLayout(tag_row)

        t = QLabel(title or "")
        t.setObjectName("title")
        t.setWordWrap(True)
        lay.addWidget(t)

        b = QLabel(body or "")
        b.setObjectName("body")
        b.setWordWrap(True)
        lay.addWidget(b)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)  # оставляем место под shadow
        outer.addWidget(card)

        self.adjustSize()

        self._closed = False
        self._move_anim: QPropertyAnimation | None = None
        self._fade_anim: QPropertyAnimation | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)

    def show_animated(self, start_pos: QPoint) -> None:
        self.move(start_pos.x() + 40, start_pos.y())
        self.setWindowOpacity(0.0)
        self.show()
        # Fade in.
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_anim.setDuration(_FADE_MS)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_anim.start()
        # Slide in.
        self.move_smooth(start_pos)
        self._timer.start(_LIFETIME_MS)

    def move_smooth(self, target: QPoint) -> None:
        if self._move_anim is not None:
            self._move_anim.stop()
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(220)
        anim.setStartValue(self.pos())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._move_anim = anim

    def dismiss(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        try:
            _stack.remove(self)
        except ValueError:
            pass
        _relayout()
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(_FADE_MS)
        anim.setStartValue(self.windowOpacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self.close)
        anim.start()
        self._fade_anim = anim

    def mousePressEvent(self, event) -> None:
        self.dismiss()
        super().mousePressEvent(event)

    def event(self, e):
        # Не даём тосту красть фокус ни при каких обстоятельствах.
        if e.type() == QEvent.FocusIn:
            return True
        return super().event(e)


def show_toast(title: str, body: str, kind: str = "notice",
               melody: str = "single", volume: float = 0.5) -> None:
    """Показать custom-тост + сыграть звук. Вызывать только из GUI-потока."""
    while len(_stack) >= _MAX_STACK:
        old = _stack[0]
        old.dismiss()
        _stack.pop(0)

    w = ToastWidget(title, body, kind)
    _stack.append(w)

    geo = _screen_geometry()
    x = geo.right() - _WIDTH - _MARGIN
    y = geo.bottom() - _MARGIN - w.sizeHint().height()
    for other in _stack[:-1]:
        y -= other.sizeHint().height() + _GAP
    w.show_animated(QPoint(x, y))
    _relayout()
    _play_sound(melody, volume)
