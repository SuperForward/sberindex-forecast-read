"""Native window hosting the HTML terminal.

The UI is a local page rendered by QtWebEngine; Python owns the data,
models and the tray icon, and talks to the page over QWebChannel. That keeps the whole thing a single double-clickable Python app -
no Node toolchain, no dev server, no browser.
"""

import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl

_perf_log = logging.getLogger("perf")
_chart_log = logging.getLogger("chart")


def _perf(label: str, since: float | None = None) -> float:
    now = time.perf_counter()
    if since is not None:
        _perf_log.info("%-30s  +%.0fms", label, (now - since) * 1000)
    else:
        _perf_log.info("%-30s  mark", label)
    return now
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QFont, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from app.bridge import Bridge
from app.desktop_toast import show_toast as _desktop_toast
from src.config import APP_NAME

UI_FILE = Path(__file__).resolve().parent / "ui" / "terminal.html"
ICON_FILE = Path(__file__).resolve().parent / "assets" / "app.ico"
# Persistent QtWebEngine profile lives next to the project so localStorage –
# chart preferences, filter state – survives a restart.
# The default profile that QWebEngineView(None) picks up when no profile is
# passed is off-the-record: every restart came back to defaults because the
# storage lived only in memory.
WEBENGINE_PROFILE_DIR = Path(__file__).resolve().parent.parent / ".webengine"


def make_icon() -> QIcon:
    """Multi-size icon for the tray, window frame and taskbar.

    Prefers the bundled .ico (built by scripts/build_icon.py) - Windows
    picks the right size from it for each context (16/32/48/256). Falls
    back to a hand-drawn pixmap only if the file is missing, so the app
    still runs from a fresh clone before the asset step has been done."""
    if ICON_FILE.exists():
        return QIcon(str(ICON_FILE))
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor(33, 160, 56)))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    p.setBrush(QBrush(QColor(255, 255, 255)))
    for x, y in ((13, 36), (26, 27), (40, 18)):
        p.drawRoundedRect(x, y, 11, 54 - y, 2, 2)
    p.end()
    return QIcon(pixmap)


class _ClickOutsideFilter(QObject):
    """Hides the popup on any mouse press outside its bounds.

    Qt.Popup loses its Win32 grab when a native menu from another app fires
    WM_CANCELMODE. Using Qt.Tool + this filter is more reliable on Windows.
    """

    def __init__(self, popup: "TrayMenu") -> None:
        super().__init__()
        self._popup = popup

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.MouseButtonPress and self._popup.isVisible():
            try:
                gpos = event.globalPosition().toPoint()
            except AttributeError:
                gpos = event.globalPos()
            if not self._popup.geometry().contains(gpos):
                self._popup.hide()
        return False


class TrayMenu(QWidget):
    """Frameless popup styled like a native tray menu.

    Two rows: show window, quit. Closes on focus loss or Escape.
    """

    _BG = QColor("#1a1b1e")
    _BORDER = QColor(255, 255, 255, 0)
    _TEXT = QColor("#ececec")
    _HOVER = QColor("#272930")
    _SEP = QColor(255, 255, 255, 55)
    _RADIUS = 12
    _PAD = 6
    _ROW_H = 32
    _H_PAD = 12  # horizontal text padding inside each button

    _LABELS = ["Открыть", "Закрыть"]

    def __init__(self, on_show, on_quit) -> None:
        super().__init__(
            None,
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        # Don't steal focus on show – lets the first click reach its real target.
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._buttons: list[QPushButton] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(self._PAD, self._PAD, self._PAD, self._PAD)
        lay.setSpacing(2)

        self._add_row(lay, self._LABELS[0], on_show)
        self._add_separator(lay)
        self._add_row(lay, self._LABELS[1], on_quit)

        # Size to the widest button text + padding
        f = QFont("Segoe UI", 10)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(f)
        max_text_w = max(fm.horizontalAdvance(t) for t in self._LABELS)
        w = max_text_w + self._H_PAD * 2 + self._PAD * 2 + 8
        rows = 2
        h = self._PAD * 2 + rows * self._ROW_H + 2 + 1
        self.setFixedSize(w, h)

        # Global click filter: hides popup on any Qt-level click outside bounds.
        self._click_filter = _ClickOutsideFilter(self)
        QApplication.instance().installEventFilter(self._click_filter)

        # Native click watcher: polls GetAsyncKeyState every 50 ms while visible.
        # Catches clicks on taskbar/tray (Win32 events Qt never sees).
        # Only hides when a mouse button is actually pressed outside our bounds.
        self._native_click_timer = QTimer(self)
        self._native_click_timer.setInterval(50)
        self._native_click_timer.timeout.connect(self._check_native_click)

    def _add_row(self, lay, text: str, cb) -> None:
        btn = QPushButton(text, self)
        btn.setFlat(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(self._ROW_H)
        f = QFont("Segoe UI", 10)
        btn.setFont(f)
        btn.setStyleSheet(
            "QPushButton{"
            f"color:{self._TEXT.name()};"
            "background:transparent;"
            "border:none;"
            "text-align:left;"
            f"padding:0 {self._H_PAD}px;"
            "border-radius:8px;"
            "}"
            "QPushButton:hover{"
            f"background:{self._HOVER.name()};"
            "}"
        )
        btn.clicked.connect(lambda: (self.hide(), cb()))
        lay.addWidget(btn)
        self._buttons.append(btn)

    def _add_separator(self, lay) -> None:
        sep = QWidget(self)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{self._SEP.name(QColor.HexArgb)};")
        lay.addWidget(sep)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRect(0, 0, self.width() - 1, self.height() - 1)
        path = QPainterPath()
        path.addRoundedRect(rect, self._RADIUS, self._RADIUS)
        p.fillPath(path, QBrush(self._BG))
        pen = QPen(self._BORDER)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawPath(path)

    def popup_at(self, global_pos: QPoint) -> None:
        screen = QApplication.screenAt(global_pos) or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = global_pos.x() + 2
        x = min(x, geo.right() - self.width() - 4)
        # 2px gap above cursor so menu bottom sits right at the tray icon
        y = global_pos.y() - self.height() + 3
        if y < geo.top() + 4:
            y = global_pos.y() + 4
        self.move(x, y)
        self.show()
        self.raise_()
        self._native_click_timer.start()

    def _check_native_click(self) -> None:
        if sys.platform != "win32":
            return
        import ctypes
        LMB, RMB = 0x01, 0x02
        pressed = (
            ctypes.windll.user32.GetAsyncKeyState(LMB) & 0x8000
            or ctypes.windll.user32.GetAsyncKeyState(RMB) & 0x8000
        )
        if pressed and not self.geometry().contains(QCursor.pos()):
            self.hide()

    def hideEvent(self, event) -> None:
        self._native_click_timer.stop()
        super().hideEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)


class TerminalWindow(QMainWindow):
    def __init__(self) -> None:
        t0 = time.perf_counter()
        _chart_log.info("terminal_window_init_start")
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        # Размер «восстановленного» окна – не больше 90% экрана: на 1366×768
        # фиксированные 1520×900 вылезали за край. Минимум – чтобы окно
        # без рамки нельзя было сжать до нечитаемого.
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1520, int(avail.width() * 0.9)), min(900, int(avail.height() * 0.9)))
        self.setMinimumSize(min(900, avail.width()), min(560, avail.height()))

        self._force_quit = False
        self._page_ready = False

        # Именованный профиль сохраняет localStorage между запусками;
        # QWebEngineView() без профиля работает off-the-record.
        WEBENGINE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self.profile = QWebEngineProfile("sberindex-terminal", self)
        self.profile.setPersistentStoragePath(str(WEBENGINE_PROFILE_DIR))
        self.profile.setCachePath(str(WEBENGINE_PROFILE_DIR / "cache"))
        self.profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.AllowPersistentCookies)

        self.view = QWebEngineView()
        self.page = QWebEnginePage(self.profile, self.view)
        self.view.setPage(self.page)
        # Без белой вспышки до первого paint; совпадает с --void в terminal.html.
        self.page.setBackgroundColor(QColor("#08090c"))
        self.setCentralWidget(self.view)

        self.bridge = Bridge(controller=self)
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("py", self.bridge)
        self.view.page().setWebChannel(self.channel)

        # DevTools поднимает второй контекст Chromium – создаём по требованию.
        self._devtools: QWebEngineView | None = None
        inspect = QAction(self)
        inspect.setShortcut(QKeySequence("Ctrl+Shift+I"))
        inspect.triggered.connect(self._open_devtools)
        self.addAction(inspect)

        self.view.loadFinished.connect(self._on_load_finished)
        self.view.setUrl(QUrl.fromLocalFile(str(UI_FILE)))

        self._build_tray()
        self.bridge.trayMessage.connect(self._show_tray_message)
        _perf("init_complete", t0)

    # ---------------------------------------------------------------- page

    def changeEvent(self, event):
        # Держим иконку max/restore в HTML синхронной, когда окно меняет ОС
        # (Win+Down, Snap), а не только клик по кнопке.
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.WindowStateChange:
            self.bridge.windowStateChanged.emit(self.isMaximized())
        super().changeEvent(event)

    def _open_devtools(self) -> None:
        if self._devtools is None:
            self._devtools = QWebEngineView()
            self._devtools.setWindowTitle("DevTools")
            self._devtools.resize(1200, 700)
            self.view.page().setDevToolsPage(self._devtools.page())
        self._devtools.show()
        self._devtools.raise_()

    def _on_load_finished(self, ok: bool) -> None:
        _chart_log.log(logging.INFO if ok else logging.ERROR,
                       "page_load_finished  ok=%s file=%s", ok, UI_FILE)
        self._page_ready = ok
        if not ok:
            print(f"Не удалось загрузить интерфейс: {UI_FILE}")

    # ---------------------------------------------------------------- tray

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip(APP_NAME)
        self._tray_menu = TrayMenu(on_show=self._on_tray_show, on_quit=self._on_tray_quit)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self._restore()
        elif reason == QSystemTrayIcon.Context:
            self._tray_menu.popup_at(QCursor.pos())

    def _log_tray_action(self, detail: str) -> None:
        _chart_log.info("tray  %s", detail)

    def _on_tray_show(self) -> None:
        self._restore()

    def _on_tray_quit(self) -> None:
        self.quit_application()

    def _show_tray_message(self, title: str, body: str) -> None:
        """Frameless-тост поверх всех окон, виден и при окне в трее."""
        try:
            _desktop_toast(title, body, "notice", "single", 0.5)
        except Exception as e:  # noqa: BLE001
            _chart_log.warning("tray_toast_failed  %s", e)

    def _restore(self) -> None:
        self.showMaximized()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        if self._force_quit:
            event.accept()
            return
        # Без значка в трее спрятанное окно не вернуть ничем, кроме Диспетчера
        # задач (Проводник ещё не поднялся или перезапускается) – тогда закрываем.
        if not (QSystemTrayIcon.isSystemTrayAvailable() and self.tray.isVisible()):
            _chart_log.warning("tray_unavailable  закрытие окна = выход из приложения")
            self.quit_application()
            event.accept()
            return
        event.ignore()
        self._tray_menu.hide()
        self.hide()
        self.tray.showMessage(
            "Свёрнуто в трей",
            "Закрыть – правый клик по значку.",
            QSystemTrayIcon.Information,
            4000,
        )

    def quit_application(self) -> None:
        _chart_log.info("engine_shutdown  по команде «Закрыть» из трея")
        self._force_quit = True
        self.close()
        self.tray.hide()
        QApplication.quit()

    # main() проверяет флаг при выходе; фоновых потоков пока нет.
    worker_lingering = False
