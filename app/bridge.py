"""Мост Python ↔ HTML (QWebChannel, объект `py` в JS).

Управление безрамочным окном и слоты данных (src/app_service.py): панель
по МО, прогнозы backtest, шоки. Слоты данных возвращают JSON-строку.
"""

import json
import logging
import sys
import time

from PySide6.QtCore import Qt, QObject, Signal, Slot

from src.config import APP_NAME, DATA_DIR

_ui_log = logging.getLogger("ui")


def _api_table() -> dict:
    """Имя запроса из JS -> (функция app_service, тихий лог). Только эти
    функции можно вызвать через request(): JS не выполняет произвольный код."""
    from src import app_service as a
    return {
        "summary": (a.summary, False), "searchMo": (a.search_mo, False), "moDetail": (a.mo_detail, False),
        "moForecast": (a.mo_forecast, False), "modelMetrics": (a.model_metrics, False),
        "worstMo": (a.worst_mo, False), "shocksOverview": (a.shocks_overview, False),
        "horizonsOverview": (a.horizons_overview, False), "cpdOverview": (a.cpd_overview, False),
        "cpdMo": (a.cpd_mo, False), "dataStatus": (a.data_status, True),
        "recalcStart": (a.recalc_start, False), "reloadData": (a.reload, False),
        "resultsOverview": (a.results_overview, False), "autoUpdateTick": (a.auto_update_tick, True),
        "newsOverview": (a.news_overview, False), "exportCsv": (a.export_csv, False),
    }


class Bridge(QObject):
    logMessage = Signal(str)
    trayMessage = Signal(str, str)
    windowStateChanged = Signal(bool)
    # ответ на request(): (номер запроса, JSON). Испускается в главном потоке.
    response = Signal(int, str)
    _done = Signal(int, str)            # из рабочего потока -> в главный (queued)

    def __init__(self, controller=None) -> None:
        super().__init__()
        self.controller = controller
        from concurrent.futures import ThreadPoolExecutor
        # Запросы данных выполняются здесь, а не в главном потоке: иначе окно
        # замирает на время запроса (первая сводка ~2 с). Два потока: долгий
        # запрос не задерживает быстрые.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="api")
        self._api = None
        self._done.connect(self.response.emit)
        # данные грузятся, пока поднимается окно и страница: первый запрос уже из кэша
        self._pool.submit(self._warm)

    @staticmethod
    def _warm() -> None:
        from src import app_service
        app_service.warm()

    @Slot(str, str, int)
    def request(self, name: str, args_json: str, req_id: int) -> None:
        """Асинхронный запрос из JS: ответ придёт сигналом response(req_id, json)."""
        if self._api is None:
            self._api = _api_table()
            from src import app_service
            self._pool.submit(app_service.log_choices)      # на чём строятся результаты – в ui.log
        entry = self._api.get(name)
        if entry is None:
            _ui_log.warning("data_api_unknown  %s", name)
            self._done.emit(req_id, json.dumps({"error": f"неизвестный запрос {name}"}, ensure_ascii=False))
            return
        try:
            args = json.loads(args_json or "[]")
        except ValueError:
            _ui_log.warning("data_api_bad_args  %s %r", name, (args_json or "")[:200])
            args = []
        fn, quiet = entry

        def work():
            out = self._call(fn, *args, quiet=quiet, name=name)
            self._done.emit(req_id, out)

        self._pool.submit(work)

    @Slot(str, str, result=str)
    def saveFile(self, filename: str, text: str) -> str:
        """Сохранить выгрузку: диалог «Сохранить как» (QtWebEngine сам загрузки
        не сохраняет). BOM – чтобы Excel открыл UTF-8 с кириллицей. Путь или ''."""
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog
        start = str(Path.home() / "Downloads" / filename)
        path, _ = QFileDialog.getSaveFileName(None, "Сохранить CSV", start, "CSV (*.csv)")
        if not path:
            return ""
        try:
            Path(path).write_text(text, encoding="utf-8-sig")
        except OSError as e:
            _ui_log.warning("save_failed  %s: %s", path, e)
            return ""
        _ui_log.info("saved  %s", path)
        return path

    @Slot(result=str)
    def appInfo(self) -> str:
        raw = DATA_DIR / "raw"
        files = sorted(p.name for p in raw.glob("*") if p.is_file()) if raw.exists() else []
        return json.dumps({"name": APP_NAME, "raw_files": files}, ensure_ascii=False)

    @Slot(str, str)
    def jsLog(self, level: str, message: str) -> None:
        """Сообщения и ошибки из JS (window.onerror, ответы API с error)."""
        _ui_log.log(getattr(logging, level.upper(), logging.INFO), "js  %s", message)

    @Slot(str)
    def openExternal(self, url: str) -> None:
        """Ссылку или файл открывает система (браузер, программа для PDF). Запуск
        программы Windows делает синхронно, иногда секунды: в главном потоке окно
        на это время замирало, поэтому – в фоновом потоке."""
        import threading

        def run() -> None:
            try:
                if sys.platform == "win32":
                    import os
                    from urllib.parse import unquote, urlparse
                    u = urlparse(url)
                    os.startfile(unquote(u.path.lstrip("/")) if u.scheme == "file" else url)  # noqa: S606
                else:
                    import webbrowser
                    webbrowser.open(url)
                _ui_log.info("open_external  %s", url)
            except Exception as e:  # noqa: BLE001
                _ui_log.warning("open_external_failed  %s: %s", url, e)
        threading.Thread(target=run, name="open-external", daemon=True).start()

    # ---------------------------------------------------------- data API
    # Каждый слот возвращает JSON-строку; ошибка – {"error": "..."}, чтобы
    # интерфейс показал её, а не упал молча.

    SLOW_MS = 1000  # запрос дольше – предупреждение в ui.log

    @staticmethod
    def _call(fn, *args, quiet: bool = False, name: str | None = None) -> str:
        """Вызов src/app_service с логом в ui.log: имя, аргументы, время, размер ответа.

        quiet – частые опросы (статус пересчёта): успешный быстрый ответ идёт
        на DEBUG, чтобы не засорять ui.log; медленный и ошибка – как обычно."""
        name = name or getattr(fn, "__name__", str(fn))
        t0 = time.perf_counter()
        try:
            # allow_nan=False: NaN в ответе – ошибка здесь, а не нечитаемый JSON в JS
            out = json.dumps(fn(*args), ensure_ascii=False, allow_nan=False)
        except Exception as e:  # noqa: BLE001
            _ui_log.exception("data_api_failed  %s args=%s", name, args)
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        ms = (time.perf_counter() - t0) * 1000
        level = logging.WARNING if ms > Bridge.SLOW_MS else logging.DEBUG if quiet else logging.INFO
        _ui_log.log(level, "data_api  %s args=%s bytes=%d%s", name, args, len(out),
                    "  МЕДЛЕННО" if level == logging.WARNING else "", extra={"elapsed_ms": ms})
        return out

    @Slot(result=str)
    def summary(self) -> str:
        from src import app_service
        return self._call(app_service.summary)

    @Slot(str, result=str)
    def searchMo(self, query: str) -> str:
        from src import app_service
        return self._call(app_service.search_mo, query)

    @Slot(str, result=str)
    def moDetail(self, oktmo: str) -> str:
        from src import app_service
        return self._call(app_service.mo_detail, oktmo)

    @Slot(str, result=str)
    def moForecast(self, oktmo: str) -> str:
        from src import app_service
        return self._call(app_service.mo_forecast, oktmo)

    @Slot(result=str)
    def modelMetrics(self) -> str:
        from src import app_service
        return self._call(app_service.model_metrics)

    @Slot(str, result=str)
    def worstMo(self, model: str) -> str:
        from src import app_service
        return self._call(app_service.worst_mo, model)

    @Slot(str, int, bool, result=str)
    def shocksOverview(self, shock_type: str, year: int, show_caution: bool) -> str:
        from src import app_service
        return self._call(app_service.shocks_overview, shock_type, year, show_caution)

    @Slot(result=str)
    def horizonsOverview(self) -> str:
        from src import app_service
        return self._call(app_service.horizons_overview)

    @Slot(result=str)
    def cpdOverview(self) -> str:
        from src import app_service
        return self._call(app_service.cpd_overview)

    @Slot(str, result=str)
    def cpdMo(self, oktmo: str) -> str:
        from src import app_service
        return self._call(app_service.cpd_mo, oktmo)

    @Slot(result=str)
    def dataStatus(self) -> str:
        from src import app_service
        return self._call(app_service.data_status, quiet=True)

    @Slot(str, result=str)
    def recalcStart(self, mode: str) -> str:
        from src import app_service
        return self._call(app_service.recalc_start, mode)

    @Slot(result=str)
    def reloadData(self) -> str:
        from src import app_service
        return self._call(app_service.reload)

    @Slot(result=str)
    def resultsOverview(self) -> str:
        from src import app_service
        return self._call(app_service.results_overview)

    # ---------------------------------------------------------- window chrome
    # Frameless window: the HTML topbar owns drag and the min/max/close buttons.

    @Slot()
    def winMinimize(self) -> None:
        if self.controller:
            _ui_log.info("window  minimize")
            self.controller.showMinimized()

    @Slot(result=bool)
    def winIsMaximized(self) -> bool:
        return bool(self.controller and self.controller.isMaximized())

    @Slot()
    def winToggleMax(self) -> None:
        if not self.controller:
            return
        if self.controller.isMaximized():
            _ui_log.info("window  restore")
            self.controller.showNormal()
        else:
            _ui_log.info("window  maximize")
            self.controller.showMaximized()
        self.windowStateChanged.emit(self.controller.isMaximized())

    @Slot()
    def winClose(self) -> None:
        if self.controller:
            _ui_log.info("window  close")
            self.controller.close()


    @Slot()
    def winStartDrag(self) -> None:
        """Hand off to the OS to move the window - one flag call, then Windows
        does the mouse tracking and redraw itself. Doing it by hand from JS
        makes the window smear on hi-DPI monitors.

        Maximised case: Windows only auto-restores a native-chrome window on
        drag; frameless ones stay pinned. We un-maximise ourselves via a
        native SC_RESTORE (Qt's showNormal + setGeometry loses the geometry
        request half the time on frameless windows) and reposition so the
        title bar lands under the cursor at the same proportional x - matches
        how Explorer, Chrome and Edge behave and keeps the drag continuous."""
        win = self.controller
        if not win:
            return
        if win.isMaximized():
            from PySide6.QtGui import QCursor
            cursor = QCursor.pos()
            max_w = max(win.width(), 1)
            rel_x = (cursor.x() - win.x()) / max_w

            # Snapshot normal geometry before restoring - once we restore, the
            # widget's own width/height will already be normal, so this value
            # is the one we want. Falls back to the current size if for some
            # reason we never had a normal state (shouldn't happen).
            norm = win.normalGeometry()
            new_w = norm.width() or 1200
            new_h = norm.height() or 800

            self._native_restore(win)

            new_x = int(cursor.x() - rel_x * new_w)
            # Keep the cursor near the top of the title bar (16 px in), so the
            # drag feels anchored to the same spot it grabbed.
            new_y = cursor.y() - 16
            win.setGeometry(new_x, new_y, new_w, new_h)

        wh = win.windowHandle()
        if wh:
            wh.startSystemMove()

    @staticmethod
    def _native_restore(win) -> None:
        """Un-maximise via the OS's own path, not Qt's.

        On Windows a frameless top-level treated with showNormal + setGeometry
        keeps reverting to the maximised bounds - Qt schedules the un-maximise
        for the next event tick but Windows still applies the maximised
        constraint to any geometry we set in the meantime. SC_RESTORE goes
        through the same code path Explorer uses when the user clicks the
        restore glyph, and it happens synchronously.

        On other platforms we fall back to Qt's showNormal, which works fine
        there because the compositor doesn't fight it."""
        import sys
        if sys.platform == "win32":
            import ctypes
            hwnd = int(win.winId())
            WM_SYSCOMMAND, SC_RESTORE = 0x0112, 0xF120
            ctypes.windll.user32.SendMessageW(hwnd, WM_SYSCOMMAND, SC_RESTORE, 0)
        else:
            win.showNormal()

    @Slot(int)
    def winStartResize(self, edges: int) -> None:
        """`edges` is a Qt.Edges bitmask sent from JS as an int."""
        if not self.controller:
            return
        wh = self.controller.windowHandle()
        if wh:
            wh.startSystemResize(Qt.Edges(edges))
