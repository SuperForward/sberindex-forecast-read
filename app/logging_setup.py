"""Unified logging for the terminal.

Loggers are named after the concern they cover:

    chart, data, perf, ui, forecast, cpd, news, worker

Each writes to its own file under `logs/`. All records also aggregate
into `session.log`; ERROR+ records also aggregate into `errors.log`.

Every record carries an optional `elapsed_ms` extra; the formatter
renders it as `[N.Nms]` or `[-]` if absent. Callers use
``log.info("msg", extra={"elapsed_ms": 3.2})`` for timing lines.

Verbose per-frame / per-tick messages are emitted at DEBUG. In normal
mode the file handlers stay at INFO so they never touch disk. Pass
``profile=True`` (``--profile`` on the CLI) to enable DEBUG.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_FILES: dict[str, str] = {
    "chart":    "chart.log",
    "data":     "data.log",
    "perf":     "performance.log",
    "ui":       "ui.log",
    "forecast": "forecast.log",
    "cpd":      "cpd.log",
    "news":     "news.log",
    "worker":   "worker.log",
}

_FMT = "[%(asctime)s.%(msecs)03d][%(levelname)s][%(module_label)s][%(caller_func)s][%(elapsed_disp)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 5

_configured = False


class _DefaultsFilter(logging.Filter):
    """Guarantee optional format fields exist on every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ms = getattr(record, "elapsed_ms", None)
        record.elapsed_disp = f"{ms:.1f}ms" if isinstance(ms, (int, float)) else "-"
        record.module_label = getattr(record, "module_label", record.name)
        record.caller_func = getattr(record, "caller_func", record.funcName)
        return True


class _RerouteRoot(logging.Filter):
    """Сторонние библиотеки, которые пишут прямо в корневой логгер
    (logging.info(...) – так делает TimesFM), – в логгер своей темы: иначе их
    записи попадают только в session.log с меткой root и без своего файла."""

    ROUTES = {"timesfm": "forecast", "chronos": "forecast"}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "root":
            return True
        path = record.pathname.replace(chr(92), "/")
        for pkg, target in self.ROUTES.items():
            if f"/{pkg}/" in path:
                record.name = target
                logging.getLogger(target).handle(record)
                return False
        return True


class _QuietDisconnect(logging.Filter):
    """asyncio на Windows пишет ERROR с traceback, когда браузер закрывает
    соединение с веб-сервером раньше, чем сервер его закрыл сам
    (_call_connection_lost, WinError 10022/10054). Данные при этом не
    теряются, а errors.log забивается. Такие записи понижаются до DEBUG."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            try:
                text = record.getMessage()
            except Exception:  # noqa: BLE001  кривые аргументы записи – не наше дело, пропускаем как есть
                return True
            if "_call_connection_lost" in text:
                record.levelno, record.levelname = logging.DEBUG, "DEBUG"
        return True


class _ErrorsOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


class _DedupFilter(logging.Filter):
    """Схлопывает повторы одной и той же ошибки в errors.log.

    Файл существует ради одного: заглянуть в него и увидеть, что сломалось.
    Он это переставал делать. Один обрыв вебсокета, логируемый через
    exception(), занял 920 записей с полным traceback из ~920 – то есть
    2.2 МБ файла, в котором больше ничего не найти.

    Первое появление ошибки проходит целиком. Дальше в течение окна
    повторы считаются молча, и следующая запись выходит с хвостом
    «повторилось N раз за M с». Ключ – логгер, шаблон сообщения и уровень:
    именно шаблон, а не готовая строка, иначе одна и та же ошибка с
    разными символами в подстановке считалась бы разными.
    """

    def __init__(self, window_sec: float = 300.0) -> None:
        super().__init__()
        self._window = window_sec
        self._seen: dict[tuple, list] = {}   # key -> [last_emit_ts, suppressed]

    def filter(self, record: logging.LogRecord) -> bool:
        import time as _time
        key = (record.name, str(record.msg), record.levelno)
        now = _time.monotonic()
        state = self._seen.get(key)
        if state is None:
            self._seen[key] = [now, 0]
            return True
        last, suppressed = state
        if now - last < self._window:
            state[1] = suppressed + 1
            return False
        if suppressed:
            # Хвост дописывается к шаблону, а не к готовой строке: аргументы
            # подставляются позже, и порядок %-плейсхолдеров ломать нельзя.
            record.msg = f"{record.msg}   [повторилось {suppressed} раз за "                         f"{now - last:.0f} с, записи подавлены]"
        self._seen[key] = [now, 0]
        return True


def _make_rotating(path: Path, level: int) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    h.addFilter(_DefaultsFilter())
    return h


def setup_logging(profile: bool = False, log_dir: Path | None = None) -> Path:
    """Idempotent. Returns the log directory used."""
    global _configured
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    if _configured:
        return log_dir

    file_level = logging.DEBUG if profile else logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    for f in list(root.filters):
        root.removeFilter(f)
    root.addFilter(_RerouteRoot())
    logging.getLogger("asyncio").addFilter(_QuietDisconnect())

    session = _make_rotating(log_dir / "session.log", file_level)
    root.addHandler(session)

    errors = _make_rotating(log_dir / "errors.log", logging.ERROR)
    errors.addFilter(_ErrorsOnly())
    errors.addFilter(_DedupFilter())
    root.addHandler(errors)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    console.addFilter(_DefaultsFilter())
    # строки, перехваченные из stderr скрипта (src/runlog.py), в консоли уже есть
    console.addFilter(lambda r: not getattr(r, "from_stream", False))
    root.addHandler(console)

    for name, filename in LOG_FILES.items():
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.addHandler(_make_rotating(log_dir / filename, file_level))
        lg.propagate = True

    boot = logging.getLogger("chart")
    boot.info("logging_initialised  profile=%s dir=%s", profile, log_dir)
    _configured = True
    return log_dir


def get(name: str) -> logging.Logger:
    """Shorthand: ``get("ws")``. Falls back to a fresh logger if the caller
    picks a name outside LOG_FILES (still captured by session.log)."""
    return logging.getLogger(name)


def timed(logger: logging.Logger, level: int, msg: str, elapsed_ms: float, *args) -> None:
    """Log ``msg % args`` at ``level`` with elapsed formatted into the header."""
    logger.log(level, msg, *args, extra={"elapsed_ms": elapsed_ms})
