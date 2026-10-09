"""Логирование запусков скриптов (scripts/*.py).

run_logged(name, logger, main): старт с аргументами и версией кода (git),
весь вывод print() – и в консоль, и в лог, длительность, исключение с
трассировкой (попадает и в errors.log). Логи – в logs/, как у приложения.
"""

import logging
import subprocess
import sys
import time

from src.config import NO_WINDOW, ROOT


def _git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, timeout=5, creationflags=NO_WINDOW)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, timeout=5, creationflags=NO_WINDOW).stdout.strip()
        return r.stdout.strip() + ("+изменения" if dirty else "")
    except Exception:  # noqa: BLE001
        return "?"


class _Tee:
    """Строки потока – в консоль как есть и в логгер построчно.

    stderr (err=True): предупреждения библиотек (lightgbm, sklearn, torch) –
    строки с warning/error идут как WARNING, остальное как INFO. Индикаторы
    прогресса перерисовывают строку через \r – в лог идёт только итог."""

    def __init__(self, stream, log: logging.Logger, prefix: str, err: bool = False) -> None:
        self._stream, self._log, self._prefix, self._buf, self._err = stream, log, prefix, "", err

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.split("\r")[-1].rstrip()
            if not line.strip():
                continue
            if self._err:
                low = line.lower()
                lvl = logging.WARNING if ("warning" in low or "error" in low) else logging.INFO
                # в консоли строка уже есть – запись лога туда не дублируем
                self._log.log(lvl, "%s  stderr: %s", self._prefix, line, extra={"from_stream": True})
            else:
                self._log.info("%s  %s", self._prefix, line)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()


def run_logged(name: str, logger_name: str, main) -> None:
    sys.path.insert(0, str(ROOT))
    from app.logging_setup import setup_logging

    setup_logging()
    log = logging.getLogger(logger_name)
    log.info("script_start  %s argv=%s git=%s", name, sys.argv[1:], _git_rev())
    t0 = time.perf_counter()
    orig, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig, log, name)
    sys.stderr = _Tee(orig_err, log, name, err=True)
    try:
        main()
    except BaseException as e:
        elapsed = time.perf_counter() - t0
        if isinstance(e, SystemExit):
            # штатный выход скрипта (например, «источник недоступен») – не авария
            (log.info if e.code in (0, None) else log.warning)(
                "script_exit  %s код=%s после %.1f с", name, e.code, elapsed)
            raise
        if isinstance(e, KeyboardInterrupt):
            log.warning("script_interrupted  %s после %.1f с", name, elapsed)
        else:
            log.exception("script_failed  %s после %.1f с: %s", name, elapsed, e)
        raise
    finally:
        if isinstance(sys.stdout, _Tee):
            sys.stdout.flush()
            sys.stdout = orig
        if isinstance(sys.stderr, _Tee):
            sys.stderr.flush()
            sys.stderr = orig_err
    log.info("script_done  %s за %.1f с", name, time.perf_counter() - t0,
             extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})
