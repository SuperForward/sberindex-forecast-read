import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PySide6 и app.web_window импортируются в main() после проверки второго
# экземпляра – повторному запуску они не нужны.
from app.logging_setup import setup_logging, get as _get_log

# Explicit Windows AppUserModelID. Without one, Explorer groups our window
# under whichever launcher started it (pythonw.exe or wscript.exe) and shows
# that launcher's icon on the taskbar. Setting our own ID before QApplication
# fires makes the taskbar pick up the window icon we set below.
APP_ID = "sberindex.terminal"


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass  # non-fatal - only affects grouping / icon on the taskbar


# Имя локального сокета единственного экземпляра. Второй запуск (ярлык,
# автозапуск) стучится сюда и завершается, а первый разворачивает окно –
# даже если оно спрятано в трей.
_INSTANCE_SERVER = f"{APP_ID}.single-instance"


def _signal_running_instance(log) -> bool:
    """True, если уже запущенный экземпляр принял просьбу показать окно.

    Без Qt: QLocalServer на Windows – это named pipe, клиенту хватает open().
    Проверка стоит миллисекунды и идёт до импорта PySide6 и app.web_window
    (~1.5 с на прогретом кэше, на холодном дольше) – раньше второй запуск
    сначала грузил всё приложение и только потом стучался в первый.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    path = "\\\\.\\pipe\\" + _INSTANCE_SERVER
    for _ in range(10):
        try:
            # Второй процесс запущен пользователем и владеет правом на
            # передний план; без передачи этого права Windows не даст первому
            # процессу вытащить окно – только мигнёт кнопкой на панели задач.
            ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
            cmd = b"quit" if "--quit" in sys.argv else b"show"
            with open(path, "wb", buffering=0) as pipe:
                pipe.write(cmd)
            log.info("single_instance  already running - sent %s, exiting", cmd.decode())
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            # ERROR_PIPE_BUSY: экземпляр пайпа занят предыдущим клиентом,
            # сервер поднимет новый за миллисекунды.
            if getattr(e, "winerror", None) != 231:
                log.warning("single_instance  pipe error: %s", e)
                return False
            ctypes.windll.kernel32.WaitNamedPipeW(path, 200)
    log.warning("single_instance  pipe busy - starting anyway")
    return False


def _listen_for_instances(window, log):
    from PySide6.QtNetwork import QLocalServer

    server = QLocalServer()
    # Сокет от упавшего процесса мешает listen() – снять перед запуском.
    QLocalServer.removeServer(_INSTANCE_SERVER)
    if not server.listen(_INSTANCE_SERVER):
        log.warning("single_instance  listen failed: %s", server.errorString())
        return server

    def _on_conn():
        conn = server.nextPendingConnection()
        if conn is None:
            return

        # Команда приходит сразу за подключением: «show» – развернуть окно,
        # «quit» – штатно закрыться (python -m app.main --quit). Штатно – это
        # важно: при снятии процесса Windows не успевает убрать значок из
        # трея, и там копятся «мёртвые» значки до наведения мыши.
        def _on_data():
            msg = bytes(conn.readAll()).strip()
            if msg == b"quit":
                log.info("single_instance  quit requested by second launch")
                window.quit_application()
            else:
                window._restore()
                log.info("single_instance  second launch - window restored")
                window._log_tray_action("повторный запуск – окно развёрнуто")

        conn.readyRead.connect(_on_data)
        conn.disconnected.connect(conn.deleteLater)
        if conn.bytesAvailable():        # команда пришла раньше подключения обработчика
            _on_data()

    server.newConnection.connect(_on_conn)
    return server


def _software_rendering(log) -> None:
    """Отрисовка без видеокарты: виртуальные машины, удалённый рабочий стол,
    старые драйверы – там QtWebEngine часто даёт чёрное или пустое окно.
    Включается флагом --software или сама, если это сеанс RDP."""
    import os
    rdp = False
    if sys.platform == "win32":
        try:
            import ctypes
            rdp = bool(ctypes.windll.user32.GetSystemMetrics(0x1000))  # SM_REMOTESESSION
        except Exception as e:  # noqa: BLE001
            log.warning("software_rendering  проверка RDP не удалась: %s", e)
    if "--software" not in sys.argv and not rdp:
        return
    from PySide6.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    os.environ["QT_OPENGL"] = "software"
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (flags + " --disable-gpu --disable-gpu-compositing").strip()
    log.info("software_rendering  %s", "сеанс удалённого рабочего стола" if rdp else "флаг --software")


def main() -> None:
    t0 = time.perf_counter()
    profile = "--profile" in sys.argv
    setup_logging(profile=profile)
    chart_log = _get_log("chart")
    chart_log.info("engine_start  argv=%s profile=%s", sys.argv[1:], profile)
    if _signal_running_instance(chart_log):
        sys.exit(0)
    if "--quit" in sys.argv:          # закрывать нечего – не запускаемся
        chart_log.info("quit_requested  приложение не запущено")
        sys.exit(0)

    from PySide6.QtWidgets import QApplication
    from app.web_window import TerminalWindow, make_icon, _perf

    _perf("app_start")
    _set_windows_app_id()
    _software_rendering(chart_log)
    if "--debug" in sys.argv:
        import os
        os.environ.setdefault("QTWEBENGINE_REMOTE_DEBUGGING", "9222")
        chart_log.info("remote_debug_enabled  port=9222")
    app = QApplication(sys.argv)
    _perf("qt_app_created", t0)
    app.setApplicationName("SberIndex Terminal")
    app.setWindowIcon(make_icon())
    app.setQuitOnLastWindowClosed(False)

    window = TerminalWindow()
    _instance_server = _listen_for_instances(window, chart_log)  # noqa: F841 - держим ссылку
    _perf("window_created", t0)
    window.showMaximized()
    _perf("show_maximized", t0)
    import logging
    from app.logging_setup import timed as _timed
    _timed(chart_log, logging.INFO, "engine_ready", (time.perf_counter() - t0) * 1000)
    code = app.exec()
    if window.worker_lingering:
        # Фоновый поток ещё в сетевом вызове: обычный выход уничтожил бы
        # живой QThread (qFatal). Сбрасываем логи и выходим напрямую.
        import os
        logging.shutdown()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
