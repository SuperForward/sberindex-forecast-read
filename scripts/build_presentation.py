"""PDF-презентация из интерактивного лендинга (app/ui/landing.html).

Запуск: python -m scripts.build_presentation [--web dist/web] [--out docs/presentation.pdf]

Лендинг открывается из статической сборки (scripts/build_static.py; если её
нет – собирается) в безголовом Edge или Chrome и печатается в PDF: каждый
раздел – страница 16:9. Цифры те же, что в приложении и на сайте.
"""

import argparse
import functools
import http.server
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT  # noqa: E402

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser", "chrome",
]


def find_browser() -> str:
    for b in BROWSERS:
        p = b if os.path.isabs(b) else shutil.which(b)
        if p and Path(p).exists():
            return p
    raise SystemExit("не найден Edge или Chrome: установите один из них или откройте "
                     "лендинг в браузере и сохраните в PDF (Ctrl+P)")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    """Локальный сервер только на 127.0.0.1: fetch() не работает с file://."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                          functools.partial(_Quiet, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def _wait_file(path: Path, timeout: float = 120) -> None:
    """Edge на Windows передаёт работу своему процессу и выходит сразу:
    PDF появляется позже. Ждём, пока файл появится и перестанет расти."""
    t0, last = time.time(), -1
    while time.time() - t0 < timeout:
        size = path.stat().st_size if path.exists() else -1
        if size > 0 and size == last:
            return
        last = size
        time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="dist/web")
    ap.add_argument("--out", default="docs/presentation.pdf")
    a = ap.parse_args()
    web, out = ROOT / a.web, ROOT / a.out
    if not (web / "data" / "summary.json").exists():
        print("статической сборки нет – собираю (scripts.build_static)", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.build_static", "--out", a.web], cwd=ROOT, check=True)
    # лендинг берём свежий из app/ui: правки видны без пересборки всех данных
    shutil.copy2(ROOT / "app" / "ui" / "landing.html", web / "ui" / "landing.html")

    browser = find_browser()
    srv, port = serve(web)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdf = out.with_suffix(".part.pdf")
    profile = tempfile.mkdtemp(prefix="sberindex_pdf_")      # отдельный профиль: не трогаем браузер пользователя
    try:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={profile}", "--no-pdf-header-footer", "--window-size=1280,720",
               "--run-all-compositor-stages-before-draw", "--virtual-time-budget=30000",
               f"--print-to-pdf={tmp_pdf}", f"http://127.0.0.1:{port}/ui/landing.html?print=1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        _wait_file(tmp_pdf)
        if not tmp_pdf.exists() or tmp_pdf.stat().st_size < 10_000:
            raise SystemExit(f"браузер не создал PDF (код {r.returncode}): {r.stderr[-800:]}")
        os.replace(tmp_pdf, out)
    finally:
        srv.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        tmp_pdf.unlink(missing_ok=True)
    print(f"готово: {out.relative_to(ROOT)} – {out.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_presentation", "ui", main)
