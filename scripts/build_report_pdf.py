"""PDF методологического отчёта: docs/report.md -> docs/report.pdf (A4).

Запуск: python -m scripts.build_report_pdf

Markdown переводится в HTML (markdown-it-py, таблицы включены), картинки
берутся из docs/figures (scripts/build_report_figures.py), печать – тем же
безголовым Edge/Chrome, что и презентация (scripts/build_presentation.py).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_it import MarkdownIt  # noqa: E402

from scripts.build_presentation import _wait_file, find_browser, serve  # noqa: E402
from src.config import ROOT  # noqa: E402

DOCS = ROOT / "docs"

CSS = """
@page{size:A4;margin:16mm 16mm 18mm}
*{box-sizing:border-box}
body{font-family:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;color:#15191e;font-size:10.5pt;line-height:1.5;margin:0}
h1{font-size:21pt;line-height:1.15;margin:0 0 6pt;color:#15191e}
h2{font-size:14.5pt;margin:18pt 0 6pt;padding-top:6pt;border-top:2px solid #21a038;break-after:avoid}
h3{font-size:11.5pt;margin:12pt 0 4pt;break-after:avoid}
p,li{orphans:3;widows:3}
ul,ol{padding-left:18pt;margin:4pt 0 8pt}
li{margin:0 0 3pt}
hr{display:none}
code{font-family:Consolas,"Cascadia Mono",monospace;font-size:9pt;background:#f1f3f0;padding:0 3px;border-radius:3px}
pre{background:#f1f3f0;padding:8pt;border-radius:6px;overflow:hidden;white-space:pre-wrap}
pre code{background:none;padding:0}
table{border-collapse:collapse;width:100%;margin:6pt 0 10pt;font-size:9.2pt;break-inside:avoid}
th,td{border-bottom:1px solid #dfe3dd;padding:3.5pt 5pt;text-align:left;vertical-align:top}
th{background:#f4f6f3;font-weight:600}
td:not(:first-child){font-variant-numeric:tabular-nums}
img{max-width:100%;display:block;margin:6pt auto 2pt;break-inside:avoid}
p:has(> img){break-inside:avoid;margin:0}
em{color:#555}
strong{font-weight:650}
.cover{color:#6b7480;margin:0 0 10pt}
"""


def render_html(md_text: str) -> str:
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")
    body = md.render(md_text)
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Методологический отчёт</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main() -> None:
    src, out = DOCS / "report.md", DOCS / "report.pdf"
    page = DOCS / "_report_print.html"          # рядом с report.md: пути к figures/ те же
    page.write_text(render_html(src.read_text(encoding="utf-8")), encoding="utf-8")
    srv, port = serve(DOCS)
    tmp = out.with_suffix(".part.pdf")
    profile = tempfile.mkdtemp(prefix="sberindex_pdf_")
    try:
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-first-run",
                        "--no-default-browser-check", f"--user-data-dir={profile}", "--no-pdf-header-footer",
                        "--virtual-time-budget=10000", f"--print-to-pdf={tmp}",
                        f"http://127.0.0.1:{port}/{page.name}"], capture_output=True, timeout=180)
        _wait_file(tmp)
        if not tmp.exists() or tmp.stat().st_size < 10_000:
            raise SystemExit("браузер не создал PDF")
        os.replace(tmp, out)
    finally:
        srv.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        tmp.unlink(missing_ok=True)
        page.unlink(missing_ok=True)
    print(f"готово: {out.relative_to(ROOT)} – {out.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_report_pdf", "data", main)
