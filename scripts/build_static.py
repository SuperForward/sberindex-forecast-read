"""Статическая веб-версия для GitHub Pages: интерфейс + готовые ответы в JSON.

Запуск: python -m scripts.build_static [--out dist/web]

Сервер в интернете не нужен: браузер берёт ответы из файлов data/*.json.
Только чтение – кнопок пересчёта нет, ломать нечего. Данные – те же, что в
приложении (опубликованная база data/sberindex.duckdb).

Структура:
  index.html            -> переадресация на ui/landing.html (итоги; оттуда – терминал)
  ui/terminal.html      (с флагом SBERINDEX_MODE='static')
  assets/               (ECharts, иконки)
  data/summary.json, models.json, …, mo/<ОКТМО>.json, cpd/<ОКТМО>.json
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import app_service as svc  # noqa: E402
from src.config import ROOT  # noqa: E402


def _dump(path: Path, data) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(body, encoding="utf-8")
    return len(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/web")
    out = ROOT / ap.parse_args().out
    t0 = time.time()
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "app" / "ui", out / "ui")
    shutil.copytree(ROOT / "app" / "assets", out / "assets",
                    ignore=shutil.ignore_patterns("*.wav", "*_preview.png"))   # звуки и превью вебу не нужны
    page = (out / "ui" / "terminal.html").read_text(encoding="utf-8")
    page = page.replace("<script>", "<script>window.SBERINDEX_MODE='static';</script>\n<script>", 1)
    (out / "ui" / "terminal.html").write_text(page, encoding="utf-8")
    # главная – лендинг с итогами; из него ссылка на терминал
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" '
                                    'content="0; url=ui/landing.html"><title>SberIndex</title>'
                                    '<link rel="icon" href="assets/app.svg">'
                                    '<a href="ui/landing.html">SberIndex: итоги</a>', encoding="utf-8")
    shutil.copy2(out / "ui" / "404.html", out / "404.html")       # GitHub Pages отдаёт его по любому неизвестному адресу
    (out / ".nojekyll").write_text("", encoding="utf-8")          # GitHub Pages: отдавать файлы как есть
    # отчёт и презентация в PDF – на них ссылается подвал лендинга (../docs/*.pdf)
    (out / "docs").mkdir()
    for name in ("report.pdf", "presentation.pdf"):
        if (ROOT / "docs" / name).exists():
            shutil.copy2(ROOT / "docs" / name, out / "docs" / name)

    d = out / "data"
    size = 0
    size += _dump(d / "summary.json", svc.summary())
    size += _dump(d / "models.json", svc.model_metrics())
    size += _dump(d / "models_worst.json", svc.worst_mo(""))
    size += _dump(d / "horizons.json", svc.horizons_overview())
    size += _dump(d / "cpd.json", svc.cpd_overview())
    size += _dump(d / "results.json", svc.results_overview())
    nw = svc.news_overview()
    if not nw.get("error"):
        size += _dump(d / "news.json", nw)
    st = svc.data_status()
    st.update(busy=False, current=None, read_only=True, steps=[], runs=st["runs"][:3])
    size += _dump(d / "status.json", st)

    # шоки: все сочетания фильтров раздела
    years = [0] + sorted(int(y) for y in svc.shocks_overview("", 0, True)["counts"])
    for t in ["", "fiscal_budget", "production_local_market", "socio_demographic"]:
        for y in years:
            for c in (False, True):
                size += _dump(d / "shocks" / f"{t or 'all'}_{y}_{1 if c else 0}.json", svc.shocks_overview(t, y, c))

    # выгрузки CSV (кроме прогноза одного МО – он собирается из карточки)
    for kind in ("forecast", "forecast_all_models", "metrics"):
        size += _dump(d / f"export_{kind}.json", svc.export_csv(kind))

    # индекс для поиска в браузере + карточки и прогнозы всех МО
    mo = svc._mo_table()
    index = [{k: svc._clean(v) for k, v in r.items()} for r in
             mo[["oktmo", "mo_name", "region", "region_short", "pop", "spend_last", "growth_last", "year", "months"]]
             .to_dict("records")]
    size += _dump(d / "mo_index.json", index)
    for i, o in enumerate(mo["oktmo"], 1):
        size += _dump(d / "mo" / f"{o}.json", svc.mo_detail(o))
        size += _dump(d / "mo" / f"{o}_forecast.json", svc.mo_forecast(o))
        c = svc.cpd_mo(o)
        if not c.get("error"):
            size += _dump(d / "cpd" / f"{o}.json", c)
        if i % 500 == 0:
            print(f"  МО: {i}/{len(mo)}", flush=True)
    files = sum(1 for _ in out.rglob("*") if _.is_file())
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"готово: {out.relative_to(ROOT)} – файлов {files}, {total / 1e6:.1f} МБ (данные {size / 1e6:.1f} МБ), "
          f"{time.time() - t0:.0f} с")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_static", "data", main)
