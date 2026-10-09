"""Загрузка внешних открытых данных.

Запуск: python -m scripts.download_external [--only cbr,wikidata]

- cbr: курсы USD/EUR/CNY ЦБ РФ за 2018–2026 -> data/raw/external/cbr_<вал>.xml
- wikidata: координаты объектов с 8-значным ОКТМО -> wikidata_oktmo_coords.json,
  координаты адм. центров (11-значный код на «001») -> wikidata_oktmo_centres.csv
- weather: Open-Meteo archive, дневная погода 2019–2025 по точке на субъект ->
  weather/region_<код>.json (по МО не укладывается в бесплатный лимит API)
- nominatim: геокодер OSM для МО панели, оставшихся без координат ->
  nominatim_mo.csv (не чаще 1 запроса в секунду по правилам сервиса)
"""

import argparse
import json
import subprocess
import time
from datetime import date, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR  # noqa: E402
from src.data.external import (CBR_CODES, CBR_URL, EXT_DIR, NOMINATIM_URL,  # noqa: E402
                               OPEN_METEO_URL, WEATHER_VARS, WIKIDATA_CENTRES_QUERY,
                               WIKIDATA_QUERY, load_mo_coords, region_points)

UA = "sberindex-forecast-research/0.1"


def curl(args: list[str], out: Path) -> None:
    from src.io_util import write_if_changed
    r = subprocess.run(["curl", "-sS", "-f", "--max-time", "180", "-A", UA] + args, capture_output=True)
    if r.returncode:
        # источник недоступен (нет сети, сайт лежит) – штатная ситуация, не
        # авария: код 3, воркер отложит скачивание и продолжит на имеющихся данных
        print(f"источник недоступен, скачивание отложено: {out.name}: "
              f"{r.stderr.decode(errors='replace').strip()[:300]}", flush=True)
        raise SystemExit(3)
    changed = write_if_changed(out, r.stdout)
    print(f"{out.name}: {len(r.stdout) / 1024:.0f} КБ{'' if changed else ' (без изменений)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="cbr,wikidata,nominatim,weather")
    only = set(ap.parse_args().only.split(","))
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    if "cbr" in only:
        for cur, code in CBR_CODES.items():
            # до сегодняшнего дня: ряд берётся целиком, так появляются и новые дни
            url = CBR_URL.format(d1="01/01/2018", d2=f"{date.today():%d/%m/%Y}", code=code)
            curl([url], EXT_DIR / f"cbr_{cur.lower()}.xml")
    if "wikidata" in only:
        for query, name in [(WIKIDATA_QUERY, "wikidata_oktmo_coords.csv"),
                            (WIKIDATA_CENTRES_QUERY, "wikidata_oktmo_centres.csv")]:
            sparql_csv(query, EXT_DIR / name)
    if "nominatim" in only:
        geocode_missing()
    if "weather" in only:
        weather()


WEATHER_LAG_DAYS = 6      # архив Open-Meteo отстаёт от сегодня на ~5 дней
WEATHER_REFRESH_DAYS = 7  # чаще перекачивать нет смысла: погода – помесячные средние


def weather() -> None:
    # установка из снимка данных: сырых координат МО нет, а погода уже есть в
    # готовых данных – пропускаем без ошибки (курсы ЦБ при этом обновляются)
    if not (EXT_DIR / "wikidata_oktmo_coords.csv").exists():
        print("weather: нет координат МО (data/raw/external/wikidata_oktmo_coords.csv) – погода не обновляется; "
              "полная сборка исходников: python -m scripts.download_external")
        return
    out = EXT_DIR / "weather"
    out.mkdir(exist_ok=True)
    pts = region_points()
    end = date.today() - timedelta(days=WEATHER_LAG_DAYS)
    print(f"weather: {len(pts)} субъектов, до {end}")
    for r in pts.itertuples():
        path = out / f"region_{r.region_code}.json"
        if path.exists() and time.time() - path.stat().st_mtime < WEATHER_REFRESH_DAYS * 86400:
            continue
        curl(["-G", "--data-urlencode", f"latitude={r.lat:.4f}", "--data-urlencode", f"longitude={r.lon:.4f}",
              "--data-urlencode", "start_date=2019-01-01", "--data-urlencode", f"end_date={end}",
              "--data-urlencode", f"daily={','.join(WEATHER_VARS)}", "--data-urlencode", "timezone=auto",
              OPEN_METEO_URL], path)
        time.sleep(1)


def sparql_csv(query: str, out: Path, attempts: int = 3) -> None:
    """SPARQL -> CSV с проверкой: ответ Wikidata иногда обрывается на полуслове."""
    for i in range(attempts):
        curl(["-G", "-H", "Accept: text/csv", "--data-urlencode", f"query={query}",
              "https://query.wikidata.org/sparql"], out)
        text = out.read_text(encoding="utf-8").replace("\r\n", "\n")
        lines = text.rstrip("\n").split("\n")
        ncol = lines[0].count(",")
        if text.endswith("\n") and all(ln.count(",") == ncol for ln in lines[-50:]):
            print(f"  строк: {len(lines) - 1}")
            return
        print(f"  ответ оборван, попытка {i + 2}")
        time.sleep(10)
    raise RuntimeError(f"{out.name}: неполный ответ Wikidata")


def geocode_missing() -> None:
    """Nominatim для МО панели без координат: «<название МО>, <регион>»."""
    panel = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet").drop_duplicates("oktmo")
    have = set(load_mo_coords()["oktmo"])
    out = EXT_DIR / "nominatim_mo.csv"
    done = pd.read_csv(out, dtype={"oktmo": str}) if out.exists() else pd.DataFrame(columns=["oktmo"])
    todo = panel[~panel["oktmo"].isin(have) & ~panel["oktmo"].isin(done["oktmo"])]
    print(f"nominatim: к геокодированию {len(todo)} МО")
    rows = []
    for r in todo.itertuples():
        name = r.mo_name.replace("внутригородская территория города федерального значения", "").strip()
        q = f"{name}, {r.region}"
        res = subprocess.run(["curl", "-sS", "-G", "--max-time", "30", "-A", UA,
                              "--data-urlencode", f"q={q}", "--data-urlencode", "format=json",
                              "--data-urlencode", "limit=1", "--data-urlencode", "countrycodes=ru",
                              NOMINATIM_URL], capture_output=True)
        hit = json.loads(res.stdout or b"[]") if res.returncode == 0 else []
        rows.append({"oktmo": r.oktmo, "query": q,
                     "lat": float(hit[0]["lat"]) if hit else None,
                     "lon": float(hit[0]["lon"]) if hit else None,
                     "display_name": hit[0]["display_name"] if hit else None})
        time.sleep(1.1)
    pd.concat([done, pd.DataFrame(rows)]).to_csv(out, index=False)
    print(f"nominatim: найдено {sum(r['lat'] is not None for r in rows)} из {len(rows)}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("download_external", "data", main)
