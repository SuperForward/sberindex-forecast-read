"""События GDELT 2.0 (поток translation) по России (configs/news.yaml, gdelt).

Запуск: python -m scripts.download_gdelt [--from 2023-01-01] [--to 2024-12-31]

По месяцу – файл data/raw/gdelt/gdelt_ГГГГ-ММ.parquet: события с местом
действия в России (src/news/gdelt.py). Выгрузка – каждые 15 минут, ~60 КБ;
скачивается и сразу фильтруется, целиком не хранится. Какие выгрузки уже
разобраны – data/.cache/gdelt_done.txt (служебный файл не в data/raw:
иначе меняются «исходные данные»); повторный запуск докачивает только новые.
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import DATA_DIR, RAW_DIR, load_config  # noqa: E402
from src.news import gdelt  # noqa: E402

OUT = RAW_DIR / "gdelt"
DONE = DATA_DIR / ".cache" / "gdelt_done.txt"
UA = "sberindex-forecast research (GDELT events for Russia)"


def _get(url: str, timeout: int = 120) -> bytes | None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:                   # в GDELT бывают пропущенные 15-минутки
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: {err}")


def _one(item: tuple[str, str]) -> tuple[str, pd.DataFrame | None]:
    stamp, url = item
    raw = _get(url)
    if raw is None:
        return stamp, None
    try:
        return stamp, gdelt.parse_export(raw)
    except Exception as e:                      # битый zip – пропускаем, но не помечаем разобранным
        print(f"{stamp}: {e}", flush=True)
        return stamp, "error"


def main() -> None:
    cfg = load_config("news").get("gdelt", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default=cfg.get("date_from", "2023-01-01"))
    ap.add_argument("--to", dest="d1", default=str(date.today()))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DONE.parent.mkdir(parents=True, exist_ok=True)
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    master = _get(gdelt.MASTER, timeout=600).decode("utf-8", errors="replace")
    todo = [x for x in gdelt.export_urls(master, a.d0, a.d1) if x[0] not in done]
    print(f"выгрузок к разбору: {len(todo)} (уже разобрано {len(done)})", flush=True)
    t0 = time.time()
    by_month: dict[str, list[str]] = {}
    for stamp, _ in todo:
        by_month.setdefault(f"{stamp[:4]}-{stamp[4:6]}", []).append(stamp)
    urls = dict(todo)
    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 8))) as pool:
        for month, stamps in sorted(by_month.items()):
            res = list(pool.map(_one, [(s, urls[s]) for s in stamps]))
            parts = [d for _, d in res if isinstance(d, pd.DataFrame) and len(d)]
            f = OUT / f"gdelt_{month}.parquet"
            if parts:
                new = pd.concat(parts, ignore_index=True)
                old = pd.read_parquet(f) if f.exists() else None
                df = pd.concat([old, new], ignore_index=True) if old is not None else new
                df = df.drop_duplicates("event_id").sort_values("date_added")
                df.to_parquet(f, index=False)
            ok = [s for s, d in res if not isinstance(d, str)]
            with DONE.open("a") as fh:
                fh.write("".join(f"{s}\n" for s in ok))
            n = sum(len(d) for d in parts)
            print(f"{month}: выгрузок {len(stamps)}, событий в России +{n}, {time.time() - t0:.0f} с", flush=True)
    print(f"готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("download_gdelt", "news", main)
