"""Архив новостей Lenta.ru по дням (configs/news.yaml).

Запуск: python -m scripts.download_news [--from 2024-04-01] [--to 2024-04-30]

По месяцу – файл data/raw/news/lenta_ГГГГ-ММ.parquet (заголовок, время,
рубрика, ссылка). Готовые прошлые месяцы не перекачиваются; последние
refetch_days дней – перекачиваются (за день новости дописываются).
Между запросами – пауза pause_s: сайт чужой, нагружать его нельзя.
"""

import argparse
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import RAW_DIR, load_config  # noqa: E402
from src.news import lenta  # noqa: E402

OUT = RAW_DIR / "news"
UA = "Mozilla/5.0 (research project; sberindex-forecast news archive)"


def _get(url: str) -> str | None:
    for attempt in range(3):
        r = subprocess.run(["curl", "-sS", "-f", "-L", "-A", UA, "--max-time", "60", url], capture_output=True)
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace")
        if b"404" in r.stderr:
            return None
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: {r.stderr.decode(errors='replace').strip()}")


def day(d: date, rubric: str, pause: float) -> pd.DataFrame:
    frames, n = [], 1
    while n <= 20:
        url = f"https://lenta.ru/rubrics/{rubric}/{d:%Y/%m/%d}/" + (f"page/{n}/" if n > 1 else "")
        page = _get(url)
        time.sleep(pause)
        if page is None:
            break
        df = lenta.parse(page, rubric)
        frames.append(df[df["published"].dt.date == d])
        if not lenta.has_next(page, n) or df.empty:
            break
        n += 1
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["published", "rubric", "title", "url"])


def main() -> None:
    cfg = load_config("news")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default=cfg["date_from"])
    ap.add_argument("--to", dest="d1", default=str(date.today()))
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.d0), date.fromisoformat(a.d1)
    fresh_from = date.today() - timedelta(days=cfg["refetch_days"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0, total = time.time(), 0
    month = date(d0.year, d0.month, 1)
    while month <= d1:
        f = OUT / f"lenta_{month:%Y-%m}.parquet"
        nxt = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["published", "rubric", "title", "url"])
        have = set(pd.to_datetime(old["published"]).dt.date) if len(old) else set()
        days = [month + timedelta(i) for i in range((nxt - month).days)]
        todo = [x for x in days if d0 <= x <= d1 and (x not in have or x >= fresh_from)]
        if todo:
            parts = [old[~pd.to_datetime(old["published"]).dt.date.isin(todo)]] if len(old) else []
            for x in todo:
                for rub in cfg["rubrics"]:
                    parts.append(day(x, rub, cfg["pause_s"]))
            df = pd.concat(parts, ignore_index=True).drop_duplicates("url").sort_values("published")
            df.to_parquet(f, index=False)
            total += len(df) - len(old)
            print(f"{month:%Y-%m}: {len(df)} новостей (+{len(df) - len(old)}), {time.time() - t0:.0f} с", flush=True)
        month = nxt
    print(f"готово: +{total} новостей за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("download_news", "news", main)
