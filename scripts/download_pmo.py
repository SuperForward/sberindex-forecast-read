"""Загрузка годовых показателей по МО из БД ПМО Росстата.

Запуск: python -m scripts.download_pmo [--bases munst01,munst22] [--workers 2]

По каждому субъекту и показателю из configs/pmo_indicators.yaml сохраняет
data/raw/rosstat/pmo/<база>/<код>.csv (сырой CSV Росстата) и <код>_ref.csv
(коды ОКТМО и подписи МО). Уже скачанное пропускает – можно перезапускать.
Ошибки пишет в data/raw/rosstat/pmo/errors.log и идёт дальше.
"""

import argparse
import logging
import sys
import os
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data import pmo  # noqa: E402


REFRESH_DAYS = 25   # файл старше – перекачать: Росстат добавляет новые годы и уточняет прошлые


def job(base: str, code: str, y0: int, y1: int) -> str:
    out = pmo.PMO_DIR / base / f"{code}.csv"
    if out.exists() and out.stat().st_size > 0 and time.time() - out.stat().st_mtime < REFRESH_DAYS * 86400:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            raw, ref = pmo.fetch_indicator(base, code, y0, y1)
            from src.io_util import write_if_changed
            write_if_changed(out.with_name(f"{code}_ref.csv"), ref.to_csv(index=False).encode("utf-8"))
            if not write_if_changed(out, raw):
                os.utime(out)          # данные те же, но проверены – свежие для REFRESH_DAYS
            return "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(5 * (attempt + 1))
    with open(pmo.PMO_DIR / "errors.log", "a", encoding="utf-8") as f:
        f.write(f"{base}\t{code}\t{err}\n")
    logging.getLogger("data").warning("pmo_fetch_failed  %s/%s: %s", base, code, err)
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    ok, why = pmo.available()
    if not ok:
        print(f"БД ПМО недоступна, скачивание отложено: {why}", flush=True)
        raise SystemExit(3)
    cfg = load_config("pmo_indicators")
    y0, y1 = cfg["years"]
    # по текущий год: сервер сам ограничит диапазон доступными годами базы
    y1 = y1 or date.today().year
    codes = [str(c) for group in cfg["indicators"].values() for c in group]
    bases = args.bases.split(",") if args.bases else pmo.list_bases()
    tasks = [(b, c) for b in bases for c in codes]
    print(f"баз: {len(bases)}, показателей: {len(codes)}, задач: {len(tasks)}", flush=True)
    stats = {"ok": 0, "skip": 0, "error": 0}
    t = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(job, b, c, y0, y1): (b, c) for b, c in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            stats[f.result()] += 1
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {stats} {time.time() - t:.0f} с", flush=True)


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("download_pmo", "data", main)
