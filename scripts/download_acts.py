"""Официальные акты о чрезвычайных ситуациях с портала publication.pravo.gov.ru.

Запуск: python -m scripts.download_acts

Портал официального опубликования правовых актов, открытый API. Берём все
документы, в названии которых есть «чрезвычайной ситуации»: введение и отмена
режима ЧС регионами и муниципалитетами, выплаты пострадавшим, восстановление.
Фильтр по дате API игнорирует, поэтому скачиваем всё (≈2,5 тыс. документов,
26 страниц) и режем по дате при сборке (scripts/build_acts.py). Хранятся только
реквизиты: название, орган, даты, номер – без текстов.

Файл data/raw/acts/pravo_chs.json перезаписывается, только если изменился.
Сеть недоступна – код 3: воркер отложит шаг и продолжит на имеющихся данных.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR  # noqa: E402
from src.io_util import write_if_changed  # noqa: E402

API = "http://publication.pravo.gov.ru/api/Documents?"
QUERY = "чрезвычайной ситуации"
PAGE = 100
OUT = RAW_DIR / "acts" / "pravo_chs.json"
UA = "Mozilla/5.0 (research project; sberindex-forecast official acts)"
KEEP = ["id", "eoNumber", "name", "complexName", "number", "documentDate", "publishDateShort",
        "signatoryAuthorityId", "documentTypeId"]


def page(i: int) -> dict:
    url = API + urllib.parse.urlencode({"PageSize": PAGE, "Index": i, "Name": QUERY})
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


def main() -> None:
    try:
        first = page(1)
    except Exception as e:
        print(f"портал недоступен, скачивание отложено: {e}")
        sys.exit(3)
    items = list(first.get("items", []))
    pages = first.get("pagesTotalCount") or 1
    for i in range(2, pages + 1):
        items += page(i).get("items", [])
        time.sleep(0.5)                          # сайт государственный, не нагружаем
    items = [{k: it.get(k) for k in KEEP} for it in items]
    items.sort(key=lambda it: (it["documentDate"] or "", it["id"]))
    data = json.dumps(items, ensure_ascii=False, indent=0).encode("utf-8")
    changed = write_if_changed(OUT, data)
    print(f"актов: {len(items)} ({'обновлено' if changed else 'без изменений'}) -> {OUT}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("download_acts", "data", main)
