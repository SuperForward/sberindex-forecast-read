"""Официальные акты о ЧС -> таблицы для анализа (src/news/acts.py).

Запуск: python -m scripts.build_acts

Вход: data/raw/acts/pravo_chs.json (scripts/download_acts.py).
Выход:
  data/processed/acts.parquet                 – акт: дата, вид, регион, МО, причина;
  data/processed/acts_region_monthly.parquet  – регион × месяц: введено ЧС,
                                                действует режим, акты о поддержке.
Акт «существует» с даты документа: для решения на месяц t используются только
акты, подписанные до конца месяца t.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.news.acts import parse, region_monthly  # noqa: E402

RAW = RAW_DIR / "acts" / "pravo_chs.json"


def main() -> None:
    items = json.loads(RAW.read_text(encoding="utf-8"))
    acts = parse(items)
    acts = acts[(acts["date"] >= "2022-01-01")].reset_index(drop=True)
    acts.to_parquet(PROCESSED_DIR / "acts.parquet", index=False)
    months = pd.date_range("2022-01-01", acts["date"].max().to_period("M").to_timestamp(), freq="MS")
    rm = region_monthly(acts, months)
    rm.to_parquet(PROCESSED_DIR / "acts_region_monthly.parquet", index=False)
    per = acts[acts["date"].dt.year.isin([2023, 2024])]
    print(f"актов с 2022 г.: {len(acts)}; 2023–2024: {len(per)}, с регионом: {per['cov_region'].notna().sum()}")
    print("вид (2023–2024):", per["kind"].value_counts().to_dict())
    print("причина введения ЧС:", per[per["kind"] == "intro"]["cause"].value_counts().to_dict())
    print(f"регион × месяц: {len(rm)}, месяцев с действующим ЧС: {int(rm['chs_active'].sum())}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_acts", "data", main)
