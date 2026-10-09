"""Годовые показатели МО из БД ПМО -> data/processed/mo_annual.parquet.

Запуск: python -m scripts.build_mo_features

Широкая таблица oktmo × year × показатель (коды из configs/pmo_indicators.yaml,
колонки вида pmo_<код>), плюс покрытие панели расходов.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, load_config  # noqa: E402
from src.data import pmo  # noqa: E402


def main() -> None:
    cfg = load_config("pmo_indicators")
    names = {str(c): n for group in cfg["indicators"].values() for c, n in group.items()}
    groups = {str(c): g for g, group in cfg["indicators"].items() for c in group}
    long = pmo.load_pmo(names)
    # один МО может встретиться в двух базах (напр. автономный округ и край) –
    # берём первое значение
    long = long.drop_duplicates(["oktmo", "code", "year"])
    wide = long.pivot_table(index=["oktmo", "year"], columns="code", values="value", aggfunc="first")
    wide.columns = [f"pmo_{c}" for c in wide.columns]
    wide = wide.reset_index()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(PROCESSED_DIR / "mo_annual.parquet", index=False)

    from src.features.build import PMO_USED
    panel = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet")["oktmo"].unique()
    print(f"МО в БД ПМО: {wide['oktmo'].nunique()}, лет {wide['year'].min()}–{wide['year'].max()}")
    print(f"покрытие панели расходов ({len(panel)} МО):")
    for code in sorted(names, key=lambda c: (groups[c], c)):
        col = f"pmo_{code}"
        if col not in wide:
            print(f"  [{groups[code]}] {code} {names[code][:60]}: нет данных")
            continue
        sub = wide[wide["oktmo"].isin(panel) & wide[col].notna()]
        yrs = sub.groupby("year")["oktmo"].nunique()
        last = f"{yrs.index.max()}: {yrs.iloc[-1]} МО" if len(yrs) else "–"
        print(f"  [{groups[code]}] {code} {names[code][:55]:<55} МО {sub['oktmo'].nunique():>4}, "
              f"годы {yrs.index.min() if len(yrs) else '–'}–{last}{'' if col in PMO_USED else '  (не в модели)'}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_mo_features", "data", main)
