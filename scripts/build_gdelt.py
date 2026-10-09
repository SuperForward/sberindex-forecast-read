"""События GDELT -> признаки МО по месяцам.

Запуск: python -m scripts.build_gdelt

- data/processed/gdelt_mo_monthly.parquet – МО × месяц (все МО панели,
  месяцы с первого по последний в архиве):
    gdelt_mo_events     – событий с местом в самом МО (log1p);
    gdelt_mo_conflict   – из них материальный конфликт (QuadClass 4: удары,
                          нападения, насилие; log1p);
    gdelt_mo_tone       – средний тон статей, взвешенный числом статей
                          (нет событий – 0);
    gdelt_mo_surge      – всплеск: log1p событий минус среднее за 6
                          предыдущих месяцев;
    gdelt_reg_*         – то же по региону МО (события в его городах и о
                          субъекте целиком).
- data/processed/gdelt_matches.parquet – привязанные события (для проверки).

Место – src/news/gdelt.assign (название города = транслитерация МО, субъект
совпадает). Месяц – месяц добавления события в GDELT (date_added): к концу
месяца t событие уже известно, признак за t не заглядывает в будущее.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import store  # noqa: E402
from src.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.news import gdelt  # noqa: E402

COLS = ["event_id", "root_code", "quad_class", "num_articles", "avg_tone", "geo_type", "geo_name",
        "geo_adm1", "date_added", "url"]


def _agg(x: pd.DataFrame, key: str) -> pd.DataFrame:
    g = x.assign(wt=x["avg_tone"] * x["num_articles"], conflict=(x["quad_class"] == 4).astype(int))
    a = g.groupby([key, "month"]).agg(events=("event_id", "size"), conflict=("conflict", "sum"),
                                      wt=("wt", "sum"), articles=("num_articles", "sum"))
    a["tone"] = a["wt"] / a["articles"].clip(lower=1)
    return a[["events", "conflict", "tone"]]


def _features(grid: pd.DataFrame, a: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    x = grid.merge(a.reset_index(), on=[key, "month"], how="left").fillna({"events": 0, "conflict": 0, "tone": 0})
    x = x.sort_values([key, "month"])
    ev = np.log1p(x["events"])
    base = ev.groupby(x[key]).transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    return pd.DataFrame({f"{prefix}_events": ev, f"{prefix}_conflict": np.log1p(x["conflict"]),
                         f"{prefix}_tone": x["tone"], f"{prefix}_surge": ev - base}, index=x.index) \
        .assign(**{key: x[key], "month": x["month"]})


def main() -> None:
    from src.data.oktmo import cov_region_of_oktmo
    files = sorted((RAW_DIR / "gdelt").glob("gdelt_*.parquet"))
    if not files:
        raise SystemExit("нет событий в data/raw/gdelt (python -m scripts.download_gdelt)")
    ev = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True).drop_duplicates("event_id")
    print(f"событий в России: {len(ev)} ({ev['date_added'].min():%Y-%m}…{ev['date_added'].max():%Y-%m})")
    m = gdelt.assign(ev)
    m["month"] = m["date_added"].dt.to_period("M").dt.to_timestamp()
    m.drop(columns=["geo_adm1"]).to_parquet(PROCESSED_DIR / "gdelt_matches.parquet", index=False)
    print(f"привязано: к МО {int((m['level'] == 'mo').sum())} (МО: {m['oktmo'].nunique()}), "
          f"к субъекту {int((m['level'] == 'region').sum())}")

    panel = store.load("spending_mo")[["oktmo"]].drop_duplicates()
    panel["cov_region"] = panel["oktmo"].map(cov_region_of_oktmo)
    months = pd.date_range(m["month"].min(), m["month"].max(), freq="MS")
    grid = panel.merge(pd.DataFrame({"month": months}), how="cross")

    mo = _features(grid[["oktmo", "month"]], _agg(m[m["level"] == "mo"], "oktmo"), "oktmo", "gdelt_mo")
    rgrid = grid[["cov_region", "month"]].drop_duplicates()
    reg = _features(rgrid, _agg(m, "cov_region"), "cov_region", "gdelt_reg")
    out = grid.merge(mo, on=["oktmo", "month"]).merge(reg, on=["cov_region", "month"]).drop(columns="cov_region")
    out.to_parquet(PROCESSED_DIR / "gdelt_mo_monthly.parquet", index=False)
    cols = [c for c in out if c.startswith("gdelt_")]
    print(f"gdelt_mo_monthly: {len(out)} строк, МО с событиями: "
          f"{int(out.groupby('oktmo')['gdelt_mo_events'].max().gt(0).sum())}, столбцы {cols}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_gdelt", "news", main)
