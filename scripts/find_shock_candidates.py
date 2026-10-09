"""Кандидаты в шоки по годовым показателям МО (БД ПМО Росстата).

Запуск: python -m scripts.find_shock_candidates

Для каждого МО, года и метрики – изменение к прошлому году; кандидат,
если ухудшение экстремально относительно всех МО того же года
(устойчивый z по медиане/MAD <= -Z) и превышает порог по величине.
Сравнение внутри года убирает общие для страны сдвиги (инфляция, кризис).

Выход: reports/shock_candidates.csv и сводка в консоль. Это кандидаты
для проверки детекторов и для чтения новостей, а не готовая разметка.

Колонка caution – почему кандидату не стоит доверять:
  data_error    – невозможное значение (доля изменилась больше чем на 100 п.п.);
  small_base    – внутригородская территория (малая база, учётные перетоки);
  reorganized   – МО преобразовано: код не из ОКТМО-2023 или отсутствует в
                  актуальном справочнике (упразднено, объединено).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from src.data.oktmo import load_oktmo  # noqa: E402

Z = 3.5
# метрика: (тип шока, столбец/формула, вид изменения, мин. величина ухудшения)
METRICS = {
    "own_revenue_share_pp": ("fiscal_budget", "pmo_8313015", "diff", -10.0),
    "shipments_growth": ("production_local_market", "pmo_8401011", "logdiff", np.log(0.7)),
    "employees_growth": ("production_local_market", "pmo_8423005", "logdiff", np.log(0.9)),
    "retail_growth": ("production_local_market", "pmo_8401003", "logdiff", np.log(0.8)),
    "migration_per_1000": ("socio_demographic", "migration_rate", "level", -15.0),
    "deaths_growth": ("socio_demographic", "pmo_8112001", "logdiff_up", np.log(1.25)),
}


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / (mad if mad > 0 else np.nan)


def main() -> None:
    a = pd.read_parquet(PROCESSED_DIR / "mo_annual.parquet").sort_values(["oktmo", "year"])
    a["migration_rate"] = a["pmo_8112023"] / a["pmo_8112013"] * 1000
    panel = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet").drop_duplicates("oktmo")
    names = panel.set_index("oktmo")[["mo_name", "region"]]

    rows = []
    for metric, (stype, col, kind, thr) in METRICS.items():
        if col not in a:
            continue
        g = a.groupby("oktmo")[col]
        if kind == "diff":
            ch = a[col] - g.shift(1)
        elif kind in ("logdiff", "logdiff_up"):
            ch = np.log(a[col].where(a[col] > 0)) - np.log(g.shift(1).where(g.shift(1) > 0))
        else:
            ch = a[col]
        d = pd.DataFrame({"oktmo": a["oktmo"], "year": a["year"], "change": ch}).dropna()
        d["z"] = d.groupby("year")["change"].transform(robust_z)
        if kind == "logdiff_up":  # рост смертности – ухудшение
            hit = d[(d["z"] >= Z) & (d["change"] >= thr)]
        else:
            hit = d[(d["z"] <= -Z) & (d["change"] <= thr)]
        for r in hit.itertuples():
            rows.append({"oktmo": r.oktmo, "year": r.year, "shock_type": stype, "metric": metric,
                         "change": r.change, "z": r.z})
    c = pd.DataFrame(rows)
    c = c.join(names, on="oktmo")
    c["in_panel"] = c["mo_name"].notna()

    mp = pd.read_csv(INTERIM_DIR / "mo_oktmo_map.csv", dtype=str).drop_duplicates("oktmo").set_index("oktmo")
    caution = pd.Series("", index=c.index)
    caution[(c["metric"] == "own_revenue_share_pp") & (c["change"].abs() > 100)] = "data_error"
    vt = c["oktmo"].map(mp["mo_type"]) == "vt"
    caution[(caution == "") & vt] = "small_base"
    # преобразованные МО: код взят не из ОКТМО-2023 или исчез из актуального
    # справочника (упразднено/объединено, как Алатырь в 2023)
    alive = set(load_oktmo("20260901")["oktmo"])
    reorg = (c["oktmo"].map(mp["oktmo_version"]).fillna("20230112") != "20230112") | ~c["oktmo"].isin(alive)
    caution[(caution == "") & reorg] = "reorganized"
    c["caution"] = caution
    c = c.sort_values(["year", "shock_type", "z"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    c.to_csv(REPORTS_DIR / "shock_candidates.csv", index=False)

    p = c[c["in_panel"]]
    print(f"кандидатов: {len(c)}, из них МО панели расходов: {len(p)}")
    print("пометки качества (МО панели):", p["caution"].replace("", "ok").value_counts().to_dict())
    p = p[p["caution"] == ""]
    print(p.pivot_table(index="year", columns="shock_type", values="oktmo", aggfunc="count", fill_value=0))
    print("\nпериод панели (2023–2024), примеры:")
    ex = p[p["year"].isin([2023, 2024])].groupby("metric").head(3)
    for r in ex.itertuples():
        chg = f"{r.change:+.1f} п.п." if r.metric.endswith("pp") or r.metric.startswith("migration") \
            else f"{np.expm1(r.change):+.0%}"
        print(f"  {r.year} {r.shock_type:<24} {r.metric:<20} {chg:>10}  z={r.z:+.1f}  {r.mo_name} ({r.region})")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("find_shock_candidates", "cpd", main)
