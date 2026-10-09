"""Наборы признаков CatBoost по 4 seed: в среднем по МО и там, где были события.

Запуск: python -m scripts.run_event_eval [--configs features_events_catboost features_forms_catboost]

Берёт прогнозы backtest-ов (reports/backtest/<config>/predictions.parquet),
где модели названы catboost_<набор>_s<seed>, и сравнивает каждый набор с
catboost_cur (признаки основной модели) на тех же seed:
  - delta – разница MAE набора и cur при одном seed, среднее по seed, %;
  - seeds_better – у скольких seed набор лучше cur;
  - spread – разброс MAE cur между seed (стандартное отклонение), %;
  - ci – 95% ДИ среднего по seed выигрыша в точке, бутстрэп по регионам
    (МО одного региона не независимы).
Вердикт «лучше» – если набор лучше cur при всех seed и средний выигрыш
больше разброса cur между seed (правило проекта для признаков основной модели).

Срезы – точки прогноза (МО, целевой месяц), где что-то случилось. Срезы
определены по факту (что было в целевом месяце), а не по тому, что модель
знала в момент прогноза: вопрос – точнее ли прогноз там, где событие было.
  - all            – все точки;
  - chs_active     – в регионе действует режим ЧС (акты, по дате акта);
  - chs_intro      – в регионе введено ЧС в целевом месяце или месяцем раньше;
  - floods_2024    – регионы паводков апреля 2024 г. (Оренбургская, Курганская,
                     Тюменская), апрель–май;
  - flooded_mo     – затопленные МО из реестра (configs/cpd.yaml), апрель–июнь;
  - news_emergency – у МО есть новости о ЧС за целевой месяц;
  - news_any       – у МО есть новости любого типа за целевой месяц;
  - gdelt_surge    – всплеск событий GDELT по МО (log1p числа событий выше
                     среднего за 6 мес. больше чем на 1).

Выход: reports/backtest/events/event_eval.csv, event_eval.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, REPORTS_DIR, load_config  # noqa: E402

OUT = REPORTS_DIR / "backtest" / "events"
FLOOD_REGIONS = {"53", "37", "71"}
SLICES = {
    "all": "все точки",
    "chs_active": "режим ЧС в регионе",
    "chs_intro": "введено ЧС (месяц или следующий)",
    "floods_2024": "регионы паводков, 04–05.2024",
    "flooded_mo": "затопленные МО, 04–06.2024",
    "news_emergency": "новости о ЧС по МО",
    "news_any": "новости по МО",
    "gdelt_surge": "всплеск GDELT по МО",
}


def load_predictions(configs: list[str]) -> pd.DataFrame:
    parts = []
    for c in configs:
        p = pd.read_parquet(REPORTS_DIR / "backtest" / c / "predictions.parquet")
        parts.append(p[p["model"].str.match(r"catboost_.+_s\d+$")])
    p = pd.concat(parts, ignore_index=True).drop_duplicates(["model", "series_id", "origin", "date", "h"])
    p["set"] = p["model"].str.extract(r"catboost_(.+)_s\d+$")[0]
    p["seed"] = p["model"].str.extract(r"_s(\d+)$")[0].astype(int)
    p["oktmo"] = p["series_id"].str.split("|").str[0]
    p["ae"] = (p["y"] - p["yhat"]).abs()
    return p


def event_flags(points: pd.DataFrame) -> pd.DataFrame:
    """Точки (oktmo, date) -> срезы SLICES как столбцы bool."""
    reg = pd.read_parquet(PROCESSED_DIR / "mo_cov_region.parquet").drop_duplicates("oktmo")
    x = points.merge(reg[["oktmo", "cov_region"]], on="oktmo", how="left")
    x["cov_region"] = x["cov_region"].astype(str)
    rm = pd.read_parquet(PROCESSED_DIR / "acts_region_monthly.parquet")
    on = set(zip(rm.loc[rm["chs_active"] > 0, "cov_region"], rm.loc[rm["chs_active"] > 0, "month"]))
    x["chs_active"] = [(c, d) in on for c, d in zip(x["cov_region"], x["date"])]
    intro = rm[rm["chs_intro"] > 0]
    hit = set(zip(intro["cov_region"], intro["month"])) | \
        set(zip(intro["cov_region"], intro["month"] + pd.DateOffset(months=1)))
    x["chs_intro"] = [(c, d) in hit for c, d in zip(x["cov_region"], x["date"])]
    apr_may = x["date"].isin(pd.to_datetime(["2024-04-01", "2024-05-01"]))
    x["floods_2024"] = apr_may & x["cov_region"].isin(FLOOD_REGIONS)
    flooded = {e["oktmo"] for e in load_config("cpd")["real_events"]}
    x["flooded_mo"] = x["oktmo"].isin(flooded) & \
        x["date"].isin(pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]))
    n = pd.read_parquet(PROCESSED_DIR / "news_mo_monthly.parquet").rename(columns={"month": "date"})
    cols = [c for c in n.columns if c.startswith("news_")]
    n = n.assign(news_any_n=n[cols].sum(axis=1))[["oktmo", "date", "news_emergency", "news_any_n"]]
    x = x.merge(n, on=["oktmo", "date"], how="left")
    x["news_emergency"] = x["news_emergency"].fillna(0) >= 1
    x["news_any"] = x.pop("news_any_n").fillna(0) >= 1
    g = pd.read_parquet(PROCESSED_DIR / "gdelt_mo_monthly.parquet").rename(columns={"month": "date"})
    x = x.merge(g[["oktmo", "date", "gdelt_mo_surge"]], on=["oktmo", "date"], how="left")
    x["gdelt_surge"] = x.pop("gdelt_mo_surge").fillna(0) > 1
    x["all"] = True
    return x


def bootstrap_ci(d: pd.DataFrame, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% ДИ средней разницы ошибок (набор − cur) в точке, по регионам."""
    by = d.groupby("cov_region")["diff"].agg(["sum", "count"])
    if len(by) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), size=(n, len(by)))
    s, c = by["sum"].to_numpy()[idx].sum(axis=1), by["count"].to_numpy()[idx].sum(axis=1)
    lo, hi = np.percentile(s / c, [2.5, 97.5])
    return lo, hi


def evaluate(p: pd.DataFrame) -> pd.DataFrame:
    keys = ["oktmo", "origin", "date", "h"]
    wide = p.pivot_table(index=keys, columns=["set", "seed"], values="ae")
    flags = event_flags(wide.index.to_frame(index=False)).set_index(keys)
    cur = wide["cur"]
    seeds = list(cur.columns)
    rows = []
    for sl in SLICES:
        m = flags[sl].to_numpy()
        if not m.any():
            continue
        cur_mae = cur[m].mean()
        spread = cur_mae.std(ddof=1) / cur_mae.mean()
        for s in wide.columns.get_level_values("set").unique():
            mae = wide[s][m][seeds].mean()
            delta = (mae - cur_mae) / cur_mae
            row = {"slice": sl, "set": s, "points": int(m.sum()), "MAE": mae.mean(),
                   "MAE_min": mae.min(), "MAE_max": mae.max(), "delta_pct": delta.mean() * 100,
                   "seeds_better": int((delta < 0).sum()), "spread_cur_pct": spread * 100}
            if s != "cur":
                d = pd.DataFrame({"diff": (wide[s][m][seeds] - cur[m][seeds]).mean(axis=1).to_numpy(),
                                  "cov_region": flags.loc[m, "cov_region"].to_numpy()})
                lo, hi = bootstrap_ci(d)
                row |= {"ci_lo_pct": lo / cur_mae.mean() * 100, "ci_hi_pct": hi / cur_mae.mean() * 100,
                        "verdict": ("лучше" if (delta < 0).all() and -delta.mean() > spread
                                    else "хуже" if (delta > 0).all() and delta.mean() > spread
                                    else "в пределах шума")}
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["features_events_catboost", "features_forms_catboost"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    p = load_predictions([c for c in args.configs
                          if (REPORTS_DIR / "backtest" / c / "predictions.parquet").exists()])
    res = evaluate(p)
    res.to_csv(OUT / "event_eval.csv", index=False)

    md = ["# Наборы признаков CatBoost: в среднем и на МО с событиями", "",
          f"Конфиги: {', '.join(args.configs)}. Seed: {', '.join(map(str, sorted(p['seed'].unique())))}, окна: "
          f"{', '.join(f'{o:%Y-%m}' for o in sorted(p['origin'].unique()))}.", "",
          "delta – изменение MAE к cur при том же seed (среднее по seed); seeds_better – у скольких seed "
          "набор лучше; spread_cur – разброс MAE cur между seed; ci – 95% ДИ выигрыша, бутстрэп по регионам. "
          "«Лучше» – лучше при всех seed и выигрыш больше разброса.", ""]
    for sl, title in SLICES.items():
        r = res[res["slice"] == sl]
        if r.empty:
            continue
        md += [f"## {title} ({sl}): точек {int(r['points'].iloc[0])}, разброс cur "
               f"{r['spread_cur_pct'].iloc[0]:.1f}%", "",
               r.drop(columns=["slice", "points", "spread_cur_pct"]).sort_values("delta_pct")
               .to_markdown(index=False, floatfmt=".1f"), ""]
    (OUT / "event_eval.md").write_text("\n".join(md), encoding="utf-8")
    show = res[res["set"] != "cur"].pivot_table(index="set", columns="slice", values="delta_pct")
    print("изменение MAE к cur, % (среднее по seed):")
    print(show[[s for s in SLICES if s in show]].round(1).to_string())
    print(res[res["verdict"] == "лучше"][["slice", "set", "delta_pct", "seeds_better"]].to_string(index=False))


if __name__ == "__main__":
    main()
