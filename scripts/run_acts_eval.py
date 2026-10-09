"""Официальные акты о ЧС: видны ли они в расходах и помогают ли детектору.

Запуск: python -m scripts.run_acts_eval

1. Ошибка прогноза в месяцы ЧС. Для каждой точки проверки прогноза на 1 месяц
   (reports/backtest/models) – относительная ошибка главной модели
   (факт − прогноз) / прогноз. Сравниваем МО регионов, где режим ЧС введён в
   целевом месяце или месяцем раньше, с остальными МО в те же месяцы: если ЧС
   бьёт по расходам, факт у них ниже прогноза. 95% ДИ – бутстрэп по регионам
   (МО одного региона не независимы).
2. Детектор по двум ключам. Тревога, если сигнал выше строгого порога (5%
   ложных) или выше мягкого (20%) и в регионе действует режим ЧС в этом или
   прошлом месяце. Сколько МО добавляется и ловятся ли паводки апреля 2024 г.
3. Меры поддержки: акты о выплатах пострадавшим в регионах паводков –
   объяснение, почему провал расходов быстро компенсируется.

Выход: reports/news/acts_summary.md, acts_event_study.csv, acts_two_key.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, REPORTS_DIR, load_config  # noqa: E402

OUT = REPORTS_DIR / "news"
FLOOD_REGIONS = {"53": "Оренбургская", "37": "Курганская", "71": "Тюменская"}


def main_model(pred: pd.DataFrame) -> str:
    solo = pred[~pred["model"].str.startswith("ens_")]
    return (solo.assign(e=(solo["yhat"] - solo["y"]).abs()).groupby("model")["e"].mean().idxmin())


def bootstrap_diff(df: pd.DataFrame, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Разница средних (ЧС − остальные), 95% ДИ по регионам."""
    rng = np.random.default_rng(seed)
    regs = df["cov_region"].unique()
    by = {r: g for r, g in df.groupby("cov_region")}
    point = df[df["chs"]]["r"].mean() - df[~df["chs"]]["r"].mean()
    diffs = []
    for _ in range(n):
        s = pd.concat([by[r] for r in rng.choice(regs, len(regs))])
        if s["chs"].any() and (~s["chs"]).any():
            diffs.append(s[s["chs"]]["r"].mean() - s[~s["chs"]]["r"].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acts = pd.read_parquet(PROCESSED_DIR / "acts.parquet")
    rm = pd.read_parquet(PROCESSED_DIR / "acts_region_monthly.parquet")
    reg = pd.read_parquet(PROCESSED_DIR / "mo_cov_region.parquet").drop_duplicates("oktmo")

    # --- 1. ошибка прогноза в месяцы введения ЧС
    pred = pd.read_parquet(REPORTS_DIR / "backtest" / "models" / "predictions.parquet")
    best = main_model(pred)
    p = pred[(pred["model"] == best) & (pred["h"] == 1)].copy()
    p["oktmo"] = p["series_id"].str.split("|").str[0]
    p = p.merge(reg, on="oktmo", how="inner")
    p["r"] = (p["y"] - p["yhat"]) / p["yhat"]
    intro = rm[rm["chs_intro"] > 0][["cov_region", "month"]]
    hit = set(zip(intro["cov_region"], intro["month"])) | \
        set(zip(intro["cov_region"], intro["month"] + pd.DateOffset(months=1)))
    p["chs"] = [(c, d) in hit for c, d in zip(p["cov_region"], p["date"])]
    all_pt, all_lo, all_hi = bootstrap_diff(p)
    fl = p[p["date"].isin(pd.to_datetime(["2024-04-01", "2024-05-01"]))]
    fl = fl.assign(chs=fl["cov_region"].isin(FLOOD_REGIONS))
    fl_pt, fl_lo, fl_hi = bootstrap_diff(fl)
    ev = pd.DataFrame([
        {"group": "все введения ЧС (целевой месяц или следующий)", "points_chs": int(p["chs"].sum()),
         "regions_chs": p[p["chs"]]["cov_region"].nunique(), "diff_pct": all_pt * 100,
         "ci_lo_pct": all_lo * 100, "ci_hi_pct": all_hi * 100},
        {"group": "паводки 04.2024 (Оренбургская, Курганская, Тюменская), апрель–май",
         "points_chs": int(fl["chs"].sum()), "regions_chs": fl[fl["chs"]]["cov_region"].nunique(),
         "diff_pct": fl_pt * 100, "ci_lo_pct": fl_lo * 100, "ci_hi_pct": fl_hi * 100}]).round(2)
    # точечно: затопленные МО из реестра событий против остальных МО, апрель–май
    flooded = [e["oktmo"] for e in load_config("cpd")["real_events"]]
    fm = p[p["date"].isin(pd.to_datetime(["2024-04-01", "2024-05-01"]))]
    f_in, f_out = fm[fm["oktmo"].isin(flooded)], fm[~fm["oktmo"].isin(flooded)]
    ev.loc[len(ev)] = {"group": f"затопленные МО из реестра ({len(flooded)}), апрель–май", "points_chs": len(f_in),
                       "regions_chs": f_in["cov_region"].nunique(),
                       "diff_pct": round((f_in["r"].mean() - f_out["r"].mean()) * 100, 2),
                       "ci_lo_pct": None, "ci_hi_pct": None}
    ev["model"] = best
    ev.to_csv(OUT / "acts_event_study.csv", index=False)

    # --- 2. детектор по двум ключам
    al = pd.read_parquet(REPORTS_DIR / "cpd" / "alarms.parquet").merge(reg, on="oktmo", how="left")
    comp = pd.read_csv(REPORTS_DIR / "cpd" / "comparison.csv")
    method = al["method"].iloc[0]
    thr = comp[(comp["method"] == method)].set_index("far_target")["threshold"]
    thr_main, thr_soft = float(thr.loc[0.05]), float(thr.loc[0.2])
    act = rm[rm["chs_active"] > 0][["cov_region", "month"]]
    on = set(zip(act["cov_region"], act["month"])) | \
        set(zip(act["cov_region"], act["month"] + pd.DateOffset(months=1)))
    al["chs"] = [(c, d) in on for c, d in zip(al["cov_region"], al["date"])]
    al["one_key"] = al["score"] > thr_main
    al["two_key"] = al["one_key"] | ((al["score"] > thr_soft) & al["chs"])
    cfg = load_config("cpd")
    floods = []
    for e in cfg["real_events"]:
        g = al[(al["oktmo"] == e["oktmo"]) & (al["date"] >= e["month"])].sort_values("date")
        first = lambda col: g[g[col]]["date"].min()
        floods.append({"mo": e["name"], "event": e["month"],
                       "one_key": None if pd.isna(first("one_key")) else f"{first('one_key'):%Y-%m}",
                       "two_key": None if pd.isna(first("two_key")) else f"{first('two_key'):%Y-%m}",
                       "chs_in_region": bool(g["chs"].any())})
    floods = pd.DataFrame(floods)
    two = pd.DataFrame([{"method": method, "thr_main": round(thr_main, 2), "thr_soft": round(thr_soft, 2),
                         "mo_one_key": int(al.groupby("oktmo")["one_key"].any().sum()),
                         "mo_two_key": int(al.groupby("oktmo")["two_key"].any().sum()),
                         "mo_with_chs_2024": int(al.groupby("oktmo")["chs"].any().sum())}])
    two["mo_added"] = two["mo_two_key"] - two["mo_one_key"]
    two.to_csv(OUT / "acts_two_key.csv", index=False)

    # --- 3. меры поддержки в регионах паводков
    a24 = acts[(acts["date"] >= "2024-04-01") & (acts["date"] < "2024-10-01")]
    sup = (a24[a24["cov_region"].isin(FLOOD_REGIONS) & (a24["kind"] == "support")]
           .assign(region=lambda d: d["cov_region"].map(FLOOD_REGIONS))
           .groupby("region").agg(acts=("id", "size"), first=("date", "min")).reset_index())
    intro_fl = acts[(acts["kind"] == "intro") & acts["cov_region"].isin(FLOOD_REGIONS)
                    & (acts["date"].dt.year == 2024)][["date", "cov_region", "title"]]

    per = acts[acts["date"].dt.year.isin([2023, 2024])]
    md = ["# Официальные акты о ЧС (publication.pravo.gov.ru)", "",
          f"Актов с «чрезвычайной ситуации» в названии за 2023–2024 гг.: {len(per)}, привязано к региону "
          f"панели: {per['cov_region'].notna().sum()}. Вид: " +
          ", ".join(f"{k} – {v}" for k, v in per["kind"].value_counts().items()) + ".", "",
          "## Паводки апреля 2024 г.: дата введения режима ЧС", "",
          intro_fl.assign(date=intro_fl["date"].dt.strftime("%Y-%m-%d")).to_markdown(index=False), "",
          "## Ошибка прогноза в месяцы ЧС", "",
          f"Модель {best}, прогноз на 1 месяц; разница относительной ошибки (факт − прогноз)/прогноз, "
          "МО регионов с ЧС минус остальные МО в те же месяцы, %; 95% ДИ – бутстрэп по регионам.", "",
          ev.drop(columns=["model"]).to_markdown(index=False), "",
          "## Детектор по двум ключам (акты вместо новостей)", "",
          two.to_markdown(index=False), "", floods.to_markdown(index=False), "",
          "## Меры поддержки пострадавших в регионах паводков (04–09.2024)", "",
          sup.assign(first=sup["first"].dt.strftime("%Y-%m-%d")).to_markdown(index=False) if len(sup) else "–"]
    (OUT / "acts_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_acts_eval", "news", main)
