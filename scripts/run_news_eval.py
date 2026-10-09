"""Польза новостей: реестр событий, детектор «по двум ключам», признаки прогноза.

Запуск: python -m scripts.run_news_eval

1. События. Для каждого события из reference/shocks_events.csv – сколько
   новостей о шоке нашлось в МО или его регионе в первые 30 дней.
2. Детектор сдвигов по двум ключам (реальные данные 2024 г.). Тревога – если
   сигнал EWMA выше основного порога (5% ложных тревог) ИЛИ выше мягкого
   порога (20% ложных тревог) и в этом или прошлом месяце вышла новость о шоке
   в МО или его регионе. Сравнение с одним ключом: зафиксированы ли паводки апреля
   2024 г. и сколько МО добавилось в тревоги.
3. Прогноз. LightGBM и случайный лес с признаками новостей против тех же
   моделей без них – те же окна, те же МО (reports/backtest/models).

Результаты: reports/news/summary.md, events.csv, cpd_two_key.csv, forecast.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import store  # noqa: E402
from src.config import PROCESSED_DIR, REPORTS_DIR, load_config  # noqa: E402

# отрицательные шоки: второй ключ детектора (build_news: взвешенное число новостей)
SHOCK_TYPES = ["news_emergency", "news_attack", "news_production_neg"]


def events_check(matches: pd.DataFrame) -> pd.DataFrame:
    from scripts.build_covariates import cov_region_of_oktmo
    from src.news import gazetteer
    ev = store.load_fresh("events").drop_duplicates("event_id")
    rows = []
    for e in ev.itertuples():
        start = pd.Timestamp(e.start_date)
        win = matches[(matches["published"] >= start - pd.Timedelta(days=2)) &
                      (matches["published"] <= start + pd.Timedelta(days=30))]
        cov = cov_region_of_oktmo(e.oktmo) if isinstance(e.oktmo, str) else \
            next(iter(gazetteer.find_regions(str(e.region))), None)
        mo_n = int((win["oktmo"] == e.oktmo).sum()) if isinstance(e.oktmo, str) else 0
        reg_n = int(((win["cov_region"] == cov) & (win["level"] == "region")).sum()) if cov else 0
        rows.append({"event": e.event_id, "date": start.date(), "region": e.region, "mo": e.mo_name,
                     "in_panel": bool(e.in_panel), "news_mo": mo_n, "news_region": reg_n,
                     "found": mo_n + reg_n > 0})
    return pd.DataFrame(rows)


def cpd_two_key(news_mo: pd.DataFrame, matches: pd.DataFrame, own_only: bool) -> tuple[pd.DataFrame, dict]:
    """own_only – второй ключ только по новостям о самом МО (без новостей о регионе)."""
    cfg = load_config("cpd")
    # из файлов, а не из DuckDB: детектор пересчитан в этом же прогоне воркера
    comp = store.load_fresh("cpd_comparison")
    a = store.load_fresh("cpd_alarms")
    best = a["method"].iloc[0]
    thr = comp[comp["method"] == best].set_index("far_target")["threshold"]
    hi, lo = float(thr[cfg["main_far"]]), float(thr[0.2])
    if own_only:
        flag = matches[matches["level"] == "mo"][["oktmo", "month"]].drop_duplicates()
    else:
        flag = news_mo.assign(shock=news_mo[[c for c in SHOCK_TYPES if c in news_mo]].sum(axis=1) > 0)
        flag = flag[flag["shock"]][["oktmo", "month"]]
    prev = flag.assign(month=flag["month"] + pd.offsets.MonthBegin(1))      # новость прошлого месяца
    keys = pd.concat([flag, prev]).drop_duplicates().assign(news=True)
    x = a.merge(keys, left_on=["oktmo", "date"], right_on=["oktmo", "month"], how="left")
    x["news"] = x["news"].fillna(False).astype(bool)
    x["one_key"] = x["score"] > hi
    x["two_key"] = x["one_key"] | ((x["score"] > lo) & x["news"])
    rows = []
    for e in cfg["real_events"]:
        g = x[(x["oktmo"] == e["oktmo"]) & (x["date"] >= pd.Timestamp(e["month"] + "-01"))
              & (x["date"] < pd.Timestamp(e["month"] + "-01") + pd.DateOffset(months=3))]
        rows.append({"mo": e["name"], "oktmo": e["oktmo"], "news_months": int(g["news"].sum()),
                     "one_key": bool(g["one_key"].any()), "two_key": bool(g["two_key"].any()),
                     "max_score": round(float(g["score"].max()), 2)})
    one = x.groupby("oktmo")["one_key"].any()
    two = x.groupby("oktmo")["two_key"].any()
    supported = x[x["one_key"]]["news"].mean() if x["one_key"].any() else float("nan")
    stats = {"variant": "новости о самом МО" if own_only else "новости о МО и регионе",
             "method": best, "thr_main": round(hi, 2), "thr_news": round(lo, 2),
             "mo_one_key": int(one.sum()), "mo_two_key": int(two.sum()), "mo_added": int((two & ~one).sum()),
             "alarms_with_news": round(float(supported), 3), "mo_with_news_2024": int(x[x["news"]]["oktmo"].nunique())}
    return pd.DataFrame(rows), stats


def forecast_effect() -> pd.DataFrame:
    from src.eval.metrics import mae
    from src.forecast.backtest import make_origins
    from src.forecast.global_model import run_global_backtest
    cfg = load_config("models")
    bc = cfg["backtest"]
    p = store.load_fresh("spending_mo")
    p = p[p["category"] == "all"]
    n = p.groupby("oktmo")["date"].transform("nunique")
    series = set(p.loc[n == p["date"].nunique(), "oktmo"])
    origins = make_origins(pd.DatetimeIndex(p["date"].unique()), bc["horizon"], bc["n_windows"], bc["step"])
    news_mo = pd.read_parquet(PROCESSED_DIR / "news_mo_monthly.parquet")
    own = store.load_fresh("mo_oktmo_map")  # noqa: F841  (для наглядности: МО с собственными упоминаниями ниже)
    mentioned = set(pd.read_parquet(PROCESSED_DIR / "news_matches.parquet").query("level == 'mo'")["oktmo"])
    if (PROCESSED_DIR / "gdelt_matches.parquet").exists():
        mentioned |= set(pd.read_parquet(PROCESSED_DIR / "gdelt_matches.parquet", columns=["oktmo"])["oktmo"].dropna())
    variants = {"новости": {"news_features": True}, "GDELT": {"gdelt_features": True},
                "новости + GDELT": {"news_features": True, "gdelt_features": True}}
    rows = []
    for m in [x for x in cfg["models"] if x["name"] in ("lightgbm", "random_forest")]:
        # база – тот же прогон без новостей: сохранённый backtest мог считаться
        # на других данных, и разница смешалась бы с эффектом новостей
        b, _ = run_global_backtest(origins, bc["horizon"], dict(m["params"]), category="all",
                                   name=m["name"], series_filter=series)
        for vname, flags in variants.items():
            params = dict(m["params"], **flags)
            pr, imp = run_global_backtest(origins, bc["horizon"], params, category="all",
                                          name=f"{m['name']}+{vname}", series_filter=series)
            key = ["series_id", "origin", "h"]
            j = pr.merge(b[key + ["yhat"]], on=key, suffixes=("_news", "_base"))
            j["oktmo"] = j["series_id"].str.split("|").str[0]
            share = imp[imp["feature"].str.startswith(("news_", "gdelt_"))]["gain"].sum() / max(imp["gain"].sum(), 1e-9)
            for subset, mask in [("все МО", None), ("МО с упоминаниями", j["oktmo"].isin(mentioned))]:
                s = j if mask is None else j[mask]
                rows.append({"model": m["name"], "features": vname, "subset": subset, "points": len(s),
                             "mae_base": round(mae(s["y"].to_numpy(), s["yhat_base"].to_numpy())),
                             "mae_news": round(mae(s["y"].to_numpy(), s["yhat_news"].to_numpy())),
                             "news_importance": round(float(share), 4)})
    out = pd.DataFrame(rows)
    out["change_%"] = ((out["mae_news"] / out["mae_base"] - 1) * 100).round(1)
    return out


def main() -> None:
    out = REPORTS_DIR / "news"
    out.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet(PROCESSED_DIR / "news_matches.parquet")
    news_mo = pd.read_parquet(PROCESSED_DIR / "news_mo_monthly.parquet")
    raw = sum(len(pd.read_parquet(f)) for f in sorted((Path(PROCESSED_DIR).parent / "raw" / "news").glob("*.parquet")))

    ev = events_check(matches)
    ev.to_csv(out / "events.csv", index=False)
    cpd, st = cpd_two_key(news_mo, matches, own_only=False)
    cpd_own, st_own = cpd_two_key(news_mo, matches, own_only=True)
    pd.concat([cpd.assign(variant=st["variant"]), cpd_own.assign(variant=st_own["variant"])]) \
        .to_csv(out / "cpd_two_key.csv", index=False)
    pd.DataFrame([st, st_own]).to_csv(out / "cpd_two_key_stats.csv", index=False)
    fc = forecast_effect()
    fc.to_csv(out / "forecast.csv", index=False)

    panel_ev = ev[ev["in_panel"]]
    md = ["# Новости: польза для детектора сдвигов и прогноза", "",
          f"Новостей в архиве: {raw}; с признаками шока и привязкой: к МО {int((matches['level'] == 'mo').sum())} "
          f"(МО: {matches['oktmo'].nunique()}), к региону {int((matches['level'] == 'region').sum())}.", "",
          "## События из реестра", "",
          f"Найдено в новостях: {int(ev['found'].sum())} из {len(ev)} (в панели СберИндекса: "
          f"{int(panel_ev['found'].sum())} из {len(panel_ev)}).", "",
          ev.to_markdown(index=False), "",
          "## Детектор по двум ключам (2024 г.)", "",
          f"Основной порог {st['method']}: {st['thr_main']} (5% ложных тревог); с новостью: {st['thr_news']} (20%).", "",
          pd.DataFrame([st, st_own]).drop(columns=["method", "thr_main", "thr_news"]).to_markdown(index=False), "",
          "### Паводки апреля 2024 г.: новости о МО и регионе", "", cpd.to_markdown(index=False), "",
          "### Паводки апреля 2024 г.: только новости о самом МО", "", cpd_own.to_markdown(index=False), "",
          "## Прогноз: признаки новостей и GDELT", "", fc.to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_news_eval", "news", main)
