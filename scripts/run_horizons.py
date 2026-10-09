"""Точность прогноза на горизонтах 1, 3, 6, 12 месяцев (см. configs/horizons.yaml).

Запуск: python -m scripts.run_horizons --config horizons

Для горизонта H и целевого месяца m прогноз строится из месяца m − H
(модель видит только данные до него) и берётся ровно H-й шаг. Целевые
месяцы одни и те же для всех H, поэтому MAE сравнимы между горизонтами.

Результаты в reports/horizons/: predictions.parquet, metrics.csv, metrics.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.config import REPORTS_DIR, ROOT, load_config, n_jobs  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.forecast.global_model import run_global_backtest  # noqa: E402
from src.forecast.models import MODELS  # noqa: E402

MODEL_ALIAS = {"seasonal_naive_drift": "baseline"}


def _local_series(sid: str, s: pd.Series, H: int, origins, models: list[dict], min_train: int) -> list[dict]:
    rows = []
    for origin in origins:
        hist = s[s.index <= origin]
        target = origin + pd.DateOffset(months=H)
        if len(hist) < min_train or target not in s.index:
            continue
        for m in models:
            try:
                yhat = MODELS[m["model"]](hist, H, m.get("params", {}))
            except (ValueError, IndexError):
                continue          # модели не хватает истории – честное «н/д»
            if len(yhat) < H or not np.isfinite(yhat[H - 1]):
                continue
            rows.append({"series_id": sid, "model": m["name"], "origin": origin, "date": target,
                         "h": H, "y": s[target], "yhat": float(yhat[H - 1])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="horizons")
    ap.add_argument("--only", default="", help="пересчитать только эти модели (через запятую), "
                                                "остальные взять из прошлого прогона")
    args = ap.parse_args()
    only = [m for m in args.only.split(",") if m]
    cfg = load_config(args.config)
    dc = cfg["data"]

    models = []
    for name in cfg["model_configs"]:
        models += load_config(name)["models"]
    models = [m for m in models if m["name"] not in cfg.get("skip_models", [])]
    models = list({m["name"]: m for m in models}.values())
    all_ens = [m for m in models if m["model"] == "ensemble"]
    if only:
        models = [m for m in models if m["name"] in only or MODEL_ALIAS.get(m["name"]) in only]
    models += [m for m in cfg.get("extra_models", []) if not only or m["name"] in only]
    local = [m for m in models if m["model"] not in ("global", "foundation", "factor", "ensemble")]
    glob = [m for m in models if m["model"] == "global"]
    found = [m for m in models if m["model"] == "foundation"]
    fact = [m for m in models if m["model"] == "factor"]
    fixed_ens = [m for m in models if m["model"] == "ensemble"]
    print("модели:", [m["name"] for m in models])

    p = pd.read_parquet(ROOT / dc["panel"])
    p = p[p["category"].isin(dc["categories"])]
    if dc.get("full_history_only"):
        n = p.groupby(["oktmo", "category"])["date"].transform("nunique")
        p = p[n == p["date"].nunique()]
    p = p.assign(series_id=p["oktmo"] + "|" + p["category"])
    groups = [(sid, g.set_index("date")["value"].sort_index()) for sid, g in p.groupby("series_id")]
    targets = pd.to_datetime(cfg["targets"])

    parts = []
    for H in cfg["horizons"]:
        origins = [t - pd.DateOffset(months=H) for t in targets]
        chunks = Parallel(n_jobs=n_jobs(cfg.get("n_jobs", -1)))(
            delayed(_local_series)(sid, s, H, origins, local, cfg["min_train"]) for sid, s in groups)
        loc = pd.DataFrame([r for c in chunks for r in c])
        print(f"H={H}: локальные модели – {len(loc)} прогнозов", flush=True)
        parts.append(loc)
        from src.forecast import foundation
        for m in found:
            try:
                pr = foundation.run_backtest(p[["series_id", "date", "value"]], origins, H, m["name"],
                                             m.get("params", {}))
                pr = pr[(pr["h"] == H) & pr["date"].isin(targets)]
                parts.append(pr)
                print(f"H={H}: {m['name']} – {len(pr)} прогнозов", flush=True)
            except foundation.Unavailable as e:
                import logging
                logging.getLogger("forecast").warning("model_skipped  H=%d: %s – %s", H, m["name"], e)
        from src.forecast import factor
        for m in fact:
            pr = factor.run_backtest(p[["series_id", "date", "value"]], origins, H, m["name"], m.get("params", {}))
            pr = pr[(pr["h"] == H) & pr["date"].isin(targets)]
            parts.append(pr)
            print(f"H={H}: {m['name']} – {len(pr)} прогнозов", flush=True)
        for m in glob:
            for cat in dc["categories"]:
                ser = set(p.loc[p["category"] == cat, "oktmo"])
                pr, _ = run_global_backtest(origins, H, m.get("params", {}), category=cat,
                                            name=m["name"], series_filter=ser)
                if len(pr):
                    pr = pr[(pr["h"] == H) & pr["date"].isin(targets)]
                    parts.append(pr)
                print(f"H={H}: {m['name']} – {len(pr)} прогнозов", flush=True)

    out = REPORTS_DIR / "horizons"
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.concat(parts, ignore_index=True)
    pred["model"] = pred["model"].replace(MODEL_ALIAS)
    pred["H"] = pred["h"]
    if only:
        old = pd.read_parquet(out / "predictions.parquet")
        # ансамбли пересобираются заново из старых и новых прогнозов членов
        fixed_ens = all_ens
        old = old[~old["model"].isin(set(pred["model"]) | set(only) | {m["name"] for m in all_ens})
                  & ~old["model"].str.startswith("ens_")]
        pred = pd.concat([old, pred], ignore_index=True)
        glob = [m for m in load_config("models")["models"] if m["model"] == "global"]

    # Ансамбли с составом из models.yaml (ens_eq[factor_mix]): среднее членов по
    # общим точкам; если какой-то член не покрыл горизонт – ансамбль тоже «н/д»
    from scripts.run_backtest import fixed_ensemble
    for m in fixed_ens:
        parts_e = [fixed_ensemble(g.drop(columns="H"), m["name"], m["params"]["members"], expected=True,
                                  min_members=m["params"].get("min_members"))
                   for _, g in pred[pred["model"].isin(m["params"]["members"])].groupby("H")]
        parts_e = [x.assign(H=x["h"]) for x in parts_e if x is not None and len(x)]
        if parts_e:
            pred = pd.concat([pred] + parts_e, ignore_index=True)

    # Ансамбль: среднее глобальных моделей по общим точкам
    ec = cfg.get("ensemble")
    if ec:
        members = [m["name"] for m in glob if m["name"] not in ec.get("exclude", [])]
        sub = pred[pred["model"].isin(members)]
        keys = ["series_id", "origin", "date", "h", "H"]
        wide = sub.pivot_table(index=keys, columns="model", values="yhat")
        wide = wide[wide.notna().sum(axis=1) == len(members)]
        if len(wide):
            y = sub.drop_duplicates(keys).set_index(keys)["y"].reindex(wide.index)
            ens = pd.DataFrame({"y": y, "yhat": wide.mean(axis=1)}).reset_index().assign(model=ec["name"])
            pred = pd.concat([pred, ens], ignore_index=True)

    pred.to_parquet(out / "predictions.parquet", index=False)

    # R2_within центрируем по ряду на 6 целевых месяцах (у каждого окна по одной точке)
    met = summarize(pred.assign(origin=pred["H"]), ["model", "H"])
    # Модель, покрывшая не все целевые месяцы (не хватило истории в ранних
    # окнах), сравнивалась бы на других точках – для такого H это «н/д».
    full = p["series_id"].nunique() * len(targets)
    met["coverage"] = met["n"] / full
    met.loc[met["coverage"] < 1, ["MAE", "WAPE", "R2", "R2_within"]] = np.nan
    met.to_csv(out / "metrics.csv", index=False)

    tab = met.pivot_table(index="model", columns="H", values="MAE")
    tab = tab.reindex(columns=cfg["horizons"])
    ref = tab.loc["prophet_default"] if "prophet_default" in tab.index else None
    # сортировка по горизонту 3 мес. – там есть все основные модели; среднее по
    # горизонтам при «н/д» сравнивало бы разные наборы точек
    key = 3 if 3 in tab.columns else tab.columns[0]
    order = tab.sort_values(key, na_position="last").index
    fmt = tab.loc[order].map(lambda v: "н/д" if pd.isna(v) else f"{v:.0f}")
    md = ["# Точность по горизонтам прогноза", "",
          f"Целевые месяцы: {cfg['targets'][0]}…{cfg['targets'][-1]}; рядов: {p['series_id'].nunique()}. "
          "Для горизонта H прогноз строится из месяца «цель − H» и берётся H-й шаг.", "",
          "## MAE (руб. на жителя в месяц)", "", fmt.to_markdown(), ""]
    if ref is not None:
        gain = (1 - tab.loc[order].div(ref, axis=1)) * 100
        md += ["## Насколько лучше Prophet, %", "",
               gain.map(lambda v: "н/д" if pd.isna(v) else f"{v:+.0f}").to_markdown(), ""]
    md += ["## Все метрики", "", met.to_markdown(index=False, floatfmt=".3f")]
    (out / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(fmt.to_string())


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_horizons", "forecast", main)
