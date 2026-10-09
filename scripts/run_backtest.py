"""Backtest моделей прогноза по конфигу.

Запуск: python -m scripts.run_backtest --config default [--limit 50]

Результаты в reports/backtest/<config>/: predictions.parquet, metrics.csv,
forecast.parquet (прогноз вперёд из последнего месяца, если forecast: true),
metrics.md, feature_importance.csv, а для нескольких моделей – ещё
error_correlation.csv и ensembles.csv (см. src/eval/ensemble.py).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR, ROOT, load_config  # noqa: E402
from src.eval.ensemble import ensemble_report, holdout_selection, rolling_selection  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.forecast.backtest import make_origins, run_backtest  # noqa: E402
from src.forecast.global_model import run_global_backtest  # noqa: E402

log = logging.getLogger("forecast")


ENSEMBLE_USED: list[dict] = []     # фактический состав ансамблей прогона -> ensemble_members.csv


def fixed_ensemble(pred: pd.DataFrame, name: str, members: list[str], expected: bool = False,
                   min_members: int | None = None) -> pd.DataFrame | None:
    """Среднее прогнозов членов по общим точкам; состав задан в конфиге.

    Нет каких-то членов (например, весов chronos-bolt-base на новой машине):
    при min_members – ансамбль из оставшихся, если их не меньше min_members,
    с WARNING; иначе ансамбля нет. expected=True – нехватка штатная (горизонт,
    где члену не хватает истории): INFO вместо WARNING."""
    missing = sorted(set(members) - set(pred["model"]))
    used = [m for m in members if m not in missing]
    need = len(members) if min_members is None else min_members
    ENSEMBLE_USED.append({"model": name, "configured": ",".join(members), "used": ",".join(used),
                          "missing": ",".join(missing), "built": len(used) >= need})
    if len(used) < need:
        (log.info if expected else log.warning)("ensemble_skipped  %s: нет прогнозов %s", name, missing)
        return None
    if missing:
        (log.info if expected else log.warning)(
            "ensemble_partial  %s: собран из %d моделей из %d, нет %s (веса фундаментальных моделей: "
            "python -m worker models)", name, len(used), len(members), missing)
    members = used
    keys = ["series_id", "origin", "date", "h"]
    sub = pred[pred["model"].isin(members)]
    wide = sub.pivot_table(index=keys, columns="model", values="yhat")[members].dropna()
    y = sub.drop_duplicates(keys).set_index(keys)["y"].reindex(wide.index)
    print(f"  {name}: среднее {len(members)} моделей, {len(wide)} прогнозов", flush=True)
    return wide.mean(axis=1).rename("yhat").reset_index().assign(model=name, y=y.to_numpy())


def forward_forecast(p: pd.DataFrame, cfg: dict, horizon: int) -> pd.DataFrame:
    """Прогноз вперёд: из последнего месяца данных на horizon месяцев за краем
    – теми же моделями и ансамблями, что в backtest. Факта ещё нет (y = NaN)."""
    from src.forecast import factor, foundation
    from src.forecast.models import MODELS
    last = p["date"].max()
    panel = p[["series_id", "date", "value"]]
    dates = [last + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    parts = []
    for m in cfg["models"]:
        kind, prm = m["model"], m.get("params", {})
        if kind == "global":
            for cat in cfg["data"]["categories"]:
                pr, _ = run_global_backtest([last], horizon, prm, category=cat, name=m["name"],
                                            series_filter=set(p.loc[p["category"] == cat, "oktmo"]), future=True)
                parts.append(pr)
        elif kind == "foundation":
            try:
                parts.append(foundation.run_backtest(panel, [last], horizon, m["name"], prm, future=True))
            except foundation.Unavailable as e:
                log.warning("model_skipped  прогноз вперёд: %s – %s", m["name"], e)
        elif kind == "factor":
            parts.append(factor.run_backtest(panel, [last], horizon, m["name"], prm, future=True))
        elif kind != "ensemble":
            rows = []
            for sid, g in panel.groupby("series_id"):
                s = g.set_index("date")["value"].sort_index()
                try:
                    yhat = MODELS[kind](s, horizon, prm)
                except ValueError:          # не хватает истории
                    continue
                rows += [{"series_id": sid, "model": m["name"], "origin": last, "date": d, "h": h,
                          "y": float("nan"), "yhat": float(v)} for h, (d, v) in enumerate(zip(dates, yhat), 1)]
            parts.append(pd.DataFrame(rows))
    fc = pd.concat([x for x in parts if len(x)], ignore_index=True)
    for m in cfg["models"]:
        if m["model"] == "ensemble":
            pr = fixed_ensemble(fc, m["name"], m["params"]["members"], min_members=m["params"].get("min_members"))
            if pr is not None:
                fc = pd.concat([fc, pr], ignore_index=True)
    fc["category"] = fc["series_id"].str.split("|").str[1]
    log_forward(fc, p, cfg)
    return fc


def log_forward(fc: pd.DataFrame, p: pd.DataFrame, cfg: dict, max_growth: float = 0.5) -> None:
    """Прогноз вперёд в лог: из какого месяца, каких моделей нет, и проверка на
    правдоподобие – медианный рост к тому же месяцу год назад. Больше ±50% –
    WARNING: так выглядит поломка (признаки, масштаб), а не прогноз."""
    last = fc["origin"].max()
    want = [m["name"] for m in cfg["models"]]
    missing = sorted(set(want) - set(fc["model"]))
    prev = p.assign(date=p["date"] + pd.DateOffset(years=1)).set_index(["series_id", "date"])["value"]
    ratio = fc["yhat"] / fc.set_index(["series_id", "date"]).index.map(prev) - 1
    med = ratio.groupby(fc["model"]).median()
    log.info("forecast_forward  из %s на %s: моделей %d из %d, прогнозов %d; медианный рост г/г: %s",
             f"{last:%Y-%m}", ", ".join(sorted(fc["date"].dt.strftime("%Y-%m").unique())), fc["model"].nunique(),
             len(want), len(fc), ", ".join(f"{m} {v:+.1%}" for m, v in med.items()))
    if missing:
        log.warning("forecast_forward_missing  прогноза вперёд нет у: %s", missing)
    odd = med[med.abs() > max_growth]
    if len(odd):
        log.warning("forecast_forward_implausible  медианный рост г/г больше ±%.0f%%: %s", max_growth * 100,
                    ", ".join(f"{m} {v:+.1%}" for m, v in odd.items()))


def log_summary(config: str, p: pd.DataFrame, bc: dict, origins, overall: pd.DataFrame, has_global: bool) -> None:
    """Итог прогона одной строкой: на каких данных и окнах, сколько признаков,
    и по моделям MAE / MAE последнего окна / худшее окно – чтобы сравнивать
    прогоны между собой по логу."""
    feats = "-"
    if has_global:
        from src.features.build import feature_columns
        from src.forecast.global_model import feature_rows
        feats = len(feature_columns(feature_rows("all", tuple(range(1, bc["horizon"] + 1)))))
    models = "; ".join(f"{r.model} {r.MAE:.0f}/{r.MAE_last:.0f}/{r.MAE_worst:.0f}"
                       for r in overall.sort_values("MAE_last").itertuples())
    log.info("backtest_summary  config=%s данные %s..%s, рядов %d, окна %s, горизонт %d, признаков %s | "
             "модель MAE/последнее окно/худшее окно: %s", config, f"{p['date'].min():%Y-%m}",
             f"{p['date'].max():%Y-%m}", p["series_id"].nunique(), ",".join(f"{o:%Y-%m}" for o in origins),
             bc["horizon"], feats, models)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--limit", type=int, default=None, help="только первые N МО (для отладки)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, bc = cfg["data"], cfg["backtest"]

    p = pd.read_parquet(ROOT / dc["panel"])
    p = p[p["category"].isin(dc["categories"])]
    if dc.get("full_history_only"):
        n = p.groupby(["oktmo", "category"])["date"].transform("nunique")
        p = p[n == p["date"].nunique()]
    mos = sorted(p["oktmo"].unique())
    if args.limit:
        p = p[p["oktmo"].isin(mos[: args.limit])]
    p = p.assign(series_id=p["oktmo"] + "|" + p["category"])
    print(f"рядов: {p['series_id'].nunique()}, моделей: {len(cfg['models'])}")

    # "global" – одна модель на все МО (backend в params), "factor" – модели на
    # отклонении от общего фактора, "ensemble" – среднее готовых прогнозов,
    # остальное – локальные
    local = [m for m in cfg["models"] if m["model"] not in ("global", "foundation", "factor", "ensemble")]
    glob = [m for m in cfg["models"] if m["model"] == "global"]
    found = [m for m in cfg["models"] if m["model"] == "foundation"]
    fact = [m for m in cfg["models"] if m["model"] == "factor"]
    ensembles = [m for m in cfg["models"] if m["model"] == "ensemble"]
    t = time.time()
    parts = []
    if local:
        parts.append(run_backtest(p[["series_id", "date", "value"]], local, bc["horizon"],
                                  bc["n_windows"], bc["step"], bc["min_train"], bc.get("n_jobs", -1)))
    origins = make_origins(pd.DatetimeIndex(p["date"].unique()), bc["horizon"], bc["n_windows"], bc["step"])
    importances = []
    for m in glob:
        for cat in dc["categories"]:
            pr, imp = run_global_backtest(origins, bc["horizon"], m.get("params", {}), category=cat,
                                          name=m["name"], series_filter=set(p.loc[p["category"] == cat, "oktmo"]))
            parts.append(pr)
            importances.append(imp.assign(model=m["name"], category=cat))
    from src.forecast import foundation
    for m in found:
        try:
            t0 = time.perf_counter()
            pr = foundation.run_backtest(p[["series_id", "date", "value"]], origins, bc["horizon"], m["name"],
                                         m.get("params", {}))
            parts.append(pr)
            log.info("model_done  %s (фундаментальная, %s): %d прогнозов", m["name"], m["params"].get("backend"),
                     len(pr), extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})
        except foundation.Unavailable as e:
            log.warning("model_skipped  %s – %s", m["name"], e)
    from src.forecast import factor
    for m in fact:
        t0 = time.perf_counter()
        pr = factor.run_backtest(p[["series_id", "date", "value"]], origins, bc["horizon"], m["name"],
                                 m.get("params", {}))
        parts.append(pr)
        log.info("model_done  %s (фактор, %s): %d прогнозов", m["name"], m["params"].get("method"), len(pr),
                 extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})
    pred = pd.concat(parts, ignore_index=True)
    for m in ensembles:
        pr = fixed_ensemble(pred, m["name"], m["params"]["members"], min_members=m["params"].get("min_members"))
        if pr is not None:
            parts.append(pr)
            pred = pd.concat([pred, pr], ignore_index=True)
    print(f"backtest: {time.time() - t:.0f} с, прогнозов: {len(pred)}")
    pred["category"] = pred["series_id"].str.split("|").str[1]

    out = REPORTS_DIR / "backtest" / args.config
    out.mkdir(parents=True, exist_ok=True)
    if importances:
        imp = pd.concat(importances).groupby(["model", "category", "feature"])["gain"].mean()
        imp = (imp / imp.groupby(level=[0, 1]).transform("sum")).rename("share").reset_index()
        imp.sort_values("share", ascending=False).to_csv(out / "feature_importance.csv", index=False)

    # пары и тройки подбираются из одиночных моделей, не из заданных ансамблей;
    # ensemble_report: false – без них (сравнение наборов признаков по seed:
    # десятки моделей дают сотни пар, которые там не нужны)
    ens = (ensemble_report(pred[~pred["model"].isin([m["name"] for m in ensembles])], out)
           if cfg.get("ensemble_report", True) else None)
    if ens is not None:
        pred = pd.concat([pred, ens], ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    if bc.get("holdout"):
        # честная проверка ансамбля: состав – по известному к дате прогноза, оценка – на поздних окнах
        # кандидаты – одиночные модели: ансамбли из конфига (mean_all) уже
        # средние других моделей и подбор по ним был бы по кругу
        solo = pred[~pred["model"].isin([m["name"] for m in ensembles])]
        ho, members = holdout_selection(solo, **bc["holdout"])
        ho.assign(members=" + ".join(members)).to_csv(out / "holdout.csv", index=False)
        print(f"holdout: ансамбль {members} – MAE {ho.set_index('model').loc['holdout_ensemble', 'MAE_holdout']:.0f}", flush=True)
        # то же по всем окнам с 3-го: способы собрать прогноз, а не один состав
        rs = rolling_selection(solo)
        rs.to_csv(out / "rolling_selection.csv", index=False)
        print("скользящая проверка способов (MAE в среднем по окнам):", flush=True)
        print(rs[["way", "mean"]].head(8).to_string(index=False), flush=True)
        # покрытие интервалов 80% (src/eval/intervals.py) – тоже скользящее
        from src.eval.intervals import rolling_coverage
        names = [m["name"] for m in cfg["models"] if m["name"] in set(pred["model"])]
        cov = rolling_coverage(pred, names)
        cov.to_csv(out / "interval_coverage.csv", index=False)
        print(cov.groupby(["model", "way"])[["coverage", "width"]].mean().round(3).to_string(), flush=True)
    if cfg.get("forecast"):
        forward_forecast(p, cfg, bc["horizon"]).to_parquet(out / "forecast.parquet", index=False)
    if ensembles:
        # состав по backtest (первая запись на ансамбль) – его читают приложение и воркер
        pd.DataFrame(ENSEMBLE_USED).drop_duplicates("model").to_csv(out / "ensemble_members.csv", index=False)

    overall = summarize(pred, ["category", "model"]).sort_values(["category", "MAE"])
    by_h = summarize(pred, ["category", "model", "h"])
    by_origin = summarize(pred, ["category", "model", "origin"])
    pd.concat([overall.assign(cut="overall"), by_h.assign(cut="h"), by_origin.assign(cut="origin")]) \
        .to_csv(out / "metrics.csv", index=False)

    # Ансамбли (ens_*): веса или состав подобраны на окнах этого же backtest-а,
    # их MAE по всем окнам занижена. В итоговой таблице – отдельным блоком
    # и с MAE последнего окна, честной для всех моделей.
    last = by_origin["origin"].max()
    overall = overall.merge(by_origin[by_origin["origin"] == last][["category", "model", "MAE"]]
                            .rename(columns={"MAE": "MAE_last"}), on=["category", "model"], how="left")
    by_o = by_origin.pivot_table(index=["category", "model"], columns="origin", values="MAE")
    overall["MAE_worst"] = overall.set_index(["category", "model"]).index.map(by_o.max(axis=1))
    tuned = overall["model"].str.startswith("ens_")
    md = [f"# Backtest: {args.config}", "",
          f"Окна: horizon={bc['horizon']}, n_windows={bc['n_windows']}, step={bc['step']}; "
          f"рядов: {p['series_id'].nunique()}", "", "## Итог: модели без подбора на этих окнах", "",
          overall[~tuned].to_markdown(index=False, floatfmt=".3f"), "",
          f"MAE_last – окно {last:%Y-%m}, MAE_worst – худшее окно.", "",
          "## Ансамбли: веса или состав подобраны на окнах этого backtest-а", "",
          "MAE по всем окнам занижена; сравнивать с моделями выше честно только по MAE_last.", "",
          overall[tuned].sort_values("MAE_last").to_markdown(index=False, floatfmt=".3f"), "",
          "## MAE по горизонту", "",
          by_h.pivot_table(index=["category", "model"], columns="h", values="MAE")
              .to_markdown(floatfmt=".0f"), "",
          "## MAE по окну (origin)", "",
          by_origin.pivot_table(index=["category", "model"], columns="origin", values="MAE")
              .to_markdown(floatfmt=".0f")]
    (out / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(overall.to_string(index=False))
    log_summary(args.config, p, bc, origins, overall, bool(glob))


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_backtest", "forecast", main)
