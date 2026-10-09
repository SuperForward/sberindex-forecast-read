"""Null-тест: действительно ли глобальные модели прогнозируют, или это случайность.

Запуск: python -m scripts.run_null_test [--config models] [--seeds 3] [--models lightgbm,xgboost]

Те же окна и модели, что в backtest, но в нескольких сценариях:
  real            – настоящие данные;
  noise_features  – все признаки заменены случайными числами (числовые –
                    N(0,1), категории перемешаны). Цель настоящая;
  shuffled_target – цель (рост г/г) перемешана между строками обучения:
                    связь признаков с целью разорвана;
  shuffled_within_month – цель перемешана между МО внутри одного целевого
                    месяца: общий по стране темп остаётся, различия между МО
                    – нет. Проверяет, что модель различает МО, а не угадывает
                    только общий тренд.
Прогноз уровня везде y_{T−12}·exp(ĝ), оценка – по настоящему факту.

Ориентиры, считаются из тех же строк: seasonal_naive (ĝ = 0), const_growth
(ĝ = медиана роста в обучении окна) и baseline (ĝ = drift3, это
seasonal_naive_drift k=3 из backtest).

Если модель знает что-то настоящее, в null-сценариях её MAE заметно хуже
real и не лучше ориентиров. Для real дополнительно: bootstrap по МО
доверительного интервала выигрыша у baseline и доля МО, где модель лучше.

Результаты: reports/null_test/<config>/runs.csv, summary.md.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR, ROOT, load_config  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.features.build import add_covariates, build_rows, feature_columns  # noqa: E402
from src.forecast.backtest import make_origins  # noqa: E402
from src.forecast.global_model import CAT_COLS, run_global_backtest  # noqa: E402

SCENARIOS = ["noise_features", "shuffled_target", "shuffled_within_month"]


def perturb(data: pd.DataFrame, scenario: str, seed: int, origin: pd.Timestamp) -> pd.DataFrame:
    """Цель перемешивается только среди строк обучения окна origin: иначе в
    обучение попадают значения роста из тестового периода (утечка)."""
    rng = np.random.default_rng(seed)
    d = data.copy()
    if scenario == "noise_features":
        for f in feature_columns(d):
            if f in CAT_COLS:
                d[f] = pd.Categorical(rng.permutation(d[f].to_numpy()), categories=d[f].cat.categories)
            else:
                d[f] = rng.standard_normal(len(d))
    elif scenario == "shuffled_target":
        ok = (d["target"].notna() & (d["target_date"] <= origin)).to_numpy()
        d.loc[ok, "target"] = rng.permutation(d.loc[ok, "target"].to_numpy())
    elif scenario == "shuffled_within_month":
        ok = d["target"].notna() & (d["target_date"] <= origin)
        d.loc[ok, "target"] = (d[ok].groupby("target_date")["target"]
                               .transform(lambda s: rng.permutation(s.to_numpy())))
    else:
        raise ValueError(scenario)
    return d


def references(data: pd.DataFrame, origins, mos: set) -> pd.DataFrame:
    """seasonal_naive, const_growth и baseline на тех же строках, что и
    глобальные модели. const_growth – прошлогодний уровень × медианный рост
    г/г по обучающим строкам окна: одна константа на всё окно, без признаков."""
    t = data[data["t"].isin(origins) & data["y_log"].notna() & data["oktmo"].isin(mos)]
    med = {o: data.loc[(data["target_date"] <= o) & data["target"].notna(), "target"].median() for o in origins}
    base = dict(series_id=t["oktmo"].astype(str) + "|all", origin=t["t"], date=t["target_date"],
                h=t["h"], y=t["y_level"])
    return pd.concat([
        pd.DataFrame({**base, "model": "seasonal_naive", "yhat": np.exp(t["base_log"])}),
        pd.DataFrame({**base, "model": "const_growth", "yhat": np.exp(t["base_log"] + t["t"].map(med))}),
        pd.DataFrame({**base, "model": "baseline", "yhat": np.exp(t["base_log"] + t["drift3"].fillna(t["g1"]))}),
    ], ignore_index=True)


def bootstrap_skill(pred: pd.DataFrame, model: str, ref: str = "baseline", n: int = 2000,
                    seed: int = 0) -> dict:
    """Выигрыш MAE у ref по МО: 1 − MAE_model/MAE_ref, 95% CI bootstrap по рядам."""
    a = pred[pred["model"].isin([model, ref])].assign(ae=lambda x: (x["y"] - x["yhat"]).abs())
    s = a.pivot_table(index=["series_id", "origin", "date"], columns="model", values="ae").dropna()
    per = s.groupby(level="series_id").sum()
    m, r = per[model].to_numpy(), per[ref].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n, len(per)))
    boot = 1 - m[idx].sum(1) / r[idx].sum(1)
    return {"skill": 1 - m.sum() / r.sum(), "ci_lo": np.quantile(boot, 0.025),
            "ci_hi": np.quantile(boot, 0.975), "share_mo_better": float((m < r).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models")
    ap.add_argument("--seeds", type=int, default=3, help="повторов каждого null-сценария")
    ap.add_argument("--models", default=None, help="через запятую; по умолчанию все глобальные")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, bc = cfg["data"], cfg["backtest"]
    models = [m for m in cfg["models"] if m["model"] == "global"]
    if args.models:
        keep = set(args.models.split(","))
        models = [m for m in models if m["name"] in keep]

    p = pd.read_parquet(ROOT / dc["panel"])
    p = p[p["category"] == "all"]
    if dc.get("full_history_only"):
        n = p.groupby(["oktmo", "category"])["date"].transform("nunique")
        p = p[n == p["date"].nunique()]
    mos = set(p["oktmo"])
    origins = make_origins(pd.DatetimeIndex(p["date"].unique()), bc["horizon"], bc["n_windows"], bc["step"])
    data = add_covariates(build_rows("all", horizons=tuple(range(1, bc["horizon"] + 1))))
    print(f"МО: {len(mos)}, строк признаков: {len(data)}, моделей: {len(models)}, "
          f"null-повторов: {args.seeds}", flush=True)

    ref = references(data, origins, mos)
    runs, preds_real = [], [ref]
    t0 = time.time()
    plan = [("real", 0)] + [(s, k) for s in SCENARIOS for k in range(args.seeds)]
    for scenario, seed in plan:
        for m in models:
            if scenario == "real":
                pr, _ = run_global_backtest(origins, bc["horizon"], m.get("params", {}), category="all",
                                            name=m["name"], series_filter=mos, data=data)
            else:  # перемешивание своё для каждого окна
                pr = pd.concat([run_global_backtest(
                    [o], bc["horizon"], m.get("params", {}), category="all", name=m["name"],
                    series_filter=mos, data=perturb(data, scenario, 1000 + seed, o))[0] for o in origins],
                    ignore_index=True)
            if pr.empty:
                continue
            if scenario == "real":
                preds_real.append(pr)
            s = summarize(pr, ["model"]).iloc[0]
            runs.append({"scenario": scenario, "seed": seed, "model": m["name"],
                         "MAE": s["MAE"], "R2_within": s["R2_within"]})
            print(f"[{time.time() - t0:.0f} с] {scenario}#{seed} {m['name']}: MAE {s['MAE']:.0f}", flush=True)

    real = pd.concat(preds_real, ignore_index=True)
    refm = summarize(ref, ["model"]).set_index("model")
    runs = pd.DataFrame(runs)
    out = REPORTS_DIR / "null_test" / args.config
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)

    # сводка: real против лучшего из null-повторов; p – доля null не хуже real
    rows = []
    for name, g in runs.groupby("model"):
        r = g[g["scenario"] == "real"].iloc[0]
        row = {"model": name, "MAE_real": r["MAE"], "R2w_real": r["R2_within"]}
        for sc in SCENARIOS:
            nul = g[g["scenario"] == sc]
            row[f"MAE_{sc}"] = nul["MAE"].mean()
            row[f"p_{sc}"] = (1 + (nul["MAE"] <= r["MAE"]).sum()) / (1 + len(nul))
        row.update(bootstrap_skill(real, name))
        row.update({f"{k}_const": v for k, v in bootstrap_skill(real, name, "const_growth").items()
                    if k != "share_mo_better"})
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values("MAE_real")

    md = [f"# Null-тест: {args.config}", "",
          f"Окна {', '.join(f'{o:%Y-%m}' for o in origins)}, горизонт {bc['horizon']}, МО {len(mos)}, "
          f"null-повторов {args.seeds}.", "",
          f"Ориентиры: seasonal_naive MAE {refm.loc['seasonal_naive', 'MAE']:.0f}, "
          f"const_growth MAE {refm.loc['const_growth', 'MAE']:.0f}, "
          f"baseline MAE {refm.loc['baseline', 'MAE']:.0f} (R2_within "
          f"{refm.loc['baseline', 'R2_within']:.3f}).", "",
          "## MAE: настоящие данные против случайных", "",
          summ[["model", "MAE_real"] + [f"MAE_{s}" for s in SCENARIOS] + [f"p_{s}" for s in SCENARIOS]]
          .to_markdown(index=False, floatfmt=".3f"), "",
          "p – доля null-повторов с MAE не хуже real (минимально возможное 1/(повторы+1)).", "",
          "## Выигрыш у baseline на настоящих данных", "",
          summ[["model", "R2w_real", "skill", "ci_lo", "ci_hi", "share_mo_better",
                "skill_const", "ci_lo_const", "ci_hi_const"]]
          .to_markdown(index=False, floatfmt=".3f"), "",
          "skill = 1 - MAE_model/MAE_baseline, CI – 95% bootstrap по МО; "
          "share_mo_better – доля МО, где MAE модели ниже; *_const – то же против const_growth."]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_null_test", "forecast", main)
