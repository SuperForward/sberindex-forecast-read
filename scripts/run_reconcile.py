"""Согласование прогнозов категорий с прогнозом всех расходов.

Запуск: python -m scripts.run_reconcile [--config reconcile]

Прогнозы всех расходов и пяти категорий строятся отдельно, и сумма
категорий не сходится с общим. Иерархия: все = пять категорий + прочее
(прочее – все минус категории, ~28% расходов; src/features/build.py).
Глобальная модель прогнозирует все 7 рядов, затем прогнозы в каждой точке
(МО, окно, месяц) согласуются: ỹ = S (SᵀWS)⁻¹ SᵀW ŷ, S – сумматор (все =
сумма шести нижних), W – веса рядов:
  bottom_up – общий = сумма прогнозов шести нижних рядов;
  ols       – W = I;
  wls_level – W = 1/ŷ²: у всех рядов одинаковая относительная ошибка;
  wls_err   – W = 1/(s_i ŷ_i)², s_i – относительная ошибка ряда i в прошлых
              окнах (только точки с датой ≤ origin – известное к прогнозу).
Сравнение – MAE до и после, по рядам и окнам.

Выход: reports/backtest/reconcile/metrics.csv, metrics.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR, ROOT, load_config  # noqa: E402
from src.forecast.backtest import make_origins  # noqa: E402
from src.forecast.global_model import run_global_backtest  # noqa: E402

CATS = ["food", "health", "horeca", "marketplaces", "transport"]
BOTTOM = CATS + ["other"]
ALL = ["all"] + BOTTOM


def reconcile(Yhat: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Yhat, w – точки × 7 рядов (all, затем BOTTOM). Возвращает согласованные."""
    S = np.vstack([np.ones(len(BOTTOM)), np.eye(len(BOTTOM))])          # 7 × 6
    out = np.empty_like(Yhat)
    for i in range(len(Yhat)):
        SW = S.T * w[i]                                                  # 6 × 7
        b = np.linalg.solve(SW @ S, SW @ Yhat[i])
        out[i] = S @ b
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="reconcile")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bc = cfg["backtest"]
    p = pd.read_parquet(ROOT / cfg["data"]["panel"])
    n = p.groupby(["oktmo", "category"])["date"].transform("nunique")
    full = p[(n == p["date"].nunique()) & (p["value"] > 0)].groupby("oktmo")["category"].nunique()
    mos = set(full[full == len(CATS) + 1].index)
    origins = make_origins(pd.DatetimeIndex(p["date"].unique()), bc["horizon"], bc["n_windows"], bc["step"])
    print(f"МО с полной историей во всех категориях: {len(mos)}, окна: {[f'{o:%Y-%m}' for o in origins]}")

    parts = []
    for c in ALL:
        pr, _ = run_global_backtest(origins, bc["horizon"], cfg["model"], category=c, name=c, series_filter=mos)
        parts.append(pr.assign(oktmo=pr["series_id"].str.split("|").str[0]))
        print(f"  {c}: {len(pr)} прогнозов", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    keys = ["oktmo", "origin", "date", "h"]
    yhat = pred.pivot_table(index=keys, columns="model", values="yhat")[ALL].dropna()
    y = pred.pivot_table(index=keys, columns="model", values="y")[ALL].reindex(yhat.index)
    ok = y.notna().all(axis=1) & (yhat[BOTTOM] > 0).all(axis=1)
    yhat, y = yhat[ok], y[ok]
    Yh = yhat.to_numpy()
    org = yhat.index.get_level_values("origin")
    dates = yhat.index.get_level_values("date")

    rec = {"base": Yh}
    bu = Yh.copy()
    bu[:, 0] = Yh[:, 1:].sum(axis=1)
    rec["bottom_up"] = bu
    rec["ols"] = reconcile(Yh, np.ones_like(Yh))
    rec["wls_level"] = reconcile(Yh, 1 / Yh ** 2)
    # относительная ошибка ряда по известному к origin; в первом окне её нет – как wls_level
    rel = np.abs(y.to_numpy() / Yh - 1)
    w_err = 1 / Yh ** 2
    for o in sorted(org.unique()):
        known = np.asarray(dates <= o)
        if known.any():
            s = rel[known].mean(axis=0)
            m = np.asarray(org == o)
            w_err[m] = 1 / (s * Yh[m]) ** 2
    rec["wls_err"] = reconcile(Yh, w_err)

    rows = []
    Y = y.to_numpy()
    for way, R in rec.items():
        for j, c in enumerate(ALL):
            e = np.abs(R[:, j] - Y[:, j])
            rows.append({"way": way, "series": c, "origin": "все", "MAE": e.mean()})
            for o in sorted(org.unique()):
                rows.append({"way": way, "series": c, "origin": f"{o:%Y-%m}", "MAE": e[np.asarray(org == o)].mean()})
    res = pd.DataFrame(rows)
    out = REPORTS_DIR / "backtest" / "reconcile"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "metrics.csv", index=False)
    tab = res[res["origin"] == "все"].pivot_table(index="way", columns="series", values="MAE")[ALL]
    chg = (tab / tab.loc["base"] - 1) * 100
    by_o = res[(res["series"] == "all") & (res["origin"] != "все")].pivot_table(index="way", columns="origin",
                                                                                values="MAE")
    md = ["# Согласование прогнозов категорий с общим", "",
          f"Модель: {cfg['model'].get('backend')}, МО: {len(mos)}, точек: {len(yhat)}, окна: "
          f"{', '.join(f'{o:%Y-%m}' for o in origins)}.", "",
          "## MAE, руб. на жителя", "", tab.to_markdown(floatfmt=".1f"), "",
          "## Изменение MAE к прогнозу без согласования, %", "", chg.to_markdown(floatfmt="+.1f"), "",
          "## Все расходы: MAE по окнам", "", by_o.to_markdown(floatfmt=".0f")]
    (out / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(tab.round(1).to_string())
    print(chg.round(1).to_string())


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_reconcile", "forecast", main)
