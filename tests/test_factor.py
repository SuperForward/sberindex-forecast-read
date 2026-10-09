"""Модели на отклонении от общего фактора (src/forecast/factor.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.forecast import factor  # noqa: E402


def _panel(n=40, months=24, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=months, freq="MS")
    season = 0.1 * np.sin(2 * np.pi * np.arange(months) / 12)
    level = rng.normal(10, 1, n)[:, None]
    L = level + season + 0.005 * np.arange(months) + rng.normal(0, 0.01, (n, months))
    idx = pd.Index([f"{i:08d}|all" for i in range(n)])
    return pd.DataFrame(L, index=idx, columns=dates)


def test_national_factor_is_median_per_month():
    L = _panel()
    F = factor.factor(L)
    assert F.shape == L.shape
    assert np.allclose(F.iloc[0], L.median(axis=0))


def test_regional_factor_shrinks_to_national(monkeypatch):
    L = _panel()
    reg = pd.Series(np.repeat([1, 2], len(L) // 2), index=L.index)
    monkeypatch.setattr(factor, "_regions", lambda idx: reg.reindex(idx))
    nat = factor.factor(L)
    assert np.allclose(factor.factor(L, 1e12), nat)                   # k -> inf
    pure = factor.factor(L, 0)
    assert np.allclose(pure.iloc[0], L[reg == 1].median(axis=0))       # k = 0
    mid = factor.factor(L, 20)                                         # между ними
    assert (np.abs(mid - nat) <= np.abs(pure - nat) + 1e-12).all().all()


def test_ets_follows_level_and_trend():
    y = np.linspace(0, 1, 24)
    fc = factor.ets_forecast(y, 3)
    assert fc.shape == (3,)
    assert fc[0] > y[-1] - 0.05 and fc[2] > fc[0]                    # тренд продолжается
    flat = factor.ets_forecast(np.full(24, 0.3) + np.random.default_rng(1).normal(0, 0.01, 24), 3)
    assert np.allclose(flat, 0.3, atol=0.02)


def test_factor_forecast_repeats_last_year_plus_growth():
    L = _panel()
    F = factor.factor(L)
    fc = factor.factor_forecast(F.iloc[:, :-3], 3, 1)
    A = F.iloc[:, :-3].to_numpy()
    g = A[:, -1] - A[:, -13]
    assert np.allclose(fc[:, 0], A[:, -12] + g)


def test_future_forecast_goes_past_the_data():
    L = _panel()
    panel = np.exp(L).stack().rename("value").reset_index()
    panel.columns = ["series_id", "date", "value"]
    last = panel["date"].max()
    out = factor.run_backtest(panel, [last], 3, "fo", {"method": "factor_only"}, future=True)
    assert len(out) == 3 * len(L)
    assert out["y"].isna().all() and out["yhat"].notna().all()
    assert sorted(out["date"].unique()) == [last + pd.DateOffset(months=h) for h in (1, 2, 3)]
