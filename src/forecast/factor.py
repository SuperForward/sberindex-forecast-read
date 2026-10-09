"""Модели на отклонении МО от общего фактора.

Фактор – медиана log-уровня расходов по МО в каждом месяце. При 24 месяцах
истории сезонность внутри одного ряда не оценить, а медиана по тысяче рядов
несёт её почти без шума. Ряд МО делится на фактор и отклонение: отклонение
прогнозирует модель без сезонности, фактор – «прошлый год × рост г/г».

Фактор бывает национальным (k = inf) и региональным: медиана по МО региона,
стянутая к национальной с весом n/(n+k), где n – число МО в регионе (k = 0 –
чистый региональный). Модели на национальном и региональном факторе ошибаются
по-разному, поэтому в ансамбле полезны обе.

Модели (params["method"]):
  factor_only – отклонение держится на последнем значении;
  ets         – экспоненциальное сглаживание без сезонности (как AutoETS
                из statsforecast, выбор модели по AICc).
Chronos на отклонении – src/forecast/foundation.py, mode "relative".

Фактор считается по всем МО панели – в момент прогноза известны уровни всех
МО, поэтому это не утечка.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import PROCESSED_DIR, n_jobs


def _regions(index: pd.Index) -> pd.Series:
    r = pd.read_parquet(PROCESSED_DIR / "mo_cov_region.parquet").drop_duplicates("oktmo")
    oktmo = index.str.split("|").str[0] if index.str.contains("|", regex=False).any() else index
    return pd.Series(r.set_index("oktmo")["cov_region"].reindex(oktmo).to_numpy(), index=index)


def factor(L: pd.DataFrame, k: float = float("inf")) -> pd.DataFrame:
    """L – log-уровни (ряды × месяцы). Матрица фактора той же формы."""
    nat = L.median(axis=0)
    if np.isinf(k):
        return pd.DataFrame(np.tile(nat.to_numpy(), (len(L), 1)), index=L.index, columns=L.columns)
    reg = _regions(L.index)
    g = L.groupby(reg).median()
    n = reg.value_counts()
    w = (n / (n + k)).reindex(g.index)
    mix = g.sub(nat, axis=1).mul(w, axis=0).add(nat, axis=1)
    # МО без региона – национальный фактор
    out = mix.reindex(reg.fillna(-1).to_numpy())
    out.index = L.index
    return out.fillna(pd.DataFrame(np.tile(nat.to_numpy(), (len(L), 1)), index=L.index, columns=L.columns))


def factor_forecast(F: pd.DataFrame, horizon: int, window: int) -> np.ndarray:
    """Фактор на T+1..T+h: прошлогоднее значение + средний рост г/г за window мес."""
    A = F.to_numpy(float)
    g = (A[:, -window:] - A[:, -window - 12:-12]).mean(axis=1)
    return np.column_stack([A[:, -12 + h - 1] + g for h in range(1, horizon + 1)])


# --- ETS без сезонности -------------------------------------------------
# Повторяет AutoETS(season_length=1) из statsforecast (сам пакет требует
# pandas < 3): модели ANN, AAN, AAdN и при положительном ряде – с
# мультипликативной ошибкой; начальные значения, границы параметров, beta <
# alpha, оптимизация Нелдера–Мида по правдоподобию, выбор по AICc.

LOWER, UPPER = np.array([1e-4, 1e-4, 0.8]), np.array([0.9999, 0.9999, 0.98])


def _ets_run(par, y, err, trend, damped):
    """Правдоподобие (как у statsforecast: меньше – лучше) и конечные состояния."""
    a = par[0]
    b_ = par[1] if trend else 0.0
    phi = par[2] if damped else 1.0
    lvl = par[-2] if trend else par[-1]
    tr = par[-1] if trend else 0.0
    s2, slog = 0.0, 0.0
    for v in y:
        f = lvl + phi * tr
        if err == "A":
            e = v - f
            lvl, tr = f + a * e, phi * tr + b_ * e
            s2 += e * e
        else:
            if f <= 0:
                return np.inf, lvl, tr
            e = (v - f) / f
            lvl, tr = f * (1 + a * e), phi * tr + b_ * f * e
            s2 += e * e
            slog += np.log(abs(f))
    lik = len(y) * np.log(max(s2, 1e-300)) + 2 * slog
    return lik, lvl, tr


def _ets_fit(y: np.ndarray, err: str, trend: bool, damped: bool):
    m = min(10, len(y))
    alpha = LOWER[0] + 0.2 * (UPPER[0] - LOWER[0])
    par = [alpha]
    if trend:
        par.append(LOWER[1] + 0.1 * (min(UPPER[1], alpha) - LOWER[1]))
    if damped:
        par.append(LOWER[2] + 0.99 * (UPPER[2] - LOWER[2]))
    if trend:
        b, l0 = np.polyfit(np.arange(1, m + 1), y[:m], 1)
        par += [l0, b]
    else:
        par.append(y[:m].mean())

    def target(p):
        # за границами – большое число, не inf: иначе Нелдер–Мид считает inf − inf
        if not (LOWER[0] <= p[0] <= UPPER[0]):
            return 1e10
        if trend and not (LOWER[1] <= p[1] <= min(UPPER[1], p[0])):
            return 1e10
        if damped and not (LOWER[2] <= p[2] <= UPPER[2]):
            return 1e10
        return min(_ets_run(p, y, err, trend, damped)[0], 1e10)

    res = minimize(target, np.array(par), method="Nelder-Mead", options={"maxiter": 2000})
    npar = len(par) + 1
    aicc = res.fun + 2 * npar + 2 * npar * (npar + 1) / (len(y) - npar - 1) if len(y) - npar - 1 > 0 else np.inf
    return aicc, res.x


def ets_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    """Лучшая по AICc модель ETS без сезонности; прогноз на 1..horizon."""
    errs = ("A", "M") if (y > 0).all() else ("A",)
    best = None
    for err in errs:
        for trend, damped in ((False, False), (True, False), (True, True)):
            aicc, p = _ets_fit(y, err, trend, damped)
            if aicc < 1e9 and (best is None or aicc < best[0]):
                best = (aicc, p, err, trend, damped)
    _, p, err, trend, damped = best
    _, lvl, tr = _ets_run(p, y, err, trend, damped)
    phi = p[2] if damped else 1.0
    return np.array([lvl + (sum(phi ** i for i in range(1, h + 1)) if trend else 0.0) * tr
                     for h in range(1, horizon + 1)])


def _ets_many(rows: np.ndarray, horizon: int) -> np.ndarray:
    return np.vstack([ets_forecast(r, horizon) for r in rows])


# --- backtest -----------------------------------------------------------

def relative_context(Y: pd.DataFrame, t: int, k: float):
    """Ряды с полной историей по месяц t: (отклонение, фактор) в логарифмах."""
    hist = Y.iloc[:, : t + 1]
    ok = hist.notna().all(axis=1) & (hist > 0).all(axis=1)
    L = np.log(hist[ok])
    F = factor(L, k)
    return L - F, F


def run_backtest(panel: pd.DataFrame, origins: list, horizon: int, name: str, params: dict,
                 future: bool = False) -> pd.DataFrame:
    """Прогнозы в формате остальных моделей. panel: series_id, date, value.
    future – прогноз вперёд из последнего месяца: факта нет (y = NaN)."""
    from src.forecast.foundation import extend_future
    method, k = params.get("method", "factor_only"), float(params.get("k", "inf"))
    Y = panel.pivot_table(index="series_id", columns="date", values="value").sort_index(axis=1)
    if future:
        Y = extend_future(Y, horizon)
    dates = list(Y.columns)
    parts = []
    for origin in origins:
        if origin not in dates:
            continue
        t = dates.index(origin)
        # фактору нужен рост г/г за window мес.: история не короче 12 + window
        window = 3 if method == "factor_only" else 1
        if t < 11 + window or t + horizon >= len(dates):
            continue
        R, F = relative_context(Y, t, k)
        if method == "factor_only":
            # как у остальных: рост фактора за 3 месяца
            rel = np.repeat(R.iloc[:, -1].to_numpy()[:, None], horizon, axis=1)
            fc = factor_forecast(F, horizon, 3)
        elif method == "ets":
            from joblib import Parallel, delayed
            rows = R.to_numpy(float)
            chunks = Parallel(n_jobs=n_jobs(params.get("n_jobs", -1)))(
                delayed(_ets_many)(rows[i:i + 100], horizon) for i in range(0, len(rows), 100))
            rel = np.vstack(chunks)
            fc = factor_forecast(F, horizon, 1)
        else:
            raise ValueError(f"неизвестный method: {method}")
        yhat = np.exp(rel + fc)
        fut = Y.loc[R.index, dates[t + 1: t + 1 + horizon]]
        for h in range(horizon):
            parts.append(pd.DataFrame({"series_id": R.index, "model": name, "origin": origin,
                                       "date": dates[t + 1 + h], "h": h + 1,
                                       "y": fut.iloc[:, h].to_numpy(), "yhat": yhat[:, h]}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["series_id", "model", "origin", "date", "h", "y", "yhat"])
    return out if future else out.dropna(subset=["y"])
