"""Rolling-origin backtest по всем рядам панели.

Для каждого origin модель видит только данные с датой <= origin и
прогнозирует следующие horizon месяцев. Никаких данных из будущего.
"""

import pandas as pd
from joblib import Parallel, delayed

from src.config import n_jobs as cap_jobs
from src.forecast.models import MODELS


def make_origins(dates: pd.DatetimeIndex, horizon: int, n_windows: int, step: int) -> list[pd.Timestamp]:
    """Последнее окно заканчивается на последней дате данных."""
    last = dates.max()
    return [last - pd.DateOffset(months=horizon + step * i) for i in range(n_windows)][::-1]


def _run_series(series_id: str, s: pd.Series, origins, horizon: int, model_cfgs: list[dict],
                min_train: int) -> list[dict]:
    rows = []
    for origin in origins:
        hist = s[s.index <= origin]
        future = s[(s.index > origin)].iloc[:horizon]
        if len(hist) < min_train or len(future) < horizon:
            continue
        for cfg in model_cfgs:
            yhat = MODELS[cfg["model"]](hist, horizon, cfg.get("params", {}))
            for h, (d, y) in enumerate(future.items(), start=1):
                rows.append({"series_id": series_id, "model": cfg["name"], "origin": origin,
                             "date": d, "h": h, "y": y, "yhat": float(yhat[h - 1])})
    return rows


def run_backtest(panel: pd.DataFrame, model_cfgs: list[dict], horizon: int, n_windows: int,
                 step: int, min_train: int, n_jobs: int = -1) -> pd.DataFrame:
    """panel: series_id, date, value (месячные даты). Возвращает прогнозы по всем окнам."""
    origins = make_origins(pd.DatetimeIndex(panel["date"].unique()), horizon, n_windows, step)
    groups = [(sid, g.set_index("date")["value"].sort_index()) for sid, g in panel.groupby("series_id")]
    chunks = Parallel(n_jobs=cap_jobs(n_jobs))(
        delayed(_run_series)(sid, s, origins, horizon, model_cfgs, min_train) for sid, s in groups)
    return pd.DataFrame([r for c in chunks for r in c])
