"""Метрики прогноза. MAE обязательна по условиям задачи, R² – опционально.

R² на уровнях по всем МО сразу завышен: большая часть дисперсии – разница
уровней между МО, которую угадывает любая модель. Поэтому считаем ещё
r2_within: R² после вычитания среднего каждого ряда на тестовом окне.
"""

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)))


def summarize(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """pred: колонки series_id, origin, date, h, y, yhat + ключи из by.

    Центрирование для R2_within – по ряду и окну (series_id, origin, model)
    до группировки, чтобы разрезы по горизонту тоже были корректны.
    """
    pred = pred.copy()
    center = pred.groupby(["series_id", "origin", "model"])["y"].transform("mean")
    pred["_yc"] = pred["y"] - center
    pred["_yhc"] = pred["yhat"] - center

    def agg(g: pd.DataFrame) -> pd.Series:
        y, yhat = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
        return pd.Series({"n": len(g), "MAE": mae(y, yhat), "WAPE": wape(y, yhat),
                          "R2": r2(y, yhat),
                          "R2_within": r2(g["_yc"].to_numpy(float), g["_yhc"].to_numpy(float))})
    return pred.groupby(by)[["y", "yhat", "_yc", "_yhc"]].apply(agg).reset_index()
