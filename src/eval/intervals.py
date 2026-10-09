"""Интервалы прогноза по ошибкам backtest-а и проверка их покрытия.

Ошибка прогноза МО – r = log(факт / прогноз) – складывается из общей части
(медиана r по всем МО в одном окне и горизонте: модель промахнулась по всей
стране, как в мае–июне 2024 г., когда рост г/г прыгнул с ~9% до ~17%) и
своей части МО (r минус эта медиана). Квантили r целиком смешивают их:
общих промахов в backtest-е всего несколько, а точек тысячи, поэтому
интервал выходит узким и в месяц общего промаха покрывает половину МО.
Здесь квантили берутся у свёртки: каждая своя ошибка МО плюс каждый
прошлый общий промах – интервал учитывает, что общий промах повторится.

Скользящая проверка (rolling_coverage): для каждого окна интервал строится
только по точкам с датой ≤ origin (что известно к прогнозу) и проверяется
на прогнозах из этого origin. Окна 2024-06…09, номинал 80%: квантили r
целиком – покрытие 74–78%, свёртка – 78–81% при той же ширине.
"""

import numpy as np
import pandas as pd

LEVELS = (0.1, 0.9)      # интервал 80%


def _parts(p: pd.DataFrame) -> pd.DataFrame:
    p = p.assign(r=np.log(p["y"] / p["yhat"]))
    p = p[np.isfinite(p["r"])]
    c = p.groupby(["origin", "h"])["r"].transform("median")
    return p.assign(common=c, own=p["r"] - c)


def error_band(p: pd.DataFrame, levels: tuple[float, float] = LEVELS) -> pd.DataFrame:
    """Интервал по горизонту: lo, hi – границы для факт / прогноз − 1.
    p – прогнозы одной модели с фактом (backtest)."""
    p = _parts(p.dropna(subset=["y", "yhat"]))
    if p.empty:
        return pd.DataFrame(columns=["lo", "hi"])
    common = p.drop_duplicates(["origin", "h"])["common"].to_numpy()
    rows = {}
    for h, g in p.groupby("h"):
        own = g["own"].to_numpy()
        # свёртка: при ~6 тыс. точек на горизонт и ~7 окнах – до 40 тыс. сумм
        s = (own[:, None] + common[None, :]).ravel()
        lo, hi = np.quantile(s, levels)
        rows[h] = {"lo": np.expm1(lo), "hi": np.expm1(hi)}
    return pd.DataFrame.from_dict(rows, orient="index")


def raw_band(p: pd.DataFrame, levels: tuple[float, float] = LEVELS) -> pd.DataFrame:
    """Прежний способ: квантили факт / прогноз − 1 целиком, по горизонту."""
    r = (p["y"] / p["yhat"] - 1).groupby(p["h"])
    return pd.DataFrame({"lo": r.quantile(levels[0]), "hi": r.quantile(levels[1])})


def rolling_coverage(pred: pd.DataFrame, models: list[str], first_eval: int = 3,
                     levels: tuple[float, float] = LEVELS) -> pd.DataFrame:
    """Покрытие и ширина интервала по окнам: строится по известному к origin,
    проверяется на прогнозах из origin. first_eval – с какого окна: в ранних
    окнах для дальних горизонтов ещё нет ни одной известной ошибки."""
    rows = []
    for model in models:
        m = pred[pred["model"] == model].dropna(subset=["y", "yhat"])
        for o in sorted(m["origin"].unique())[first_eval:]:
            known, test = m[m["date"] <= o], m[m["origin"] == o]
            for way, fn in (("свёртка", error_band), ("квантили ошибки", raw_band)):
                b = fn(known, levels)
                lo, hi = test["h"].map(b["lo"]), test["h"].map(b["hi"])
                r = test["y"] / test["yhat"] - 1
                rows.append({"model": model, "way": way, "origin": f"{o:%Y-%m}",
                             "coverage": float(((r >= lo) & (r <= hi)).mean()),
                             "width": float((hi - lo).mean()), "n": len(test)})
    return pd.DataFrame(rows)
