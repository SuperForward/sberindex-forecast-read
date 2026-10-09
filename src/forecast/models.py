"""Локальные модели прогноза: по одному ряду за раз.

Каждая модель – функция (history, horizon, params) -> np.ndarray прогнозов,
где history – pd.Series значений с месячным DatetimeIndex, упорядоченная.
"""

import logging

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def naive(history: pd.Series, horizon: int, params: dict) -> np.ndarray:
    """Последнее значение."""
    return np.repeat(history.iloc[-1], horizon)


def seasonal_naive(history: pd.Series, horizon: int, params: dict) -> np.ndarray:
    """Значение того же месяца год назад (горизонт до 12 месяцев)."""
    if horizon > 12:
        raise ValueError("seasonal_naive: горизонт больше 12 месяцев")
    # Берём по дате, а не по позиции: при истории короче года месяц
    # «цель − 12» может быть известен (при h = 12 это последний месяц).
    last = history.index[-1]
    return np.array([history.get(last + pd.DateOffset(months=h - 12), np.nan)
                     for h in range(1, horizon + 1)], dtype=float)


def seasonal_naive_drift(history: pd.Series, horizon: int, params: dict) -> np.ndarray:
    """Тот же месяц год назад × рост г/г за последние k месяцев.

    Сохраняет сезонность ряда и переносит текущий темп роста – сильный
    базовый уровень при 2 годовых циклах в истории.
    """
    k = params.get("k", 3)
    if len(history) < k + 12:
        raise ValueError(f"seasonal_naive_drift: нужно {k + 12} мес. истории, есть {len(history)}")
    last = history.iloc[-k:].to_numpy(float)
    prev = history.iloc[-k - 12:-12].to_numpy(float)
    growth = last.sum() / prev.sum()
    return seasonal_naive(history, horizon, params) * growth


_NATIONAL_GROWTH = None


def seasonal_naive_national(history: pd.Series, horizon: int, params: dict) -> np.ndarray:
    """Тот же месяц год назад × рост расходов по России из дашборда СберИндекса.

    Рост г/г самого МО в 2023 г. неизвестен (панель с 01.2023), поэтому
    baseline на горизонте 12 месяцев не построить. Общероссийский рост г/г
    СберИндекс публикует с 2018 г. Берётся последнее значение, известное на
    дату прогноза: месяц «origin − lag» (lag – запас на задержку публикации).
    """
    global _NATIONAL_GROWTH
    if _NATIONAL_GROWTH is None:
        from src.data.sberindex import load_national_growth
        _NATIONAL_GROWTH = load_national_growth()
    known = _NATIONAL_GROWTH[:history.index[-1] - pd.DateOffset(months=params.get("lag", 1))]
    if known.empty:
        raise ValueError("seasonal_naive_national: нет роста по России на дату прогноза")
    return seasonal_naive(history, horizon, params) * (1 + known.iloc[-1])


def _ascii_tempdir() -> None:
    """cmdstan (движок Prophet) не пишет во временную папку с не-ASCII путём,
    а у пользователя «Иван» она лежит в его профиле (…/Иван/AppData/Local/Temp).
    Подменяем временную папку до импорта cmdstanpy (он запоминает её при
    импорте). Порядок: короткое 8.3-имя той же папки, папка в проекте,
    C:/sberindex-tmp."""
    import os
    import sys
    import tempfile
    from pathlib import Path

    cur = tempfile.gettempdir()
    if cur.isascii():
        return
    candidates = []
    if sys.platform == "win32":
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(cur, buf, 1024):
            candidates.append(Path(buf.value))
    root = Path(__file__).resolve().parents[2]
    candidates += [root / "data" / ".tmp", Path(os.environ.get("SystemDrive", "C:") + os.sep) / "sberindex-tmp"]
    for c in candidates:
        if not str(c).isascii():
            continue
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_test"
            probe.write_text("ok")
            probe.unlink()
        except OSError:
            continue
        os.environ["TMP"] = os.environ["TEMP"] = str(c)
        tempfile.tempdir = str(c)
        logging.getLogger("forecast").info("prophet_tempdir  %s -> %s (путь с не-ASCII символами)", cur, c)
        return
    logging.getLogger("forecast").warning("prophet_tempdir  нет ASCII-папки для временных файлов, "
                                          "Prophet может упасть: %s", cur)


def prophet(history: pd.Series, horizon: int, params: dict) -> np.ndarray:
    """Prophet, месячные данные. params передаются в конструктор Prophet.

    yearly_seasonality: "auto" (по умолчанию Prophet отключает её при истории
    короче 2 лет), либо число – порядок Фурье.
    """
    _ascii_tempdir()
    from prophet import Prophet

    kw = {"weekly_seasonality": False, "daily_seasonality": False}
    kw.update(params)
    df = pd.DataFrame({"ds": history.index, "y": history.to_numpy(float)})
    try:
        m = Prophet(**kw)
    except AttributeError as e:          # движок cmdstan не загрузился; причина – только в DEBUG
        raise RuntimeError(
            "Prophet: не загрузился движок cmdstan. На русской Windows частая причина – "
            "переменная PYTHONUTF8=1 (cmdstan не читает вывод where.exe в cp866); "
            "запускайте без неё. Подробности: logging.DEBUG логгера prophet.") from e
    m = m.fit(df)
    future = pd.DataFrame({"ds": pd.date_range(history.index[-1], periods=horizon + 1, freq="MS")[1:]})
    return m.predict(future)["yhat"].to_numpy()


MODELS = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_drift": seasonal_naive_drift,
    "seasonal_naive_national": seasonal_naive_national,
    "prophet": prophet,
}
