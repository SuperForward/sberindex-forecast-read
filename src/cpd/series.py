"""Ряды для обнаружения структурных сдвигов.

Детекторы смотрят не на уровень расходов, а на «локальный рост»:
  x_{i,t} = (log y_{i,t} − log y_{i,t−12}) − медиана по всем МО того же месяца.
Годовое сравнение убирает сезонность, вычитание медианы – общие для страны
сдвиги (январь 2024 у всех МО). Остаётся то, что случилось именно с этим МО.

Мера шума σ_i – по 2023 году (до периода мониторинга): устойчивый разброс
месячных изменений локального уровня; для годового роста умножается на √2
(разность двух независимых ошибок). Снизу ограничена половиной медианы по МО,
чтобы очень «гладкие» МО не давали ложных тревог на мелочах.
"""

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

MAD_K = 1.4826  # MAD -> стандартное отклонение для нормального распределения


def load_levels(category: str = "all") -> pd.DataFrame:
    """Матрица уровней МО × месяц (только МО с полными 24 месяцами)."""
    p = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet")
    p = p[p["category"] == category]
    Y = p.pivot_table(index="oktmo", columns="date", values="value")
    return Y.dropna()


def local_growth(Y: pd.DataFrame, remove_common: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """(x – локальный рост за 2024 г., МО × 12 мес.; σ – шум по МО).

    remove_common=False – абляция: без вычитания общего для всех МО сдвига.
    """
    L = np.log(Y)
    yoy = L.iloc[:, 12:].to_numpy() - L.iloc[:, :-12].to_numpy()
    x = yoy - np.median(yoy, axis=0, keepdims=True) if remove_common else yoy
    X = pd.DataFrame(x, index=Y.index, columns=Y.columns[12:])

    mom = np.diff(L.iloc[:, :12].to_numpy(), axis=1)
    if remove_common:
        mom = mom - np.median(mom, axis=0, keepdims=True)
    med = np.median(mom, axis=1, keepdims=True)
    s = MAD_K * np.median(np.abs(mom - med), axis=1) * np.sqrt(2)
    s = np.maximum(s, 0.5 * np.median(s))
    return X, pd.Series(s, index=Y.index)


def inject_shifts(Y: pd.DataFrame, share: float, sizes: list[float], months: list[int],
                  seed: int = 42, duration: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетические сдвиги уровня с известной датой.

    Случайной доле МО (share) с месяца τ уровень умножается на (1 + s) до
    конца ряда – устойчивый сдвиг, как после закрытия предприятия или
    паводка; duration – временный шок: уровень возвращается через duration
    месяцев. months – допустимые τ как индексы столбцов Y.
    Возвращает (изменённая матрица, разметка: oktmo, tau, size, end).
    """
    rng = np.random.default_rng(seed)
    Y2 = Y.copy()
    idx = rng.choice(len(Y), size=int(len(Y) * share), replace=False)
    rows = []
    for i in idx:
        tau = int(rng.choice(months))
        s = float(rng.choice(sizes))
        end = Y.shape[1] if duration is None else min(tau + duration, Y.shape[1])
        Y2.iloc[i, tau:end] *= (1 + s)
        rows.append({"oktmo": Y.index[i], "tau": tau, "size": s, "end": end})
    return Y2, pd.DataFrame(rows)


def apply_shifts(Y: pd.DataFrame, lab: pd.DataFrame, factor=None) -> pd.DataFrame:
    """Те же сдвиги (разметка inject_shifts), что и в ряде «все расходы», –
    для ряда категории: МО и τ совпадают, уровень умножается на (1 + s·f).
    factor – множитель f для каждого сдвига (по умолчанию 1: категория
    сдвигается так же, как все расходы)."""
    A = Y.to_numpy(copy=True)
    f = np.ones(len(lab)) if factor is None else np.asarray(factor)
    end = lab["end"] if "end" in lab else [A.shape[1]] * len(lab)
    for i, tau, e, s, k in zip(Y.index.get_indexer(lab["oktmo"]), lab["tau"], end, lab["size"], f):
        A[i, tau:e] *= 1 + s * k
    return pd.DataFrame(A, index=Y.index, columns=Y.columns)


def category_z(levels: dict[str, pd.DataFrame], clip: float = 8.0) -> np.ndarray:
    """Сигнал по нескольким категориям: локальный рост каждой, делённый на шум
    МО в ней, среднее по категориям × √k (при независимом шуме – снова
    единичная дисперсия). Сдвиг всех расходов (паводок: −2…−5 п.п. к росту
    при шуме ~3 п.п.) в отдельных категориях бывает сильнее (маркетплейсы,
    здоровье, транспорт в апреле 2024 г.: −7…−14%), а шум категорий частично
    независим. z каждой категории ограничен ±clip: в редких тратах мелких МО
    бывают скачки в сотни σ."""
    zs = []
    for Y in levels.values():
        X, s = local_growth(Y)
        zs.append(np.clip(X.to_numpy() / s.to_numpy()[:, None], -clip, clip))
    return np.mean(zs, axis=0) * np.sqrt(len(zs))
