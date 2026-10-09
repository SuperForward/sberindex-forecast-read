"""Детекторы структурных сдвигов.

Каждый детектор получает z – локальный рост, делённый на шум МО
(МО × месяц), и возвращает «силу сигнала» той же формы; тревога – когда
сила выше порога. Так порог можно подобрать сразу под нужную долю ложных
тревог, не прогоняя детектор заново. Мониторинг начинается с месяца
START: для опорного уровня нужны хотя бы два предыдущих месяца.

Все, кроме PELT, работают онлайн: решение в месяц t принимается только по
данным до t включительно. PELT – офлайн: ищет разломы по всему ряду сразу,
поэтому для раннего предупреждения не годится и оценивается для сравнения.
У PELT нет «силы сигнала», только штраф за лишний разлом, поэтому он
возвращает тревоги при заданном штрафе (pelt_alarms).
"""

import numpy as np

def _start() -> int:
    """Первый месяц мониторинга (индекс в ряду 2024 г.) – configs/cpd.yaml → monitor_start."""
    try:
        from src.config import load_config
        return int(load_config("cpd").get("monitor_start", 2))
    except (OSError, ValueError):
        return 2


START = _start()


def zscore(z: np.ndarray) -> np.ndarray:
    """Отклонение от обычного: |z_t − медиана предыдущих месяцев|."""
    out = np.zeros_like(z)
    for t in range(START, z.shape[1]):
        ref = np.median(z[:, :t], axis=1)
        out[:, t] = np.abs(z[:, t] - ref)
    return out


def ewma(z: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Отклонение от сглаженного прогноза: EWMA прошлых значений как ожидание."""
    out = np.zeros_like(z)
    level = z[:, :START].mean(axis=1)
    for t in range(START, z.shape[1]):
        out[:, t] = np.abs(z[:, t] - level)
        level = alpha * z[:, t] + (1 - alpha) * level
    return out


def cusum(z: np.ndarray, k: float = 0.5) -> np.ndarray:
    """Двусторонний CUSUM относительно уровня первых месяцев.

    Копит отклонения больше k и поднимает флаг, когда накопленная сумма
    превышает порог: ловит небольшие, но устойчивые сдвиги.
    """
    out = np.zeros_like(z)
    ref = z[:, :START].mean(axis=1)
    up = np.zeros(len(z))
    dn = np.zeros(len(z))
    for t in range(START, z.shape[1]):
        e = z[:, t] - ref
        up = np.maximum(0, up + e - k)
        dn = np.maximum(0, dn - e - k)
        out[:, t] = np.maximum(up, dn)
    return out


def bocpd(z: np.ndarray, hazard: float = 1 / 12, prior_var: float = 4.0) -> np.ndarray:
    """Байесовский онлайн-детектор (Adams & MacKay, 2007).

    Держит распределение «сколько месяцев назад был последний сдвиг».
    Модель: нормальное распределение с неизвестным средним и дисперсией 1
    (ряд уже нормирован на шум МО). Сила сигнала – вероятность того, что
    сдвиг случился в текущем или прошлом месяце.
    """
    n, T = z.shape
    out = np.zeros_like(z)
    for i in range(n):
        mu0 = z[i, :START].mean()
        R = np.array([1.0])                  # P(длина серии без сдвига = r)
        mu = np.array([mu0])                  # апостериорное среднее для каждой r
        v = np.array([prior_var])             # апостериорная дисперсия среднего
        for t in range(T):
            x = z[i, t]
            var = 1.0 + v
            pred = np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
            growth = R * pred * (1 - hazard)
            cp = (R * pred * hazard).sum()
            R = np.append(cp, growth)
            R /= R.sum()
            # обновление: новая серия начинается с априорного распределения
            v_new = 1.0 / (1.0 / v + 1.0)
            mu_new = v_new * (mu / v + x)
            mu = np.append(mu0, mu_new)
            v = np.append(prior_var, v_new)
            if t >= START:
                out[i, t] = R[:2].sum()
    return out


def segmentations(z: np.ndarray, max_bkps: int = 5, min_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Лучшие разбиения каждого ряда на k+1 кусков (k = 0..max_bkps), модель l2.

    Возвращает (стоимость [n, K], разломы [n, K, max_bkps], −1 – нет разлома).
    PELT решает задачу «стоимость + штраф × число разломов»; её оптимум всегда
    одно из этих разбиений, поэтому решение для любого штрафа выбирается из
    таблицы без пересчёта. Для 12 точек перебор точный и быстрый.
    """
    n, T = z.shape
    c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(z, axis=1)], axis=1)
    c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(z ** 2, axis=1)], axis=1)

    def cost(a, b):  # сумма квадратов отклонений от среднего на [a, b)
        s = c1[:, b] - c1[:, a]
        return c2[:, b] - c2[:, a] - s * s / (b - a)

    K = max_bkps + 1
    F = np.full((K, n, T + 1), np.inf)
    arg = np.zeros((K, n, T + 1), dtype=int)
    for b in range(min_size, T + 1):
        F[0, :, b] = cost(0, b)
    for k in range(1, K):
        for b in range(min_size * (k + 1), T + 1):
            cands = np.stack([F[k - 1, :, a] + cost(a, b) for a in range(min_size * k, b - min_size + 1)], axis=1)
            j = np.argmin(cands, axis=1)
            F[k, :, b] = cands[np.arange(n), j]
            arg[k, :, b] = j + min_size * k
    costs = F[:, :, T].T
    bkps = np.full((n, K, max_bkps), -1)
    for k in range(1, K):
        pos = np.full(n, T)
        for kk in range(k, 0, -1):
            pos = arg[kk, np.arange(n), pos]
            bkps[:, k, kk - 1] = pos
    return costs, bkps


def pelt_alarms(seg: tuple[np.ndarray, np.ndarray], pen: float, T: int) -> np.ndarray:
    """Решение PELT при штрафе pen: тревоги в месяцы найденных разломов. Офлайн."""
    costs, bkps = seg
    k = np.argmin(costs + pen * np.arange(costs.shape[1]), axis=1)
    out = np.zeros((len(costs), T), dtype=bool)
    b = bkps[np.arange(len(costs)), k]
    rows, cols = np.nonzero(b >= START)
    out[rows, b[rows, cols]] = True
    return out


def pelt_ruptures(x: np.ndarray, pen: float) -> list[int]:
    """Тот же PELT через ruptures – для сверки на части рядов."""
    import ruptures as rpt
    return rpt.Pelt(model="l2", min_size=2, jump=1).fit(x.reshape(-1, 1)).predict(pen=pen)[:-1]


def chronos_interval(z: np.ndarray) -> np.ndarray:
    """Фундаментальная модель как детектор: Chronos-Bolt по истории до t−1
    прогнозирует месяц t (без дообучения), сила сигнала – насколько факт
    выходит за середину прогноза в долях половины 80%-го интервала (10–90%,
    configs/foundation.yaml → quantiles). Модель
    сама оценивает, насколько ряд обычно «гуляет», – это и есть порог.
    Нет весов модели – src.forecast.foundation.Unavailable (метод пропускается)."""
    from src.forecast import foundation
    out = np.zeros_like(z)
    for t in range(START, z.shape[1]):
        q = foundation.predict("chronos", list(z[:, :t]), 1)[:, 0, :]      # [рядов, (нижний, медиана, верхний)]
        half = np.maximum((q[:, 2] - q[:, 0]) / 2, 1e-6)
        out[:, t] = np.abs(z[:, t] - q[:, 1]) / half
    return out


def chronos_median(z: np.ndarray) -> np.ndarray:
    """То же ожидание от Chronos-Bolt, но отклонение – в «сигмах» самого МО
    (z уже нормирован на шум МО), как у z-порога и EWMA. Отделяет качество
    прогноза модели от качества её интервалов."""
    from src.forecast import foundation
    out = np.zeros_like(z)
    for t in range(START, z.shape[1]):
        q = foundation.predict("chronos", list(z[:, :t]), 1)[:, 0, :]
        out[:, t] = np.abs(z[:, t] - q[:, 1])
    return out


SCORES = {"zscore": zscore, "ewma": ewma, "cusum": cusum, "bocpd": bocpd,
          "chronos_interval": chronos_interval, "chronos_median": chronos_median}
ONLINE = {"zscore": True, "ewma": True, "cusum": True, "bocpd": True, "pelt": False,
          "chronos_interval": True, "chronos_median": True}
