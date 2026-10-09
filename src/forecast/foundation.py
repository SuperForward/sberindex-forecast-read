"""Фундаментальные модели временных рядов: Chronos-Bolt (Amazon), TimesFM 2.5 (Google).

Предобученные нейросети, прогноз без дообучения на наших данных (zero-shot).
Веса лежат в data/models/ (python -m worker models – скачать), работа – без
интернета. Нет весов или библиотек – модель пропускается с предупреждением,
остальной расчёт не ломается.

Две постановки:
- "growth" – модель прогнозирует ряд роста г/г (log y_t − log y_{t−12}),
  уровень = прошлогоднее значение × exp(прогноз). Годовую сезонность при
  двух годах истории модель сама не выделит, а так она берётся из прошлого
  года – как у baseline и глобальных моделей. Нужен хотя бы один известный
  рост г/г (история ≥ 13 месяцев).
- "level" – прогноз прямо по уровням. Хуже, но работает при истории короче
  года (горизонт 12 месяцев из 2023 г., где рост г/г ещё не известен).
- "relative" – отклонение МО от общего фактора (src/forecast/factor.py,
  параметр k – национальный или региональный фактор).

Чекпойнт Chronos – params["weights"] (папка в data/models), по умолчанию small.
"""

import logging
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import DATA_DIR, NO_WINDOW, cpu_budget

log = logging.getLogger("forecast")
MODELS_DIR = DATA_DIR / "models"
WEIGHTS = {"chronos": MODELS_DIR / "chronos-bolt-small", "timesfm": MODELS_DIR / "timesfm-2.5-200m-pytorch"}
# все скачиваемые чекпойнты: папка в data/models -> репозиторий Hugging Face
REPOS = {"chronos-bolt-small": "amazon/chronos-bolt-small", "chronos-bolt-base": "amazon/chronos-bolt-base",
         "timesfm-2.5-200m-pytorch": "google/timesfm-2.5-200m-pytorch",
         # эмбеддинги заголовков новостей (src/news/model.py)
         "multilingual-e5-base": "intfloat/multilingual-e5-base"}


def _weights(backend: str, weights: str | None = None):
    return MODELS_DIR / weights if weights else WEIGHTS[backend]
def _cfg() -> dict:
    """Параметры запуска моделей – configs/foundation.yaml."""
    from src.config import load_config
    return load_config("foundation")


QUANTILES = tuple(_cfg().get("quantiles", [0.1, 0.5, 0.9]))


class Unavailable(RuntimeError):
    """Нет весов или библиотек – модель пропускается."""


def available(backend: str, weights: str | None = None) -> tuple[bool, str]:
    w = _weights(backend, weights)
    if not (w / "model.safetensors").exists():
        return False, f"нет весов {w} (скачать: python -m worker models)"
    try:
        import torch  # noqa: F401
        __import__("chronos" if backend == "chronos" else "timesfm")
    except ImportError as e:
        return False, f"не установлена библиотека: {e.name}"
    return True, "ok"


@lru_cache(maxsize=3)
def _load(backend: str, weights: str | None = None):
    ok, why = available(backend, weights)
    if not ok:
        raise Unavailable(why)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # веса локальные, в сеть не ходим
    import torch
    torch.set_num_threads(cpu_budget())
    if backend == "chronos":
        from chronos import BaseChronosPipeline
        dtype = getattr(torch, _cfg().get("chronos", {}).get("dtype", "float32"))
        return BaseChronosPipeline.from_pretrained(str(_weights(backend, weights)), device_map="cpu",
                                                   torch_dtype=dtype)
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(_weights(backend, weights)), local_files_only=True)
    m.compile(timesfm.ForecastConfig(**_cfg().get("timesfm", {})))
    return m


def predict(backend: str, contexts: list[np.ndarray], horizon: int, weights: str | None = None) -> np.ndarray:
    """Квантили QUANTILES для каждого ряда: массив [рядов, horizon, 3]."""
    model = _load(backend, weights)
    if backend == "chronos":
        import torch
        q, _ = model.predict_quantiles([torch.tensor(np.asarray(c, float)) for c in contexts],
                                       prediction_length=horizon, quantile_levels=list(QUANTILES))
        return q.numpy()
    point, qs = model.forecast(horizon=horizon, inputs=[np.asarray(c, float) for c in contexts])
    # TimesFM: qs[..., 0] – среднее, дальше квантили 0.1…0.9 шагом 0.1
    idx = [int(round(q * 10)) for q in QUANTILES]
    return np.stack([qs[..., i] for i in idx], axis=-1)


def extend_future(Y: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Матрица рядов × месяцы + horizon пустых месяцев за краем данных."""
    last = Y.columns.max()
    return Y.reindex(columns=list(Y.columns) + [last + pd.DateOffset(months=h) for h in range(1, horizon + 1)])


def run_backtest(panel: pd.DataFrame, origins: list, horizon: int, name: str, params: dict,
                 future: bool = False) -> pd.DataFrame:
    """Прогнозы в формате остальных моделей (+ q_lo, q_hi). panel: series_id, date, value.
    future – прогноз вперёд из последнего месяца: факта нет (y = NaN)."""
    backend, mode, weights = params.get("backend", "chronos"), params.get("mode", "growth"), params.get("weights")
    ok_, why = available(backend, weights)
    if not ok_:
        raise Unavailable(why)
    Y = panel.pivot_table(index="series_id", columns="date", values="value").sort_index(axis=1)
    if future:
        Y = extend_future(Y, horizon)
    dates = list(Y.columns)
    rows = []
    for origin in origins:
        t = dates.index(origin) if origin in dates else None
        if t is None or t + horizon >= len(dates):
            continue
        hist = Y.iloc[:, : t + 1]
        ok = hist.notna().all(axis=1)
        if not future:
            ok &= Y.iloc[:, t + 1: t + 1 + horizon].notna().all(axis=1)
        H, F = hist[ok].to_numpy(float), Y.iloc[:, t + 1: t + 1 + horizon][ok].to_numpy(float)
        if mode == "growth":
            if t < 12:
                continue                     # роста г/г ещё нет
            L = np.log(H)
            q = predict(backend, list(L[:, 12:] - L[:, :-12]), horizon, weights)
            base = np.column_stack([Y.iloc[:, t + h - 11][ok].to_numpy(float) for h in range(horizon)])
            yq = base[..., None] * np.exp(q)
        elif mode == "relative":
            if t < 12:
                continue                     # фактору нужен хотя бы один рост г/г
            # отклонение от общего фактора (src/forecast/factor.py): сезонность
            # несёт фактор, модель видит ряд без неё
            from src.forecast.factor import factor_forecast, relative_context
            R, Fm = relative_context(Y, t, float(params.get("k", "inf")))
            R = R.reindex(hist[ok].index)
            q = predict(backend, list(R.to_numpy(float)), horizon, weights)
            fc = factor_forecast(Fm.reindex(hist[ok].index), horizon, 1)
            yq = np.exp(q + fc[..., None])
        else:
            yq = predict(backend, list(H), horizon, weights)
        for i, sid in enumerate(hist[ok].index):
            for h in range(horizon):
                rows.append((sid, name, origin, dates[t + 1 + h], h + 1, F[i, h],
                             float(yq[i, h, 1]), float(yq[i, h, 0]), float(yq[i, h, 2])))
        log.debug("foundation  %s (%s/%s) окно %s: %d рядов", name, backend, mode, f"{origin:%Y-%m}", int(ok.sum()))
    return pd.DataFrame(rows, columns=["series_id", "model", "origin", "date", "h", "y", "yhat", "q_lo", "q_hi"])


def download(echo=print) -> None:
    """Скачать веса в data/models (≈3 ГБ). Проверка целостности – sha256 с Hugging Face."""
    import hashlib
    import json
    import subprocess
    for folder, repo in REPOS.items():
        d = MODELS_DIR / folder
        d.mkdir(parents=True, exist_ok=True)
        meta = json.loads(subprocess.run(["curl", "-sS", "-f", f"https://huggingface.co/api/models/{repo}?blobs=true"],
                                         capture_output=True, check=True, creationflags=NO_WINDOW).stdout)
        for s in meta["siblings"]:
            f = s["rfilename"]
            if "/" in f or not (f.endswith((".json", ".safetensors")) or f == "README.md"):
                continue                        # подпапки (onnx/, openvino/) не нужны
            out = d / f
            if out.exists() and out.stat().st_size == s.get("size"):
                continue
            echo(f"{repo}/{f}: {s.get('size', 0) / 1e6:.0f} МБ")
            # curl – хранилище сертификатов Windows (у Python свой список, и при
            # проверке HTTPS антивирусом он отвергает соединение)
            subprocess.run(["curl", "-sS", "-f", "-L", "--retry", "5", "--retry-all-errors", "-o", str(out) + ".part",
                            f"https://huggingface.co/{repo}/resolve/main/{f}"], check=True)
            os.replace(str(out) + ".part", out)
            want = (s.get("lfs") or {}).get("sha256")
            if want and hashlib.sha256(out.read_bytes()).hexdigest() != want:
                out.unlink()
                raise RuntimeError(f"{repo}/{f}: контрольная сумма не совпала")
        echo(f"{repo}: готово")
