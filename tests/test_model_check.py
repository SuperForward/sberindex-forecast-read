"""Проверка воркера перед публикацией: модели не ухудшились резко (worker.runner.model_check)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worker import runner  # noqa: E402


def _pred(scale: dict, y_shift: float = 0.0, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.uniform(20000, 40000, n)
    rows = []
    for model, k in scale.items():
        err = rng.normal(0, 1000, n) * k
        rows.append(pd.DataFrame({"backtest": "models", "model": model, "series_id": [f"{i}|all" for i in range(n)],
                                  "origin": pd.Timestamp("2024-09-01"), "h": 1, "y": y + y_shift, "yhat": y + err}))
    return pd.concat(rows, ignore_index=True)


def _run(monkeypatch, old, new, tmp_path):
    db = tmp_path / "db.duckdb"
    db.write_text("")
    monkeypatch.setattr(runner.store, "DATA_DB", db)
    monkeypatch.setattr(runner.store, "load", lambda name: old)
    monkeypatch.setitem(runner.store.SOURCES, "predictions", lambda: new)
    return runner.model_check({"max_model_mae_rise": 0.25, "min_model_points": 1000})


def test_broken_model_is_caught(monkeypatch, tmp_path):
    old = _pred({"lightgbm": 1.0, "ridge": 1.0})
    new = _pred({"lightgbm": 1.05, "ridge": 2.0})          # ridge сломался
    problems = _run(monkeypatch, old, new, tmp_path)
    assert len(problems) == 1 and "ridge" in problems[0]


def test_revised_data_is_not_compared(monkeypatch, tmp_path):
    old = _pred({"ridge": 1.0})
    new = _pred({"ridge": 2.0}, y_shift=500)               # факт пересмотрен – точки не общие
    assert _run(monkeypatch, old, new, tmp_path) == []
