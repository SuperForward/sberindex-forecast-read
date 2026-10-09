"""Быстрые проверки, что проект собирается и работает на чистой машине.

Данные не нужны: тесты на синтетике и на конфигах. Запуск: python -m pytest -q
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_imports():
    import src.app_service  # noqa: F401
    import src.cpd.methods  # noqa: F401
    import src.store  # noqa: F401
    import worker.runner  # noqa: F401


def test_pipeline_config_is_portable():
    from src.config import load_config
    cfg = load_config("pipeline")
    names = [s["name"] for s in cfg["steps"]]
    assert len(names) == len(set(names)), "имена шагов повторяются"
    for s in cfg["steps"]:
        assert s["group"] in {"fetch", "data", "model", "heavy"}, s["name"]
        mod = ROOT / (s["module"].replace(".", "/") + ".py")
        assert mod.exists(), f"{s['name']}: нет модуля {mod}"
        for p in s.get("needs", []) + s.get("code", []) + s.get("makes", []):
            assert not re.match(r"^([A-Za-z]:|/|\\)", p), f"{s['name']}: абсолютный путь {p}"
        for p in s.get("code", []):
            if p.startswith("data/models/"):        # веса скачиваются отдельно (python -m worker models)
                continue
            assert list(ROOT.glob(p)), f"{s['name']}: нет файла кода {p}"


def test_no_machine_specific_paths():
    """В коде и конфигах нет путей конкретного компьютера."""
    bad = re.compile(r"[A-Za-z]:\\\\?(Users|Main\s)|/home/\w+|C:/Users")
    for pattern in ("src/**/*.py", "app/**/*.py", "scripts/*.py", "worker/*.py", "configs/*.yaml",
                    "app/ui/*.html", "*.ps1", "*.vbs"):
        for f in ROOT.glob(pattern):
            text = f.read_text(encoding="utf-8-sig", errors="ignore")
            assert not bad.search(text), f"путь конкретной машины в {f.relative_to(ROOT)}"


def test_reference_events_tracked():
    """Реестр событий должен быть в репозитории (раньше его скрывало правило *.csv)."""
    f = ROOT / "reference" / "shocks_events.csv"
    assert f.exists()
    ev = pd.read_csv(f)
    assert {"event_id", "start_date", "region", "source_url"} <= set(ev.columns)


def test_seasonal_naive_uses_dates():
    from src.forecast.models import seasonal_naive, seasonal_naive_drift
    s = pd.Series(np.arange(24.0), index=pd.date_range("2023-01-01", periods=24, freq="MS"))
    assert list(seasonal_naive(s, 3, {})) == [12.0, 13.0, 14.0]
    assert seasonal_naive(s.iloc[:7], 12, {})[-1] == 6.0      # h=12: последний известный
    with pytest.raises(ValueError):
        seasonal_naive_drift(s.iloc[:13], 1, {"k": 3})       # мало истории – честный отказ


def test_seasonal_naive_national_uses_known_growth(monkeypatch):
    from src.forecast import models
    g = pd.Series([0.10, 0.20, 0.30], index=pd.date_range("2023-05-01", periods=3, freq="MS"))
    monkeypatch.setattr(models, "_NATIONAL_GROWTH", g)
    s = pd.Series(np.arange(1.0, 8.0), index=pd.date_range("2023-01-01", periods=7, freq="MS"))
    # origin 2023-07, lag 1: известен рост за 2023-06 (0.20), не за 2023-07
    assert models.seasonal_naive_national(s, 12, {"lag": 1})[-1] == pytest.approx(7.0 * 1.2)
    with pytest.raises(ValueError):
        models.seasonal_naive_national(s.iloc[:3], 12, {"lag": 1})   # роста ещё нет


def test_ridge_regularization_ignores_weight_scale():
    """Веса – уровни расходов (~25 тыс. ₽): штраф ridge не должен от них зависеть."""
    from src.forecast.global_model import _fit_predict
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
    y = pd.Series(2 * x["a"] + rng.standard_normal(200) * 0.1)
    w = pd.Series(rng.uniform(0.5, 1.5, 200))
    p1, _ = _fit_predict("ridge", {"alpha": 50.0}, x, y, w, x.head(5))
    p2, _ = _fit_predict("ridge", {"alpha": 50.0}, x, y, w * 25_000, x.head(5))
    assert np.allclose(p1, p2)


def test_detectors_react_to_shift():
    from src.cpd.methods import SCORES, START
    from src.forecast import foundation
    chronos_ok = foundation.available("chronos")[0]        # без весов детекторы на Chronos не проверяются
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 12))
    z[:100, 6:] -= 4                                          # сдвиг у половины рядов
    for name, fn in SCORES.items():
        if name.startswith("chronos") and not chronos_ok:
            continue
        s = fn(z)
        assert s.shape == z.shape
        shifted, clean = s[:100, 6:].max(axis=1).mean(), s[100:, 6:].max(axis=1).mean()
        assert shifted > clean, f"{name} не отличает сдвиг от шума"
        assert np.all(s[:, :START] == 0)


def test_exact_pelt_is_optimal():
    """Перебор разбиений даёт минимум целевой функции (сверка с полным перебором)."""
    import itertools
    from src.cpd.methods import segmentations
    rng = np.random.default_rng(1)
    z = rng.standard_normal((20, 8))
    costs, _ = segmentations(z, max_bkps=2, min_size=2)

    def cost(x, bk):
        b = [0, *bk, len(x)]
        return sum(((x[a:e] - x[a:e].mean()) ** 2).sum() for a, e in zip(b[:-1], b[1:]))
    for i in range(len(z)):
        for k in range(3):
            best = min(cost(z[i], list(c)) for c in itertools.combinations(range(2, 7), k)
                       if all(b - a >= 2 for a, b in zip([0, *c], [*c, 8])))
            assert costs[i, k] == pytest.approx(best)


def test_worker_plan_runs_without_data(tmp_path, monkeypatch):
    """План строится даже без данных: шаги без исходников пропускаются, а не падают."""
    from src import store
    from src.config import load_config
    from worker import runner
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(runner, "ROOT", tmp_path)            # пустой «проект»
    plan = runner.plan(load_config("pipeline"), force=True)
    assert all(s["action"] == "skip" for s in plan if s.get("needs"))


def test_downstream_reruns_when_upstream_runs(tmp_path, monkeypatch):
    """Пересчитанный шаг тянет за собой зависящие от него, даже если их выходы ещё есть.
    И шаг, чьи входы создаст предыдущий шаг, не пропускается как «нет данных»."""
    from src import store
    from worker import runner
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.csv").write_text("1")
    cfg = {"steps": [
        {"name": "build", "group": "data", "module": "m", "needs": ["raw"], "makes": ["proc/x.parquet"]},
        {"name": "model", "group": "model", "module": "m", "needs": ["proc/x.parquet"], "makes": ["rep/y.csv"]},
    ]}
    plan = {s["name"]: s for s in runner.plan(cfg)}
    assert plan["build"]["action"] == "run"
    assert plan["model"]["action"] == "run", plan["model"]["why"]    # вход появится после build
    assert "build" in plan["model"]["why"]


def test_child_env_has_no_utf8_mode(monkeypatch):
    """Дочерние процессы без PYTHONUTF8: иначе Prophet падает на русской Windows."""
    from worker import runner
    monkeypatch.setenv("PYTHONUTF8", "1")
    env = runner.child_env()
    assert "PYTHONUTF8" not in env and env["PYTHONIOENCODING"] == "utf-8"
