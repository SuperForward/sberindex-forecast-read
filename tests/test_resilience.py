"""Устойчивость к сбоям: нет данных, битые файлы, плохие архивы, пути с кириллицей.

Первый тест нужен с данными (пропускается на чистой машине), остальные – нет.
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import store  # noqa: E402

HAS_DATA = not store.load("spending_mo").empty


@pytest.mark.skipif(not HAS_DATA, reason="нужны данные (снимок или сборка)")
@pytest.mark.parametrize("missing", list(store.SOURCES))
def test_every_view_survives_a_missing_table(missing, monkeypatch):
    """Нет любой одной таблицы – ни один раздел не падает (кроме «нет панели» → NoData)."""
    from src import app_service as a
    real = store.load
    monkeypatch.setattr(store, "load", lambda n: store.empty(n) if n == missing else real(n))
    a.reload()
    calls = [a.summary, lambda: a.search_mo("ишим"), lambda: a.mo_detail("71705000"), a.model_metrics,
             lambda: a.mo_forecast("71705000"), lambda: a.worst_mo(""),
             lambda: a.shocks_overview("", 0, False), a.horizons_overview, a.cpd_overview,
             lambda: a.cpd_mo("71705000"), a.results_overview, a.data_status]
    try:
        for f in calls:
            try:
                json.dumps(f(), ensure_ascii=False, allow_nan=False)
            except a.NoData:
                assert missing == "spending_mo"
    finally:
        monkeypatch.undo()
        a.reload()


def test_corrupt_duckdb_falls_back_to_files(tmp_path, monkeypatch):
    bad = tmp_path / "bad.duckdb"
    bad.write_bytes(b"not a database" * 100)
    monkeypatch.setattr(store, "DATA_DB", bad)
    monkeypatch.setitem(store.SOURCES, "cpd_by_size", lambda: pd.DataFrame({"size": [-0.1], "recall": [0.5],
                                                                             "method": ["ewma"],
                                                                             "far_target": [0.05]}))
    assert len(store.load("cpd_by_size")) == 1
    assert store.published_at() is None


def test_corrupt_journal_is_recreated(tmp_path, monkeypatch):
    st = tmp_path / "state.sqlite"
    st.write_bytes(b"garbage" * 200)
    monkeypatch.setattr(store, "STATE_DB", st)
    assert store.last_runs() == []
    assert any(p.name.startswith("state.sqlite.corrupt-") for p in tmp_path.iterdir())


def test_publish_and_read_in_cyrillic_folder(tmp_path, monkeypatch):
    """Путь проекта с кириллицей и пробелами (C:\\Users\\Иван\\Мои проекты\\...)."""
    d = tmp_path / "Мои проекты" / "СберИндекс"
    d.mkdir(parents=True)
    monkeypatch.setattr(store, "DATA_DB", d / "база.duckdb")
    monkeypatch.setattr(store, "SOURCES", {"cpd_by_size": lambda: pd.DataFrame(
        {"size": [-0.1], "recall": [0.5], "method": ["ewma"], "far_target": [0.05]})})
    assert store.publish() == {"cpd_by_size": 1}
    assert store.load("cpd_by_size")["method"].tolist() == ["ewma"]


# ------------------------------------------------------------ снимок данных

def _make_zip(path: Path, members: dict) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


@pytest.fixture
def snap(tmp_path, monkeypatch):
    from worker import snapshot
    proj = tmp_path / "проект"
    proj.mkdir()
    monkeypatch.setattr(snapshot, "ROOT", proj)
    monkeypatch.setattr(snapshot, "DIST", proj / "dist")
    monkeypatch.setattr(snapshot, "github_repo", lambda: None)
    return snapshot, tmp_path, proj


def test_snapshot_rejects_paths_outside_data(snap):
    snapshot, tmp, proj = snap
    z = tmp / "evil.zip"
    _make_zip(z, {"data/ok.txt": "1", "../outside.txt": "x", "src/app_service.py": "x"})
    with pytest.raises(SystemExit, match="недопустимые пути"):
        snapshot.fetch(z.as_uri())
    assert not (tmp / "outside.txt").exists() and not (proj / "src").exists()


def test_snapshot_rejects_broken_archive(snap):
    snapshot, tmp, proj = snap
    z = tmp / "broken.zip"
    z.write_bytes(b"PK\x03\x04 not a zip at all")
    with pytest.raises(SystemExit, match="повреждён"):
        snapshot.fetch(z.as_uri())


def test_snapshot_checks_sha256(snap):
    snapshot, tmp, proj = snap
    z = tmp / "data.zip"
    _make_zip(z, {"data/processed/x.txt": "1"})
    (tmp / "data.zip.sha256").write_text("0" * 64 + "  data.zip\n", encoding="ascii")
    with pytest.raises(SystemExit, match="контрольная сумма"):
        snapshot.fetch(z.as_uri())
    (tmp / "data.zip.sha256").write_text(hashlib.sha256(z.read_bytes()).hexdigest() + "  data.zip\n",
                                         encoding="ascii")
    snapshot.fetch(z.as_uri())
    assert (proj / "data" / "processed" / "x.txt").read_text() == "1"


# ------------------------------------------------------------ воркер и автообновление

@pytest.fixture
def journal(tmp_path, monkeypatch):
    """Свой журнал и своя блокировка: тесты не должны мешать настоящему
    воркеру (и зависеть от того, запущен ли он сейчас)."""
    from worker import runner
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(runner, "LOCK_FILE", tmp_path / ".worker.lock")
    return tmp_path


def test_fetch_step_respects_its_interval(journal):
    import time
    from worker import runner
    step = {"name": "fetch_pmo", "group": "fetch", "every_h": 720, "module": "m"}
    store.set_meta("fetch_ok:fetch_pmo", str(time.time() - 3600))            # час назад
    assert runner._decide(step, {}, False, False, True, None, [])[0] == "skip"
    store.set_meta("fetch_ok:fetch_pmo", str(time.time() - 800 * 3600))      # больше месяца
    assert runner._decide(step, {}, False, False, True, None, [])[0] == "run"


def test_fetch_retry_waits_after_failed_attempt(journal):
    import time
    from worker import runner
    sched = {"fetch_every_h": 24, "retry_after_fail_min": 60}
    store.set_meta("last_fetch", "0")
    store.set_meta("last_fetch_attempt", str(time.time() - 5 * 60))           # попытка 5 мин назад
    assert not runner.fetch_due(sched)
    store.set_meta("last_fetch_attempt", str(time.time() - 2 * 3600))
    assert runner.fetch_due(sched)


def test_stale_running_rows_become_interrupted(journal, monkeypatch):
    from contextlib import closing
    from worker import runner
    with closing(store.state()) as con, con:
        con.execute("insert into runs (trigger, started, status) values ('manual', '2026-01-01T00:00:00', 'running')")
    monkeypatch.setattr(runner, "plan", lambda *a, **k: [])
    monkeypatch.setattr(runner, "load_config", lambda n: {"steps": []})
    monkeypatch.setattr(store, "DATA_DB", journal / "db.duckdb")
    monkeypatch.setattr(store, "publish", lambda: {})
    runner.run(echo=lambda *_: None)
    statuses = [r["status"] for r in store.last_runs(5)]
    assert "interrupted" in statuses and "running" not in statuses


def test_auto_update_decisions(journal, monkeypatch):
    import time
    from src import app_service as a
    from worker import runner
    # решения проверяются при включённом автообновлении, независимо от значения в configs/pipeline.yaml
    cfg = a.load_config("pipeline")
    sched = {**cfg["schedule"], "app_auto_update": True, "app_auto_recalc": True}
    monkeypatch.setattr(a, "load_config", lambda n: {**cfg, "schedule": sched})
    started = []
    monkeypatch.setattr(a, "recalc_start", lambda mode, trigger="manual": started.append((mode, trigger)) or {})
    monkeypatch.setattr(runner, "is_busy", lambda: False)
    store.set_meta("last_fetch", str(time.time()))
    store.set_meta("last_fetch_attempt", str(time.time()))
    a._PLAN["steps"] = [{"name": "cpd", "group": "model", "action": "run", "why": "код"}]
    assert a.auto_update_tick()["action"] == "recalc" and started[-1] == ("auto", "app")
    a._PLAN["steps"] = [{"name": "horizons", "group": "heavy", "action": "run", "why": "код"}]
    assert a.auto_update_tick()["action"] == "none"            # тяжёлое – не автоматически
    store.set_meta("last_fetch", "0")
    store.set_meta("last_fetch_attempt", "0")
    assert a.auto_update_tick()["action"] == "fetch" and started[-1] == ("fetch", "app")
    monkeypatch.setattr(runner, "is_busy", lambda: True)
    assert a.auto_update_tick()["action"] == "busy"
    sched["app_auto_update"] = False
    assert a.auto_update_tick()["action"] == "off"            # автообновление выключено в конфиге
    a._PLAN["steps"] = None


@pytest.mark.skipif(not HAS_DATA, reason="нужны данные")
def test_default_backtest_ensembles_do_not_shadow_main_ones():
    """Ансамбли базового backtest-а не должны подменять одноимённые из основного."""
    from src import app_service as a
    a.reload()
    p = a._predictions()
    assert not ((p["backtest"] == "default") & p["model"].str.startswith("ens_")).any()
    assert p.groupby(["model", "series_id", "origin", "h"]).size().max() == 1


def test_foundation_models_skip_without_weights(tmp_path, monkeypatch):
    """Нет весов фундаментальной модели – Unavailable с понятной причиной, а не падение."""
    from src.forecast import foundation
    monkeypatch.setitem(foundation.WEIGHTS, "chronos", tmp_path / "нет")
    foundation._load.cache_clear()
    ok, why = foundation.available("chronos")
    assert not ok and "worker models" in why
    with pytest.raises(foundation.Unavailable):
        foundation.predict("chronos", [pd.Series([1.0, 2.0]).to_numpy()], 1)
    foundation._load.cache_clear()
