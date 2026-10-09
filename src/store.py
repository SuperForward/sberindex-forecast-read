"""Хранилище: опубликованные данные (DuckDB) и журнал воркера (SQLite).

data/sberindex.duckdb – всё, что показывает приложение: панель расходов,
прогнозы, метрики, тревоги детекторов. Файл пересобирается целиком и
подменяется атомарно (os.replace), поэтому приложение никогда не видит
«полуобновлённые» данные. Приложение открывает его только на чтение и сразу
закрывает, чтобы воркер мог подменить файл.

data/state.sqlite – журнал запусков и отпечатки входов шагов. SQLite, а не
DuckDB: воркер пишет, а приложение в это же время читает статус (WAL).

Если базы ещё нет (свежий клон без снимка), load() читает те же таблицы
прямо из файлов в data/ и reports/ – приложение работает и так.
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime

import pandas as pd

from src.config import DATA_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, ROOT

log = logging.getLogger("data")

DATA_DB = DATA_DIR / "sberindex.duckdb"
STATE_DB = DATA_DIR / "state.sqlite"


# ------------------------------------------------------------ таблицы данных

def _csv(path, **kw):
    return lambda: pd.read_csv(path, **kw) if path.exists() else None


def _parquet(path):
    return lambda: pd.read_parquet(path) if path.exists() else None


def _predictions():
    frames = []
    for name in ("default", "models"):
        f = REPORTS_DIR / "backtest" / name / "predictions.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f).assign(backtest=name))
    return pd.concat(frames, ignore_index=True) if frames else None


def _events():
    if not (ROOT / "reference" / "shocks_events.csv").exists():
        return None
    from src.data import shocks
    try:
        return shocks.load_events()
    except FileNotFoundError:     # нет панели или карты МО – событиям не к чему привязаться
        return None


# имя таблицы -> как собрать её из файлов
SOURCES = {
    "spending_mo": _parquet(PROCESSED_DIR / "spending_mo.parquet"),
    "mo_population": _parquet(PROCESSED_DIR / "mo_population.parquet"),
    "mo_annual": _parquet(PROCESSED_DIR / "mo_annual.parquet"),
    "mo_oktmo_map": _csv(INTERIM_DIR / "mo_oktmo_map.csv", dtype=str),
    "predictions": _predictions,
    "forecast": _parquet(REPORTS_DIR / "backtest" / "models" / "forecast.parquet"),
    "ensemble_members": _csv(REPORTS_DIR / "backtest" / "models" / "ensemble_members.csv"),
    "feature_importance": _csv(REPORTS_DIR / "backtest" / "models" / "feature_importance.csv"),
    "shock_candidates": _csv(REPORTS_DIR / "shock_candidates.csv", dtype={"oktmo": str}),
    "events": _events,
    "horizons_metrics": _csv(REPORTS_DIR / "horizons" / "metrics.csv"),
    "cpd_comparison": _csv(REPORTS_DIR / "cpd" / "comparison.csv"),
    "cpd_by_size": _csv(REPORTS_DIR / "cpd" / "by_size.csv"),
    "cpd_real_events": _csv(REPORTS_DIR / "cpd" / "real_events.csv", dtype={"oktmo": str}),
    "cpd_alarms": _parquet(REPORTS_DIR / "cpd" / "alarms.parquet"),
    "news_matches": _parquet(PROCESSED_DIR / "news_matches.parquet"),
    "news_events": _csv(REPORTS_DIR / "news" / "events.csv"),
    "news_cpd": _csv(REPORTS_DIR / "news" / "cpd_two_key.csv", dtype={"oktmo": str}),
    "news_cpd_stats": _csv(REPORTS_DIR / "news" / "cpd_two_key_stats.csv"),
    "news_forecast": _csv(REPORTS_DIR / "news" / "forecast.csv"),
    "news_event_eval": _csv(REPORTS_DIR / "backtest" / "events" / "event_eval.csv"),
    "null_test_runs": _csv(REPORTS_DIR / "null_test" / "models" / "runs.csv"),
}


# Колонки, которые ждёт приложение. Нет таблицы – отдаётся пустая, но с
# этими колонками: раздел покажет «нет данных», а не упадёт с KeyError.
COLUMNS = {
    "spending_mo": ["date", "mo_name", "category", "value", "oktmo", "region_code", "region", "match_type"],
    "mo_population": ["oktmo", "mo_name", "region_code", "pop", "pop_urban", "pop_rural", "pop_source"],
    "mo_annual": ["oktmo", "year"],
    "mo_oktmo_map": ["mo_name", "mo_type", "match_type", "score", "n_candidates", "oktmo", "oktmo_name",
                     "region_code", "region", "candidates", "oktmo_version"],
    "predictions": ["series_id", "model", "origin", "date", "h", "y", "yhat", "category", "backtest"],
    "forecast": ["series_id", "model", "origin", "date", "h", "yhat", "q_lo", "q_hi", "category"],
    "ensemble_members": ["model", "configured", "used", "missing", "built"],
    "feature_importance": ["model", "category", "feature", "share"],
    "shock_candidates": ["oktmo", "year", "shock_type", "metric", "change", "z", "mo_name", "region",
                         "in_panel", "caution"],
    "events": ["event_id", "start_date", "region", "mo_query", "shock_type", "subtype", "description",
               "source_url", "oktmo", "mo_name", "in_panel"],
    "horizons_metrics": ["model", "H", "n", "MAE", "WAPE", "R2", "R2_within", "coverage"],
    "cpd_comparison": ["method", "online", "far_target", "threshold", "far_clean", "recall", "delay_months",
                       "early_alarm", "n_shifts"],
    "cpd_by_size": ["size", "recall", "method", "far_target"],
    "cpd_real_events": ["method", "far_target", "oktmo", "name", "event", "first_alarm", "caught_3m",
                        "alarm_before"],
    "cpd_alarms": ["oktmo", "date", "x", "z", "score", "alarm", "method", "threshold"],
    "news_matches": ["published", "month", "level", "oktmo", "cov_region", "types", "p_type", "p_severe",
                     "title", "url"],
    "news_events": ["event", "date", "region", "mo", "in_panel", "news_mo", "news_region", "found"],
    "news_cpd": ["mo", "oktmo", "news_months", "one_key", "two_key", "max_score", "variant"],
    "news_cpd_stats": ["variant", "method", "thr_main", "thr_news", "mo_one_key", "mo_two_key", "mo_added",
                       "alarms_with_news", "mo_with_news_2024"],
    "news_forecast": ["model", "features", "subset", "points", "mae_base", "mae_news", "news_importance", "change_%"],
    "news_event_eval": ["slice", "set", "points", "MAE", "MAE_min", "MAE_max", "delta_pct", "seeds_better",
                        "spread_cur_pct", "ci_lo_pct", "ci_hi_pct", "verdict"],
    "null_test_runs": ["scenario", "seed", "model", "MAE", "R2_within"],
}
DATE_COLUMNS = {"date", "origin", "start_date"}


def empty(name: str) -> pd.DataFrame:
    cols = COLUMNS.get(name, [])
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in DATE_COLUMNS else object)
                         for c in cols})


def _duck(read_only: bool = True, path=None):
    import duckdb
    return duckdb.connect(str(path or DATA_DB), read_only=read_only)


def load_fresh(name: str) -> pd.DataFrame:
    """Таблица прямо из файлов шагов, минуя DuckDB: для шагов воркера, которые
    читают результаты других шагов того же прогона (публикация – в конце)."""
    df = SOURCES[name]()
    return empty(name) if df is None or df.empty else df


def load(name: str) -> pd.DataFrame:
    """Таблица для приложения: из DuckDB, если опубликована, иначе из файлов.

    Нет ни там, ни там – пустая таблица (раздел покажет «нет данных»).
    """
    if DATA_DB.exists():
        for attempt in range(5):
            try:
                with closing(_duck()) as con:
                    if con.execute("select count(*) from information_schema.tables where table_name = ?",
                                   [name]).fetchone()[0]:
                        df = con.execute(f'select * from "{name}"').df()
                        return df if not df.empty else empty(name)
                break
            except Exception as e:  # noqa: BLE001  файл подменяется прямо сейчас
                if "lock" not in str(e).lower() and "being used" not in str(e).lower() or attempt == 4:
                    log.warning("db_read_failed  %s: %s – читаю из файлов", name, e)
                    break
                time.sleep(0.2)
    try:
        df = SOURCES[name]() if name in SOURCES else None
    except Exception as e:  # noqa: BLE001  битый файл не должен ронять весь раздел
        log.warning("file_read_failed  %s: %s", name, e)
        df = None
    return df if df is not None and not df.empty else empty(name)


def publish() -> dict:
    """Собирает все таблицы из файлов в новую базу и подменяет старую.

    Возвращает {таблица: число строк}; отсутствующие таблицы пропускаются.
    """
    t0 = time.perf_counter()
    tmp = DATA_DB.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)
    counts = {}
    with closing(_duck(read_only=False, path=tmp)) as con:
        for name, build in SOURCES.items():
            df = build()
            if df is None or df.empty:
                continue
            if name == "spending_mo" and "date" in df:
                log.info("publish_data  расходы МО: %s..%s, МО %d", f"{df['date'].min():%Y-%m}",
                         f"{df['date'].max():%Y-%m}", df["oktmo"].nunique())
            con.register("_df", df)
            con.execute(f'create table "{name}" as select * from _df')
            con.unregister("_df")
            counts[name] = len(df)
        con.execute("create table _meta as select ? as published_at, ? as tables",
                    [datetime.now().isoformat(timespec="seconds"), json.dumps(counts, ensure_ascii=False)])
    for attempt in range(20):   # приложение могло как раз читать файл – ждём
        try:
            os.replace(tmp, DATA_DB)
            missing = sorted(set(SOURCES) - set(counts))
            log.info("publish  таблиц %d, строк %d%s", len(counts), sum(counts.values()),
                     f"; нет данных для: {missing}" if missing else "",
                     extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})
            return counts
        except PermissionError:
            if attempt in (0, 10):
                log.warning("publish_wait  %s занят (приложение читает), жду", DATA_DB.name)
            time.sleep(0.5)
    log.error("publish_failed  не удалось заменить %s: файл занят", DATA_DB)
    raise RuntimeError(f"не удалось заменить {DATA_DB}: файл занят")


def published_at() -> str | None:
    if not DATA_DB.exists():
        return None
    try:
        with closing(_duck()) as con:
            return con.execute("select published_at from _meta").fetchone()[0]
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------ журнал воркера

SCHEMA = """
create table if not exists runs (
  id integer primary key autoincrement, trigger text, mode text, git text,
  started text, finished text, status text, message text, current_step text);
create table if not exists run_steps (
  run_id integer, step text, status text, started text, finished text,
  seconds real, message text);
create table if not exists step_state (
  step text primary key, fingerprint text, finished text);
create table if not exists meta (key text primary key, value text);
"""


def _open_state() -> sqlite3.Connection:
    con = sqlite3.connect(STATE_DB, timeout=10)
    try:
        con.row_factory = sqlite3.Row
        con.execute("pragma journal_mode=wal")
        con.executescript(SCHEMA)
    except Exception:
        con.close()          # иначе Windows держит файл и его не переместить
        raise
    return con


def state() -> sqlite3.Connection:
    """Журнал воркера. Испорченный файл (сбой диска, обрыв питания) убирается
    в сторону и создаётся новый: журнал – история запусков, а не данные;
    без него воркер просто пересчитает то, что сочтёт устаревшим."""
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _open_state()
    except sqlite3.DatabaseError as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            raise
        aside = STATE_DB.with_name(f"{STATE_DB.name}.corrupt-{datetime.now():%Y%m%d-%H%M%S}")
        log.error("state_corrupt  %s: %s – перенесён в %s, создаю новый журнал", STATE_DB.name, e, aside.name)
        for suffix in ("", "-wal", "-shm"):
            f = STATE_DB.with_name(STATE_DB.name + suffix)
            if f.exists():
                os.replace(f, aside.with_name(aside.name + suffix))
        return _open_state()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_meta(key: str) -> str | None:
    with closing(state()) as con:
        r = con.execute("select value from meta where key = ?", [key]).fetchone()
        return r["value"] if r else None


def set_meta(key: str, value: str) -> None:
    with closing(state()) as con, con:
        con.execute("insert into meta values (?, ?) on conflict(key) do update set value = excluded.value",
                    [key, value])


def last_runs(limit: int = 10) -> list[dict]:
    if not STATE_DB.exists():
        return []
    with closing(state()) as con:
        runs = [dict(r) for r in con.execute("select * from runs order by id desc limit ?", [limit])]
        for r in runs:
            r["steps"] = [dict(s) for s in con.execute(
                "select step, status, seconds, message from run_steps where run_id = ? order by rowid", [r["id"]])]
        return runs
