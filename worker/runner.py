"""Прогон конвейера из configs/pipeline.yaml.

Порядок запуска:
1. Блокировка: второй воркер (или кнопка в приложении во время прогона
   по расписанию) не стартует, пока идёт первый.
2. Для каждого шага – отпечаток входов (файлы needs и code, размер и время
   изменения). Совпал с прошлым успешным и все выходы на месте – шаг
   пропускается.
3. Перед первым шагом, который будет что-то пересчитывать, текущие
   data/processed, data/interim и reports копируются в резерв.
4. После шагов data – проверка качества. Не прошла или упал любой шаг –
   всё возвращается из резерва: приложение продолжает показывать последние
   хорошие результаты.
5. В конце – публикация в data/sberindex.duckdb (src/store.py).

Каждый шаг – отдельный процесс «python -m scripts.<...>»: так же, как их
запускает человек, и падение одного шага не роняет воркер.
"""

import fnmatch
import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path

import pandas as pd

from src import store
from src.config import DATA_DIR, LOGS_DIR, ROOT, console_python, cpu_budget, load_config

BACKUP_DIR = DATA_DIR / ".backup" / "pre_run"
BACKUP_PATHS = ["data/processed", "data/interim", "reports"]
LOCK_FILE = DATA_DIR / ".worker.lock"
STEP_TIMEOUT_S = 3 * 3600
SOURCE_UNAVAILABLE = 3        # код выхода скрипта скачивания: источник недоступен, отложено
KEEP_STEP_LOGS = 300          # логов шагов в logs/worker/ – старые удаляются

log = logging.getLogger("worker")


class Busy(RuntimeError):
    """Уже идёт другой прогон."""


@contextmanager
def lock():
    """Блокировка средствами ОС: снимается сама, даже если процесс убит."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_FILE, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                log.debug("lock_busy  pid=%s", os.getpid())
                raise Busy("пересчёт уже идёт") from None
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise Busy("пересчёт уже идёт") from None
        yield
    finally:
        f.close()


def is_busy() -> bool:
    try:
        with lock():
            return False
    except Busy:
        return True


# ------------------------------------------------------------------ шаги

_SCAN: dict | None = None      # кэш проверки файлов на время одного решения


@contextmanager
def scan_cache():
    """Проверка папок (в data/raw тысячи файлов) – один раз на план, а не на
    каждую проверку каждого шага: без кэша план считался ~12 с."""
    global _SCAN
    outer = _SCAN is not None
    if not outer:
        _SCAN = {}
    try:
        yield
    finally:
        if not outer:
            _SCAN = None


def _expand(pattern: str) -> list[Path]:
    if _SCAN is not None and pattern in _SCAN:
        return _SCAN[pattern]
    paths = [Path(p) for p in glob.glob(str(ROOT / pattern))]
    out = []
    for p in paths:
        if p.is_dir():
            out += [q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts]
        elif p.is_file():
            out.append(p)
    if _SCAN is not None:
        _SCAN[pattern] = out
    return out


def _stat(p: Path):
    key = ("stat", p)
    if _SCAN is not None and key in _SCAN:
        return _SCAN[key]
    st = p.stat()
    if _SCAN is not None:
        _SCAN[key] = st
    return st


def fingerprint(step: dict) -> str:
    # собственные выходы шага не входят в отпечаток: audit_data пишет
    # MANIFEST.csv внутрь data/raw, которую сам же читает
    own = {q for m in step.get("makes", []) for q in _expand(m)}
    h = hashlib.sha1(repr(step.get("args", [])).encode())
    for pattern in step.get("needs", []) + step.get("code", []):
        for p in sorted(set(_expand(pattern)) - own):
            st = _stat(p)
            h.update(f"{p.relative_to(ROOT).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def missing_inputs(step: dict) -> list[str]:
    return [p for p in step.get("needs", []) if not _expand(p)]


def missing_outputs(step: dict) -> list[str]:
    return [p for p in step.get("makes", []) if not glob.glob(str(ROOT / p))]


def _feeds(make: str, need: str) -> bool:
    """Выход make одного шага – вход need другого (путь, папка или шаблон)."""
    m, n = make.rstrip("/"), need.rstrip("/")
    return fnmatch.fnmatch(m, n) or m.startswith(n + "/") or n.startswith(m + "/") or m == n


def _decide(s: dict, prev: dict, force: bool, heavy: bool, fetch: bool, only, upstream: list[dict]):
    """(run|skip, причина) для шага. upstream – шаги, которые в этом прогоне
    уже пересчитаны (или будут): их выходы считаются появившимися/изменёнными."""
    g = s["group"]
    chosen = bool(only and s["name"] in only)
    fresh = [u["name"] for u in upstream
             if any(_feeds(m, n) for m in u.get("makes", []) for n in s.get("needs", []))]
    if only and not chosen:
        return "skip", "не выбран"
    if g == "fetch" and not (fetch or chosen):
        return "skip", "скачивание не по расписанию"
    if g == "fetch" and not (force or chosen) and s.get("every_h"):
        age = time.time() - float(store.get_meta(f"fetch_ok:{s['name']}") or 0)
        if age < s["every_h"] * 3600:
            return "skip", f"скачано {age / 3600:.0f} ч назад (интервал {s['every_h']} ч)"
    if g == "heavy" and not (heavy or chosen):
        return "skip", "тяжёлый шаг, запускается только с --heavy"
    miss = [n for n in missing_inputs(s)
            if not any(_feeds(m, n) for u in upstream for m in u.get("makes", []))]
    if miss:
        return "skip", "нет исходных данных: " + ", ".join(miss)
    if g == "fetch" or force or chosen:
        return "run", "принудительно" if force or chosen else "по расписанию"
    if fresh:
        return "run", "обновились входные данные: " + ", ".join(fresh)
    if miss := missing_outputs(s):
        return "run", "нет результата: " + ", ".join(miss)
    if prev.get(s["name"]) != fingerprint(s):
        return "run", "изменились входные данные или код"
    return "skip", "актуально"


def _prev() -> dict:
    with closing(store.state()) as con:
        return {r["step"]: r["fingerprint"] for r in con.execute("select step, fingerprint from step_state")}


def plan(cfg: dict, force=False, heavy=False, fetch=False, only: list[str] | None = None) -> list[dict]:
    """Что будет сделано с каждым шагом: run / skip + причина (прогноз на весь прогон)."""
    prev, out, upstream = _prev(), [], []
    with scan_cache():
        return _plan(cfg, prev, out, upstream, force, heavy, fetch, only)


def _plan(cfg, prev, out, upstream, force, heavy, fetch, only):
    for s in cfg["steps"]:
        action, why = _decide(s, prev, force, heavy, fetch, only, upstream)
        if action == "run" and s["group"] != "fetch":
            upstream.append(s)
        out.append({**s, "action": action, "why": why})
    return out


# ------------------------------------------------------------ качество

def quality_check(q: dict) -> list[str]:
    """Список нарушений (пусто – всё хорошо)."""
    f = ROOT / "data/processed/spending_mo.parquet"
    if not f.exists():
        return ["нет data/processed/spending_mo.parquet"]
    p = pd.read_parquet(f)
    problems = []
    n_mo = p["oktmo"].nunique()
    if n_mo < q.get("min_mo", 0):
        problems.append(f"МО в панели {n_mo} < {q['min_mo']}")
    prev = store.load("spending_mo") if store.DATA_DB.exists() else pd.DataFrame()
    if not prev.empty:
        n_prev = prev["oktmo"].nunique()
        if n_mo < n_prev * (1 - q.get("max_mo_drop", 1)):
            problems.append(f"МО стало {n_mo} против {n_prev} в прошлой публикации")
    dup = int(p.duplicated(["oktmo", "date", "category"]).sum())
    if dup > q.get("max_duplicates", 0):
        problems.append(f"дублей (МО, дата, категория): {dup}")
    s = p.sort_values("date").groupby(["oktmo", "category"])["value"].pct_change().abs()
    jump = float((s > 1).mean())
    if jump > q.get("max_jump_share", 1):
        problems.append(f"доля скачков м/м > 100%: {jump:.4f}")
    (log.warning if problems else log.info)(
        "quality  mo=%d dup=%d jump=%.5f period=%s..%s problems=%s", n_mo, dup, jump,
        p["date"].min().date(), p["date"].max().date(), problems or "нет")
    return problems


def model_check(q: dict) -> list[str]:
    """Модели не ухудшились резко против опубликованных: MAE каждой модели на
    общих точках (МО, окно, горизонт, backtest) выросла не больше чем на
    max_model_mae_rise. Ловит поломки (ошибка в признаках, сломанный бэкенд),
    а не шум: сравниваются одни и те же прогнозные точки."""
    rise = q.get("max_model_mae_rise")
    if rise is None or not store.DATA_DB.exists():
        return []
    old = store.load("predictions")
    new = store.SOURCES["predictions"]()
    if old.empty or new is None or new.empty:
        return []
    keys = ["backtest", "model", "series_id", "origin", "h"]
    m = old[keys + ["y", "yhat"]].merge(new[keys + ["y", "yhat"]], on=keys, suffixes=("_old", "_new"))
    m = m[(m["y_old"] - m["y_new"]).abs() <= 1e-6 * m["y_old"].abs()]   # факт тот же – данные не пересмотрены
    if m.empty:
        return []
    g = m.assign(ae_old=(m["y_old"] - m["yhat_old"]).abs(), ae_new=(m["y_new"] - m["yhat_new"]).abs())         .groupby(["backtest", "model"])[["ae_old", "ae_new"]].agg(["mean", "size"])
    g = g[g[("ae_old", "size")] >= q.get("min_model_points", 1000)]
    ratio = g[("ae_new", "mean")] / g[("ae_old", "mean")] - 1
    bad = ratio[ratio > rise].sort_values(ascending=False)
    log.info("model_check  моделей сравнено %d, худший рост MAE %s", len(ratio),
             f"{ratio.max():+.1%} ({ratio.idxmax()[1]})" if len(ratio) else "–")
    return [f"MAE {model} ({bt}) выросла на {r:.0%}" for (bt, model), r in bad.items()]


# ------------------------------------------------------------ резерв

def backup() -> None:
    t0 = time.perf_counter()
    shutil.rmtree(BACKUP_DIR, ignore_errors=True)
    for rel in BACKUP_PATHS:
        src = ROOT / rel
        if src.exists():
            shutil.copytree(src, BACKUP_DIR / rel)
    size = sum(f.stat().st_size for f in BACKUP_DIR.rglob("*") if f.is_file()) if BACKUP_DIR.exists() else 0
    log.info("backup  %s -> %s (%.1f МБ)", BACKUP_PATHS, BACKUP_DIR.relative_to(ROOT), size / 1e6,
             extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})


def restore() -> None:
    """Вернуть состояние до прогона; папки, которых тогда не было, удалить."""
    log.warning("restore  откат %s из %s", BACKUP_PATHS, BACKUP_DIR.relative_to(ROOT))
    for rel in BACKUP_PATHS:
        saved = BACKUP_DIR / rel
        shutil.rmtree(ROOT / rel, ignore_errors=True)
        if saved.exists():
            shutil.copytree(saved, ROOT / rel)


# ------------------------------------------------------------ прогон

def _partial_ensembles() -> str:
    """Ансамбли прогона, собранные не из всех моделей: «ens: нет a, b» или ''."""
    f = ROOT / "reports/backtest/models/ensemble_members.csv"
    if not f.exists():
        return ""
    e = pd.read_csv(f)
    e = e[e["missing"].fillna("") != ""]
    return "; ".join(f"{r.model}: нет {r.missing}" for r in e.itertuples())


def log_weights() -> None:
    """Какие веса фундаментальных моделей на диске: без них модели (и ансамбли
    с ними) молча пропадут из прогноза – причина видна сразу в начале прогона."""
    from src.forecast import foundation
    have = [f for f in foundation.REPOS if (foundation.MODELS_DIR / f / "model.safetensors").exists()]
    miss = sorted(set(foundation.REPOS) - set(have))
    (log.warning if miss else log.info)("weights  есть: %s; нет: %s%s", have or "–", miss or "–",
                                        " (скачать: python -m worker models)" if miss else "")


def child_env() -> dict:
    """Окружение для дочерних процессов: вывод в UTF-8, но без PYTHONUTF8.

    Режим PYTHONUTF8 ломает Prophet на русской Windows: cmdstan читает вывод
    системной where.exe (он в cp866) как UTF-8 и падает, а Prophet остаётся без
    stan_backend. run_app.vbs включает этот режим, поэтому убираем его явно.

    Потоки OpenMP/BLAS ограничены cpu_budget(): по умолчанию numpy, sklearn и
    бустинги занимают все ядра, и компьютер на время прогона виснет.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "utf-8"
    threads = str(cpu_budget())
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, threads)
    return env


def _run_step(step: dict, run_id: int) -> tuple[bool, str, float]:
    log_dir = LOGS_DIR / "worker"
    log_dir.mkdir(parents=True, exist_ok=True)
    step_log = log_dir / f"{run_id:05d}_{step['name']}.log"
    logger = log
    env = child_env()
    # консольный python со скрытой консолью: curl в шагах не открывает окон
    cmd = [console_python(), "-m", step["module"], *map(str, step.get("args", []))]
    # пониженный приоритет наследуют и процессы joblib: интерфейс не тормозит
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
             if sys.platform == "win32" else 0)
    t0 = time.perf_counter()
    code = None
    logger.info("step_start  run=%d %s: %s  лог=%s", run_id, step["name"], " ".join(cmd[1:]),
                step_log.relative_to(ROOT))
    with open(step_log, "w", encoding="utf-8") as out:
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=STEP_TIMEOUT_S, creationflags=flags)
            code = r.returncode
        except subprocess.TimeoutExpired:
            out.write(f"\nпревышен лимит времени {STEP_TIMEOUT_S} с\n")
    ok = code == 0
    sec = time.perf_counter() - t0
    tail = step_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
    msg = "" if ok else " | ".join(tail)[-500:]
    if ok:
        logger.info("step_done  run=%d %s", run_id, step["name"], extra={"elapsed_ms": sec * 1000})
    elif step.get("group") == "fetch":
        # скачивание не удалось (нет сети, источник недоступен) – воркер продолжит
        # на имеющихся данных; это не авария и не должно попадать в errors.log
        logger.warning("step_deferred  run=%d %s код=%s%s: %s  (полный вывод: %s)", run_id, step["name"],
                       "таймаут" if code is None else code,
                       " (источник недоступен)" if code == SOURCE_UNAVAILABLE else "", msg,
                       step_log.relative_to(ROOT), extra={"elapsed_ms": sec * 1000})
    else:
        logger.error("step_failed  run=%d %s код=%s: %s  (полный вывод: %s)", run_id, step["name"],
                     "таймаут" if code is None else code, msg, step_log.relative_to(ROOT),
                     extra={"elapsed_ms": sec * 1000})
    return ok, msg, sec


def _prune_step_logs() -> None:
    files = sorted((LOGS_DIR / "worker").glob("*.log"))
    for f in files[:-KEEP_STEP_LOGS]:
        f.unlink(missing_ok=True)


def run(trigger="manual", force=False, heavy=False, fetch=False, only=None,
        echo=print) -> dict:
    """Один прогон. Возвращает {"id", "status", "message"}."""
    cfg = load_config("pipeline")
    try:
        cm = lock()
        cm.__enter__()
    except Busy:
        log.warning("run_rejected  trigger=%s: пересчёт уже идёт", trigger)
        raise
    t_run = time.perf_counter()
    run_id = None
    backed_up = False
    try:
        # Блокировка наша – значит, «идущих» запусков нет. Записи со статусом
        # running остались от процессов, снятых извне (выключение ПК, диспетчер задач).
        with closing(store.state()) as con, con:
            n = con.execute("update runs set status='interrupted', finished=?, current_step=null, "
                            "message=coalesce(message, 'прерван извне (процесс снят или ПК выключен)') "
                            "where status='running'", [store.now()]).rowcount
        if n:
            log.warning("runs_interrupted  помечено прерванными: %d", n)
        steps = plan(cfg, force=force, heavy=heavy, fetch=fetch, only=only)
        from src.runlog import _git_rev
        mode = ",".join(k for k, v in {"force": force, "heavy": heavy, "fetch": fetch}.items() if v) or "auto"
        git = _git_rev()
        with closing(store.state()) as con, con:
            run_id = con.execute("insert into runs (trigger, mode, git, started, status) values (?,?,?,?,?)",
                                 [trigger, mode + (f" [{','.join(only)}]" if only else ""), git,
                                  store.now(), "running"]).lastrowid
        if fetch:
            store.set_meta("last_fetch_attempt", str(time.time()))
        if any(s["action"] == "run" and s["group"] in ("model", "heavy") for s in steps):
            log_weights()
        log.info("run_start  #%d trigger=%s mode=%s only=%s git=%s pid=%d python=%s", run_id, trigger,
                 mode, only or "-", git, os.getpid(), sys.executable)
        log.info("run_plan  #%d к пересчёту: %s; пропуск: %s", run_id,
                 [f"{s['name']} ({s['why']})" for s in steps if s["action"] == "run"] or "нет",
                 sum(s["action"] == "skip" for s in steps))

        def finish(status, message):
            with closing(store.state()) as con, con:
                con.execute("update runs set finished=?, status=?, message=?, current_step=null where id=?",
                            [store.now(), status, message, run_id])
            {"ok": log.info, "warning": log.warning}.get(status, log.error)(
                "run_done  #%d status=%s: %s", run_id, status, message,
                extra={"elapsed_ms": (time.perf_counter() - t_run) * 1000})
            echo(f"[{status}] {message}")
            return {"id": run_id, "status": status, "message": message}

        def record(step, status, sec=0.0, msg=""):
            with closing(store.state()) as con, con:
                con.execute("insert into run_steps values (?,?,?,?,?,?,?)",
                            [run_id, step, status, None, store.now(), round(sec, 1), msg])

        todo = [s for s in steps if s["action"] == "run"]
        if not todo:
            for s in steps:
                record(s["name"], "skip", msg=s["why"])
            if not store.DATA_DB.exists():
                store.publish()
            return finish("ok", "всё актуально, пересчёт не нужен")

        data_done = False
        ran = []
        fetch_failed = []
        prints = {}
        prev = _prev()
        for s in cfg["steps"]:
            # решение – прямо перед запуском: шаги выше могли создать или
            # изменить входы этого шага
            with scan_cache():       # свежая проверка: шаги выше могли изменить файлы
                action, why = _decide(s, prev, force, heavy, fetch, only,
                                      [x for x in ran if x["group"] != "fetch"])
            s = {**s, "action": action, "why": why}
            if action != "run":
                record(s["name"], "skip", msg=why)
                log.debug("step_skip  #%d %s: %s", run_id, s["name"], why)
                continue
            if s["group"] != "fetch" and not backed_up:
                backup()
                backed_up = True
            if s["group"] in ("model", "heavy") and not data_done and any(x["group"] == "data" for x in ran):
                problems = quality_check(cfg.get("quality", {}))
                data_done = True
                if problems:
                    record("quality", "failed", msg="; ".join(problems))
                    restore()
                    return finish("failed", "проверка качества не пройдена, данные откатаны: " + "; ".join(problems))
                record("quality", "ok")
            with closing(store.state()) as con, con:
                con.execute("update runs set current_step=? where id=?", [s["name"], run_id])
            echo(f"→ {s['name']} ({s['why']})")
            ok, msg, sec = _run_step(s, run_id)
            record(s["name"], "ok" if ok else "failed", sec, msg)
            if not ok:
                if s["group"] == "fetch":       # интернет недоступен – работаем на том, что есть
                    fetch_failed.append(s["name"])
                    log.warning("fetch_failed  #%d %s – продолжаю на имеющихся данных", run_id, s["name"])
                    echo(f"  скачивание не удалось, продолжаю: {msg}")
                    continue
                if backed_up:
                    restore()
                return finish("failed", f"шаг {s['name']} упал, результаты откатаны: {msg}")
            ran.append(s)
            if s["group"] == "fetch":
                store.set_meta(f"fetch_ok:{s['name']}", str(time.time()))
            # отпечаток запоминаем сейчас (входы ещё те, на которых считали),
            # а записываем только если весь прогон успешен: после отката
            # шаг должен остаться устаревшим
            with scan_cache():
                prints[s["name"]] = fingerprint(s)

        if any(x["group"] == "data" for x in ran) and not data_done:
            problems = quality_check(cfg.get("quality", {}))
            if problems:
                record("quality", "failed", msg="; ".join(problems))
                restore()
                return finish("failed", "проверка качества не пройдена, данные откатаны: " + "; ".join(problems))
            record("quality", "ok")

        # модели пересчитаны – не ухудшились ли резко (ручной --force проверку пропускает)
        if not force and any(x["group"] in ("model", "heavy") for x in ran):
            problems = model_check(cfg.get("quality", {}))
            if problems:
                record("model_check", "failed", msg="; ".join(problems))
                restore()
                return finish("failed", "модели резко ухудшились, результаты откатаны "
                                        "(если изменение осознанное, запустите с --force): " + "; ".join(problems))
            record("model_check", "ok")

        with closing(store.state()) as con, con:
            for name, fp in prints.items():
                if name in {x["name"] for x in ran if x["group"] != "fetch"}:
                    con.execute("insert into step_state values (?,?,?) on conflict(step) do update set "
                                "fingerprint=excluded.fingerprint, finished=excluded.finished",
                                [name, fp, store.now()])
        counts = store.publish()
        shutil.rmtree(BACKUP_DIR, ignore_errors=True)
        try:                     # на чём теперь строятся результаты приложения
            from src import app_service
            app_service.log_choices("worker")
        except Exception:  # noqa: BLE001  лог решений не должен ронять публикацию
            log.warning("choices_failed  не удалось записать выбор моделей", exc_info=True)
        names = ", ".join(s["name"] for s in ran if s["group"] != "fetch") or "ничего"
        partial = _partial_ensembles()
        if partial:
            return finish("warning", f"пересчитано: {names}; ансамбль собран не из всех моделей: {partial} "
                                     "(нет весов фундаментальных моделей: python -m worker models)")
        if fetch_failed:
            return finish("warning", f"скачать не удалось: {', '.join(fetch_failed)} (нет сети или сайт "
                                     f"недоступен), работаю на имеющихся данных; пересчитано: {names}")
        if fetch:
            store.set_meta("last_fetch", str(time.time()))
        got = [s["name"] for s in ran if s["group"] == "fetch"]
        return finish("ok", (f"скачано: {', '.join(got)}; " if got else "") +
                      f"пересчитано: {names}; опубликовано таблиц: {len(counts)}")
    except Exception as e:  # noqa: BLE001  сбой самого воркера (не шага): откат и запись в журнал
        log.exception("run_crashed  #%s: %s", run_id, e)
        if backed_up:
            restore()
        if run_id is not None:
            with closing(store.state()) as con, con:
                con.execute("update runs set finished=?, status='failed', message=?, current_step=null "
                            "where id=?", [store.now(), f"сбой воркера: {e}", run_id])
        return {"id": run_id, "status": "failed", "message": f"сбой воркера: {e}"}
    finally:
        cm.__exit__(None, None, None)
        _prune_step_logs()


def fetch_due(sched: dict) -> bool:
    """Пора ли скачивать: прошло fetch_every_h после удачного скачивания и
    не меньше retry_after_fail_min после последней попытки (нет сети – не
    долбим источники каждую минуту)."""
    t = time.time()
    ok = float(store.get_meta("last_fetch") or 0)
    tried = float(store.get_meta("last_fetch_attempt") or 0)
    return t - ok >= sched["fetch_every_h"] * 3600 and t - tried >= sched.get("retry_after_fail_min", 60) * 60


def last_run_failed_recently(sched: dict) -> bool:
    runs = store.last_runs(1)
    if not runs or runs[0]["status"] != "failed" or not runs[0]["finished"]:
        return False
    from datetime import datetime
    age = (datetime.now() - datetime.fromisoformat(runs[0]["finished"])).total_seconds()
    return age < sched.get("retry_after_fail_min", 60) * 60


def loop(echo=print) -> None:
    """Работа по расписанию: просыпаться, скачивать и пересчитывать по мере надобности."""
    cfg = load_config("pipeline")["schedule"]
    log.info("loop_start  pid=%d каждые %d мин., скачивание раз в %d ч, тяжёлые раз в %d дн.",
             os.getpid(), cfg["check_every_min"], cfg["fetch_every_h"], cfg["heavy_every_d"])
    echo(f"воркер запущен: проверка каждые {cfg['check_every_min']} мин.")
    while True:
        t = time.time()
        due_fetch = fetch_due(cfg)
        due_heavy = t - float(store.get_meta("last_heavy") or 0) >= cfg["heavy_every_d"] * 86400
        log.info("loop_tick  fetch=%s heavy=%s", due_fetch, due_heavy)
        try:
            res = run(trigger="schedule", fetch=due_fetch, heavy=due_heavy, echo=echo)
            if due_heavy and res["status"] in ("ok", "warning"):
                store.set_meta("last_heavy", str(t))
        except Busy:
            echo("пропуск: идёт ручной пересчёт")
        except Exception as e:  # noqa: BLE001  воркер не должен умирать из-за одного прогона
            log.exception("loop_error  %s", e)
            echo(f"ошибка прогона: {e}")
        time.sleep(cfg["check_every_min"] * 60)



def adopt(echo=print) -> list[str]:
    """Считать текущие результаты актуальными (после снимка или ручного прогона).

    Шаг «принимается», если все его выходы на месте; его входы запоминаются,
    и следующий run пересчитает его только когда они изменятся.
    """
    cfg = load_config("pipeline")
    done = []
    with closing(store.state()) as con, con:
        for s in cfg["steps"]:
            if s["group"] == "fetch" or missing_outputs(s):
                continue
            con.execute("insert into step_state values (?,?,?) on conflict(step) do update set "
                        "fingerprint=excluded.fingerprint, finished=excluded.finished",
                        [s["name"], fingerprint(s), store.now()])
            done.append(s["name"])
    log.info("adopt  приняты как актуальные: %s", done or "нет")
    echo("приняты как актуальные: " + (", ".join(done) or "нет"))
    return done


def task(name: str, fn, echo=print) -> dict:
    """Разовая операция (например, скачивание снимка) с записью в журнал и блокировкой."""
    with lock():
        with closing(store.state()) as con, con:
            run_id = con.execute("insert into runs (trigger, mode, started, status, current_step) "
                                 "values ('manual', ?, ?, 'running', ?)", [name, store.now(), name]).lastrowid
        log.info("task_start  #%d %s", run_id, name)
        t0 = time.perf_counter()
        try:
            fn()
            status, msg = "ok", f"{name}: готово"
        except (Exception, SystemExit) as e:  # noqa: BLE001
            log.exception("task_failed  #%d %s: %s", run_id, name, e)
            status, msg = "failed", f"{name}: {e}"
        log.info("task_done  #%d %s status=%s", run_id, name, status,
                 extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})
        with closing(store.state()) as con, con:
            con.execute("update runs set finished=?, status=?, message=?, current_step=null where id=?",
                        [store.now(), status, msg, run_id])
        echo(f"[{status}] {msg}")
        return {"id": run_id, "status": status, "message": msg}
