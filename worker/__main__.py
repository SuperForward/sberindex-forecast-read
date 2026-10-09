"""Воркер: пересчёт по расписанию и вручную. Справка: python -m worker -h"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from worker import runner, snapshot  # noqa: E402


PANEL_SOURCE = "https://sberindex.ru/ru/dashboards/potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij"


def panel_age() -> str | None:
    """Свежесть панели расходов: последний месяц и сколько месяцев прошло.
    Новые месяцы СберИндекс выкладывает выгрузкой на странице набора
    (PANEL_SOURCE) – их надо скачать вручную в data/raw/. На 2026-10 там
    по-прежнему январь 2023 – декабрь 2024."""
    import pandas as pd
    f = ROOT / "data/processed/spending_mo.parquet"
    if not f.exists():
        return None
    last = pd.read_parquet(f, columns=["date"])["date"].max()
    now = pd.Timestamp.today()
    months = (now.year - last.year) * 12 + now.month - last.month
    return (f"панель расходов по {last:%Y-%m}, прошло {months} мес." +
            (f" – проверьте новую выгрузку: {PANEL_SOURCE}" if months > 3 else ""))


def _status() -> None:
    print(f"данные опубликованы: {store.published_at() or 'ещё нет'}")
    age = panel_age()
    if age:
        print(age)
    print(f"пересчёт идёт сейчас: {'да' if runner.is_busy() else 'нет'}")
    print("\nшаги:")
    for s in runner.plan(load_config("pipeline")):
        print(f"  {s['name']:<18} {s['group']:<6} {s['action']:<4} {s['why']}")
    print("\nпоследние запуски:")
    for r in store.last_runs(5):
        print(f"  #{r['id']} {r['started']} {r['trigger']:<8} {r['status']:<8} {r['message'] or ''}")


def _install_task() -> None:
    """Задача Планировщика Windows: воркер стартует при входе в систему.
    Интервал проверок задаётся в configs/pipeline.yaml → schedule."""
    import subprocess
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = pyw if pyw.exists() else Path(sys.executable)
    action = f'cmd /c cd /d "{ROOT}" && "{exe}" -m worker loop'
    subprocess.run(["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", "SberIndexWorker",
                    "/TR", action, "/RL", "LIMITED"], check=True)
    print("готово: воркер будет стартовать при входе в Windows (удалить: "
          "schtasks /Delete /TN SberIndexWorker /F)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m worker", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="пересчитать устаревшее (ручной запуск)")
    r.add_argument("--force", action="store_true", help="пересчитать всё, даже актуальное")
    r.add_argument("--heavy", action="store_true", help="включить тяжёлые шаги (горизонты)")
    r.add_argument("--fetch", action="store_true", help="сначала скачать свежие данные из интернета")
    r.add_argument("--steps", default="", help="только эти шаги, через запятую")
    r.add_argument("--trigger", default="manual", help=argparse.SUPPRESS)
    sub.add_parser("loop", help="работать по расписанию (configs/pipeline.yaml → schedule)")
    sub.add_parser("status", help="состояние шагов и последние запуски")
    sub.add_parser("publish", help="только пересобрать базу для приложения")
    sub.add_parser("adopt", help="считать текущие результаты актуальными (не пересчитывать)")
    sn = sub.add_parser("snapshot", help="снимок данных для GitHub Release")
    sn.add_argument("action", choices=["pack", "fetch"])
    sn.add_argument("--raw", action="store_true", help="pack: ещё и архив исходников")
    sn.add_argument("--url", default=None, help="fetch: адрес архива")
    sub.add_parser("install-task", help="запускать воркер при входе в Windows")
    sub.add_parser("models", help="скачать веса моделей (прогноз и новости) в data/models (~3 ГБ)")
    rl = sub.add_parser("release", help="подготовить релиз в dist/ (без публикации)")
    rl.add_argument("--raw", action="store_true", help="добавить архив исходных выгрузок")
    a = ap.parse_args()
    from app.logging_setup import setup_logging
    setup_logging()
    import logging
    logging.getLogger("worker").info("cli  %s", " ".join(sys.argv[1:]))

    if a.cmd == "run":
        only = [s for s in a.steps.split(",") if s] or None
        try:
            res = runner.run(trigger=a.trigger, force=a.force, heavy=a.heavy, fetch=a.fetch, only=only)
        except runner.Busy as e:
            raise SystemExit(str(e))
        raise SystemExit(0 if res["status"] in ("ok", "warning") else 1)
    if a.cmd == "loop":
        runner.loop()
    elif a.cmd == "status":
        _status()
    elif a.cmd == "publish":
        print(store.publish())
    elif a.cmd == "adopt":
        runner.adopt()
    elif a.cmd == "snapshot":
        if a.action == "pack":
            snapshot.pack(raw=a.raw)
        else:
            def fetch_and_adopt():
                snapshot.fetch(a.url)
                store.publish() if not store.DATA_DB.exists() else None
                runner.adopt()
            res = runner.task("snapshot", fetch_and_adopt)
            raise SystemExit(0 if res["status"] == "ok" else 1)
    elif a.cmd == "release":
        snapshot.prepare_release(raw=a.raw)
    elif a.cmd == "models":
        from src.forecast import foundation
        foundation.download()
    elif a.cmd == "install-task":
        _install_task()


if __name__ == "__main__":
    main()
