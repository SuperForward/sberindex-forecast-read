"""Пути и общие настройки проекта.

Гиперпараметры моделей живут в configs/*.yaml, здесь только то, что
нужно коду до чтения конфига.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

APP_NAME = "SberIndex Terminal"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

# Приложение и воркер идут под pythonw.exe без консоли: любая консольная
# программа (curl, git, gh) из них открывает своё окно. Флаг его прячет.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def console_python() -> str:
    """python.exe рядом с pythonw.exe. Запущенный с NO_WINDOW, он получает
    скрытую консоль, и её наследуют все его дочерние программы (у pythonw
    консоли нет, флаг на нём не действует)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def cpu_budget() -> int:
    """Сколько ядер отдаём расчётам: все. Шаги воркера идут с пониженным
    приоритетом, поэтому система и приложение не зависают. Переопределяется
    переменной SBERINDEX_CPU (например, половина ядер на слабом ноутбуке)."""
    env = os.environ.get("SBERINDEX_CPU", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


def n_jobs(value: int | None = -1) -> int:
    """n_jobs из конфига с потолком cpu_budget(): -1 (и прочие отрицательные,
    None) – весь бюджет, положительное число – не больше бюджета."""
    budget = cpu_budget()
    if value is None or value < 0:
        return budget
    return max(1, min(value, budget))


def load_config(name: str = "default") -> dict:
    """Читает configs/<name>.yaml."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
