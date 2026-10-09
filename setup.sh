#!/usr/bin/env bash
# Установка на Linux / macOS (основная платформа – Windows 10/11, см. setup.ps1).
#   bash setup.sh
# Работают расчёты, воркер и приложение (PySide6); уведомления Windows и
# задача планировщика – только на Windows (здесь – cron: см. README).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee -a logs/setup.log) 2>&1

PY=""
for c in python3.12 python3; do
  if command -v "$c" >/dev/null && [ "$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])')" = "3.12" ]; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Нужен Python 3.12"; exit 1; }

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.lock.txt --quiet

if [ ! -f data/processed/spending_mo.parquet ] && [ ! -f data/sberindex.duckdb ]; then
  URL=$(.venv/bin/python -c "from worker.snapshot import resolve_url; print(resolve_url() or '')")
  if [ -n "$URL" ]; then .venv/bin/python -m worker snapshot fetch
  else echo "Данных нет и адрес снимка неизвестен: положите выгрузки в data/raw и выполните .venv/bin/python -m worker run --force --fetch"
  fi
fi
.venv/bin/python -m worker status
echo "Готово. Запуск: .venv/bin/python -m app.main"
