# SberIndex Terminal: веб-сервер (API + интерфейс) и воркер пересчёта в одном образе.
#   docker compose up        – см. docker-compose.yml
FROM python:3.12-slim

# curl – загрузчики данных и весов моделей (системное хранилище сертификатов)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt .
# без настольного интерфейса (PySide6) и уведомлений Windows – в контейнере они не нужны
RUN grep -viE "^(pyside6|shiboken6|winrt|colorama)" requirements.lock.txt > req.txt \
    && pip install --no-cache-dir -r req.txt fastapi uvicorn

COPY . .
ENV PYTHONIOENCODING=utf-8
EXPOSE 8000
# данные (data/), отчёты (reports/) и логи (logs/) – в томах, см. docker-compose.yml
CMD ["python", "-m", "server", "--host", "0.0.0.0", "--port", "8000"]
