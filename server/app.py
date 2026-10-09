"""Веб-сервер SberIndex Terminal: REST API поверх src/app_service + веб-интерфейс.

Запуск:  python -m server            (http://127.0.0.1:8000, документация – /docs)
         python -m server --host 0.0.0.0 --port 8000 --allow-write

Тот же интерфейс, что у настольного приложения (app/ui/terminal.html), в
браузере; данные – те же функции app_service, что и в приложении. Запись
(ручной пересчёт, автообновление) по умолчанию выключена: сервер только
читает. Включается флагом --allow-write – для своего компьютера, не для
публичного адреса.
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import app_service as svc
from src.config import APP_NAME, ROOT

log = logging.getLogger("ui")
WRITE = os.environ.get("SBERINDEX_ALLOW_WRITE") == "1"

@asynccontextmanager
async def _lifespan(_app):
    # данные грузятся в фоне при запуске сервера, а не на первом запросе страницы
    threading.Thread(target=svc.warm, name="warm", daemon=True).start()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title=f"{APP_NAME} API",
    version="1.0",
    description=("Прогноз безналичных потребительских расходов по муниципалитетам и обнаружение "
                 "структурных сдвигов. Данные: СберИндекс, Росстат, ЦБ; прогнозы и детекторы "
                 "считает воркер (data/sberindex.duckdb). Только чтение, если сервер запущен "
                 "без --allow-write."),
)


_FRESH = {"at": None, "checked": 0.0}


def _ensure_fresh() -> None:
    """Кэш app_service – в памяти процесса сервера. Воркер мог опубликовать
    новые данные, пока сервер работал, а страницу открыли уже после этого:
    её сверка даты публикации изменения не увидит. Поэтому сверяет сам сервер
    (не чаще раза в 10 с) и при новой публикации перечитывает данные."""
    now = time.monotonic()
    if now - _FRESH["checked"] < 10:
        return
    _FRESH["checked"] = now
    at = svc.store.published_at()
    if _FRESH["at"] is not None and at != _FRESH["at"]:
        log.info("web_reload  новая публикация %s (была %s)", at, _FRESH["at"])
        svc.reload()
    _FRESH["at"] = at


def _json(fn, *args):
    """Ответ функции app_service как JSON; ошибка – {"error": …}, как в приложении."""
    t0 = time.perf_counter()
    _ensure_fresh()
    try:
        data = fn(*args)
        body = json.dumps(data, ensure_ascii=False, allow_nan=False)
    except svc.NoData as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:  # noqa: BLE001
        log.exception("web_api_failed  %s args=%s", fn.__name__, args)
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    ms = (time.perf_counter() - t0) * 1000
    slow = ms > 1000                 # как в приложении: медленный запрос – WARNING
    log.log(logging.WARNING if slow else logging.INFO, "web_api  %s args=%s bytes=%d%s", fn.__name__, args,
            len(body), "  МЕДЛЕННО" if slow else "", extra={"elapsed_ms": ms})
    return Response(content=body, media_type="application/json")   # уже готовый JSON, без повторной сериализации


# ---------------------------------------------------------------- чтение

@app.get("/api/summary", tags=["сводка"], summary="Ключевые цифры: МО, период, лучшая модель, выигрыш у Prophet")
def summary():
    return _json(svc.summary)


@app.get("/api/mo", tags=["МО"], summary="Поиск МО по названию или региону")
def search_mo(query: str = Query("", description="часть названия МО или региона")):
    return _json(svc.search_mo, query)


@app.get("/api/mo/{oktmo}", tags=["МО"], summary="Карточка МО: ряды расходов, события, кандидаты в шоки")
def mo_detail(oktmo: str):
    return _json(svc.mo_detail, oktmo)


@app.get("/api/mo/{oktmo}/forecast", tags=["МО"], summary="Прогнозы всех моделей по окнам backtest")
def mo_forecast(oktmo: str):
    return _json(svc.mo_forecast, oktmo)


@app.get("/api/export/{kind}", tags=["выгрузка"], summary="Таблица для выгрузки в CSV: {filename, csv}")
def export(kind: str, oktmo: str = Query("", description="ОКТМО – для forecast_mo")):
    return _json(svc.export_csv, kind, oktmo)


@app.get("/api/models", tags=["модели"], summary="Сравнение моделей прогноза (MAE, WAPE, R², оценка без подбора)")
def models():
    return _json(svc.model_metrics)


@app.get("/api/models/worst", tags=["модели"], summary="МО с наибольшей ошибкой модели")
def worst(model: str = Query("", description="код модели; пусто – лучшая")):
    return _json(svc.worst_mo, model)


@app.get("/api/horizons", tags=["модели"], summary="Точность на горизонтах 1, 3, 6, 12 месяцев")
def horizons():
    return _json(svc.horizons_overview)


@app.get("/api/shocks", tags=["шоки"], summary="Задокументированные события и кандидаты в шоки")
def shocks(type: str = "", year: int = 0, caution: bool = False):  # noqa: A002
    return _json(svc.shocks_overview, type, year, caution)


@app.get("/api/cpd", tags=["сдвиги"], summary="Сравнение детекторов сдвигов и МО с тревогами")
def cpd():
    return _json(svc.cpd_overview)


@app.get("/api/cpd/{oktmo}", tags=["сдвиги"], summary="Сигнал лучшего детектора по МО")
def cpd_mo(oktmo: str):
    return _json(svc.cpd_mo, oktmo)


@app.get("/api/news", tags=["новости"], summary="Новости о шоках: итоги проверки и лента (для МО – его и региона)")
def news(oktmo: str = Query("", description="ОКТМО; пусто – все")):
    return _json(svc.news_overview, oktmo)


@app.get("/api/results", tags=["результаты"], summary="Выводы, ограничения, воспроизводимость")
def results():
    return _json(svc.results_overview)


@app.get("/api/status", tags=["данные"], summary="Состояние данных и воркера")
def status():
    resp = _json(svc.data_status)
    if resp.status_code == 200:
        data = json.loads(resp.body)
        data["read_only"] = not WRITE
        return JSONResponse(data)
    return resp


# ---------------------------------------------------------------- запись

def _guard():
    if not WRITE:
        raise HTTPException(status_code=403, detail="сервер запущен только для чтения (--allow-write выключен)")


@app.post("/api/recalc/{mode}", tags=["данные"], summary="Запустить пересчёт (только с --allow-write)")
def recalc(mode: str):
    _guard()
    return _json(svc.recalc_start, mode, "web")


@app.post("/api/reload", tags=["данные"], summary="Перечитать опубликованные данные")
def reload():
    return _json(svc.reload)


@app.post("/api/auto-update", tags=["данные"], summary="Такт автообновления (только с --allow-write)")
def auto_update():
    if not WRITE:
        return JSONResponse({"action": "off"})
    return _json(svc.auto_update_tick)


@app.post("/api/log", include_in_schema=False)
async def js_log(request: Request):
    """Ошибки и события интерфейса в браузере – в ui.log, как у приложения."""
    try:
        d = await request.json()
        level = getattr(logging, str(d.get("level", "info")).upper(), logging.INFO)
        log.log(level, "web_js  %s", str(d.get("message", ""))[:2000])
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(StarletteHTTPException)
async def not_found(request: Request, exc: StarletteHTTPException):
    """Ненайденная страница интерфейса – понятная страница 404, а не JSON; API отвечает JSON."""
    if exc.status_code == 404 and not request.url.path.startswith("/api"):
        return FileResponse(ROOT / "app" / "ui" / "404.html", status_code=404)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# ---------------------------------------------------------------- интерфейс

@app.middleware("http")
async def ui_revalidate(request: Request, call_next):
    """Страницы интерфейса браузер сверяет с сервером при каждом открытии – правки видны сразу."""
    resp = await call_next(request)
    if request.url.path.startswith(("/ui/", "/assets/")):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


app.mount("/ui", StaticFiles(directory=ROOT / "app" / "ui"), name="ui")
app.mount("/assets", StaticFiles(directory=ROOT / "app" / "assets"), name="assets")
app.mount("/docs", StaticFiles(directory=ROOT / "docs"), name="docs")   # отчёт и презентация (PDF)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/ui/terminal.html")
