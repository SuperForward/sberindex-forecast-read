"""Данные для интерфейса приложения: чтение готовых таблиц и ответы в виде
словарей (мост отдаёт их в JS как JSON).

Всё читается один раз и кэшируется: таблицы небольшие (до сотен тысяч
строк). Источник – база data/sberindex.duckdb, которую публикует воркер
(src/store.py); без неё – те же файлы в data/ и reports/. После пересчёта
воркером приложение вызывает reload(), и кэш читается заново.
"""

import re
import threading
from functools import lru_cache, wraps

import numpy as np
import pandas as pd

from src import store
from src.config import RAW_DIR, REPORTS_DIR, load_config

CAT_RU = {"all": "Все категории", "food": "Продовольствие", "health": "Здоровье",
          "horeca": "Общепит", "transport": "Транспорт", "marketplaces": "Маркетплейсы"}
TYPE_RU = {"go": "городской округ", "mr": "муниципальный район", "mo": "муниципальный округ",
           "vt": "внутригородская территория"}
MODEL_RU = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
            "hist_gbm": "HistGradientBoosting", "random_forest": "Случайный лес",
            "ridge": "Линейная (ridge)", "baseline": "Прошлый год × недавний рост",
            "lgbm_global_nomonth": "LightGBM", "lgbm_global": "LightGBM (с номером месяца)",
            "seasonal_naive_drift": "Прошлый год × недавний рост", "prophet_default": "Prophet",
            "prophet_yearly3": "Prophet (годовая сезонность)", "naive": "Последний месяц без изменений",
            "seasonal_naive": "Прошлый год без учёта роста",
            "national_growth": "Прошлый год × рост по России", "chronos": "Chronos-Bolt (Amazon)",
            "timesfm": "TimesFM 2.5 (Google)", "chronos_level": "Chronos-Bolt по уровням",
            "timesfm_level": "TimesFM по уровням",
            "factor_only": "Общий фактор", "ets_rel": "ETS на отклонении от фактора",
            "ets_rel_reg": "ETS на отклонении от регионального фактора",
            "chronos_base_rel": "Chronos-Bolt base на отклонении от фактора",
            "chronos_base_rel_reg": "Chronos-Bolt base на отклонении от регионального фактора",
            "ens_eq[factor_mix]": "Ансамбль: модели на национальном и региональном факторе",
            "mean_all": "Среднее всех моделей"}


def model_name(code: str) -> str:
    """Человеческое имя модели; для ансамблей – «Ансамбль: A + B»."""
    if code in MODEL_RU:
        return MODEL_RU[code]
    m = re.fullmatch(r"ens_(eq|w)\[(.+)\]", code)
    if m:
        if m.group(2) == "top3":
            return "Ансамбль: три лучшие модели"
        if m.group(2) == "global":
            return "Ансамбль: среднее бустингов и леса"
        parts = " и ".join(MODEL_RU.get(x, x) for x in m.group(2).split("+"))
        return f"Ансамбль{'' if m.group(1) == 'eq' else ' (веса)'}: {parts}"
    return code
SHOCK_RU = {"fiscal_budget": "финансово-бюджетный", "production_local_market": "производственный",
            "socio_demographic": "социально-демографический"}
BACKTESTS = ["default", "models"]
# один и тот же baseline назван по-разному в default.yaml и models.yaml
MODEL_ALIAS = {"seasonal_naive_drift": "baseline"}


def _clean(v):
    if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


class NoData(RuntimeError):
    """Данных нет: свежий клон без снимка и без сборки."""


NO_DATA = ("Данных нет. Скачайте снимок данных или соберите их из исходников "
           "в разделе «Данные» (python -m worker snapshot fetch / python -m worker run --force).")


_LOAD_LOCK = threading.RLock()


def _cached(fn):
    """lru_cache(maxsize=1) для загрузчиков без аргументов, с блокировкой на
    промахе: запросы страницы идут из нескольких потоков, и без блокировки
    холодный кэш считался параллельно дважды (сводка при старте 7 с вместо 4)."""
    c = lru_cache(maxsize=1)(fn)

    @wraps(fn)
    def w():
        if c.cache_info().currsize:
            return c()
        with _LOAD_LOCK:
            return c()
    w.cache_clear = c.cache_clear
    return w


def _oktmo(series_id: pd.Series) -> pd.Series:
    """Код МО из series_id «oktmo|категория». Строки разбираются по уникальным
    рядам (их ~2 тыс.), а не по каждой из ~1 млн строк прогнозов."""
    c = series_id.astype("category")
    codes = c.cat.codes.to_numpy()
    ok = c.cat.categories.str.split("|").str[0].to_numpy(dtype=object)
    return pd.Series(np.where(codes >= 0, ok[codes], None), index=series_id.index, dtype=object)


def warm(logger: str = "ui") -> None:
    """Прогрев кэша при запуске приложения или сервера, до первого запроса
    страницы: загрузка данных и метрик моделей занимает несколько секунд."""
    import logging
    import time
    t0 = time.perf_counter()
    try:
        summary()
    except NoData:
        return
    except Exception:  # noqa: BLE001  ошибку покажет сам запрос страницы
        logging.getLogger(logger).warning("data_warm_failed", exc_info=True)
        return
    logging.getLogger(logger).info("data_warm  данные и метрики загружены",
                                   extra={"elapsed_ms": (time.perf_counter() - t0) * 1000})


def reload() -> dict:
    """Сбросить кэш после пересчёта воркером."""
    import logging
    at = store.published_at()
    logging.getLogger("ui").info("data_reload  опубликовано %s", at)
    for f in (_panel, _mo_table, _predictions, _events, _candidates, _annual, _horizons, _cpd_alarms,
              model_metrics, _forecast, _error_band):
        f.cache_clear()
    log_choices()
    return {"ok": True, "published_at": at}


def log_choices(logger: str = "ui") -> dict:
    """Записать в лог, на чём строятся результаты: период данных, модель прогноза
    вперёд и её честная ошибка, модель графика важности. Смена модели прогноза
    с прошлого раза – WARNING: например, ансамбль пропал без весов chronos-bolt-base."""
    import logging
    ui = logging.getLogger(logger)
    try:
        p = _panel()
    except NoData:
        ui.warning("choices  данных нет")
        return {}
    fm, im = forecast_model(), importance_model()
    mm = model_metrics()
    row = next((x for x in mm["models"] if x["model"] == fm), None)
    f = _forecast()
    out = {"period": f"{p['date'].min():%Y-%m}..{p['date'].max():%Y-%m}", "forecast_model": fm,
           "forecast_mae_last": row["MAE_test"] if row else None, "forecast_from": None if f.empty else
           f"{f['origin'].max():%Y-%m}", "importance_model": im, "best_honest": mm.get("best")}
    ui.info("choices  данные %s; прогноз вперёд: %s (MAE окна %s: %s) из %s; важность признаков: %s; "
            "лучшая без подбора: %s", out["period"], fm, mm.get("test_origin"), out["forecast_mae_last"],
            out["forecast_from"], im, out["best_honest"])
    if fm is None:
        ui.warning("choices  прогноза вперёд нет (нет reports/backtest/models/forecast.parquet)")
    for ens, miss in ensemble_missing().items():
        ui.warning("ensemble_partial  %s собран не из всех моделей, нет: %s (python -m worker models)", ens, miss)
    try:
        prev = store.get_meta("forecast_model")
        if prev and fm and prev != fm:
            ui.warning("forecast_model_changed  модель прогноза вперёд сменилась: %s -> %s", prev, fm)
        if fm and prev != fm:
            store.set_meta("forecast_model", fm)
    except Exception:  # noqa: BLE001  состояние воркера недоступно – не мешаем показу данных
        ui.debug("choices  meta недоступна", exc_info=True)
    return out


@_cached
def _panel() -> pd.DataFrame:
    p = store.load("spending_mo")
    if p.empty:
        raise NoData(NO_DATA)
    return p


@_cached
def _mo_table() -> pd.DataFrame:
    p = _panel()
    mo = p.groupby("oktmo").agg(mo_name=("mo_name", "first"), region=("region", "first"),
                                region_code=("region_code", "first"),
                                months=("date", "nunique")).reset_index()
    pop = store.load("mo_population")[["oktmo", "pop", "pop_urban"]]
    mp = store.load("mo_oktmo_map")[["oktmo", "mo_type", "match_type"]]
    mo = mo.merge(pop, on="oktmo", how="left").merge(mp.drop_duplicates("oktmo"), on="oktmo", how="left")
    allc = p[p["category"] == "all"].assign(y=lambda d: d["date"].dt.year, m=lambda d: d["date"].dt.month)
    # последний год в данных (после обновления источников – следующий) и год до него
    y1 = int(allc["y"].max())
    y0 = y1 - 1
    last = allc[allc["y"] == y1].groupby("oktmo")["value"].mean()
    # Рост – только по месяцам, которые есть в обоих годах: у 163 МО ряды
    # неполные, а последний год может быть неполным (данные выходят помесячно);
    # среднее «май–декабрь» против «январь–декабрь» дало бы ложный рост из-за сезонности.
    both = allc[allc["y"].isin([y0, y1])].pivot_table(index=["oktmo", "m"], columns="y", values="value").dropna()
    g = both.groupby(level=0).sum()
    mo["year"] = y1
    mo["spend_last"] = mo["oktmo"].map(last)
    mo["growth_last"] = mo["oktmo"].map(g[y1] / g[y0] - 1) if {y0, y1} <= set(g.columns) else None
    mo["growth_months"] = mo["oktmo"].map(both.groupby(level=0).size())
    mo["region_short"] = mo["region"].str.replace(r"\s*\(.*\)", "", regex=True)
    return mo


@_cached
def _predictions() -> pd.DataFrame:
    p = store.load("predictions")
    if p.empty:
        return p.assign(oktmo=pd.Series(dtype=object))
    # Ансамбли базового backtest-а (из наивных моделей и Prophet) не показываем:
    # они называются так же, как ансамбли основного сравнения (ens_eq[top3]),
    # и подменяли бы их при совпадении кода.
    p = p[~((p["backtest"] == "default") & p["model"].str.startswith("ens_"))]
    # порядок: default раньше models – при совпадении кода остаётся первый
    p = p.sort_values("backtest", key=lambda s: s.map({b: i for i, b in enumerate(BACKTESTS)}), kind="stable")
    p["model"] = p["model"].replace(MODEL_ALIAS)
    p = p.drop_duplicates(["series_id", "model", "origin", "h"])
    p["oktmo"] = _oktmo(p["series_id"])
    return p


@_cached
def _events() -> pd.DataFrame:
    return store.load("events")


@_cached
def _candidates() -> pd.DataFrame:
    return store.load("shock_candidates")


@_cached
def _annual() -> pd.DataFrame:
    return store.load("mo_annual")


# ------------------------------------------------------------------ API

def search_mo(query: str = "", limit: int = 60) -> dict:
    """Поиск МО. Порядок: слово названия начинается с запроса («орск» -> Орск),
    затем совпадение внутри названия (Магнитогорский), затем по региону
    (Приморского края); внутри группы – по населению."""
    mo = _mo_table()
    q = (query or "").strip().lower().replace("ё", "е")
    total = len(mo)
    if q:
        name = mo["mo_name"].str.lower().str.replace("ё", "е")
        region = mo["region"].str.lower().str.replace("ё", "е")
        rank = pd.Series(9, index=mo.index)
        rank[region.str.contains(q, regex=False)] = 2
        rank[name.str.contains(q, regex=False)] = 1
        rank[name.str.contains(r"(?:^|[\s-])" + re.escape(q), regex=True)] = 0
        mo = mo.assign(_rank=rank)[rank < 9]
        total = len(mo)
        mo = mo.sort_values(["_rank", "pop"], ascending=[True, False], na_position="last")
    else:
        mo = mo.sort_values("pop", ascending=False, na_position="last")
    mo = mo.head(limit)
    return {"total": int(total), "year": int(mo["year"].iloc[0]) if len(mo) else None,
            "items": _records(mo[["oktmo", "mo_name", "region_short", "pop", "spend_last", "growth_last", "months"]])}


def mo_detail(oktmo: str) -> dict:
    mo = _mo_table()
    row = mo[mo["oktmo"] == oktmo]
    if row.empty:
        return {"error": f"МО {oktmo} не найдено"}
    info = _records(row)[0]
    info["mo_type_ru"] = TYPE_RU.get(info.get("mo_type"), info.get("mo_type"))
    p = _panel()
    p = p[p["oktmo"] == oktmo].sort_values("date")
    series = {c: [{"time": d.strftime("%Y-%m-%d"), "value": float(v)}
                  for d, v in zip(g["date"], g["value"])] for c, g in p.groupby("category")}

    annual = []
    a = _annual()
    if not a.empty:
        cfg = load_config("pmo_indicators")
        names = {f"pmo_{c}": n for grp in cfg["indicators"].values() for c, n in grp.items()}
        sub = a[a["oktmo"] == oktmo].sort_values("year")
        for col, name in names.items():
            if col in sub and sub[col].notna().any():
                s = sub[["year", col]].dropna()
                annual.append({"indicator": name, "values": {int(y): _clean(v) for y, v in zip(s["year"], s[col])}})

    ev = _events()
    ev = ev[ev["oktmo"] == oktmo]
    events = [{"date": r.start_date.strftime("%Y-%m-%d"), "title": r.description, "type": SHOCK_RU.get(r.shock_type, r.shock_type),
               "source": r.source_url} for r in ev.itertuples()]
    cand = _candidates()
    cand = cand[cand["oktmo"] == oktmo] if not cand.empty else cand
    candidates = [{"year": int(r.year), "type": SHOCK_RU.get(r.shock_type, r.shock_type), "metric": r.metric,
                   "change": _clean(r.change), "z": _clean(r.z), "caution": _caution(r.caution)}
                  for r in cand.itertuples()]
    return {"info": info, "series": series, "categories": CAT_RU, "annual": annual,
            "events": events, "candidates": candidates}


def _tuned(model: str) -> bool:
    """Ансамбли основного backtest-а: веса или состав подобраны на его окнах,
    поэтому их MAE по всем окнам занижена и в выбор лучшей модели они не входят."""
    return model.startswith("ens_")


@_cached
def model_metrics() -> dict:
    """Метрики моделей основного backtest-а.

    Порядок: сначала модели без подбора на тестовых данных (по MAE на всех
    окнах), затем ансамбли. Лучшая модель выбирается только среди первых.
    MAE_test – MAE на последнем окне: единственная оценка, честная для всех.
    """
    p = _predictions()
    if p.empty:
        return {"models": [], "best": None}
    from src.eval.metrics import summarize
    overall = summarize(p, ["model"])
    overall["tuned"] = overall["model"].map(_tuned)
    overall = overall.sort_values(["tuned", "MAE"])
    by_h = summarize(p, ["model", "h"]).pivot_table(index="model", columns="h", values="MAE")
    by_o = summarize(p, ["model", "origin"]).pivot_table(index="model", columns="origin", values="MAE")
    last = by_o.columns.max()
    # устойчивость по окнам: худшее окно и в скольких окнах модель лучше baseline
    base = by_o.loc["baseline"] if "baseline" in by_o.index else None
    missing = ensemble_missing()
    rows = []
    for r in overall.itertuples():
        rows.append({"model": r.model, "name": model_name(r.model), "n": int(r.n),
                     "MAE": _clean(r.MAE), "WAPE": _clean(r.WAPE), "R2": _clean(r.R2),
                     "R2_within": _clean(r.R2_within), "tuned": bool(r.tuned),
                     "MAE_test": _clean(by_o.at[r.model, last]),
                     "MAE_worst": _clean(by_o.loc[r.model].max()),
                     "windows_better_than_baseline": int((by_o.loc[r.model] < base).sum()) if base is not None else None,
                     "missing": [model_name(m) for m in missing.get(r.model, [])],
                     "by_h": {int(h): _clean(v) for h, v in by_h.loc[r.model].items()},
                     "by_origin": {o.strftime("%Y-%m"): _clean(v) for o, v in by_o.loc[r.model].items()}})
    honest = [x for x in rows if not x["tuned"]]
    best_test = min(rows, key=lambda x: x["MAE_test"] if x["MAE_test"] is not None else float("inf"))
    return {"models": rows, "n_series": int(p["series_id"].nunique()),
            "best": honest[0]["model"] if honest else None,
            "best_test": best_test["model"], "test_origin": last.strftime("%Y-%m"),
            "origins": [o.strftime("%Y-%m") for o in sorted(by_o.columns)],
            "horizon": int(p["h"].max())}


@_cached
def _forecast() -> pd.DataFrame:
    f = store.load("forecast")
    return f.assign(oktmo=_oktmo(f["series_id"])) if not f.empty else f.assign(oktmo=pd.Series(dtype=object))


def ensemble_missing() -> dict:
    """Ансамбли, собранные не из всех заданных моделей: {код: [нет моделей]}.
    Так бывает без весов chronos-bolt-base (python -m worker models)."""
    e = store.load("ensemble_members")
    if e.empty:
        return {}
    return {r.model: [m for m in str(r.missing).split(",") if m and m != "nan"]
            for r in e.itertuples() if isinstance(r.missing, str) and r.missing}


def forecast_model() -> str | None:
    """Модель прогноза вперёд: лучшая MAE по всем окнам среди моделей без
    подбора на окнах backtest-а (сейчас mean_all – среднее всех моделей).
    Не по последнему окну: состав ens_eq[factor_mix] выбран после просмотра
    всех окон, включая последнее, и его MAE там тоже занижена."""
    f = _forecast()
    if f.empty:
        return None
    have = set(f["model"])
    honest = [x for x in model_metrics()["models"] if not x["tuned"] and x["MAE"] is not None]
    return next((x["model"] for x in sorted(honest, key=lambda x: x["MAE"]) if x["model"] in have), None)


@lru_cache(maxsize=4)
def _error_band(model: str) -> pd.DataFrame:
    """Интервал 80% из ошибок backtest-а модели по каждому горизонту: своя
    ошибка МО плюс общий для всех МО промах (src/eval/intervals.py). Честнее
    интервалов самих моделей: на коротких рядах те слишком узкие (см. раздел
    «Сдвиги»)."""
    from src.eval.intervals import error_band
    p = _predictions()
    return error_band(p[p["model"] == model])


def mo_forecast(oktmo: str) -> dict:
    p = _predictions()
    p = p[p["oktmo"] == oktmo]
    out = {}
    for (model, origin), g in p.groupby(["model", "origin"]):
        g = g.sort_values("date")
        out.setdefault(model, []).append({
            "origin": origin.strftime("%Y-%m-%d"),
            "points": [{"time": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in zip(g["date"], g["yhat"])],
            "mae": _clean(float((g["y"] - g["yhat"]).abs().mean()))})
    future = None
    fm = forecast_model()
    if fm:
        f = _forecast()
        f = f[(f["oktmo"] == oktmo) & (f["model"] == fm)].sort_values("date")
        if not f.empty:
            band = _error_band(fm)
            future = {"model": fm, "name": model_name(fm), "origin": f["origin"].iloc[0].strftime("%Y-%m-%d"),
                      "missing": [model_name(m) for m in ensemble_missing().get(fm, [])],
                      "points": [{"time": r.date.strftime("%Y-%m-%d"), "value": _clean(r.yhat),
                                  "lo": _clean(r.yhat * (1 + band.at[r.h, "lo"])) if r.h in band.index else None,
                                  "hi": _clean(r.yhat * (1 + band.at[r.h, "hi"])) if r.h in band.index else None}
                                 for r in f.itertuples()]}
    return {"models": {m: {"name": model_name(m), "windows": w} for m, w in out.items()}, "future": future}


EXPORT_KINDS = {"forecast": "прогноз вперёд по всем МО (основная модель, интервал 80%)",
                "forecast_all_models": "прогноз вперёд по всем МО, все модели",
                "forecast_mo": "прогноз вперёд по одному МО, все модели",
                "metrics": "метрики моделей backtest"}


def export_csv(kind: str, oktmo: str = "") -> dict:
    """Таблица для выгрузки: {"filename", "csv"}. CSV под русский Excel:
    разделитель «;», десятичная запятая (BOM добавляет тот, кто сохраняет)."""
    if kind not in EXPORT_KINDS:
        return {"error": f"неизвестная выгрузка {kind}"}
    mo = _mo_table().set_index("oktmo")[["mo_name", "region_short"]]
    if kind == "metrics":
        mm = model_metrics()
        rows = []
        for x in mm["models"]:
            r = {"модель": x["name"], "код": x["model"], "подбор на окнах проверки": "да" if x["tuned"] else "нет",
                 "MAE": x["MAE"], f"MAE окно {mm['test_origin']}": x["MAE_test"], "MAE худшее окно": x["MAE_worst"],
                 "окон лучше простого прогноза": x["windows_better_than_baseline"], "WAPE": x["WAPE"], "R2 внутри ряда": x["R2_within"]}
            r.update({f"MAE h={h}": v for h, v in x["by_h"].items()})
            r.update({f"MAE окно {o}": v for o, v in x["by_origin"].items()})
            rows.append(r)
        df, name = pd.DataFrame(rows), "metrics"
    else:
        f = _forecast()
        if f.empty:
            return {"error": "прогноза вперёд нет: пересчитайте модели (раздел «Данные»)"}
        fm = forecast_model()
        if kind == "forecast":
            f = f[f["model"] == fm]
        if kind == "forecast_mo":
            f = f[f["oktmo"] == oktmo]
            if f.empty:
                return {"error": f"для МО {oktmo} прогноза вперёд нет"}
        band = _error_band(fm) if fm else pd.DataFrame(columns=["lo", "hi"])
        main = f["model"] == fm
        lo = f["h"].map(band["lo"]) if len(band) else np.nan
        hi = f["h"].map(band["hi"]) if len(band) else np.nan
        df = pd.DataFrame({
            "ОКТМО": f["oktmo"], "МО": f["oktmo"].map(mo["mo_name"]), "регион": f["oktmo"].map(mo["region_short"]),
            "месяц": f["date"].dt.strftime("%Y-%m"), "горизонт, мес.": f["h"],
            "модель": f["model"].map(model_name), "прогноз, руб. на жителя": f["yhat"].round(0).astype("Int64"),
            "нижняя граница 80%": (f["yhat"] * (1 + lo)).where(main).round(0).astype("Int64"),
            "верхняя граница 80%": (f["yhat"] * (1 + hi)).where(main).round(0).astype("Int64"),
            "последние данные": f["origin"].dt.strftime("%Y-%m")}).sort_values(["ОКТМО", "модель", "месяц"])
        name = kind + (f"_{oktmo}" if kind == "forecast_mo" else "")
    import logging
    logging.getLogger("ui").info("export  %s%s: %d строк", kind, f" {oktmo}" if oktmo else "", len(df))
    return {"filename": f"sberindex_{name}.csv", "description": EXPORT_KINDS[kind],
            "csv": df.to_csv(index=False, sep=";", decimal=",")}


def best_model() -> str | None:
    """Лучшая модель без подбора на тестовых окнах (для панелей «текущая лучшая»)."""
    return model_metrics()["best"]


def worst_mo(model: str = "", limit: int = 20) -> dict:
    """МО с наибольшей средней относительной ошибкой модели."""
    p = _predictions()
    model = model or best_model() or ""
    p = p[p["model"] == model]
    if p.empty:
        return {"items": [], "model": model, "name": model_name(model)}
    err = p.assign(ape=(p["y"] - p["yhat"]).abs() / p["y"]).groupby("oktmo")["ape"].mean()
    mo = _mo_table().set_index("oktmo")
    top = err.sort_values(ascending=False).head(limit)
    return {"model": model, "name": model_name(model),
            "items": [{"oktmo": k, "mo_name": mo.at[k, "mo_name"], "region_short": mo.at[k, "region_short"],
                       "mape": _clean(v)} for k, v in top.items() if k in mo.index]}


CAUTION_RU = {"data_error": "ошибка данных", "small_base": "малая база", "reorganized": "преобразование МО"}


def _caution(v) -> str:
    return CAUTION_RU.get(v, "") if isinstance(v, str) else ""


def shocks_overview(shock_type: str = "", year: int = 0, show_caution: bool = False,
                    panel_only: bool = True, limit: int = 300) -> dict:
    ev = _events()
    events = [{"event_id": r.event_id, "date": r.start_date.strftime("%Y-%m-%d"), "region": r.region,
               "mo_name": _clean(r.mo_name), "oktmo": _clean(r.oktmo), "in_panel": bool(r.in_panel),
               "type": SHOCK_RU.get(r.shock_type, r.shock_type), "title": r.description,
               "source": r.source_url} for r in ev.itertuples()]
    c = _candidates()
    if not c.empty:
        if panel_only:
            c = c[c["in_panel"]]
        if not show_caution:
            c = c[c["caution"].fillna("") == ""]
        if shock_type:
            c = c[c["shock_type"] == shock_type]
        if year:
            c = c[c["year"] == year]
        c = c.reindex(c["z"].abs().sort_values(ascending=False).index).head(limit)
    cand = [{"oktmo": r.oktmo, "mo_name": _clean(r.mo_name), "region": _clean(r.region), "year": int(r.year),
             "type": SHOCK_RU.get(r.shock_type, r.shock_type), "metric": r.metric,
             "change": _clean(r.change), "z": _clean(r.z), "caution": _caution(r.caution)}
            for r in c.itertuples()] if not c.empty else []
    counts = {}
    full = _candidates()
    if not full.empty:
        t = full[full["in_panel"] & (full["caution"].fillna("") == "")].pivot_table(index="year", columns="shock_type", values="oktmo",
                                               aggfunc="count", fill_value=0)
        counts = {int(y): {SHOCK_RU.get(k, k): int(v) for k, v in r.items()} for y, r in t.iterrows()}
    return {"events": events, "candidates": cand, "counts": counts}


@_cached
def _n_territories() -> int | None:
    """Число кодов территорий в официальном наборе: больше числа МО на дубли кодов
    (Павловский Посад до и после 2024 г.). Без сырых файлов – по коду территории в панели."""
    f = RAW_DIR / "hackathon" / "consumption.parquet"
    try:
        if f.exists():
            return int(pd.read_parquet(f, columns=["territory_id"])["territory_id"].nunique())
        # снимок данных без сырых файлов: код территории есть в самой панели
        p = _panel()
        return int(p["territory_id"].nunique()) if "territory_id" in p.columns else None
    except Exception:  # noqa: BLE001
        return None


def summary() -> dict:
    p = _panel()
    mo = _mo_table()
    mm = model_metrics()
    m = mm["models"]
    best = next((x for x in m if x["model"] == mm["best"]), None)
    prophet = next((x for x in m if x["model"] == "prophet_default"), None)
    return {
        "mo": int(mo["oktmo"].nunique()), "territories": _n_territories(), "regions": int(mo["region_code"].nunique()),
        "months": int(p["date"].nunique()),
        "period": f"{p['date'].min():%m.%Y}–{p['date'].max():%m.%Y}",
        "pop_covered": _clean(mo["pop"].sum()),
        "best_model": best, "prophet": prophet,
        "gain_vs_prophet": _clean(1 - best["MAE"] / prophet["MAE"]) if best and prophet else None,
        "events": int(_events()["event_id"].nunique()),
        "candidates": int((_candidates()["in_panel"] & (_candidates()["caution"].fillna("") == "")).sum())
        if not _candidates().empty else 0,
    }


# ------------------------------------------------ модели по горизонтам

@_cached
def _horizons() -> pd.DataFrame:
    return store.load("horizons_metrics")


def horizons_overview() -> dict:
    """MAE по горизонтам 1/3/6/12 и выигрыш у Prophet (reports/horizons)."""
    h = _horizons()
    if h.empty:
        return {"models": [], "horizons": [], "error": "нет reports/horizons: запустите scripts.run_horizons"}
    h = h.assign(model=h["model"].replace(MODEL_ALIAS))
    cfg = load_config("horizons")
    hs = cfg["horizons"]
    mae = h.pivot_table(index="model", columns="H", values="MAE").reindex(columns=hs)
    r2w = h.pivot_table(index="model", columns="H", values="R2_within").reindex(columns=hs)
    ref = mae.loc["prophet_default"] if "prophet_default" in mae.index else None
    key = 3 if 3 in hs else hs[0]
    order = mae.sort_values(key, na_position="last").index
    # лучшая на горизонте – только среди моделей без подбора на окнах backtest-а,
    # как в model_metrics: у ансамблей MAE занижена
    honest = mae.loc[[m for m in mae.index if not _tuned(m)]]
    best = {int(H): honest[H].idxmin() for H in hs if honest[H].notna().any()}
    rows = []
    for m in order:
        rows.append({"model": m, "name": model_name(m), "tuned": _tuned(m),
                     "mae": {int(H): _clean(mae.at[m, H]) for H in hs},
                     "r2_within": {int(H): _clean(r2w.at[m, H]) for H in hs},
                     "vs_prophet": {int(H): _clean(1 - mae.at[m, H] / ref[H]) if ref is not None else None
                                    for H in hs}})
    return {"horizons": hs, "targets": cfg["targets"], "models": rows,
            "best": {H: {"model": m, "name": model_name(m)} for H, m in best.items()}}


# ------------------------------------------------ детекторы сдвигов (CPD)

CPD_RU = {"zscore": "z-порог", "ewma": "EWMA", "cusum": "CUSUM", "bocpd": "BOCPD (байесовский)",
          "pelt": "PELT (офлайн)", "zscore_no_common": "z-порог без вычета общего сдвига",
          "chronos_interval": "Chronos: интервал модели", "chronos_median": "Chronos: прогноз + шум МО",
          "ewma_cat": "EWMA по категориям", "zscore_cat": "z-порог по категориям",
          "ewma_mix": "EWMA по двум сигналам", "zscore_mix": "z-порог по двум сигналам"}


CPD_TABLES = {"comparison.csv": "cpd_comparison", "by_size.csv": "cpd_by_size",
              "real_events.csv": "cpd_real_events", "alarms.parquet": "cpd_alarms"}


def _cpd_file(name: str) -> pd.DataFrame:
    return store.load(CPD_TABLES[name])


@_cached
def _cpd_alarms() -> pd.DataFrame:
    return _cpd_file("alarms.parquet")


def cpd_overview(limit: int = 200) -> dict:
    """Сравнение детекторов, проверка на паводках и МО с тревогами лучшего метода."""
    comp = _cpd_file("comparison.csv")
    if comp.empty:
        return {"error": "нет reports/cpd: запустите scripts.run_cpd"}
    cfg = load_config("cpd")
    main_far = cfg["main_far"]
    by_size = _cpd_file("by_size.csv")
    real = _cpd_file("real_events.csv")
    a = _cpd_alarms()
    best = a["method"].iloc[0] if not a.empty else None

    methods = []
    for m, g in comp.groupby("method", sort=False):
        g = g.set_index("far_target")
        bs = by_size[(by_size["method"] == m) & (by_size["far_target"] == main_far)]
        rm = real[(real["method"] == m) & (real["far_target"] == main_far)] if not real.empty else real
        methods.append({
            "method": m, "name": CPD_RU.get(m, m), "online": bool(g["online"].iloc[0]),
            "recall": {str(f): _clean(v) for f, v in g["recall"].items()},
            "delay": _clean(g.at[main_far, "delay_months"]), "early": _clean(g.at[main_far, "early_alarm"]),
            "far_clean": _clean(g.at[main_far, "far_clean"]),
            "f1": _clean(g.at[main_far, "f1"]) if "f1" in g else None,
            "by_size": {str(s): _clean(v) for s, v in zip(bs["size"], bs["recall"])},
            "real_caught": int(rm["caught_3m"].sum()) if len(rm) else 0})
    methods.sort(key=lambda x: (not x["online"], -(x["recall"].get(str(main_far)) or 0)))

    events = []
    if not real.empty:
        for (o, n), g in real.groupby(["oktmo", "name"], sort=False):
            events.append({"oktmo": o, "name": n, "event": g["event"].iloc[0],
                           "first": {f"{r.method}|{r.far_target}": _clean(r.first_alarm) for r in g.itertuples()}})

    flagged = []
    if not a.empty and limit:
        al = a[a["alarm"]].sort_values("date").groupby("oktmo").agg(
            first=("date", "first"), x=("x", "first"), score=("score", "max"), n=("alarm", "size"))
        mo = _mo_table().set_index("oktmo")
        al = al.join(mo[["mo_name", "region_short", "pop"]], how="left").sort_values("score", ascending=False)
        flagged = [{"oktmo": k, "mo_name": _clean(r["mo_name"]), "region_short": _clean(r["region_short"]),
                    "pop": _clean(r["pop"]), "first": r["first"].strftime("%Y-%m"), "x": _clean(r["x"]),
                    "score": _clean(r["score"]), "months": int(r["n"]),
                    "episodes": _alarm_episodes(a[a["oktmo"] == k])} for k, r in al.head(limit).iterrows()]
    return {"main_far": main_far, "far_targets": cfg["far_targets"], "best": best,
            "best_name": CPD_RU.get(best, best), "methods": methods, "events": events,
            "flagged": flagged, "n_flagged": int(a[a["alarm"]]["oktmo"].nunique()) if not a.empty else 0,
            "n_mo": int(a["oktmo"].nunique()) if not a.empty else 0,
            "threshold": _clean(a["threshold"].iloc[0]) if not a.empty else None}


def _alarm_episodes(g: pd.DataFrame) -> list[dict]:
    """Тревоги МО подряд идущими месяцами: у каждого эпизода свой пик сигнала и своё
    отклонение от страны (иначе спад весной и разворот в декабре сливаются в одну цифру)."""
    g = g.sort_values("date").reset_index(drop=True)
    run = (g["alarm"] != g["alarm"].shift()).cumsum()
    out = []
    for _, e in g[g["alarm"]].groupby(run[g["alarm"]]):
        pk = e.loc[e["score"].idxmax()]
        i0 = e.index[0]
        out.append({"first": e["date"].iloc[0].strftime("%Y-%m"), "last": e["date"].iloc[-1].strftime("%Y-%m"),
                    "score": _clean(pk["score"]), "peak": pk["date"].strftime("%Y-%m"),
                    # отклонение роста от страны за месяц до эпизода и в месяц пика сигнала
                    "x_before": _clean(g.at[i0 - 1, "x"]) if i0 > 0 else None, "x_peak": _clean(pk["x"])})
    return out


def cpd_mo(oktmo: str) -> dict:
    """Локальный рост МО (п.п. к общему по стране) и сила сигнала лучшего детектора."""
    a = _cpd_alarms()
    g = a[a["oktmo"] == oktmo].sort_values("date") if not a.empty else a
    if g.empty:
        return {"error": "для этого МО нет ряда детектора (нужны полные 24 месяца)"}
    mo = _mo_table().set_index("oktmo")
    t = [d.strftime("%Y-%m-%d") for d in g["date"]]
    return {"oktmo": oktmo, "mo_name": _clean(mo.at[oktmo, "mo_name"]) if oktmo in mo.index else oktmo,
            "method": g["method"].iloc[0], "name": CPD_RU.get(g["method"].iloc[0]),
            "threshold": _clean(g["threshold"].iloc[0]),
            "x": [{"time": d, "value": _clean(v * 100)} for d, v in zip(t, g["x"])],
            "score": [{"time": d, "value": _clean(v)} for d, v in zip(t, g["score"])],
            "alarms": [d for d, f in zip(t, g["alarm"]) if f]}


# ------------------------------------------------ результаты и воспроизводимость

def _git_rev() -> str | None:
    import subprocess
    from src.config import NO_WINDOW, ROOT
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %ad", "--date=short"],
                           capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


PY = ".venv/Scripts/python -m"
REPRO_STEPS = [
    ("Окружение (точные версии библиотек)", "powershell -ExecutionPolicy Bypass -File setup.ps1"),
    ("Всё одной командой (воркер: скачивание, сборка, проверка качества, модели, публикация)",
     f"{PY} worker run --fetch --force --heavy"),
    ("Панель МО и сопоставление с ОКТМО", f"{PY} scripts.build_mo_map"),
    ("Загрузка открытых данных (нужен интернет)", f"{PY} scripts.download_pmo; {PY} scripts.download_external"),
    ("Ковариаты (ЦБ, календарь, погода, Росстат)", f"{PY} scripts.build_covariates"),
    ("Годовые показатели МО и кандидаты в шоки",
     f"{PY} scripts.build_mo_features; {PY} scripts.find_shock_candidates"),
    ("Сверка источников и аудит данных", f"{PY} scripts.validate_sources; {PY} scripts.audit_data"),
    ("Базовые модели и Prophet", f"{PY} scripts.run_backtest --config default"),
    ("Сравнение моделей и ансамбли", f"{PY} scripts.run_backtest --config models"),
    ("Горизонты 1/3/6/12 мес.", f"{PY} scripts.run_horizons --config horizons"),
    ("Детекторы сдвигов", f"{PY} scripts.run_cpd --config cpd"),
    ("Приложение", f"{PY} app.main"),
]


FEATURE_RU = {
    "g1": "рост г/г в последнем месяце", "g2": "рост г/г месяцем раньше", "g3": "рост г/г два месяца назад",
    "drift3": "рост г/г за 3 месяца", "mom": "изменение к прошлому месяцу",
    "seas_step": "прошлогодний сезонный шаг", "log_pop": "численность населения",
    "urban_share": "доля городского населения", "mo_type": "тип МО", "h": "горизонт прогноза",
    "common_g1": "общий по стране рост г/г", "common_g1_prev": "общий рост г/г месяцем раньше",
    "cal_workdays": "рабочие дни", "cal_workdays_yoy": "рабочие дни к прошлому году",
    "cal_workdays_yoy_t": "рабочие дни к прошлому году (месяц прогноза)",
    "cal_holidays_on_weekdays": "праздники в будни", "cal_long_weekend_max": "длинные выходные",
    "wage_log": "зарплата в МО", "wage_growth": "рост зарплаты в МО", "employees_log": "число работников",
    "employees_growth": "рост числа работников", "retail_turnover_log": "оборот розницы в МО",
    "retail_turnover_growth": "рост оборота розницы", "shipments_log": "отгрузка продукции",
    "shipments_growth": "рост отгрузки", "own_revenue_share_log": "доля собственных доходов бюджета",
    "own_revenue_share_growth": "изменение доли собственных доходов", "migration_per_1000": "миграция",
    "unemployment_ilo": "безработица в регионе", "cpi_yoy": "инфляция в регионе",
    "cpi_food_yoy": "продовольственная инфляция", "services_ifo_yoy": "цены услуг",
    "catering_yoy": "оборот общепита в регионе", "usd_yoy": "курс доллара г/г", "usd_mom": "курс доллара м/м",
    "precip_anom": "аномалия осадков", "t_anom": "аномалия температуры", "cov_region": "регион",
    # ПМО (годовые показатели МО)
    "catering_turnover_log": "оборот общепита в МО", "catering_turnover_growth": "рост оборота общепита в МО",
    "investment_pc_log": "инвестиции на жителя", "investment_pc_growth": "рост инвестиций на жителя",
    "housing_pc_log": "ввод жилья на жителя", "housing_pc_growth": "рост ввода жилья",
    "admin_spend_pc_log": "расходы на органы МСУ на жителя", "admin_spend_pc_growth": "рост расходов на органы МСУ",
    "wage_large_log": "зарплата крупных и средних организаций", "wage_large_growth": "рост зарплаты крупных организаций",
    "natural_growth_per_1000": "естественный прирост", "sme_per_10k": "субъекты МСП на 10 тыс. жителей",
    "sme_emp_share": "доля работников МСП",
    # региональные ряды ЕМИСС, ЦБ, ФНС
    "deposits_households_yoy": "вклады населения г/г", "loans_households_yoy": "кредиты населению г/г",
    "income_per_capita_yoy": "доходы на душу г/г", "ndfl_agents_ytd_yoy": "НДФЛ с начала года г/г",
    "ndfl_agents_m_yoy": "НДФЛ за месяц г/г", "tax_local_ytd_yoy": "налоги в местные бюджеты с начала года г/г",
    "tax_local_m_yoy": "налоги в местные бюджеты за месяц г/г", "poverty_rate": "уровень бедности",
    "poverty_rate_diff": "изменение уровня бедности", "basket_cost_yoy": "стоимость набора товаров и услуг г/г",
    "housing_price_m2_primary_yoy": "цена жилья, первичный рынок, г/г",
    "housing_price_m2_secondary_yoy": "цена жилья, вторичный рынок, г/г",
    "loans_corporate_yoy": "кредиты юрлицам и ИП г/г", "loans_sme_issued_yoy": "выдачи кредитов МСП г/г",
    "loans_sme_debt_yoy": "кредиты МСП г/г", "loans_sme_overdue_yoy": "просрочка МСП г/г",
    "prices_fuel_utilities_ai92_yoy": "цена бензина АИ-92 г/г", "prices_fuel_utilities_diesel_yoy": "цена дизельного топлива г/г",
    "prices_fuel_utilities_power_yoy": "тариф на электроэнергию г/г",
    "prices_fuel_utilities_water_cold_yoy": "тариф на холодную воду и водоотведение г/г",
    "prices_fuel_utilities_water_hot_yoy": "тариф на горячую воду г/г",
    "prices_fuel_utilities_heating_yoy": "тариф на отопление г/г", "prices_fuel_utilities_gas_yoy": "тариф на газ г/г"}


def importance_model() -> str | None:
    """Модель для графика «вклад признаков»: лучшая MAE на последнем окне
    среди моделей, у которых есть важность признаков (бустинги, лес, ridge)."""
    d = store.load("feature_importance")
    if d.empty:
        return None
    have = set(d["model"])
    ranked = sorted((x for x in model_metrics()["models"] if x["model"] in have and x["MAE_test"] is not None),
                    key=lambda x: x["MAE_test"])
    return ranked[0]["model"] if ranked else sorted(have)[0]


def _feature_base(f: str) -> str:
    """Имя признака без префиксов one-hot ridge: num__g1 -> g1, cat__cov_region_12 -> cov_region."""
    if f.startswith("num__"):
        return f[5:]
    m = re.fullmatch(r"cat__(.+)_\d+", f)
    return m.group(1) if m else f


def _importance(model: str | None = None, top: int = 10) -> list[dict]:
    d = store.load("feature_importance")
    model = model or importance_model()
    if d.empty or not model:
        return []
    d = d[d["model"] == model]
    s = d.groupby(d["feature"].map(_feature_base))["share"].sum().sort_values(ascending=False).head(top)
    return [{"feature": f, "name": FEATURE_RU.get(f, f), "share": _clean(v)} for f, v in s.items()]


NULL_SCENARIOS_RU = {"real": "настоящие данные", "noise_features": "признаки заменены шумом",
                     "shuffled_target": "цель перемешана", "shuffled_within_month": "перемешана внутри месяца"}


def null_test() -> dict | None:
    """Null-тест (scripts/run_null_test.py): MAE моделей на настоящих данных и
    в null-сценариях; p – доля null-повторов с MAE не хуже настоящего."""
    d = store.load("null_test_runs")
    if d.empty:
        return None
    real = d[d["scenario"] == "real"].groupby("model")["MAE"].mean()
    rows = []
    for m, g in d[d["scenario"] != "real"].groupby("model"):
        if m not in real:
            continue
        r = {"model": m, "name": model_name(m), "real": _clean(real[m])}
        for sc, x in g.groupby("scenario"):
            r[sc] = _clean(x["MAE"].mean())
            r[f"p_{sc}"] = _clean(((x["MAE"] <= real[m]).sum() + 1) / (len(x) + 1))
        rows.append(r)
    rows.sort(key=lambda r: r["real"])
    return {"models": rows, "seeds": int(d.loc[d["scenario"] != "real", "seed"].nunique()),
            "scenarios": NULL_SCENARIOS_RU}


def results_overview() -> dict:
    """Ключевые выводы с цифрами из отчётов + как всё воспроизвести."""
    from src.config import CONFIGS_DIR, RAW_DIR, ROOT
    h = horizons_overview()
    c = cpd_overview(limit=0)
    reports = []
    for rel in ["reports/validation.md", "reports/data_audit.md", "reports/eda.md",
                "reports/backtest/default/metrics.md", "reports/backtest/models/metrics.md",
                "reports/backtest/models/ensembles.csv", "reports/horizons/metrics.md",
                "reports/cpd/summary.md", "reports/shock_candidates.csv"]:
        f = ROOT / rel
        reports.append({"path": rel, "exists": f.exists(),
                        "updated": pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
                        if f.exists() else None})
    manifest = RAW_DIR / "MANIFEST.csv"
    return {"summary": summary(), "horizons": None if h.get("error") else h,
            "cpd": None if c.get("error") else c, "git": _git_rev(),
            "configs": sorted(p.name for p in CONFIGS_DIR.glob("*.yaml")),
            "raw_files": int(len(pd.read_csv(manifest))) if manifest.exists() else None,
            "steps": [{"title": t, "cmd": cmd} for t, cmd in REPRO_STEPS], "reports": reports,
            "importance": _importance(), "importance_model": model_name(importance_model() or ""),
            "null_test": null_test()}


# ------------------------------------------------ данные и пересчёт

_PLAN = {"steps": None, "key": None, "thread": None}
PLAN_TTL_S = 300


def _plan_refresh(key) -> None:
    """План шагов считается секунды (проверка тысяч файлов) – только в фоне,
    иначе окно приложения замирает: слоты моста выполняются в главном потоке."""
    import logging
    import threading
    import time as _t
    from worker import runner

    def work():
        t0 = _t.perf_counter()
        try:
            steps = [{"name": s["name"], "group": s["group"], "action": s["action"], "why": s["why"]}
                     for s in runner.plan(load_config("pipeline"))]
            _PLAN.update(steps=steps, key=key)
            logging.getLogger("ui").debug("plan_refreshed", extra={"elapsed_ms": (_t.perf_counter() - t0) * 1000})
        except Exception:  # noqa: BLE001
            logging.getLogger("ui").exception("plan_refresh_failed")
        finally:
            _PLAN["thread"] = None

    if _PLAN["thread"] is None:
        _PLAN["thread"] = threading.Thread(target=work, name="plan", daemon=True)
        _PLAN["thread"].start()


def data_status() -> dict:
    """Состояние данных и воркера для раздела «Данные». Отвечает сразу: план
    шагов – из кэша, который обновляется в фоне."""
    import time as _t
    from src.config import RAW_DIR
    from worker import runner, snapshot
    cfg = load_config("pipeline")
    try:
        has = not _panel().empty
    except NoData:
        has = False
    busy = runner.is_busy()
    runs = store.last_runs(8)
    # план пересчитываем, когда закончился очередной запуск или кэш устарел;
    # во время пересчёта не трогаем (файлы как раз меняются)
    key = (runs[0]["id"], runs[0]["status"]) if runs else None
    stale = (_PLAN["steps"] is None or _PLAN["key"] is None or _PLAN["key"][0] != key
             or _t.time() - _PLAN["key"][1] > PLAN_TTL_S)
    if stale and not busy:
        _plan_refresh((key, _t.time()))
    steps = _PLAN["steps"] or []
    return {"has_data": has, "published_at": store.published_at(), "busy": busy,
            "current": runs[0] if runs and runs[0]["status"] == "running" and busy else None,
            "runs": runs, "steps": steps, "steps_pending": _PLAN["steps"] is None,
            "snapshot_url": bool(snapshot.resolve_url()),
            "raw_available": any(RAW_DIR.glob("potrebitelskie-*.csv")),
            "schedule": cfg["schedule"]}


RECALC_ARGS = {
    "auto": ["run"], "force": ["run", "--force"], "fetch": ["run", "--fetch"],
    "heavy": ["run", "--heavy"], "snapshot": ["snapshot", "fetch"],
}


def recalc_start(mode: str, trigger: str = "manual") -> dict:
    """Запустить воркер отдельным процессом (приложение не блокируется)."""
    import subprocess
    import sys
    from src.config import ROOT, console_python
    from worker import runner
    if mode not in RECALC_ARGS:
        return {"error": f"неизвестный режим {mode}"}
    import logging
    ui = logging.getLogger("ui")
    if runner.is_busy():
        ui.warning("recalc_rejected  mode=%s: пересчёт уже идёт", mode)
        return {"error": "пересчёт уже идёт"}
    env = runner.child_env()
    # скрытая консоль (не DETACHED_PROCESS): её наследуют git, gh и шаги воркера
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
    args = RECALC_ARGS[mode] + (["--trigger", trigger] if RECALC_ARGS[mode][0] == "run" else [])
    proc = subprocess.Popen([console_python(), "-m", "worker", *args], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
                            start_new_session=sys.platform != "win32")
    ui.info("recalc_started  mode=%s pid=%d cmd=worker %s", mode, proc.pid, " ".join(RECALC_ARGS[mode]))
    return {"started": mode, "pid": proc.pid}


def auto_update_tick() -> dict:
    """Вызывается приложением при старте и раз в минуту.

    Данные в источниках могли выйти, пока приложение было закрыто, или пока
    оно открыто, а воркер по расписанию не установлен. Здесь решается, нужно
    ли запустить воркер: скачать (если подошёл срок) и/или пересчитать (если
    изменились исходники или код). После пересчёта приложение само подтянет
    новые данные (watchData сравнивает дату публикации).
    """
    import logging
    from worker import runner
    sched = load_config("pipeline")["schedule"]
    ui = logging.getLogger("ui")
    if not sched.get("app_auto_update", True):
        return {"action": "off"}
    if runner.is_busy():
        return {"action": "busy"}
    if runner.last_run_failed_recently(sched):
        return {"action": "wait_after_failure"}
    if runner.fetch_due(sched):
        ui.info("auto_update  пора скачать свежие данные – запускаю воркер")
        return {"action": "fetch", **recalc_start("fetch", trigger="app")}
    stale = [s for s in (_PLAN["steps"] or []) if s["action"] == "run" and s["group"] != "heavy"]
    if sched.get("app_auto_recalc", True) and stale:
        ui.info("auto_update  устарели шаги %s – запускаю пересчёт", [s["name"] for s in stale])
        return {"action": "recalc", "steps": [s["name"] for s in stale], **recalc_start("auto", trigger="app")}
    return {"action": "none"}


# ------------------------------------------------ новости

NEWS_TYPES_RU = {"emergency": "ЧС, авария", "attack": "удар с ущербом", "production_neg": "закрытие производства",
                 "production_pos": "новое производство", "fiscal": "выплаты и бюджет", "prices": "цены и тарифы",
                 "demand": "спрос, транспорт, ограничения",
                 # словари (если модели нет)
                 "production": "производственный", "social": "социально-демографический"}


# Проверка данных о событиях как признаков прогноза (scripts/run_event_eval.py:
# CatBoost, seed 1–4, 7 окон) – те же наборы и срезы, что в отчёте (раздел 7.5)
EVENT_SETS_RU = {"news": "Новости: число", "news_flag": "Новости: флаг", "gdelt": "GDELT",
                 "news_gdelt": "Новости + GDELT", "acts": "Акты о ЧС", "emiss": "ЕМИСС: рост г/г",
                 "emiss_level": "ЕМИСС: уровень цен"}
EVENT_SLICES_RU = {"all": "все МО", "chs_active": "регионы с режимом ЧС",
                   "floods_2024": "регионы паводков, 04–05.2024", "news_emergency": "МО с новостями о ЧС"}
# Счётчики новостей – за период отчёта (новости докачиваются и дальше)
NEWS_REPORT_UNTIL = "2025-04-01"


def _event_eval() -> list[dict]:
    e = store.load("news_event_eval")
    if e.empty:
        return []
    e = e[e["set"].isin(EVENT_SETS_RU) & e["slice"].isin(EVENT_SLICES_RU)]
    return [{"set": r.set, "set_ru": EVENT_SETS_RU[r.set], "slice": r.slice, "slice_ru": EVENT_SLICES_RU[r.slice],
             "points": int(r.points), "delta_pct": float(r.delta_pct), "seeds_better": int(r.seeds_better)}
            for r in e.itertuples()]


def _event_news(m: pd.DataFrame) -> dict:
    """Новости о самом МО в месяц реального события детектора (паводки 2024 г.)."""
    ev = store.load("cpd_real_events")
    if ev.empty:
        return {}
    ev = ev.drop_duplicates("oktmo")
    mo = m[m["level"] == "mo"].assign(month=lambda d: pd.to_datetime(d["month"]).dt.strftime("%Y-%m"))
    return {r.oktmo: int(mo[(mo["oktmo"] == r.oktmo) & (mo["month"] == str(r.event)[:7])]["url"].nunique())
            for r in ev.itertuples()}


def news_overview(oktmo: str = "", limit: int = 150) -> dict:
    """Итоги проверки пользы новостей и лента привязанных новостей (для МО – его и его региона)."""
    m = store.load("news_matches")
    if m.empty:
        return {"error": "новостей нет (python -m scripts.download_news, затем scripts.build_news)"}
    mo = _mo_table().set_index("oktmo")
    feed = m
    if oktmo:
        from src.data.oktmo import cov_region_of_oktmo
        feed = m[(m["oktmo"] == oktmo) | ((m["level"] == "region") & (m["cov_region"] == cov_region_of_oktmo(oktmo)))]
    feed = feed.sort_values("published", ascending=False).head(limit)
    items = [{"published": pd.Timestamp(r.published).strftime("%Y-%m-%d %H:%M"), "level": r.level,
              "oktmo": _clean(r.oktmo), "place": _clean(mo.at[r.oktmo, "mo_name"]) if isinstance(r.oktmo, str)
              and r.oktmo in mo.index else f"регион {r.cov_region}",
              "types": [NEWS_TYPES_RU.get(t, t) for t in str(r.types).split(",")], "title": r.title, "url": r.url}
             for r in feed.itertuples()]
    top = (m[m["level"] == "mo"].groupby("oktmo").size().sort_values(ascending=False).head(15))
    rep = m[pd.to_datetime(m["published"]) < NEWS_REPORT_UNTIL]
    return {"total_matches": int(rep["url"].nunique()), "mo_matches": int(rep.loc[rep["level"] == "mo", "url"].nunique()),
            "region_matches": int(rep.loc[rep["level"] == "region", "url"].nunique()),
            "mo_mentioned": int(rep["oktmo"].nunique()),
            "period": [pd.Timestamp(rep["published"].min()).strftime("%m.%Y") if len(rep) else "",
                       (pd.Timestamp(NEWS_REPORT_UNTIL) - pd.DateOffset(months=1)).strftime("%m.%Y")],
            "event_eval": _event_eval(),
            "event_news": _event_news(m),
            "types": {NEWS_TYPES_RU.get(k, k): int(v) for k, v in
                      m["types"].str.split(",").explode().value_counts().items()},
            "top_mo": [{"oktmo": k, "mo_name": _clean(mo.at[k, "mo_name"]) if k in mo.index else k, "n": int(v)}
                       for k, v in top.items()],
            "events": _records(store.load("news_events")), "cpd": _records(store.load("news_cpd")),
            "cpd_stats": _records(store.load("news_cpd_stats")), "forecast": _records(store.load("news_forecast")),
            "feed": items, "oktmo": oktmo}
