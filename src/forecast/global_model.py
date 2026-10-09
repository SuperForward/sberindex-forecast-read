"""Глобальная модель: одна на все МО, прямой прогноз на горизонты 1–3.

Цель – рост г/г целевого месяца (см. src/features/build.py). Прогноз
уровня: y_{T−12} · exp(ĝ). Вес наблюдения – y_{T−12}: так оптимизируется
MAE в рублях, а не в логарифмах.

Бэкенды (ключ `backend` в конфиге):
  lightgbm, xgboost, catboost – градиентный бустинг с MAE как функцией
    потерь; пропуски и категории обрабатывают сами;
  hist_gbm      – sklearn HistGradientBoosting: тот же метод, что LightGBM.
    Классический GradientBoosting не берём: на 50 тыс. строк и 39 признаках
    он считается десятки минут при том же результате;
  random_forest – случайный лес. Критерий MAE в sklearn перебирает все
    пороги и на наших данных идёт часами, поэтому обучается на квадратичной
    ошибке – в отличие от остальных, не на целевой метрике;
  ridge         – линейная модель (one-hot для категорий, медианное
    заполнение пропусков, стандартизация, веса нормированы к среднему 1).
    Единственная, кто экстраполирует за пределы значений, виденных в
    обучении, поэтому признаки уровня месяца (общий рост, курс, календарь)
    ей не даём – см. configs/models.yaml.

Backtest: для каждого origin модель обучается только на строках, чей
целевой месяц <= origin, и прогнозирует origin+1..origin+h.
"""

import logging
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import n_jobs
from src.features.build import add_covariates, build_rows, feature_columns

CAT_COLS = ["mo_type", "cov_region"]
log = logging.getLogger("forecast")


def _fit_predict(backend: str, params: dict, x_tr: pd.DataFrame, y_tr: pd.Series,
                 w: pd.Series, x_te: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Обучает бэкенд и возвращает (прогноз, важность признаков).

    Важность приведена к одной шкале – доле от суммы; там, где библиотека
    её не даёт (hist_gbm), словарь пустой.
    """
    cats = [c for c in CAT_COLS if c in x_tr.columns]
    # без явного числа потоков библиотеки занимают все ядра
    if backend == "catboost":
        params = {**params, "thread_count": n_jobs(params.get("thread_count", -1))}
    elif backend in ("lightgbm", "xgboost", "random_forest"):
        params = {**params, "n_jobs": n_jobs(params.get("n_jobs", -1))}

    if backend == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="l1", verbose=-1, **params)
        m.fit(x_tr, y_tr, sample_weight=w)
        imp = dict(zip(x_tr.columns, m.booster_.feature_importance("gain")))
        return m.predict(x_te), imp

    if backend == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="reg:absoluteerror", enable_categorical=True,
                             tree_method="hist", **params)
        m.fit(x_tr, y_tr, sample_weight=w)
        return m.predict(x_te), dict(zip(x_tr.columns, m.feature_importances_))

    if backend == "catboost":
        from catboost import CatBoostRegressor
        # CatBoost не принимает NaN в категориях – переводим в строки
        tr = x_tr.copy(); te = x_te.copy()
        for c in cats:
            tr[c] = tr[c].astype(str)
            te[c] = te[c].astype(str)
        m = CatBoostRegressor(loss_function="MAE", verbose=0, allow_writing_files=False, **params)
        m.fit(tr, y_tr, sample_weight=w, cat_features=cats)
        return m.predict(te), dict(zip(x_tr.columns, m.get_feature_importance()))

    # --- sklearn: категории -> числовые коды, единые для train и test
    tr = x_tr.copy(); te = x_te.copy()
    for c in cats:
        levels = pd.Index(sorted(set(x_tr[c].dropna().astype(str)) | set(x_te[c].dropna().astype(str))))
        tr[c] = pd.Categorical(tr[c].astype(str), categories=levels).codes
        te[c] = pd.Categorical(te[c].astype(str), categories=levels).codes

    if backend == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        mask = [c in cats for c in tr.columns]
        m = HistGradientBoostingRegressor(loss="absolute_error", categorical_features=mask, **params)
        m.fit(tr, y_tr, sample_weight=w)
        return m.predict(te), {}

    if backend == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        # sklearn RF не умеет пропуски – заполняем медианой обучающей выборки
        imp = SimpleImputer(strategy="median").fit(tr)
        m = RandomForestRegressor(**params)
        m.fit(imp.transform(tr), y_tr, sample_weight=w)
        return m.predict(imp.transform(te)), dict(zip(tr.columns, m.feature_importances_))

    if backend == "ridge":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        num = [c for c in tr.columns if c not in cats]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)])
        pipe = Pipeline([("pre", pre), ("model", Ridge(**params))])
        # Веса – уровень расходов год назад (~25 тыс. ₽): без нормировки штраф
        # alpha тонет в сумме весов и регуляризация фактически выключена.
        pipe.fit(tr, y_tr, model__sample_weight=np.asarray(w) / np.mean(w))
        coef = np.abs(pipe.named_steps["model"].coef_)
        names = pipe.named_steps["pre"].get_feature_names_out()
        return pipe.predict(te), dict(zip(names, coef))

    raise ValueError(f"неизвестный backend: {backend}")


# Формы подачи счётчиков событий (новости, GDELT, акты ЧС) – ключ
# news_features / gdelt_features / acts_features в конфиге:
#   count – число за месяц t и сумма за t−1…t−2 (true – то же);
#   flag  – то же как 0/1: было ли событие;
#   dev   – отклонение от своей нормы: число за t минус среднее за t−12…t−1
#           (норма – по прошлому, без заглядывания вперёд), сумма t−1…t−2 –
#           минус две нормы;
#   lag6  – count и ещё сумма за t−3…t−5: эффект может прийти позже.
EVENT_FORMS = ("count", "flag", "dev", "lag6")


def event_forms(n: pd.DataFrame, key: str, cols: list[str], form: str) -> pd.DataFrame:
    """Помесячные счётчики (key, t, cols) -> признаки в форме form. Пропущенные
    месяцы внутри ряда – нули: событий не было."""
    if form not in EVENT_FORMS:
        raise ValueError(f"неизвестная форма признаков событий: {form}")
    full = []
    for k, g in n.sort_values([key, "t"]).groupby(key):
        g = g.set_index("t")[cols].asfreq("MS", fill_value=0)
        prev = g.shift(1, fill_value=0) + g.shift(2, fill_value=0)
        if form == "flag":
            x = pd.concat([(g > 0).astype(float), (prev > 0).add_suffix("_prev").astype(float)], axis=1)
        elif form == "dev":
            norm = g.shift(1).rolling(12, min_periods=3).mean()
            x = pd.concat([(g - norm).add_suffix("_dev"), (prev - 2 * norm).add_suffix("_prev_dev")], axis=1)
        else:
            x = g.join(prev.add_suffix("_prev"))
            if form == "lag6":
                x = x.join(sum(g.shift(i, fill_value=0) for i in (3, 4, 5)).add_suffix("_prev3_5"))
        full.append(x.assign(**{key: k}).reset_index())
    return pd.concat(full, ignore_index=True)


SEED_KEY = {"catboost": "random_seed"}     # у остальных бэкендов – random_state


def _fit_seeds(backend: str, params: dict, seeds: list[int] | None, x_tr, y_tr, w, x_te):
    """Обучение с несколькими seed и среднее прогнозов (ключ seeds в конфиге).
    У CatBoost MAE при seed 1–4 – 1003–1021 ₽: одна модель зависит от удачи
    seed, среднее четырёх – 1004 ₽ без этого разброса."""
    if not seeds:
        return _fit_predict(backend, params, x_tr, y_tr, w, x_te)
    key = SEED_KEY.get(backend, "random_state")
    preds, imps = [], []
    for s in seeds:
        g, imp = _fit_predict(backend, {**params, key: s}, x_tr, y_tr, w, x_te)
        preds.append(np.asarray(g))
        imps.append(imp)
    imp = {k: float(np.mean([i.get(k, 0.0) for i in imps])) for k in imps[0]} if imps[0] else {}
    return np.mean(preds, axis=0), imp


def _form(value) -> str | None:
    return "count" if value is True else (value or None)


def add_news_features(data: pd.DataFrame, form: str = "count") -> pd.DataFrame:
    """Новости как признаки (scripts/build_news.py): число новостей каждого типа
    о МО (и с весом – о его регионе) в месяце t и за прошлые месяцы (form).
    Месяц t – месяц публикации: к концу месяца t эти новости уже вышли, так
    что прогноз из t их видит без заглядывания в будущее."""
    from src.config import PROCESSED_DIR
    f = PROCESSED_DIR / "news_mo_monthly.parquet"
    if not f.exists():
        return data
    n = pd.read_parquet(f).rename(columns={"month": "t"})
    n = event_forms(n, "oktmo", [c for c in n.columns if c.startswith("news_")], form)
    out = data.merge(n, on=["oktmo", "t"], how="left")
    new = [c for c in n.columns if c.startswith("news_")]
    # МО без новостей за весь период: счётчики – нули; у dev нормы нет – пропуск
    if form != "dev":
        out[new] = out[new].fillna(0)
    return out


def add_gdelt_features(data: pd.DataFrame, form: str = "count") -> pd.DataFrame:
    """События GDELT (scripts/build_gdelt.py) за месяц t: по самому МО и по
    его региону. Месяц – месяц добавления события в GDELT, к концу t известен.
    count – как в таблице (только месяц t); другие формы – для числа событий и
    конфликтов, тон и всплеск остаются как есть (всплеск – уже отклонение от нормы)."""
    from src.config import PROCESSED_DIR
    f = PROCESSED_DIR / "gdelt_mo_monthly.parquet"
    if not f.exists():
        return data
    g = pd.read_parquet(f).rename(columns={"month": "t"})
    if form != "count":
        counts = [c for c in g.columns if c.endswith(("_events", "_conflict"))]
        g = g.drop(columns=counts).merge(event_forms(g, "oktmo", counts, form), on=["oktmo", "t"], how="left")
    return data.merge(g, on=["oktmo", "t"], how="left")


def acts_region_known(acts: pd.DataFrame, months: pd.DatetimeIndex, default_len: int = 3) -> pd.DataFrame:
    """Акты о ЧС (scripts/build_acts.py) по региону и месяцу t – только то, что
    опубликовано к концу t: введено ЧС, число актов о поддержке, действует ли
    режим. Причину по паводку не выделяем: в названиях актов 2024 г. её обычно
    нет («О введении режима ЧС» без слова «паводок»). Режим действует, если введён за последние
    default_len мес. и отмена к t ещё не опубликована – в отличие от
    acts_region_monthly, где конец режима берётся из будущего акта об отмене."""
    a = acts.dropna(subset=["cov_region"]).copy()
    a["t"] = a["published"].fillna(a["date"]).dt.to_period("M").dt.to_timestamp()
    rows = []
    for reg, g in a.groupby("cov_region"):
        intro, cancel = g[g["kind"] == "intro"], g[g["kind"] == "cancel"]
        n = lambda d: d.groupby("t").size().reindex(months, fill_value=0).to_numpy()  # noqa: E731
        active = np.zeros(len(months))
        for m in intro["t"]:
            ended = cancel[cancel["t"] > m]["t"]
            stop = min(m + pd.DateOffset(months=default_len), ended.min() if len(ended) else pd.Timestamp.max)
            active[(months >= m) & (months < stop)] = 1
        rows.append(pd.DataFrame({"cov_region": reg, "t": months, "chs_intro": n(intro),
                                  "chs_support": n(g[g["kind"] == "support"]), "chs_active": active}))
    return pd.concat(rows, ignore_index=True)


def add_acts_features(data: pd.DataFrame, form: str = "count") -> pd.DataFrame:
    """Официальные акты о ЧС по региону МО (acts_region_known) в форме form."""
    from src.config import PROCESSED_DIR
    f = PROCESSED_DIR / "acts.parquet"
    if not f.exists():
        return data
    months = pd.date_range("2022-01-01", data["t"].max(), freq="MS")
    a = acts_region_known(pd.read_parquet(f), months)
    cols = ["chs_intro", "chs_support", "chs_active"]
    a = event_forms(a, "cov_region", cols, form)
    out = data.assign(_reg=data["cov_region"].astype(str)).merge(
        a.rename(columns={"cov_region": "_reg"}), on=["_reg", "t"], how="left").drop(columns="_reg")
    new = [c for c in a.columns if c.startswith("chs_")]
    if form != "dev":
        out[new] = out[new].fillna(0)
    return out


@lru_cache(maxsize=4)
def feature_rows(category: str, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Строки признаков – одни на все глобальные модели прогона: сборка
    (слияния с ковариатами, ПМО, ЕМИСС) дольше обучения иной модели.
    Не изменять – вызывающие работают с копией."""
    return add_covariates(build_rows(category, horizons=horizons))


def run_global_backtest(origins: list[pd.Timestamp], horizon: int, params: dict,
                        category: str = "all", name: str = "global",
                        series_filter: set | None = None,
                        data: pd.DataFrame | None = None, future: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """data – готовые строки признаков (build_rows + add_covariates); без него
    берутся из кэша feature_rows. Нужен null-тестам, которые подменяют признаки или цель.
    future – прогноз вперёд: целевые месяцы за краем данных, факта нет (y = NaN)."""
    params = dict(params)
    backend = params.pop("backend", "lightgbm")
    offset_col = params.pop("target_offset", None)
    seeds = params.pop("seeds", None)
    drop = set(params.pop("exclude_features", []))
    news = _form(params.pop("news_features", False))
    gdelt = _form(params.pop("gdelt_features", False))
    acts = _form(params.pop("acts_features", False))
    extra_form = params.pop("extra_form", None)

    data = (feature_rows(category, tuple(range(1, horizon + 1))) if data is None else data).copy()
    if news:
        data = add_news_features(data, news)
    if gdelt:
        data = add_gdelt_features(data, gdelt)
    if acts:
        data = add_acts_features(data, acts)
    if extra_form:
        from src.features.build import region_extra_forms
        x = region_extra_forms(extra_form).rename(columns={"date": "t", "cov_region": "_reg"})
        data = data.assign(_reg=data["cov_region"].astype(str)).merge(x, on=["_reg", "t"], how="left") \
            .drop(columns="_reg")
    feats = [f for f in feature_columns(data) if f not in drop]
    # смещение: модель учит поправку к baseline, а не рост целиком
    data["_offset"] = (data[offset_col].fillna(data["g1"]).fillna(0.0) if offset_col
                       else pd.Series(0.0, index=data.index))

    preds, importances = [], []
    fit_total = 0.0
    for origin in origins:
        train = data[(data["target_date"] <= origin) & data["target"].notna()]
        test = data[(data["t"] == origin) & (future | data["y_log"].notna())]
        if series_filter is not None:
            test = test[test["oktmo"].isin(series_filter)]
        if train.empty or test.empty:
            continue
        # В раннем окне часть признаков ещё пуста (g3 и drift3 требуют 14 мес.
        # истории). Бустинги это переживают, hist_gbm на полностью пустом
        # столбце падает – отбрасываем такие признаки внутри окна.
        usable = [f for f in feats if train[f].notna().any()]
        if len(usable) < len(feats):
            log.debug("features_dropped  %s: в окне %s пропущены пустые признаки: %s", name,
                      f"{origin:%Y-%m}", sorted(set(feats) - set(usable)))
        t0 = time.perf_counter()
        g, imp = _fit_seeds(backend, params, seeds, train[usable], train["target"] - train["_offset"],
                            np.exp(train["base_log"]), test[usable])
        fit_sec = time.perf_counter() - t0
        g = np.asarray(g) + test["_offset"].to_numpy()
        preds.append(pd.DataFrame({
            "series_id": test["oktmo"].astype(str) + "|" + category,
            "model": name, "origin": origin, "date": test["target_date"], "h": test["h"],
            "y": test["y_level"], "yhat": np.exp(test["base_log"] + g)}))
        if imp:
            importances.append(pd.DataFrame({"origin": origin, "feature": list(imp),
                                             "gain": list(imp.values())}))
        log.debug("model_window  %s (%s) окно %s: обучение %d строк за %.1f с, прогноз %d", name, backend,
                  f"{origin:%Y-%m}", len(train), fit_sec, len(test))
        fit_total += fit_sec
    if preds:
        log.info("model_done  %s (%s): окон %d, прогнозов %d, обучение %.1f с", name, backend, len(preds),
                 sum(len(x) for x in preds), fit_total, extra={"elapsed_ms": fit_total * 1000})
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(
        columns=["origin", "feature", "gain"])
    if not preds:
        # ни в одном окне не нашлось обучающих строк (например, горизонт 12 при
        # годе истории: роста г/г ещё нет) – честно возвращаем пустой прогноз
        log.warning("model_empty  %s: нет окон с обучающими данными – прогноза нет", name)
        return pd.DataFrame(columns=["series_id", "model", "origin", "date", "h", "y", "yhat"]), imp_df
    return pd.concat(preds, ignore_index=True), imp_df
