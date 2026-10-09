"""Признаки для глобальной модели прогноза расходов по МО.

Строка = (МО, дата прогноза t, горизонт h). Целевой месяц T = t + h.
Цель: g_T = log y_T − log y_{T−12} (рост г/г целевого месяца). Прогноз
уровня: y_{T−12} · exp(ĝ). Сезонность берётся из прошлогоднего значения,
модель учит поправку к темпу роста – по всем МО сразу.

Только данные, известные в момент t:
  - ряды расходов – по месяц t включительно;
  - региональная статистика Росстата – за t−1 (публикуется с лагом ~месяц),
    безработица МОТ – последний квартал, закончившийся не позже t−3;
  - годовые показатели МО (БД ПМО) – за год year(t)−2 (годовые итоги
    выходят во второй половине следующего года);
  - региональные ряды ЕМИСС, ЦБ и ФНС – с лагом публикации каждого
    (REGION_EXTRA): вклады, кредиты и стоимость набора – за t−2, НДФЛ и
    налоги в местные бюджеты (1-НМ) – за t−1, доходы и цена жилья –
    последний квартал, закончившийся не позже t−3, бедность – год, итоги
    которого вышли (в апреле следующего);
  - календарь целевого месяца известен заранее.
"""

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR

PMO_FEATURES = {
    "pmo_8423007": "wage",            # среднемесячная зарплата (без МСП)
    "pmo_8423005": "employees",       # численность работников
    "pmo_8401011": "shipments",       # отгрузка собственной продукции
    "pmo_8313015": "own_revenue_share",
    "pmo_8401003": "retail_turnover",
    "pmo_8401006": "catering_turnover",
    "pmo_8109003": "investment_pc",   # инвестиции (без бюджетных) на 1 жителя
    "pmo_8215001": "housing_pc",      # ввод жилья на 1 жителя
    "pmo_8313004": "admin_spend_pc",  # расходы на содержание органов МСУ на 1 жителя
    "pmo_8213002": "wage_large",      # зарплата крупных и средних организаций
}
# Показатели ПМО, которые есть только за один год (2020): берутся как
# постоянные свойства МО – структура экономики меняется медленно.
PMO_STATIC = {
    "pmo_8215003": "sme_per_10k",     # субъектов МСП на 10 тыс. населения
    "pmo_8155040": "sme_emp_share",   # доля работников МСП
}
# В модель не идут: заполнены меньше чем у четверти МО (банкротства
# 8155023, незавершёнка 8155033, кредиторка МСУ 8313019, долги по зарплате
# 8423001) или есть только за 2024–2025 (организации 9010002 – в окнах
# бэктеста это данные из будущего). Покрытие печатает build_mo_features.
PMO_USED = set(PMO_FEATURES) | set(PMO_STATIC) | {"pmo_8112001", "pmo_8112003", "pmo_8112013", "pmo_8112023"}


def _panel_matrix(category: str) -> pd.DataFrame:
    """МО × месяц. other – расходы вне пяти категорий: все минус их сумма
    (категории покрывают ~72% всех расходов); нужна для согласования
    прогнозов категорий с общим (scripts/run_reconcile.py)."""
    p = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet")
    if category == "other":
        w = p.pivot_table(index=["oktmo", "date"], columns="category", values="value")
        rest = w["all"] - w.drop(columns="all").sum(axis=1, min_count=w.shape[1] - 1)
        return rest.unstack("date")
    p = p[p["category"] == category]
    return p.pivot_table(index="oktmo", columns="date", values="value")


def _static() -> pd.DataFrame:
    pop = pd.read_parquet(PROCESSED_DIR / "mo_population.parquet").set_index("oktmo")
    mp = pd.read_csv(INTERIM_DIR / "mo_oktmo_map.csv", dtype=str).drop_duplicates("oktmo").set_index("oktmo")
    cov = pd.read_parquet(PROCESSED_DIR / "mo_cov_region.parquet").set_index("oktmo")
    s = pd.DataFrame(index=cov.index)
    s["log_pop"] = np.log(pop["pop"].astype(float))
    s["urban_share"] = pop["pop_urban"].astype(float) / pop["pop"].astype(float)
    s["mo_type"] = mp["mo_type"]
    s["cov_region"] = cov["cov_region"]
    return s


def _pmo_by_year() -> pd.DataFrame:
    a = pd.read_parquet(PROCESSED_DIR / "mo_annual.parquet").sort_values(["oktmo", "year"])
    out = a[["oktmo", "year"]].copy()
    for col, name in PMO_FEATURES.items():
        if col not in a:
            continue
        v = a[col].where(a[col] > 0)
        out[f"{name}_log"] = np.log(v)
        out[f"{name}_growth"] = out[f"{name}_log"] - out.groupby("oktmo")[f"{name}_log"].shift(1)
    if {"pmo_8112023", "pmo_8112013"} <= set(a.columns):
        out["migration_per_1000"] = a["pmo_8112023"] / a["pmo_8112013"] * 1000
    if {"pmo_8112003", "pmo_8112001", "pmo_8112013"} <= set(a.columns):
        out["natural_growth_per_1000"] = (a["pmo_8112003"] - a["pmo_8112001"]) / a["pmo_8112013"] * 1000
    return out


def _pmo_static() -> pd.DataFrame:
    a = pd.read_parquet(PROCESSED_DIR / "mo_annual.parquet").sort_values(["oktmo", "year"])
    cols = [c for c in PMO_STATIC if c in a]
    last = a.groupby("oktmo")[cols].last()          # последнее непустое значение
    return last.rename(columns=PMO_STATIC)


def _yoy(x: pd.DataFrame, col: str) -> pd.Series:
    """log(x_m / x_{m−12}) по региону; x – строки (cov_region, date, col) без пропусков."""
    prev = x.assign(date=x["date"] + pd.DateOffset(months=12))[["cov_region", "date", col]]
    m = x.merge(prev, on=["cov_region", "date"], how="left", suffixes=("", "_prev"))
    return pd.Series(np.log(m[col] / m[f"{col}_prev"]).to_numpy(), index=x.index)


# Региональные ряды scripts/build_extra.py: столбец -> (вид, лаг публикации, мес.).
#   monthly – месячный уровень, признак – рост г/г;
#   quarterly – квартальное значение на каждом месяце квартала (лаг – от
#     первого месяца квартала: 5 = квартал известен через ~2 мес. после конца);
#   ytd – нарастающий итог с начала года (ФНС 1-НМ): рост г/г итога и месяца;
#   annual – годовое значение, выходит в апреле следующего года.
REGION_EXTRA = {
    "deposits_households": ("monthly", 2),   # ЦБ: остаток на конец месяца
    "loans_households": ("monthly", 2),      # ЕМИСС (ЦБ): долг по кредитам физлицам
    "income_per_capita": ("quarterly", 5),   # ЕМИСС: среднедушевые доходы
    "ndfl_agents": ("ytd", 1),               # ФНС 1-НМ: НДФЛ, удержанный работодателями
    "tax_local": ("ytd", 1),                 # ФНС 1-НМ: налоги в местные бюджеты
    "poverty_rate": ("annual", 4),           # ЕМИСС: уровень бедности, %
    "basket_cost": ("monthly", 2),           # ЕМИСС: стоимость фиксированного набора
    "housing_price_m2_primary": ("quarterly", 5),    # ЕМИСС: цена 1 м², первичный рынок
    "housing_price_m2_secondary": ("quarterly", 5),  # ЕМИСС: цена 1 м², вторичный рынок
    "loans_corporate": ("monthly", 2),       # ЕМИСС (ЦБ): долг юрлиц и ИП по кредитам
    "loans_sme_issued": ("monthly", 2),      # ЕМИСС (ЦБ): выдачи кредитов МСП за месяц
    "loans_sme_debt": ("monthly", 2),        # ЕМИСС (ЦБ): долг МСП по кредитам
    "loans_sme_overdue": ("monthly", 2),     # ЕМИСС (ЦБ): просроченный долг МСП
    # ЕМИСС: средние цены на топливо и ЖКУ (бензин АИ-92, дизель, электроэнергия,
    # холодная вода с водоотведением, горячая вода, отопление, газ)
    **{f"prices_fuel_utilities_{s}": ("monthly", 2)
       for s in ("ai92", "diesel", "power", "water_cold", "water_hot", "heating", "gas")},
}


def _region_extra() -> pd.DataFrame | None:
    """Региональные ряды scripts/build_extra.py -> признаки с лагом публикации
    (дата строки – месяц t, в который значение уже известно)."""
    f = PROCESSED_DIR / "covariates_region_extra.parquet"
    if not f.exists():
        print("  нет covariates_region_extra.parquet – региональные признаки ЕМИСС/ЦБ/ФНС пропущены")
        return None
    r = pd.read_parquet(f).sort_values(["cov_region", "date"])
    parts = []
    for col, (kind, lag) in REGION_EXTRA.items():
        if col not in r:
            print(f"  в covariates_region_extra нет {col} – признак пропущен")
            continue
        x = r[["cov_region", "date", col]].dropna().reset_index(drop=True)
        out = x[["cov_region", "date"]].copy()
        if kind in ("monthly", "quarterly"):
            out[f"{col}_yoy"] = _yoy(x, col)
        elif kind == "ytd":
            out[f"{col}_ytd_yoy"] = _yoy(x, col)
            # месяц = итог минус итог прошлого месяца того же года (январь – сам итог)
            prev = x.assign(date=x["date"] + pd.DateOffset(months=1))
            m = x.merge(prev, on=["cov_region", "date"], how="left", suffixes=("", "_p"))
            month = np.where(m["date"].dt.month == 1, m[col], m[col] - m[f"{col}_p"])
            xm = x.assign(**{col: month}).dropna()
            xm = xm[xm[col] > 0]
            out[f"{col}_m_yoy"] = _yoy(xm, col).reindex(x.index)
        elif kind == "annual":
            y = x.assign(year=x["date"].dt.year).groupby(["cov_region", "year"])[col].mean()
            months = pd.date_range(x["date"].min(), x["date"].max() + pd.DateOffset(years=2), freq="MS")
            grid = pd.MultiIndex.from_product([x["cov_region"].unique(), months], names=["cov_region", "date"])
            out = grid.to_frame(index=False)
            # год Y известен с апреля Y+1: до апреля – позапрошлый год
            src = out["date"].dt.year - np.where(out["date"].dt.month >= lag, 1, 2)
            out[col] = [y.get((c, yr), np.nan) for c, yr in zip(out["cov_region"], src)]
            prev = [y.get((c, yr - 1), np.nan) for c, yr in zip(out["cov_region"], src)]
            out[f"{col}_diff"] = out[col] - np.asarray(prev)
            parts.append(out.dropna(subset=[col]).set_index(["cov_region", "date"]))
            continue
        out = out.assign(date=out["date"] + pd.DateOffset(months=lag))
        parts.append(out.set_index(["cov_region", "date"]))
    if not parts:
        return None
    out = pd.concat(parts, axis=1).reset_index()
    # темпы за пределами ±150% – сбои данных (смена методики, разовые платежи)
    growth = [c for c in out if c.endswith(("_yoy", "_m_yoy"))]
    out[growth] = out[growth].clip(-1.5, 1.5)
    return out


# Ряды ЕМИСС, которые в форме роста г/г не помогли (configs/models.yaml,
# exclude_features), и другие формы для них (region_extra_forms):
#   level – log уровня минус медиана по регионам в тот же месяц: дорогой или
#           дешёвый регион; только для цен (у долга и выдач уровень – размер региона);
#   rel   – рост г/г минус медиана по регионам: регион растёт быстрее страны;
#   mom3  – log изменения за 3 месяца: свежее, чем рост г/г.
EXTRA_TRIAL = ["basket_cost", "housing_price_m2_primary", "housing_price_m2_secondary", "loans_corporate",
               "loans_sme_issued", "loans_sme_debt", "loans_sme_overdue",
               *(f"prices_fuel_utilities_{s}" for s in
                 ("ai92", "diesel", "power", "water_cold", "water_hot", "heating", "gas"))]
EXTRA_PRICES = [c for c in EXTRA_TRIAL if not c.startswith("loans_")]
EXTRA_FORMS = ("level", "rel", "mom3")


def region_extra_forms(form: str) -> pd.DataFrame:
    """Ряды EXTRA_TRIAL в форме form, с тем же лагом публикации, что и рост г/г
    в _region_extra. Столбцы: cov_region, date (месяц t, когда известно), <ряд>_<form>."""
    if form not in EXTRA_FORMS:
        raise ValueError(f"неизвестная форма рядов ЕМИСС: {form}")
    r = pd.read_parquet(PROCESSED_DIR / "covariates_region_extra.parquet").sort_values(["cov_region", "date"])
    parts = []
    for col in (EXTRA_PRICES if form == "level" else EXTRA_TRIAL):
        if col not in r:
            continue
        x = r[["cov_region", "date", col]].dropna()
        x = x[x[col] > 0].reset_index(drop=True)
        lx = np.log(x[col])
        if form == "level":
            v = lx - lx.groupby(x["date"]).transform("median")
        elif form == "rel":
            g = _yoy(x, col)
            v = g - g.groupby(x["date"]).transform("median")
        else:
            prev = x.assign(date=x["date"] + pd.DateOffset(months=3))
            m = x.merge(prev, on=["cov_region", "date"], how="left", suffixes=("", "_p"))
            v = pd.Series(np.log(m[col] / m[f"{col}_p"]).to_numpy(), index=x.index)
        lag = REGION_EXTRA[col][1]
        out = pd.DataFrame({"cov_region": x["cov_region"], "date": x["date"] + pd.DateOffset(months=lag),
                            f"{col}_{form}": v.clip(-1.5, 1.5)})
        parts.append(out.set_index(["cov_region", "date"]))
    return pd.concat(parts, axis=1).reset_index()


def _region_monthly() -> pd.DataFrame:
    r = pd.read_parquet(PROCESSED_DIR / "covariates_region_monthly.parquet").sort_values(["cov_region", "date"])
    g = r.groupby("cov_region")
    r["cpi_yoy"] = g["cpi"].transform(lambda s: np.log(s / 100).rolling(12).sum())
    r["cpi_food_yoy"] = g["cpi_food"].transform(lambda s: np.log(s / 100).rolling(12).sum())
    r["catering_yoy"] = np.log(r["catering"]) - g["catering"].shift(12).pipe(np.log)
    return r[["cov_region", "date", "cpi_yoy", "cpi_food_yoy", "services_ifo_yoy",
              "catering_yoy", "unemployment_ilo", "t_anom", "precip_anom"]]


def build_rows(category: str = "all", horizons=(1, 2, 3)) -> pd.DataFrame:
    """Все строки (МО, t, h), для которых определены y_{T−12} и y_t."""
    Y = _panel_matrix(category)
    months = list(Y.columns)
    L = np.log(Y.where(Y > 0))
    idx = {d: i for i, d in enumerate(months)}

    def col(i):
        return L.iloc[:, i] if 0 <= i < len(months) else pd.Series(np.nan, index=L.index)

    # общий по стране рост г/г на месяц (медиана по МО) – известен в момент t
    common = pd.Series({d: (col(i) - col(i - 12)).median() for d, i in idx.items()})

    frames = []
    for t_date, t in idx.items():
        for h in horizons:
            T = t + h
            if T >= len(months) + 3:  # целевой месяц дальше 3 мес. за краем данных
                continue
            base = col(T - 12)
            if base.isna().all() or col(t).isna().all():
                continue
            f = pd.DataFrame(index=L.index)
            f["t"] = t_date
            f["h"] = h
            f["target_date"] = t_date + pd.DateOffset(months=h)
            f["base_log"] = base
            f["y_log"] = col(T) if T < len(months) else np.nan
            # точный уровень: exp(log(y)) ≠ y в последних разрядах, а по y
            # прогнозы разных моделей сводятся в ансамбле
            f["y_level"] = Y.iloc[:, T] if T < len(months) else np.nan
            f["g1"] = col(t) - col(t - 12)
            f["g2"] = col(t - 1) - col(t - 13)
            f["g3"] = col(t - 2) - col(t - 14)
            s_now = sum(Y.iloc[:, i] for i in range(t - 2, t + 1) if i >= 0) if t >= 2 else np.nan
            s_prev = sum(Y.iloc[:, i] for i in range(t - 14, t - 11) if i >= 0) if t >= 14 else np.nan
            f["drift3"] = np.log(s_now / s_prev) if t >= 14 else np.nan
            f["mom"] = col(t) - col(t - 1)
            f["seas_step"] = base - col(t - 12)  # прошлогодний сдвиг от месяца t к T
            f["common_g1"] = common.get(t_date, np.nan)
            f["common_g1_prev"] = common.get(months[t - 1], np.nan) if t >= 1 else np.nan
            frames.append(f.dropna(subset=["base_log"]))
    rows = pd.concat(frames).reset_index().rename(columns={"index": "oktmo"})
    rows["target"] = rows["y_log"] - rows["base_log"]
    return rows


def add_covariates(rows: pd.DataFrame) -> pd.DataFrame:
    from src.data import calendar
    r = rows.join(_static(), on="oktmo").join(_pmo_static(), on="oktmo")
    r["target_month"] = r["target_date"].dt.month

    cal = calendar.load_monthly().set_index("date")
    for c in ["workdays", "workdays_yoy", "holidays_on_weekdays", "long_weekend_max"]:
        r[f"cal_{c}"] = r["target_date"].map(cal[c])
    r["cal_workdays_yoy_t"] = r["t"].map(cal["workdays_yoy"])

    reg = _region_monthly()
    lagged = reg.assign(date=reg["date"] + pd.DateOffset(months=1))  # данные за t−1 доступны в t
    r = r.merge(lagged.drop(columns=["unemployment_ilo"]), left_on=["cov_region", "t"],
                right_on=["cov_region", "date"], how="left").drop(columns="date")
    un = reg[["cov_region", "date", "unemployment_ilo"]].dropna()
    un = un.assign(date=un["date"] + pd.DateOffset(months=3))  # квартал известен через ~3 мес.
    r = r.merge(un, left_on=["cov_region", "t"], right_on=["cov_region", "date"], how="left").drop(columns="date")

    from src.data import external
    fx = external.load_fx_monthly().set_index("date")
    r["usd_mom"] = r["t"].map(fx["usd_mom"])
    r["usd_yoy"] = r["t"].map(np.log(fx["usd"] / fx["usd"].shift(12)))

    extra = _region_extra()
    if extra is not None:
        r = r.merge(extra.rename(columns={"date": "t"}), on=["cov_region", "t"], how="left")

    pmo = _pmo_by_year()
    r["pmo_year"] = r["t"].dt.year - 2
    r = r.merge(pmo.rename(columns={"year": "pmo_year"}), on=["oktmo", "pmo_year"], how="left")
    for c in ["mo_type", "cov_region"]:
        r[c] = r[c].astype("category")
    _log_coverage(r)
    return r


def _log_coverage(r: pd.DataFrame) -> None:
    """Доля непустых значений каждого признака: целиком по данным и в
    последнем месяце прогноза. Признак, пустой в последнем месяце, модель
    в прогнозе фактически не использует."""
    import logging
    log = logging.getLogger("forecast")
    last = r[r["t"] == r["t"].max()]
    feats = feature_columns(r)
    empty_last, sparse = [], []
    for f in feats:
        a, b = r[f].notna().mean(), last[f].notna().mean()
        if b == 0:
            empty_last.append(f)
        elif a < 0.25:
            sparse.append(f)
        # по каждому признаку – DEBUG (python -m app.main --profile или logging.DEBUG)
        log.debug("feature_coverage  %s: заполнено %.0f%%, в последнем месяце %.0f%%", f, a * 100, b * 100)
    log.info("features  признаков %d, строк %d, последний месяц прогноза %s", len(feats), len(r),
             f"{r['t'].max():%Y-%m}")
    if empty_last:
        log.warning("features_empty_last  пусты в последнем месяце (в прогнозе не работают): %s", empty_last)
    if sparse:
        log.info("features_sparse  заполнены меньше чем на 25%%: %s", sparse)


FEATURES_EXCLUDE = {"oktmo", "t", "target_date", "y_log", "y_level", "target", "pmo_year", "base_log"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDE]
