"""Дополнительные региональные данные -> таблицы по регионам (cov_region).

Запуск: python -m scripts.build_extra

- data/processed/covariates_region_extra.parquet – по месяцам: показатели ЕМИСС
  (configs/extra_sources.yaml), вклады населения ЦБ, поступления налогов по
  форме ФНС 1-НМ и доходы местных бюджетов по отчёту Казначейства 0503317
  (нарастающим итогом с начала года; дата – последний месяц, который
  покрывает отчёт «на 01.ММ»).
  Квартальные значения ставятся на каждый месяц квартала, годовые – на
  каждый месяц года. Запаздывание публикации здесь не учитывается: его
  добавляет модель при построении признаков.
- data/processed/vacancies_region.parquet – по срезам вакансий «Работы
  России»: число вакансий, рабочих мест, медианная предлагаемая зарплата.
  Срез – состояние на дату выгрузки; история копится с каждым срезом.

Регионы привязываются по названию тем же справочником, что и в
build_covariates (ключ region_key, коды из таблиц ИПЦ).
"""

import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, RAW_DIR, load_config  # noqa: E402

MONTHS = {m: i for i, m in enumerate(["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
                                     "сентябрь", "октябрь", "ноябрь", "декабрь"], start=1)}
QUARTERS = {"I квартал": 1, "II квартал": 2, "III квартал": 3, "IV квартал": 4}
KEEP_SNAPSHOTS = 12          # сырых срезов вакансий храним последние N (сводка копится вся)


def _name(n: str) -> str:
    """Варианты названий из этих источников, которых нет в других таблицах:
    «Город Москва», «Чувашская Республика - Чувашия»."""
    s = re.sub(r"^\s*город\s+", "", str(n), flags=re.I)
    return re.sub(r"\s*-\s*Чувашия\s*$", "", s)


def _months(period: str, year: int) -> list[pd.Timestamp]:
    p = str(period).strip()
    if p in MONTHS:
        return [pd.Timestamp(year, MONTHS[p], 1)]
    if p in QUARTERS:
        q = QUARTERS[p]
        return [pd.Timestamp(year, m, 1) for m in range(3 * q - 2, 3 * q + 1)]
    if "год" in p:
        return [pd.Timestamp(year, m, 1) for m in range(1, 13)]
    return []


def _emiss_files(spec: dict) -> list[Path]:
    """Ручная выгрузка (<id>_<name>.xml – SDMX, .xls – Excel) важнее
    автоматической (.xml.gz): автоматически сайт отдаёт только срез по умолчанию.
    .xls частями по годам (<id>_<name>_<годы>.xls, из-за предела 256
    столбцов) склеиваются."""
    base = RAW_DIR / "emiss" / f"{spec['id']}_{spec['name']}"
    for f in (base.with_suffix(".xml"), base.with_suffix(".xls")):
        if f.exists():
            return [f]
    parts = sorted(base.parent.glob(f"{base.name}_*.xls"))
    if parts:
        return parts
    gz = Path(f"{base}.xml.gz")
    return [gz] if gz.exists() else []


def _read_emiss(f: Path) -> pd.DataFrame:
    from src.data import emiss
    if f.suffix == ".xls":
        return emiss.parse_xls(f)
    return emiss.parse_sdmx(gzip.open(f).read() if f.suffix == ".gz" else f.read_bytes())


def _territory_col(d: pd.DataFrame) -> str | None:
    """Измерение территорий: в разных показателях у него разные коды
    (s_OKATO, 00_003, mРЕГИОН…), узнаём по значению «Российская Федерация»."""
    for c in d.columns:
        if c.endswith("_name") and d[c].astype(str).str.contains("Российская Федерация", regex=False).any():
            return c
    return None


def emiss_series(codes: dict, by_name) -> list[pd.DataFrame]:
    out = []
    for spec in load_config("extra_sources")["emiss"]:
        ind, name = spec["id"], spec["name"]
        files = _emiss_files(spec)
        if not files:
            print(f"ЕМИСС {ind} {name}: файла нет – пропуск (python -m scripts.download_extra --only emiss "
                  f"или вручную https://fedstat.ru/indicator/{ind})")
            continue
        d = pd.concat([_read_emiss(f) for f in files], ignore_index=True)
        fname = files[0].name if len(files) == 1 else f"{files[0].name} … {files[-1].name}"
        terr = _territory_col(d)
        if d.empty or terr is None or "PERIOD" not in d:
            print(f"ЕМИСС {ind} {name} ({fname}): нет территорий или периода – пропуск")
            continue
        # срезы из конфига: keep – по подписям измерений, year_from – по году
        years_all = f"{d['time'].min()}–{d['time'].max()}"
        miss = None
        for dim, pattern in (spec.get("keep") or {}).items():
            hit = {c: d[c].astype(str).map(lambda v: bool(re.search(pattern, v)))
                   for c in d.columns if c.endswith("_name")}
            col = next((c for c, m in hit.items() if m.any()), None)
            if col is None:
                miss = dim
                break
            d = d[hit[col]]
        if miss:
            print(f"ЕМИСС {ind} {name}: фильтр «{miss}» не совпал ни с одним значением – пропуск")
            continue
        if spec.get("year_from"):
            d = d[d["time"].astype(int) >= spec["year_from"]]
        if d.empty:
            print(f"ЕМИСС {ind} {name}: после фильтров пусто (в файле {years_all}, year_from "
                  f"{spec.get('year_from')}) – пропуск")
            continue
        for suffix, pattern in (spec.get("split") or {None: None}).items():
            part = d
            col = name
            if pattern:
                # список выражений – все должны совпасть (каждое в своём измерении)
                hit = pd.Series(True, index=d.index)
                for pat in [pattern] if isinstance(pattern, str) else pattern:
                    hit &= pd.concat([d[c].astype(str).str.contains(pat, regex=True)
                                      for c in d.columns if c.endswith("_name")], axis=1).any(axis=1)
                part, col = d[hit], f"{name}_{suffix}"
            until = (spec.get("ytd_until") or {}).get(suffix)
            if until:
                part = _ytd_to_month(part, terr, until)
            df = _emiss_one(part, terr, f"{ind} {col}", col, fname, codes, by_name)
            if df is not None:
                out.append(df)
    return out


def _ytd_to_month(d: pd.DataFrame, terr: str, until: int) -> pd.DataFrame:
    """Годы до until включительно – нарастающим итогом с начала года: месяц =
    итог минус итог прошлого месяца (январь – сам итог; без прошлого месяца – NaN)."""
    d = d.copy()
    old = (d["time"].astype(int) <= until) & d["PERIOD"].isin(MONTHS)
    x = d[old].assign(m=d.loc[old, "PERIOD"].map(MONTHS))
    prev = x.assign(m=x["m"] + 1)[[terr, "time", "m", "value"]]
    x = x.reset_index().merge(prev, on=[terr, "time", "m"], how="left", suffixes=("", "_prev")).set_index("index")
    d.loc[x.index, "value"] = x["value"].where(x["m"] == 1, x["value"] - x["value_prev"])
    return d


def _emiss_one(d: pd.DataFrame, terr: str, label: str, name: str, fname: str,
               codes: dict, by_name) -> pd.DataFrame | None:
    """Один ряд показателя (после keep/split) -> (date, cov_region, name)."""
    if d.empty:
        print(f"ЕМИСС {label}: split не совпал ни с одним значением – пропуск")
        return None
    # оставшиеся измерения кроме территории и периода (например, вид
    # товара) – несколько рядов на регион; их не усредняем вслепую
    extra = [c for c in d.columns if c.endswith("_name") and c != terr and d[c].nunique() > 1]
    if extra:
        print(f"ЕМИСС {label}: несколько рядов на регион по {extra} – пропуск, "
              f"уточните keep/split в configs/extra_sources.yaml")
        return None
    # в одной выгрузке бывают и кварталы, и итог за год: берём самый
    # частый период, иначе годовое значение смешается с квартальными
    kinds = d["PERIOD"].astype(str).str.strip().map(
        lambda p: "month" if p in MONTHS else "quarter" if p in QUARTERS else "year")
    finest = next(k for k in ("month", "quarter", "year") if (kinds == k).any())
    if kinds.nunique() > 1:
        print(f"ЕМИСС {label}: периоды {sorted(kinds.unique())} – беру {finest}")
        d = d[kinds == finest]
    years = d["time"].astype(int)
    rows = [(m, _name(r[terr]), r["value"]) for _, r in d.iterrows()
            for m in _months(r["PERIOD"], int(r["time"]))]
    df = pd.DataFrame(rows, columns=["date", "terr_name", name])
    unknown = set(d["PERIOD"].astype(str).str.strip()) - {p for p in d["PERIOD"].astype(str).str.strip()
                                                          if _months(p, 2000)}
    if unknown:
        print(f"ЕМИСС {label}: непонятные периоды пропущены: {sorted(unknown)}")
    df = by_name(df, codes).groupby(["date", "cov_region"], as_index=False)[name].mean()
    if df.empty:
        print(f"ЕМИСС {label} ({fname}): ни одного региона (есть только "
              f"{sorted(d[terr].astype(str).unique())[:3]}) – пропуск")
        return None
    note = ""
    if years.max() - years.min() + 1 < 3:
        note = "  (!) короткий ряд – нужна ручная выгрузка"
    elif years.max() < pd.Timestamp.today().year - 1:
        note = f"  (!) ряд кончается {years.max()} – нужна ручная выгрузка свежих лет"
    print(f"ЕМИСС {label} ({fname}): регионов {df['cov_region'].nunique()}, "
          f"{df['date'].min():%Y-%m}–{df['date'].max():%Y-%m}{note}")
    return df


def cbr_deposits(codes: dict, by_name) -> pd.DataFrame | None:
    f = RAW_DIR / "cbr" / "deposits_households.xlsx"
    if not f.exists():
        return None
    x = pd.read_excel(f, sheet_name="в рублях", header=None)
    dates = pd.to_datetime(x.iloc[1, 1:], format="%d.%m.%Y", errors="coerce")
    body = x.iloc[2:].set_axis(["terr_name", *range(len(dates))], axis=1)
    long = body.melt(id_vars="terr_name", var_name="i", value_name="deposits_households")
    # остаток «на 01.02» – это конец января: относим к предыдущему месяцу
    long["date"] = long["i"].map(dict(enumerate(dates))) - pd.offsets.MonthBegin(1)
    long["deposits_households"] = pd.to_numeric(long["deposits_households"], errors="coerce")
    long = long.dropna(subset=["date", "deposits_households"])
    long["terr_name"] = long["terr_name"].map(_name)
    df = by_name(long[["date", "terr_name", "deposits_households"]], codes)
    print(f"ЦБ вклады: регионов {df['cov_region'].nunique()}, {df['date'].min():%Y-%m}–{df['date'].max():%Y-%m}")
    return df[["date", "cov_region", "deposits_households"]]


# Форма 1-НМ: лист (по первому коду строки) -> {код графы: столбец}.
# 1130 – НДФЛ всего, 1140.3 – НДФЛ, удержанный налоговыми агентами (прокси
# фонда оплаты труда), 1000 – все доходы, 1000.4 – из них в местные бюджеты.
FNS_FIELDS = {"1130": {"1130": "ndfl_total", "1140.3": "ndfl_agents"},
              "1000": {"1000": "tax_total", "1000.4": "tax_local"}}


def _zip_name(i) -> str:
    try:
        return i.filename.encode("cp437").decode("cp866")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return i.filename


def fns_taxes(codes: dict, by_name) -> pd.DataFrame | None:
    import io
    import zipfile
    files = sorted((RAW_DIR / "fns").glob("1nm_*.zip"))
    if not files:
        print("ФНС 1-НМ: файлов нет – пропуск (python -m scripts.download_extra --only fns)")
        return None
    rows = []
    for f in files:
        rep = pd.Timestamp(re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1))
        month = rep - pd.offsets.MonthBegin(1)          # «на 01.07» – итог за январь–июнь
        z = zipfile.ZipFile(f)
        book = [i for i in z.infolist() if "Поступ" in _zip_name(i) and "террит" in _zip_name(i)
                and "ЕСН" not in _zip_name(i)]
        if len(book) != 1:
            print(f"ФНС 1-НМ {f.name}: не найдена книга «Поступление налогов по территориям» – пропуск")
            continue
        x = pd.ExcelFile(io.BytesIO(z.read(book[0])))
        got = {}
        for prefix, fields in FNS_FIELDS.items():
            sheet = next((sh for sh in x.sheet_names if sh.split("-")[0] == prefix), None)
            if sheet is None:
                print(f"ФНС 1-НМ {f.name}: нет листа {prefix} – пропуск")
                continue
            d = pd.read_excel(x, sheet_name=sheet, header=None, dtype=str)
            code_row = d.index[d[0].astype(str).str.strip() == "А"]
            if not len(code_row):
                print(f"ФНС 1-НМ {f.name}: на листе {sheet} нет строки кодов граф – пропуск")
                continue
            codes_row = d.loc[code_row[0]].astype(str).str.strip()
            body = d.loc[code_row[0] + 1:]
            for code, name in fields.items():
                col = codes_row.index[codes_row == code]
                if not len(col):
                    print(f"ФНС 1-НМ {f.name}: на листе {sheet} нет графы {code}")
                    continue
                got[name] = pd.Series(pd.to_numeric(body[col[0]], errors="coerce").to_numpy(),
                                      index=body[0].astype(str).str.strip())
        if got:
            t = pd.DataFrame(got).rename_axis("terr_name").reset_index()
            rows.append(t.assign(date=month))
    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True)
    long = long[long["terr_name"].str.len() > 0]
    long["terr_name"] = long["terr_name"].map(_name)
    cols = [c for f in FNS_FIELDS.values() for c in f.values() if c in long]
    df = by_name(long, codes).groupby(["date", "cov_region"], as_index=False)[cols].sum(min_count=1)
    months = sorted(df["date"].unique())
    full = pd.date_range(months[0], months[-1], freq="MS")
    gaps = [f"{m:%Y-%m}" for m in full.difference(pd.DatetimeIndex(months))]
    print(f"ФНС 1-НМ: регионов {df['cov_region'].nunique()}, {months[0]:%Y-%m}–{months[-1]:%Y-%m}, "
          f"столбцы {cols}" + (f"; нет месяцев {gaps}" if gaps else ""))
    return df


def vacancies(codes: dict, by_name) -> pd.DataFrame | None:
    snaps = sorted((RAW_DIR / "trudvsem").glob("vacancies_*.parquet"))
    if not snaps:
        return None
    out_file = PROCESSED_DIR / "vacancies_region.parquet"
    prev = pd.read_parquet(out_file) if out_file.exists() else pd.DataFrame()
    parts = [prev] if not prev.empty else []
    done = set(prev["snapshot"].dt.strftime("%Y-%m-%d")) if not prev.empty else set()
    for f in snaps:
        day = re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1)
        if day in done:
            continue
        v = pd.read_parquet(f, columns=["regionName", "workPlaces", "salaryMin", "deleted", "status"])
        v = v[(v["deleted"] != "True") & (v["status"] == "Одобрено")]
        agg = v.groupby("regionName").agg(vacancies=("workPlaces", "size"), work_places=("workPlaces", "sum"),
                                          salary_median=("salaryMin", lambda s: s[s > 0].median())).reset_index()
        agg = agg.assign(terr_name=agg["regionName"].map(_name), date=pd.Timestamp(day))
        agg = by_name(agg, codes)
        parts.append(agg.rename(columns={"date": "snapshot"})[["snapshot", "cov_region", "vacancies",
                                                                "work_places", "salary_median"]])
        print(f"вакансии {day}: {int(v.shape[0])} вакансий, регионов {agg['cov_region'].nunique()}")
    for f in snaps[:-KEEP_SNAPSHOTS]:          # сводка уже посчитана – сырой срез больше не нужен
        f.unlink()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["snapshot", "cov_region"]) if parts else None


def main() -> None:
    from scripts.build_covariates import by_name, region_codes
    codes = region_codes()
    parts = emiss_series(codes, by_name)
    for extra in (cbr_deposits(codes, by_name), fns_taxes(codes, by_name)):
        if extra is not None:
            parts.append(extra)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if parts:
        reg = parts[0]
        for p in parts[1:]:
            reg = reg.merge(p, on=["date", "cov_region"], how="outer")
        reg = reg.sort_values(["cov_region", "date"]).reset_index(drop=True)
        reg.to_parquet(PROCESSED_DIR / "covariates_region_extra.parquet", index=False)
        print(f"covariates_region_extra: {len(reg)} строк, столбцы {[c for c in reg if c not in ('date', 'cov_region')]}")
    vac = vacancies(codes, by_name)
    if vac is not None:
        vac.to_parquet(PROCESSED_DIR / "vacancies_region.parquet", index=False)
        print(f"vacancies_region: срезов {vac['snapshot'].nunique()}, регионов {vac['cov_region'].nunique()}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_extra", "data", main)
