"""Загрузка таблиц Росстата (rosstat.gov.ru/storage/mediabank).

- BUL_MO_<год>.xlsx, лист «Численность_по_МО» – население МО на 1 января.
  Код ТЕРСОН-МО (10 знаков): первые 8 совпадают с ОКТМО МО верхнего уровня.
- ipc_RF_fo_sub_<мм-гггг>.xlsx – ИПЦ по РФ, округам и субъектам, лист на
  месяц «ММ(ГГГГ)», % к предыдущему месяцу.
- Oborot_m_*.xlsx / Obschepit_m_*.xls – оборот розницы и общепита по РФ
  за текущий год, млн руб.
"""

import re

import pandas as pd

from src.config import RAW_DIR
from src.data.mo_match import norm_key, stem_key

ROSSTAT_DIR = RAW_DIR / "rosstat"
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь",
             "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def _one(pattern: str):
    files = sorted(ROSSTAT_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"нет {pattern} в {ROSSTAT_DIR}")
    return files[-1]


def load_population_mo(year: int = 2024) -> pd.DataFrame:
    """Население МО верхнего уровня на 1 января: oktmo, name, pop, pop_urban, pop_rural.

    Нужна версия бюллетеня с кодами ТЕРСОН-МО (есть с 2024).
    """
    raw = pd.read_excel(_one(f"BUL_MO_{year}.xlsx"), sheet_name="Численность_по_МО",
                        header=None, dtype=str)
    raw = raw.iloc[:, :5]
    raw.columns = ["code", "name", "pop", "pop_urban", "pop_rural"]
    # Часть субъектов записана «01701000 0 0» (ОКТМО + разряды через пробел).
    code = raw["code"].str.replace(r"\s+", "", regex=True)
    # Excel хранит код числом: у субъектов с кодом на 0 (01 Алтайский край,
    # 04 Красноярский…) ведущий ноль теряется – 9 знаков вместо 10.
    code = code.where(~code.str.fullmatch(r"\d{9}", na=False), "0" + code)
    raw["code"] = code
    # МО верхнего уровня: 10 знаков, поселенческая часть (знаки 6-8) нулевая,
    # а район/округ (знаки 3-5) – нет (иначе это субъект).
    # Не МО: строки-итоги групп («Муниципальные районы …», XX600000) и
    # административные округа Москвы/Петербурга (KOD1 = 2xx) – группировки
    # поверх внутригородских МО. Иркутск записан 12 знаками (257010001000).
    top = (code.str.fullmatch(r"\d{10}|\d{12}", na=False) & (code.str[5:8] == "000")
           & ~code.str[2:5].str.endswith("00") & (code.str[2] != "2"))
    df = raw[top].copy()
    df["oktmo"] = df["code"].str[:8]
    df["name"] = df["name"].str.strip()
    for c in ["pop", "pop_urban", "pop_rural"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df[["oktmo", "name", "pop", "pop_urban", "pop_rural"]].reset_index(drop=True)


_TOP_MO_RE = re.compile(r"муниципальный район|муниципальный округ|городской округ|внутригородск", re.I)


def load_population_mo_by_name(year: int = 2023) -> pd.DataFrame:
    """Население МО верхнего уровня из бюллетеня без кодов (2023 и раньше).

    Регион определяется по строкам-заголовкам субъектов (их коды берутся из
    бюллетеня 2024). Колонки: region_code, name, pop, pop_urban, pop_rural.
    """
    raw = pd.read_excel(_one(f"BUL_MO_{year}.xlsx"), sheet_name="Численность_по_МО",
                        header=None, dtype=str).iloc[:, :4]
    raw.columns = ["name", "pop", "pop_urban", "pop_rural"]
    raw["name"] = raw["name"].str.strip()

    ref = pd.read_excel(_one("BUL_MO_2024.xlsx"), sheet_name="Численность_по_МО",
                        header=None, dtype=str).iloc[:, :2]
    ref.columns = ["code", "name"]
    ref["code"] = ref["code"].str.replace(r"\s+", "", regex=True)
    ref["code"] = ref["code"].where(~ref["code"].str.fullmatch(r"\d{9}", na=False), "0" + ref["code"])
    regions = ref[ref["code"].str.fullmatch(r"\d{2}0{8}", na=False)]
    region_code = dict(zip(regions["name"].str.strip(), regions["code"].str[:2]))

    rows, current = [], None
    for name, pop, urb, rur in raw.itertuples(index=False):
        if not isinstance(name, str):
            continue
        if name in region_code:
            current = region_code[name]
        elif current and _TOP_MO_RE.search(name) and "поселени" not in name.lower():
            rows.append((current, name, pop, urb, rur))
    df = pd.DataFrame(rows, columns=["region_code", "name", "pop", "pop_urban", "pop_rural"])
    for c in ["pop", "pop_urban", "pop_rural"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def attach_population(mo: pd.DataFrame) -> pd.DataFrame:
    """Население на 01.01.2024 (или 01.01.2023) для МО панели.

    mo: колонки oktmo, mo_name. Порядок поиска:
      code2024 – по коду ОКТМО в бюллетене 2024;
      name2024 – по названию внутри региона (МО, получившие в 2023 новый код
                 при преобразовании района в округ);
      name2023 – по названию в бюллетене 2023 (упразднённые к 2024 МО).
    Возвращает mo + pop, pop_urban, pop_rural, pop_source.
    """
    out = mo[["oktmo", "mo_name"]].drop_duplicates("oktmo").copy()
    out["region_code"] = out["oktmo"].str[:2]
    out["k"] = out["mo_name"].map(lambda n: stem_key(norm_key(n)))
    cols = ["pop", "pop_urban", "pop_rural"]

    p24 = load_population_mo(2024)
    res = out.merge(p24[["oktmo"] + cols], on="oktmo", how="left")
    res["pop_source"] = res["pop"].notna().map({True: "code2024", False: None})

    p24["region_code"] = p24["oktmo"].str[:2]
    p23 = load_population_mo_by_name(2023)
    for src, ref in [("name2024", p24), ("name2023", p23)]:
        ref = ref.assign(k=ref["name"].map(lambda n: stem_key(norm_key(n))))
        ref = ref.drop_duplicates(["region_code", "k"], keep=False)  # только однозначные
        miss = res["pop"].isna()
        hit = res.loc[miss, ["region_code", "k"]].merge(ref[["region_code", "k"] + cols],
                                                         on=["region_code", "k"], how="left")
        hit.index = res.index[miss]
        found = hit["pop"].notna()
        res.loc[found[found].index, cols] = hit.loc[found, cols].to_numpy()
        res.loc[found[found].index, "pop_source"] = src
    return res.drop(columns=["k"])


def load_cpi_regions() -> pd.DataFrame:
    """ИПЦ по субъектам помесячно, % к предыдущему месяцу.

    Колонки: date, terr_code, terr_name, level (rf / fo / region),
    region_code (2 знака ОКАТО/ОКТМО для субъектов), cpi, cpi_food,
    cpi_nonfood, cpi_services.
    """
    path = _one("ipc_RF_fo_sub_*.xlsx")
    book = pd.ExcelFile(path)
    frames = []
    for sheet in book.sheet_names:
        m = re.fullmatch(r"(\d{2})\((\d{4})\)", sheet)
        if not m:
            continue
        d = pd.read_excel(book, sheet, header=None, dtype=str).iloc[4:, :6]
        d.columns = ["terr_code", "terr_name", "cpi", "cpi_food", "cpi_nonfood", "cpi_services"]
        d = d[d["terr_code"].str.fullmatch(r"\d+", na=False)].copy()
        d["date"] = pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in ["cpi", "cpi_food", "cpi_nonfood", "cpi_services"]:
        df[c] = pd.to_numeric(df[c].str.replace(",", "."), errors="coerce")
    df["terr_name"] = df["terr_name"].str.strip()
    # Excel хранит код числом: у субъектов на 0 (01 Алтайский край, 04, 05,
    # 07, 08) ведущий ноль теряется – 7 знаков вместо 8, и регион путался
    # с другим (01000000 -> «10», Амурская область).
    df["terr_code"] = df["terr_code"].where(df["terr_code"].str.len() != 7, "0" + df["terr_code"])
    df["level"] = df["terr_code"].map(lambda c: "rf" if c == "643" else ("region" if len(c) == 8 else "fo"))
    df["region_code"] = df["terr_code"].where(df["level"] == "region").str[:2]
    return df[["date", "terr_code", "terr_name", "level", "region_code",
               "cpi", "cpi_food", "cpi_nonfood", "cpi_services"]]


def _monthly_current_year(path, sheet: str, cols: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    year_cell = raw[0].astype(str).str.extract(r"^(\d{4})")[0].dropna()
    year = int(year_cell.iloc[0]) if len(year_cell) else None
    if year is None:  # общепит: год стоит во второй колонке
        year = int(str(raw[1].dropna().astype(str).str.extract(r"(\d{4})")[0].dropna().iloc[0]))
    rows = raw[raw[0].isin(MONTHS_RU)].iloc[:, :len(cols) + 1]
    rows.columns = ["month"] + cols
    rows = rows.dropna(subset=cols[:1])
    rows["date"] = [pd.Timestamp(year, MONTHS_RU.index(m) + 1, 1) for m in rows["month"]]
    for c in cols:
        rows[c] = pd.to_numeric(rows[c], errors="coerce") / 1000  # млн -> млрд руб.
    return rows[["date"] + cols].reset_index(drop=True)


def load_retail_current_year() -> pd.DataFrame:
    """Оборот розницы по РФ за текущий год, млрд руб.: total, food, nonfood."""
    return _monthly_current_year(_one("Oborot_m_*.xlsx"), "1", ["retail", "food", "nonfood"])


def load_catering_current_year() -> pd.DataFrame:
    """Оборот общепита по РФ за текущий год, млрд руб."""
    return _monthly_current_year(_one("Obschepit_m_*.xls"), "1", ["catering"])


def _label(cell) -> str:
    """«октябрь¹⁾» -> «октябрь»: только буквы и дефис, нижний регистр."""
    return re.sub(r"[^а-яё-]", "", str(cell).lower())


def _year_of(cell) -> int | None:
    m = re.match(r"^\s*(\d{4})", str(cell))
    return int(m.group(1)) if m else None


def load_retail_monthly() -> pd.DataFrame:
    """Оборот розницы по РФ помесячно с 2000 г., млрд руб.: retail, food, nonfood.

    Oborot_<год>.xls, лист «2»: строка-год, затем месяцы и нарастающие итоги
    («январь-февраль»), которые отбрасываются.
    """
    raw = pd.read_excel(_one("Oborot_2*.xls"), sheet_name="2", header=None)
    rows, year = [], None
    for r in raw.itertuples(index=False):
        label = _label(r[0])
        if _year_of(r[0]) and pd.isna(r[1]):  # строка-год: «2025», «20252)»
            year = _year_of(r[0])
        elif year and label in MONTHS_RU and pd.notna(r[1]):
            rows.append((pd.Timestamp(year, MONTHS_RU.index(label) + 1, 1), r[1], r[2], r[3]))
    df = pd.DataFrame(rows, columns=["date", "retail", "food", "nonfood"])
    for c in ["retail", "food", "nonfood"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 1000
    return df


def load_catering_monthly() -> pd.DataFrame:
    """Оборот общепита по РФ помесячно (2020–2025), млрд руб.

    Obschepit_2000-2025.xls, лист «1»: годы по столбцам, месяцы по строкам.
    """
    raw = pd.read_excel(_one("Obschepit_2000-*.xls"), sheet_name="1", header=None)
    hdr = raw.index[raw.iloc[:, 1:].map(lambda v: _year_of(v) is not None).any(axis=1)][0]
    years = {j: _year_of(v) for j, v in raw.iloc[hdr].items() if _year_of(v)}
    rows = []
    for r in raw.iloc[hdr + 1:].itertuples(index=False):
        label = _label(r[0])
        if label in MONTHS_RU:
            for j, y in years.items():
                if pd.notna(r[j]):
                    rows.append((pd.Timestamp(y, MONTHS_RU.index(label) + 1, 1), r[j]))
    df = pd.DataFrame(rows, columns=["date", "catering"])
    df["catering"] = pd.to_numeric(df["catering"], errors="coerce") / 1000
    return df.sort_values("date").reset_index(drop=True)


def load_catering_regions_monthly() -> pd.DataFrame:
    """Оборот общепита по субъектам помесячно (2022–2025), млрд руб.

    Лист «6» даёт нарастающие итоги с начала года («январь-март»);
    месячное значение – разность соседних итогов.
    Колонки: date, terr_name, catering.
    """
    raw = pd.read_excel(_one("Obschepit_2000-*.xls"), sheet_name="6", header=None)
    year_row = raw.index[raw.iloc[:, 1:].map(lambda v: _year_of(v) is not None).any(axis=1)][0]
    years = raw.iloc[year_row].map(_year_of).ffill()
    labels = raw.iloc[year_row + 1].map(_label)
    cols = []
    for j in range(1, raw.shape[1]):
        n = labels[j].count("-") and MONTHS_RU.index(labels[j].split("-")[-1]) + 1
        n = n or (MONTHS_RU.index(labels[j]) + 1 if labels[j] in MONTHS_RU else None)
        if n and pd.notna(years[j]):
            cols.append((j, int(years[j]), n))
    body = raw.iloc[year_row + 2:]
    body = body[body[0].notna() & ~body[0].astype(str).str.match(r"^\s*\d\)")]  # без сносок
    rows = []
    for r in body.itertuples(index=False):
        for j, y, n in cols:
            rows.append((str(r[0]).strip(), y, n, r[j]))
    df = pd.DataFrame(rows, columns=["terr_name", "year", "n", "cum"])
    df["cum"] = pd.to_numeric(df["cum"], errors="coerce")
    df = df.sort_values(["terr_name", "year", "n"])
    df["catering"] = df.groupby(["terr_name", "year"])["cum"].diff().fillna(df["cum"]) / 1000
    df["date"] = [pd.Timestamp(y, n, 1) for y, n in zip(df["year"], df["n"])]
    return df[["date", "terr_name", "catering"]].reset_index(drop=True)


def load_paid_services_monthly() -> pd.DataFrame:
    """Платные услуги населению по РФ, округам и субъектам помесячно, млрд руб.

    rosstat/plat/plat_ГГГГ-ММ.xls, лист «платные всего»: фактически (тыс. руб.),
    индексы физ. объёма к пред. месяцу и к тому же месяцу пред. года.
    Колонки: date, terr_name, services, ifo_mom, ifo_yoy.
    """
    frames = []
    for path in sorted((ROSSTAT_DIR / "plat").glob("plat_*.xls")):
        y, m = map(int, re.search(r"(\d{4})-(\d{2})", path.name).groups())
        raw = pd.read_excel(path, sheet_name="платные всего", header=None).iloc[:, :4]
        raw.columns = ["terr_name", "services", "ifo_mom", "ifo_yoy"]
        raw = raw[pd.to_numeric(raw["services"], errors="coerce").notna() & raw["terr_name"].notna()]
        raw = raw[~raw["terr_name"].astype(str).str.fullmatch(r"\d+")]
        raw["date"] = pd.Timestamp(y, m, 1)
        frames.append(raw)
    df = pd.concat(frames, ignore_index=True)
    df["terr_name"] = df["terr_name"].astype(str).str.strip()
    df["services"] = pd.to_numeric(df["services"]) / 1e6
    for c in ["ifo_mom", "ifo_yoy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "terr_name", "services", "ifo_mom", "ifo_yoy"]]
