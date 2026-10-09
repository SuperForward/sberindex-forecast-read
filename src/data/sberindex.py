"""Загрузка выгрузки СберИндекса «Потребительские безналичные расходы на уровне МО».

Исходный CSV: `;`-разделитель, колонки period, value, obs_status, source,
category_15, mo, freq, decimals, unit_measure, unit_mult. Кода МО нет –
только название, и у МО-тёзок из разных регионов строки слиты под одним
именем (см. mo_oktmo_map: match_type == "ambiguous").
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, ROOT

SPENDING_GLOB = "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij_*.csv"

CATEGORY_CODES = {
    "Все категории": "all",
    "Продовольствие": "food",
    "Здоровье": "health",
    "Общественное питание": "horeca",
    "Транспорт": "transport",
    "Маркетплейсы": "marketplaces",
}


# Порядок МО в фильтре на странице датасета sberindex.ru: по регионам,
# тёзки из разных регионов – отдельными строками. Сохранён как FNV-1a
# хэши названий (UTF-16 code units, base36) – см. load_site_order. Лежит в
# reference/: это справочник, собранный вручную, из данных его не получить.
SITE_ORDER_FILE = "sberindex_mo_order.hashes.txt"


def _fnv36(s: str) -> str:
    raw = s.encode("utf-16-le")
    h = 0x811C9DC5
    for i in range(0, len(raw), 2):
        h ^= raw[i] | (raw[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        h, r = divmod(h, 36)
        out = digits[r] + out
        if not h:
            return out


def load_site_order(names: pd.Series) -> pd.Series:
    """Список МО в порядке сайта (с повторами тёзок); None – хэш не опознан."""
    lookup = {_fnv36(n): n for n in pd.Series(names).drop_duplicates()}
    seq = (ROOT / "reference" / SITE_ORDER_FILE).read_text(encoding="utf-8").split()
    return pd.Series([lookup.get(h) for h in seq], name="mo_name")


def spending_path() -> Path:
    files = sorted(RAW_DIR.glob(SPENDING_GLOB))
    if not files:
        raise FileNotFoundError(f"нет файла {SPENDING_GLOB} в {RAW_DIR}")
    return files[-1]


def load_spending(path: Path | None = None) -> pd.DataFrame:
    """Длинная таблица: date, mo_name, category, value.

    Константные служебные колонки отброшены, category – латинский код.
    """
    df = pd.read_csv(path or spending_path(), sep=";",
                     usecols=["period", "value", "category_15", "mo"])
    df = df.rename(columns={"period": "date", "mo": "mo_name", "category_15": "category_ru"})
    df["date"] = pd.to_datetime(df["date"])
    df["mo_name"] = df["mo_name"].str.strip()
    df["category"] = df["category_ru"].map(CATEGORY_CODES)
    unknown = df.loc[df["category"].isna(), "category_ru"].unique()
    if len(unknown):
        raise ValueError(f"неизвестные категории: {list(unknown)}")
    return df[["date", "mo_name", "category", "value"]]


# Дашборд СберИндекса «Потребительские расходы, % г/г» (Россия, с 2018 г.):
# второй датасет СберИндекса, известен на дату прогноза – в отличие от роста
# г/г самих МО, которого в 2023 г. ещё нет (панель МО начинается с 01.2023).
GROWTH_GLOB = "consumer-spending-growth_*.csv"


def load_national_growth() -> pd.Series:
    """Номинальный рост безналичных расходов по России, доля г/г, по месяцам."""
    files = sorted((RAW_DIR / "sberindex").glob(GROWTH_GLOB))
    if not files:
        raise FileNotFoundError(f"нет файла {GROWTH_GLOB} в {RAW_DIR / 'sberindex'}")
    g = pd.read_csv(files[-1], sep=";")
    g = g[(g["value_type"] == "Номинальное") & (g["type"] == "Всего") & (g["ref_area"] == "Россия")]
    return (g.set_index(pd.to_datetime(g["period"]))["value"] / 100).sort_index()


# Официальный набор СберИндекса (hackathonlicence.zip, CC BY-SA 4.0): та же
# оценка расходов, но у каждого МО код неизменной территории territory_id –
# тёзки разведены, сопоставлять названия не нужно. Код переводится в ОКТМО
# по справочнику СберИндекса «Границы и изменения МО» (t_dict_municipal).
HACKATHON_DIR = RAW_DIR / "hackathon"
CATEGORY_CODES_HACKATHON = {**CATEGORY_CODES}
MO_TYPES = {"городской округ": "go", "муниципальный район": "mr", "муниципальный округ": "mo",
            "внутригородская территория города федерального значения": "vt"}


def hackathon_available() -> bool:
    return (HACKATHON_DIR / "consumption.parquet").exists() and \
        (HACKATHON_DIR / "t_dict_municipal_districts.xlsx").exists()


def load_territories(year: int = 2023) -> pd.DataFrame:
    """territory_id → ОКТМО (8 знаков), название, тип. У территории бывает
    несколько версий кода (преобразования МО): берётся версия, действовавшая в
    year – начало периода данных, как и в бюллетенях численности Росстата;
    если такой нет – последняя."""
    d = pd.read_excel(HACKATHON_DIR / "t_dict_municipal_districts.xlsx", dtype={"oktmo": str})
    d["valid"] = (d["year_from"] <= year) & (d["year_to"] >= year)
    d = d.sort_values(["territory_id", "valid", "year_to"]).drop_duplicates("territory_id", keep="last")
    return pd.DataFrame({"territory_id": d["territory_id"].astype(int),
                         "oktmo": d["oktmo"].str.replace("-", "").str[:8],
                         "mo_name": d["municipal_district_name"].str.strip(),
                         "mo_type": d["municipal_district_type"].map(MO_TYPES),
                         "region_dict": d["region_name"],
                         "lat": d["municipal_district_center_lat"], "lon": d["municipal_district_center_lon"]})


def load_spending_hackathon() -> pd.DataFrame:
    """Длинная таблица: date, territory_id, category (латинский код), value."""
    c = pd.read_parquet(HACKATHON_DIR / "consumption.parquet")
    c["date"] = pd.to_datetime(c["date"])
    c["category"] = c["category"].map(CATEGORY_CODES_HACKATHON)
    if c["category"].isna().any():
        raise ValueError("неизвестные категории в consumption.parquet")
    return c[["date", "territory_id", "category", "value"]].astype({"territory_id": int, "value": "int64"})
