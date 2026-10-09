"""Справочник ОКТМО (Росстат, открытые данные 7708234640-oktmo).

Берём раздел 1 (муниципальные образования) верхнего уровня: муниципальные
районы и округа, городские округа, внутригородские территории городов
федерального значения. Строки-заголовки групп («Муниципальные районы …
края», «Городские округа …/») отбрасываются.
"""

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

OKTMO_DIR = RAW_DIR / "oktmo"
COLUMNS = ["TER", "KOD1", "KOD2", "KOD3", "KC", "RAZDEL", "NAME1", "Centrum",
           "NomDescr", "NomAkt", "Status", "DateUtv", "DateVved"]

# Первая цифра KOD1 в разделе 1 – вид МО. 8xx/9xx – МО автономных округов
# внутри края/области, их вид определяется только по названию.
KOD1_TYPE = {"3": "vt", "5": "mo", "6": "mr", "7": "go"}
# Города федерального значения: всё их деление – внутригородские территории.
FEDERAL_CITIES = {"40", "45", "67"}

_HEADER_RE = re.compile(r"^(?:Муниципальные|Городские округа|Внутригородские)|/$")


def oktmo_path(version: str = "20230112") -> Path:
    files = sorted(OKTMO_DIR.glob(f"data-{version}*.csv"))
    if not files:
        raise FileNotFoundError(f"нет ОКТМО версии {version} в {OKTMO_DIR}")
    return files[-1]


def mo_type_from_name(name: str) -> str | None:
    """go / mr / mo / vt по тексту названия, None если вид не назван."""
    n = name.lower()
    if "внутригородск" in n:
        return "vt"
    if "городской округ" in n or n.startswith("город ") or n.startswith("город-"):
        return "go"
    if "муниципальный район" in n or n.endswith(" район"):
        return "mr"
    if "муниципальный округ" in n:
        return "mo"
    return None


def load_oktmo(version: str = "20230112") -> pd.DataFrame:
    """МО верхнего уровня: oktmo (8 знаков), region_code, region, name, mo_type."""
    raw = pd.read_csv(oktmo_path(version), sep=";", header=None, names=COLUMNS,
                      dtype=str, encoding="utf-8")
    raw = raw[raw["RAZDEL"] == "1"]
    top = raw[(raw["KOD2"] == "000") & (raw["KOD3"] == "000")]

    regions = (top[top["KOD1"] == "000"]
               .set_index("TER")["NAME1"]
               .str.replace(r"^Муниципальные образования\s+", "", regex=True))

    mo = top[(top["KOD1"] != "000") & ~top["NAME1"].str.contains(_HEADER_RE)].copy()
    mo["oktmo"] = mo["TER"] + mo["KOD1"] + mo["KOD2"]
    mo["region_code"] = mo["TER"]
    mo["region"] = mo["TER"].map(regions)
    mo["name"] = mo["NAME1"].str.strip()
    by_name = mo["name"].map(mo_type_from_name)
    by_code = mo["KOD1"].str[0].map(KOD1_TYPE)
    # Для 5xx название «Кунгурский муниципальный округ» и код согласованы;
    # код надёжнее для «Ивдельский» без вида в названии.
    mo["mo_type"] = by_code.fillna(by_name)
    mo.loc[mo["TER"].isin(FEDERAL_CITIES), "mo_type"] = "vt"
    return mo[["oktmo", "region_code", "region", "name", "mo_type"]].reset_index(drop=True)


# Код региона ковариат (cov_region) по ОКТМО МО: 2 знака субъекта, но МО
# автономных округов внутри краёв выделены отдельно (см. scripts/build_covariates.py).
OKTMO_PREFIX = {"118": "1110", "718": "7110", "719": "7114"}


def cov_region_of_oktmo(oktmo: str) -> str:
    return OKTMO_PREFIX.get(oktmo[:3], oktmo[:2])
