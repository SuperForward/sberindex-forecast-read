"""Строит data/interim/mo_oktmo_map.csv – МО СберИндекса → ОКТМО,
data/processed/spending_mo.parquet – панель расходов с кодом ОКТМО,
data/processed/mo_population.parquet – население МО (Росстат).

Основной источник – официальный набор (data/raw/hackathon/consumption.parquet):
у каждого МО код territory_id, ОКТМО берётся из справочника СберИндекса
(t_dict_municipal_districts.xlsx). Если его нет – выгрузка с сайта (CSV с
названиями МО): в панель попадают только однозначно сопоставленные МО, тёзки
(homonym, строки нескольких МО слиты под одним именем) не берутся.

Запуск: python -m scripts.build_mo_map

Сначала ищет в ОКТМО на начало 2023 (начало периода данных): поздние
версии уже учитывают реформу 2025 и не содержат упразднённых МО. Не
найденное – в версиях на конец 2024 и актуальной.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.config import INTERIM_DIR, PROCESSED_DIR  # noqa: E402
from src.data.mo_match import mark_conflicts, match_mo, resolve_by_order  # noqa: E402
from src.data.oktmo import load_oktmo  # noqa: E402
from src.data.rosstat import attach_population  # noqa: E402
from src.data.sberindex import (hackathon_available, load_site_order, load_spending,  # noqa: E402
                                load_spending_hackathon, load_territories)

PRIMARY, FALLBACKS = "20230112", ["20241227", "20260901"]
OK_TYPES = ["exact", "normalized", "stem", "fuzzy", "by_neighbors"]


def build_from_hackathon() -> tuple:
    """(карта МО → ОКТМО, панель) из официального набора: сопоставление по коду."""
    terr = load_territories(int(PRIMARY[:4]))
    spending = load_spending_hackathon().merge(terr, on="territory_id", how="inner")
    # объединённые территории с одним ОКТМО (Павловский Посад: до и после
    # слияния) – один ряд под последним названием; месяцы не пересекаются
    last = spending.sort_values("date").groupby("oktmo")[["mo_name", "mo_type"]].last()
    spending = spending.drop(columns=["mo_name", "mo_type"]).join(last, on="oktmo")
    ref = load_oktmo(PRIMARY)
    regions = ref.drop_duplicates("region_code").set_index("region_code")["region"]
    res = (spending.drop_duplicates("oktmo")[["mo_name", "mo_type", "oktmo", "territory_id", "region_dict"]]
           .assign(match_type="territory_id", score=None, n_candidates="1", oktmo_name=lambda d: d["mo_name"],
                   region_code=lambda d: d["oktmo"].str[:2], candidates=None))
    # версия ОКТМО – как у сопоставления по названиям: код из ОКТМО на начало
    # 2023 г. или более поздний (МО преобразовано)
    res["oktmo_version"] = res["oktmo"].isin(set(ref["oktmo"])).map({True: PRIMARY, False: "t_dict_municipal"})
    res["region"] = res["region_code"].map(regions).fillna(res["region_dict"])
    res = res.drop(columns="region_dict")
    panel = spending.merge(res[["oktmo", "region_code", "region", "match_type"]], on="oktmo")
    panel = panel[["date", "mo_name", "category", "value", "oktmo", "region_code", "region", "match_type",
                   "territory_id"]].sort_values(["oktmo", "category", "date"]).reset_index(drop=True)
    return res, panel


def save_panel(panel) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pout = PROCESSED_DIR / "spending_mo.parquet"
    panel.to_parquet(pout, index=False)
    n_mo = panel["oktmo"].nunique()
    print(f"панель: {len(panel)} строк, {n_mo} МО, {panel['region_code'].nunique()} регионов -> {pout}")

    pop = attach_population(panel)
    popout = PROCESSED_DIR / "mo_population.parquet"
    pop.to_parquet(popout, index=False)
    print(f"население: {pop['pop'].notna().sum()} из {len(pop)} МО "
          f"({pop['pop_source'].value_counts().to_dict()}) -> {popout}")


def main() -> None:
    if hackathon_available():
        res, panel = build_from_hackathon()
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        out = INTERIM_DIR / "mo_oktmo_map.csv"
        res.to_csv(out, index=False, encoding="utf-8")
        print(f"МО официального набора: {len(res)} (сопоставление по territory_id) -> {out}")
        save_panel(panel)
        return

    spending = load_spending()
    names = spending["mo_name"].drop_duplicates()
    oktmo = load_oktmo(PRIMARY)

    res = match_mo(names, oktmo)
    res["oktmo_version"] = PRIMARY
    for version in FALLBACKS:
        miss = res["match_type"] == "none"
        if not miss.any():
            break
        alt = match_mo(res.loc[miss, "mo_name"], load_oktmo(version))
        alt["oktmo_version"] = version
        found = alt[alt["match_type"] != "none"].set_index("mo_name")
        res = res.set_index("mo_name")
        res.update(found)
        res = res.reset_index()

    # Тёзки: регион по соседям в списке МО на сайте СберИндекса.
    res = resolve_by_order(res, load_site_order(names))
    ref = oktmo.set_index("oktmo")
    fix = res["match_type"] == "by_neighbors"
    res.loc[fix, "oktmo_name"] = res.loc[fix, "oktmo"].map(ref["name"])
    res.loc[fix, "region"] = res.loc[fix, "oktmo"].map(ref["region"])
    res.loc[fix, "candidates"] = None
    res = mark_conflicts(res, OK_TYPES)

    # Тёзки, которых сопоставление не заметило (второй нашёлся в ОКТМО под
    # новым именем): у слитого имени одна и та же дата встречается в ряду
    # дважды. Число строк не показатель: два неполных ряда дают те же 24.
    merged = spending.loc[spending.duplicated(["mo_name", "category", "date"]), "mo_name"].unique()
    hidden = res["mo_name"].isin(merged) & res["match_type"].isin(OK_TYPES)
    res.loc[hidden, "match_type"] = "homonym"

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "mo_oktmo_map.csv"
    res.to_csv(out, index=False, encoding="utf-8")

    total = len(res)
    print(f"МО СберИндекса: {total}")
    for how, n in res["match_type"].value_counts().items():
        print(f"  {how:<11} {n:>5}  {n / total:6.1%}")
    print(f"сохранено: {out}")

    good = res[res["match_type"].isin(OK_TYPES)][["mo_name", "oktmo", "region_code", "region", "match_type"]]
    save_panel(spending.merge(good, on="mo_name", how="inner"))


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_mo_map", "data", main)
