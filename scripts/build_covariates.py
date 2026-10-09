"""Помесячные ковариаты -> data/processed/covariates_{region,national}_monthly.parquet.

Запуск: python -m scripts.build_covariates

Регион (region_code – 2 знака ОКАТО/ОКТМО субъекта), помесячно:
  ИПЦ (всего/продовольствие/непрод./услуги), платные услуги, общепит,
  погода (температура, осадки, аномалии), безработица МОТ (квартал ->
  месяцы квартала).
Страна, помесячно: производственный календарь, курсы ЦБ.

Код региона ковариат (cov_region): 2 знака субъекта, но автономные округа
внутри краёв выделены отдельно – 1110 НАО, 7110 ХМАО, 7114 ЯНАО; «11» и
«71» тогда означают Архангельскую и Тюменскую области без округов. МО
привязываются по префиксу ОКТМО (mo_cov_region.parquet).

Соглашение о датах: значение за месяц t датировано t (1-е число). Сдвиг на
лаг публикации делается при сборке признаков модели, не здесь.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR  # noqa: E402
from src.data import calendar, emiss, external, rosstat  # noqa: E402
from src.data.oktmo import cov_region_of_oktmo  # noqa: E402,F401  (импортируют и другие скрипты)

# латиница, похожая на кириллицу, встречается в названиях («Hенецкого»)
LATIN_LOOKALIKES = str.maketrans("aehkmoptcxyb", "аенкмортсхуь")
QUARTERS = {"I квартал": 1, "II квартал": 2, "III квартал": 3, "IV квартал": 4}


def region_key(name: str) -> str:
    """Ключ названия субъекта: без скобок, «г.», сокращений «авт.» и небукв.

    Разные таблицы Росстата пишут один субъект по-разному: «г.Москва» и
    «Москва», «Чукотский авт.округ» и «Чукотский автономный округ»,
    «Республика Северная Осетия» и «… Осетия-Алания».
    """
    s = str(name).lower().replace("ё", "е").translate(LATIN_LOOKALIKES)
    s = re.sub(r"\(.*?\)", " ", s, flags=re.S).strip()
    # ЕМИСС: «Город Москва столица Российской Федерации город федерального
    # значения», «Город федерального значения Севастополь»
    s = re.sub(r"город федерального значения|столица российской федерации", " ", s).strip()
    s = re.sub(r"^в том числе\s*|^город\s+|^г\.?\s*", "", s)
    s = re.sub(r"[-\u2014–]\s*(кузбасс|югр[аы]|алания|чувашия)", " ", s)
    s = re.sub(r"автономн(ая|ый)|авт\.?", "авт", s)
    return re.sub(r"[^а-я]", "", s)


# Автономные округа внутри краёв/областей: код ОКАТО -> cov_region.
NESTED = {"11100000": "1110", "71100000": "7110", "71140000": "7114",
          "11001000": "11", "71001000": "71"}
WHOLE = {"11000000", "71000000"}  # области вместе с округами – не используем


def cov_region_of_terr(code: str) -> str | None:
    if code in WHOLE:
        return None
    return NESTED.get(code, code[:2])


def region_codes() -> dict[str, str]:
    """Ключ названия -> cov_region, по справочнику ИПЦ (там есть коды).

    «Архангельская область (кроме НАО)» и «Архангельская область» дают один
    ключ (скобки отбрасываются) – побеждает вариант «кроме», код 11.
    """
    cpi = rosstat.load_cpi_regions()
    r = cpi[cpi["level"] == "region"].drop_duplicates("terr_code")
    r = r.assign(cov=r["terr_code"].map(cov_region_of_terr)).dropna(subset=["cov"])
    return {region_key(n): c for n, c in zip(r["terr_name"], r["cov"])}


def by_name(df: pd.DataFrame, codes: dict[str, str]) -> pd.DataFrame:
    """cov_region по названию; из пары «область» / «область (кроме …)» –
    строка «кроме»: она без вложенных округов."""
    df = df.assign(cov_region=df["terr_name"].map(region_key).map(codes),
                   _prio=df["terr_name"].str.contains("кроме").astype(int))
    df = df.dropna(subset=["cov_region"]).sort_values("_prio", ascending=False)
    return df.drop_duplicates(["date", "cov_region"]).drop(columns="_prio")


def unemployment_ilo(codes: dict[str, str]) -> pd.DataFrame:
    """Безработица МОТ, 15 лет и старше: квартал -> месяцы квартала.

    Ручная выгрузка .xls (свежее, без кодов ОКАТО – субъект по названию)
    важнее старой SDMX .xml (субъект по ОКАТО)."""
    xls = emiss.EMISS_DIR / "43062_unemployment_ilo.xls"
    if xls.exists():
        un = emiss.parse_xls(xls)
        age = next(c for c in un if c.endswith("_name") and un[c].eq("15 лет и старше").any())
        terr = next(c for c in un if c.endswith("_name") and un[c].eq("Российская Федерация").any())
        un = un[(un[age] == "15 лет и старше") & un["PERIOD"].isin(QUARTERS)]
        # by_name снимает дубли по (date, регион): date здесь – год и квартал
        un = un.assign(terr_name=un[terr], date=un["time"].astype(str) + un["PERIOD"])
        un = by_name(un, codes)
    else:
        un = emiss.parse_sdmx((emiss.EMISS_DIR / "43062_unemployment_ilo.xml").read_bytes())
        un = un[(un["s_vozr_name"] == "15 лет и старше") & un["PERIOD"].isin(QUARTERS)].copy()
        # ОКАТО субъекта – 8 знаков; числом теряет ведущий ноль (1000000 = 01000000)
        okato = un["s_OKATO"].str.zfill(11).str[:8]  # в ЕМИСС ОКАТО 11 знаков
        un["cov_region"] = okato.map(cov_region_of_terr).where(un["s_OKATO"].str.len() >= 7)
        un = un.dropna(subset=["cov_region"])
    rows = []
    for r in un.itertuples():
        q = QUARTERS[r.PERIOD]
        for m in range(3 * q - 2, 3 * q + 1):
            rows.append((pd.Timestamp(int(r.time), m, 1), r.cov_region, r.value))
    un = pd.DataFrame(rows, columns=["date", "cov_region", "unemployment_ilo"])
    return un.drop_duplicates(["date", "cov_region"])


def main() -> None:
    codes = region_codes()

    cpi = rosstat.load_cpi_regions()
    cpi = cpi[cpi["level"] == "region"].copy()
    cpi["cov_region"] = cpi["terr_code"].map(cov_region_of_terr)
    cpi = cpi.dropna(subset=["cov_region"])[["date", "cov_region", "cpi", "cpi_food", "cpi_nonfood", "cpi_services"]]

    sv = by_name(rosstat.load_paid_services_monthly(), codes)
    sv = sv[["date", "cov_region", "services", "ifo_mom", "ifo_yoy"]].rename(
        columns={"ifo_mom": "services_ifo_mom", "ifo_yoy": "services_ifo_yoy"})

    cat = by_name(rosstat.load_catering_regions_monthly(), codes)[["date", "cov_region", "catering"]]

    # погода считалась по 2-значному субъекту: для округов берём точку субъекта
    wth = external.load_weather_monthly().rename(columns={"region_code": "cov_region"})
    extra = []
    for sub, parent in [("1110", "11"), ("7110", "71"), ("7114", "71")]:
        if sub not in set(wth["cov_region"]):
            extra.append(wth[wth["cov_region"] == parent].assign(cov_region=sub))
    wth = pd.concat([wth] + extra)

    un = unemployment_ilo(codes)

    reg = cpi
    for part in (sv, cat, wth, un):
        reg = reg.merge(part, on=["date", "cov_region"], how="outer")
    reg = reg.sort_values(["cov_region", "date"]).reset_index(drop=True)
    dup = reg.duplicated(["date", "cov_region"]).sum()
    if dup:
        raise RuntimeError(f"дубли (месяц, регион): {dup}")

    nat = calendar.load_monthly().merge(external.load_fx_monthly(), on="date", how="outer")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reg.to_parquet(PROCESSED_DIR / "covariates_region_monthly.parquet", index=False)
    nat.to_parquet(PROCESSED_DIR / "covariates_national_monthly.parquet", index=False)

    mo = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet").drop_duplicates("oktmo")[["oktmo"]]
    mo["cov_region"] = mo["oktmo"].map(cov_region_of_oktmo)
    mo.to_parquet(PROCESSED_DIR / "mo_cov_region.parquet", index=False)
    panel_regions = mo["cov_region"].unique()
    print(f"регион × месяц: {len(reg)} строк, регионов {reg['cov_region'].nunique()}")
    for c in [c for c in reg.columns if c not in ("date", "cov_region")]:
        sub = reg.dropna(subset=[c])
        cov = pd.Series(panel_regions).isin(sub["cov_region"]).mean()
        print(f"  {c:<18} {sub['date'].min():%Y-%m}..{sub['date'].max():%Y-%m}  "
              f"покрытие регионов панели {cov:.0%}")
    print(f"страна × месяц: {len(nat)} строк, {nat['date'].min():%Y-%m}..{nat['date'].max():%Y-%m}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_covariates", "data", main)
