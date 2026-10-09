"""Разбор официальных актов о ЧС (publication.pravo.gov.ru): вид, регион, МО, причина.

Всё по названию и реквизитам акта, словарями – как и классификация новостей:
каждое решение объяснимо и воспроизводимо.

- вид: введение режима ЧС, отмена, меры поддержки пострадавших (выплаты,
  компенсации, восстановление жилья), прочее;
- регион – по названию органа и тексту названия (тот же справочник, что у
  новостей, src/news/gazetteer.py), МО – если назван в названии акта;
- причина: паводок, пожар, погода и урожай, эпизоотия, обстрелы, авария
  инфраструктуры.
"""

import re

import pandas as pd

from src.news.gazetteer import find_mo, find_regions

KIND = [
    ("cancel", re.compile(r"(?:отмен|прекращен|снят)\w*[^.]{0,80}режим\w*\s+(?:ЧС|чрезвычайн)", re.I)),
    ("intro", re.compile(r"(?:введени|установлени|объявлени)\w*[^.]{0,120}режим\w*\s+(?:ЧС|чрезвычайн)"
                         r"|режим\w*\s+чрезвычайн\w*\s+ситуац\w*[^.]{0,60}(?:введен|установлен)", re.I)),
    ("support", re.compile(r"выплат|помощ|компенсац|поддержк|возмещен|субсиди|жил\w*\s+помещен|"
                           r"восстановлен|размещени\w*\s+и\s+питани|пострадавш", re.I)),
]
NOT_INTRO = re.compile(r"внесени\w*\s+изменени|признани\w*\s+утратив|правил\w*\s+поведени", re.I)

CAUSE = {
    "flood": r"паводк|наводнен|подтопл|половод|затоплен|прорыв\w*\s+дамб",
    "fire": r"пожар|возгоран|в\s+лесах",
    "weather_crop": r"засух|заморозк|переувлажн|град\w*\b|ливн|сельскохозяйств|агропромышл|посев|урожа",
    "epizootic": r"дерматит|чум\w*\b|грипп\w*\s+птиц|бешенств|эпизоот|ящур",
    "attack": r"обстрел|беспилотн|атак|ракетн|диверси|вооружен",
    "infrastructure": r"авари|обрушен|отоплен|теплоснабж|электроснабж|водоснабж|газоснабж|взрыв",
}
_CAUSE = {k: re.compile(v, re.I) for k, v in CAUSE.items()}
MUNICIPAL = re.compile(r"муниципальн\w*\s+характер|локальн\w*\s+характер", re.I)


def kind(name: str) -> str:
    for k, rx in KIND:
        if rx.search(name) and not (k == "intro" and NOT_INTRO.search(name)):
            return k
    return "other"


def parse(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        name = re.sub(r"\s+", " ", str(it.get("name") or "")).strip()
        full = re.sub(r"\s+", " ", str(it.get("complexName") or "")).strip()
        regions = find_regions(full)
        rows.append({
            "id": it["id"], "date": pd.to_datetime(it["documentDate"]).normalize(),
            "published": pd.to_datetime(it.get("publishDateShort")).normalize(),
            "name": name, "title": full.split(" \"")[0][:200],
            "kind": kind(name),
            "level": "municipal" if MUNICIPAL.search(name) else "region",
            "cov_region": regions[0] if regions else None,
            "mos": ",".join(find_mo(name)),
            "cause": ",".join(k for k, rx in _CAUSE.items() if rx.search(name)) or "other",
        })
    return pd.DataFrame(rows)


def region_monthly(acts: pd.DataFrame, months: pd.DatetimeIndex, default_len: int = 3) -> pd.DataFrame:
    """Регион × месяц: введено ли ЧС в этом месяце, действует ли режим (от
    введения до отмены; без акта об отмене – default_len месяцев), число актов
    о поддержке пострадавших. Регион без привязки не учитывается."""
    a = acts.dropna(subset=["cov_region"]).assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
    rows = []
    for reg, g in a.groupby("cov_region"):
        intro = g[g["kind"] == "intro"].sort_values("month")
        cancel = g[g["kind"] == "cancel"].sort_values("month")
        active = pd.Series(0, index=months)
        for m in intro["month"]:
            later = cancel[cancel["month"] >= m]["month"]
            end = later.iloc[0] if len(later) else m + pd.DateOffset(months=default_len - 1)
            active[(months >= m) & (months <= end)] = 1
        n_intro = intro.groupby("month").size().reindex(months, fill_value=0)
        n_sup = g[g["kind"] == "support"].groupby("month").size().reindex(months, fill_value=0)
        rows.append(pd.DataFrame({"cov_region": reg, "month": months, "chs_intro": n_intro.to_numpy(),
                                  "chs_active": active.to_numpy(), "support_acts": n_sup.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["cov_region", "month", "chs_intro", "chs_active", "support_acts"])
