"""Сопоставление названий МО СберИндекса с кодами ОКТМО.

СберИндекс пишет вид МО в названии («городской округ город Казань»,
«муниципальный округ Кунгурский»), ОКТМО часто без него («город Казань»,
«Ивдельский»). Обе стороны сводятся к ключу без видовых слов, затем к
основе без прилагательного окончания («Ирбитское» ~ «Ирбитский»); остаток –
нечётко (rapidfuzz). Среди нескольких кандидатов выигрывает совпадающий
по виду МО и признаку «город» (см. _pick); код, доставшийся двум
названиям, помечается conflict (см. mark_conflicts).

Каждая строка результата несёт match_type, чтобы было видно, насколько
сопоставлению можно верить.
"""

import re

import pandas as pd
from rapidfuzz import fuzz, process

from src.data.oktmo import mo_type_from_name

_TYPE_WORDS = [
    "внутригородская территория города федерального значения",
    "внутригородское муниципальное образование",
    "национальный эвенкийский", "эвенкийский национальный", "национальный",
    "муниципальный район", "муниципальный округ", "городской округ",
    "муниципальное образование", "город-курорт", "город-герой", "город",
    "округ", "район", "г", "пгт",
]
_TYPE_RE = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, _TYPE_WORDS)) + r")(?![\w-])")
_ADJ_END_RE = re.compile(r"(ский|ское|ская|ской|ские|цкий|цкое|цкая|ий|ое|ая|ый)$")
_CITY_RE = re.compile(r"(?<![\w-])город(-курорт|-герой)?(?![\w-])", re.I)

# Реформа 2023–2025: муниципальные районы и городские округа массово
# становились муниципальными округами, поэтому mr~mo и go~mo совместимы.
# vt (внутригородские территории Москвы, Петербурга, Севастополя) с
# остальными не смешиваются.
_COMPATIBLE = {"mr": {"mr", "mo"}, "mo": {"mo", "mr", "go"}, "go": {"go", "mo"}, "vt": {"vt"}}


def norm_key(name: str) -> str:
    s = name.lower().replace("ё", "е")
    s = _TYPE_RE.sub(" ", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stem_key(key: str) -> str:
    return " ".join(_ADJ_END_RE.sub("", w) for w in key.split())


def is_city(name: str) -> bool:
    return bool(_CITY_RE.search(name))


def _pick(cand: list, ok: pd.DataFrame, si_type: str | None, si_city: bool) -> list:
    """Оставляет лучших кандидатов по согласию вида МО и признака «город».

    Жёстко отсекается только смешение внутригородских территорий (vt) с
    остальными. Вид и «город» – предпочтения, не фильтры: ОКТМО пишет
    города ХМАО без слова «город» («Нягань»), а реформа 2025 уже перевела
    ГО Свердловской области в муниципальные округа («Серовский»).
    Когда в одном регионе есть и город, и округ с тем же корнем
    («город Ленинск-Кузнецкий» / «Ленинск-Кузнецкий муниципальный округ»),
    побеждает совпадающий по виду.
    """
    if si_type:
        cand = [i for i in cand if (ok.at[i, "mo_type"] == "vt") == (si_type == "vt")]
    if len(cand) <= 1:
        return cand

    def score(i):
        t = ok.at[i, "mo_type"]
        s = 2 if t == si_type else (1 if si_type and t in _COMPATIBLE[si_type] else 0)
        return s + (1 if ok.at[i, "city"] == si_city else 0)

    best = max(score(i) for i in cand)
    return [i for i in cand if score(i) == best]


def resolve_by_order(res: pd.DataFrame, order: pd.Series, window: int = 3) -> pd.DataFrame:
    """Разбирает ambiguous по региону соседей в списке МО с сайта.

    Список на сайте идёт блоками по регионам. Для каждого вхождения
    неоднозначного названия берутся ближайшие однозначно сопоставленные
    соседи слева и справа; если их регион один и среди кандидатов есть МО
    этого региона – вхождение разрешено (match_type = "by_neighbors").

    Название, встречающееся на сайте несколько раз, – настоящие тёзки: их
    строки в CSV СберИндекса слиты под одним именем, и разделить их без
    кода МО нельзя. Такие получают match_type = "homonym" и список ОКТМО.
    """
    res = res.set_index("mo_name")
    region_of = res.loc[res["match_type"].isin(["exact", "normalized", "stem", "fuzzy"]),
                        "region_code"].to_dict()
    names = order.tolist()
    counts = order.value_counts()

    def neighbour_regions(i: int) -> set:
        regs = set()
        for step in (-1, 1):
            j, found = i + step, 0
            while 0 <= j < len(names) and found < window:
                r = region_of.get(names[j])
                if r:
                    regs.add(r)
                    found += 1
                j += step
        return regs

    def cand_list(name: str) -> list[tuple[str, str]]:
        """[(oktmo, region_code)] из колонки candidates."""
        out = []
        for c in (res.at[name, "candidates"] or "").split("; "):
            code = c.split(" ", 1)[0]
            if code:
                out.append((code, code[:2]))
        return out

    resolved = {}
    for i, name in enumerate(names):
        if name is None or name not in res.index or res.at[name, "match_type"] != "ambiguous":
            continue
        regs = neighbour_regions(i)
        hits = [c for c in cand_list(name) if c[1] in regs]
        if len(regs) == 1 and len(hits) == 1:
            resolved.setdefault(name, []).append(hits[0][0])

    for name, codes in resolved.items():
        if counts.get(name, 0) == 1:
            res.at[name, "oktmo"] = codes[0]
            res.at[name, "region_code"] = codes[0][:2]
            res.at[name, "match_type"] = "by_neighbors"
            res.at[name, "n_candidates"] = 1
        else:
            res.at[name, "match_type"] = "homonym"
            res.at[name, "candidates"] = "; ".join(sorted(set(codes)))
            res.at[name, "n_candidates"] = counts[name]
    return res.reset_index()


def mark_conflicts(res: pd.DataFrame, ok_types: list[str]) -> pd.DataFrame:
    """Код ОКТМО, доставшийся нескольким названиям, – ошибка сопоставления
    у кого-то из них; все такие строки помечаются conflict."""
    res = res.copy()
    good = res["match_type"].isin(ok_types) & res["oktmo"].notna()
    dup = res.loc[good, "oktmo"].duplicated(keep=False)
    idx = dup[dup].index
    res.loc[idx, "candidates"] = res.loc[idx, "oktmo"] + " (конфликт)"
    res.loc[idx, "match_type"] = "conflict"
    return res


def match_mo(names: pd.Series, oktmo: pd.DataFrame, fuzzy_cutoff: float = 92.0) -> pd.DataFrame:
    """Для каждого уникального названия СберИндекса – строка сопоставления.

    match_type:
      exact      – полное совпадение названия;
      normalized – совпадение ключа без видовых слов;
      stem       – совпадение основы («Ирбитское» ~ «Ирбитский»);
      fuzzy      – нечёткое совпадение ключа, score >= fuzzy_cutoff;
      ambiguous  – несколько кандидатов одного вида (тёзки из разных регионов);
      none       – не найдено.
    """
    ok = oktmo.copy()
    ok["key"] = ok["name"].map(norm_key)
    ok["stem"] = ok["key"].map(stem_key)
    ok["city"] = ok["name"].map(is_city)
    ok = ok[ok["key"] != ""].reset_index(drop=True)
    by_name = ok.groupby("name").indices
    by_key = ok.groupby("key").indices
    by_stem = ok.groupby("stem").indices
    keys = list(by_key)

    rows = []
    for name in pd.Series(names).drop_duplicates():
        si_type = mo_type_from_name(name)
        si_city = is_city(name)
        key = norm_key(name)
        cand, how, score = [], "none", None

        if name in by_name:
            cand, how = list(by_name[name]), "exact"
            if len(cand) > 1:
                cand = _pick(cand, ok, si_type, si_city)
        if not cand and key in by_key:
            cand, how = _pick(list(by_key[key]), ok, si_type, si_city), "normalized"
        if not cand and stem_key(key) in by_stem:
            # По основе город с округом не сводим: «город Канаш» и
            # «Канашский муниципальный округ» – разные МО.
            c = [i for i in by_stem[stem_key(key)]
                 if not (si_city and "муниципальн" in ok.at[i, "name"].lower())]
            cand, how = _pick(c, ok, si_type, si_city), "stem"
        if not cand:
            hit = process.extractOne(key, keys, scorer=fuzz.token_sort_ratio,
                                     score_cutoff=fuzzy_cutoff)
            if hit:
                cand, how, score = _pick(list(by_key[hit[0]]), ok, si_type, si_city), "fuzzy", hit[1]

        if not cand:
            how = "none"
        elif len(cand) > 1:
            how = "ambiguous"

        first = ok.loc[cand[0]] if len(cand) == 1 else None
        rows.append({
            "mo_name": name,
            "mo_type": si_type,
            "match_type": how,
            "score": score,
            "n_candidates": len(cand),
            "oktmo": first["oktmo"] if first is not None else None,
            "oktmo_name": first["name"] if first is not None else None,
            "region_code": first["region_code"] if first is not None else None,
            "region": first["region"] if first is not None else None,
            "candidates": "; ".join(f"{ok.at[i, 'oktmo']} {ok.at[i, 'region']}" for i in cand)
                          if len(cand) > 1 else None,
        })
    return pd.DataFrame(rows)
