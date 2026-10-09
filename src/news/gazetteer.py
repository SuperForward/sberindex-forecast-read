"""Справочник топонимов МО: как найти муниципалитет в тексте новости.

Из официального названия МО строится шаблон основы слова с учётом падежей:
«городской округ город Орск» -> Орск, Орске, Орска, Орском…;
«Абатский муниципальный район» -> «Абатский район», «Абатском районе»…
Город и одноимённый район различаются: «Ишим» не совпадает с «Ишимский».

Одноимённые МО разных регионов (Советский район – их десятки) без
упоминания региона в тексте не привязываются: лучше пропустить новость, чем
приписать её не тому муниципалитету. Внутригородские территории Москвы и
Петербурга не привязываются вовсе – их названия (Арбат, Южное Бутово) редко
означают муниципалитет в экономическом смысле.
"""

import re
from functools import lru_cache

import pandas as pd

from src import store

# «Город» или «район» в падежах: остаток после основы слова
CITY_TAIL = r"(?:а|е|у|ом|ой|ы|и|я|ю|ем)?"
ADJ_TAIL = r"(?:ий|ый|ой|ого|ому|ом|ая|ую|ое|ой|им|ым)"
UNIT = r"(?:район\w*|округ\w*|муниципальн\w*)"
REGION_WORDS = r"(?:област\w*|кра[йяею]\w*|республик\w*|округ\w*)"
# «Смоленской области», «Курганская область» – это регион, не город Смоленск / Курган
NOT_REGION = r"(?!\s+(?:област|кра[йяею]|район|округ|губерни)\w*)"


def _base(mo_name: str) -> tuple[str, str] | None:
    """(основа, вид): вид "city" – город/поселение, "adj" – «Абатский район»."""
    n = re.sub(r"\s+", " ", str(mo_name)).strip()
    if n.startswith("внутригородская территория"):
        return None
    m = re.match(r"городской округ (?:город |поселок |посёлок |закрытое административно-территориальное "
                 r"образование (?:город |поселок )?)?(.+)$", n)
    if m:
        word = m.group(1).strip()
        if re.search(r"(ский|цкий|ской)$", word):          # «городской округ Охинский»
            return word, "adj"
        return word, "city"
    m = re.match(r"(.+?) (?:муниципальный|городской) (?:район|округ)$", n)
    if m:
        return m.group(1).strip(), "adj"
    return None


def _pattern(word: str, kind: str) -> str | None:
    w = word.replace("ё", "е")
    if kind == "adj":
        stem = re.sub(r"(ий|ый|ой)$", "", w)                   # Абатск-
        if len(stem) < 4:
            return None
        # «Абатский район» или «в Абатском / у Абатского» (так называют райцентр)
        return (rf"\b{re.escape(stem)}{ADJ_TAIL}\s+{UNIT}"
                rf"|\b(?i:в|во|у|из|под|около|возле)\s+{re.escape(stem)}(?:ом|ого|ое)\b")
    if " " in w or "-" in w:                                    # Кирово-Чепецк, Нижний Новгород
        parts = re.split(r"([ -])", w)
        last = parts[-1]
        head = "".join(re.escape(p) for p in parts[:-1])        # склоняется только последнее слово
        stem = re.escape(last[:-1] if len(last) > 5 and last[-1] in "аяоеьы" else last)
        return rf"\b{head}{stem}\w{{0,3}}\b{NOT_REGION}"
    stem = w[:-1] if len(w) > 5 and w[-1] in "аяоеьы" else w
    if len(stem) < 4:
        return None
    # не прилагательное «Ишимский», не часть другого слова
    return rf"\b{re.escape(stem)}{CITY_TAIL}\b(?!\s*-){NOT_REGION}"


def _region_pattern(region: str) -> str | None:
    """Регион в тексте: «Оренбургской области», «Оренбуржье» – по основе прилагательного."""
    r = str(region).replace("ё", "е")
    m = re.match(r"(?:город |г\.\s*)?([А-ЯЁ][а-яё-]+)", r)
    if not m:
        return None
    word = m.group(1)
    stem = re.sub(r"(ской|ского|ская|ский|ой|ая|ий)$", "", word)
    return rf"\b{re.escape(stem[:max(4, len(stem))])}\w*" if len(stem) >= 4 else None


@lru_cache(maxsize=1)
def gazetteer() -> pd.DataFrame:
    """Таблица: oktmo, mo_name, region, kind, pattern, ambiguous (одноимённые в других регионах)."""
    p = store.load("spending_mo")[["oktmo", "mo_name", "region"]].drop_duplicates("oktmo")
    rows = []
    for r in p.itertuples():
        b = _base(r.mo_name)
        if not b:
            continue
        pat = _pattern(*b)
        if pat:
            rows.append({"oktmo": r.oktmo, "mo_name": r.mo_name, "region": r.region, "word": b[0],
                         "kind": b[1], "pattern": pat, "region_pattern": _region_pattern(r.region)})
    g = pd.DataFrame(rows)
    g["ambiguous"] = g.duplicated(["word", "kind"], keep=False)
    return g


@lru_cache(maxsize=1)
def _compiled():
    g = gazetteer()
    return [(r.oktmo, re.compile(r.pattern), r.ambiguous,
             re.compile(r.region_pattern, re.I) if isinstance(r.region_pattern, str) else None)
            for r in g.itertuples()]


def find_mo(text: str) -> list[str]:
    """ОКТМО муниципалитетов, упомянутых в тексте. Регистр важен: «Мирный» –
    город, «мирный» – прилагательное; поэтому ищем с учётом заглавной буквы."""
    t = str(text).replace("ё", "е")
    out = []
    for oktmo, rx, ambiguous, region_rx in _compiled():
        if rx.search(t):
            if ambiguous and not (region_rx and region_rx.search(t)):
                continue                         # тёзка без указания региона – не угадываем
            out.append(oktmo)
    return out


@lru_cache(maxsize=1)
def region_gazetteer() -> list:
    """(cov_region, шаблон) для упоминаний субъекта: «в Белгородской области»,
    «Белгородская область», «в Москве», «Петербурге», «Республика Татарстан»."""
    from src.data.oktmo import cov_region_of_oktmo
    p = store.load("spending_mo")[["oktmo", "region"]].drop_duplicates("region")
    out, seen = [], set()
    special = {"Москв": r"\bМоскв(?:а|е|у|ы|ой)\b", "Санкт-Петербург": r"\b(?:Санкт-)?Петербург\w{0,2}\b",
               "Севастопол": r"\bСевастопол\w{1,2}\b"}
    # как регион называют в новостях, помимо официального имени
    aliases = {"Кемеров": r"Кузбасс\w*", "Башкортостан": r"Башкири\w*", "Саха": r"Якути\w*",
               "Удмурт": r"Удмурти\w*", "Чуваш": r"Чуваши\w*", "Ханты": r"Югр\w*", "Ямало": r"Ямал\w*",
               "Мордови": r"Мордови\w*", "Бурят": r"Бурятии|Бурятия", "Коми": r"Коми\b",
               "Карел": r"Карели\w*", "Хакас": r"Хакаси\w*", "Тыв": r"Тыв\w*", "Алтай": r"Алта\w*",
               "Омск": r"Омск\w*\s+област\w*", "Томск": r"Томск\w*\s+област\w*",
               "Курск": r"Курск\w*\s+област\w*", "Марий": r"Марий\s+Эл",
               "Дагестан": r"Дагестан\w*", "Татарстан": r"Татарстан\w*", "Крым": r"Крым\w*"}
    for r in p.itertuples():
        cov = cov_region_of_oktmo(r.oktmo)
        if cov in seen:
            continue
        seen.add(cov)
        name = str(r.region)
        sp = next((v for k, v in special.items() if k in name), None)
        if sp:
            out.append((cov, re.compile(sp)))
            continue
        alias = next((v for k, v in aliases.items() if k in name), None)
        m = re.match(r"(?:Республики?\s+)?([А-ЯЁ][а-яё-]+)", name)
        if not m:
            if alias:
                out.append((cov, re.compile(rf"\b(?:{alias})")))
            continue
        word = m.group(1)
        stem = re.sub(r"(ской|ского|ская|ский|цкой|цкого|цкая|ой|ая|ого|ий|ы|и|а)$", "", word)
        if len(stem) < 4:
            if alias:
                out.append((cov, re.compile(rf"\b(?:{alias})")))
            continue
        extra = rf"|\b(?:{alias})" if alias else ""
        if "Республик" in name:
            out.append((cov, re.compile(rf"\b(?:Республик\w*\s+{re.escape(stem)}\w*|{re.escape(stem)}\w*\s+Республик\w*){extra}")))
        else:
            out.append((cov, re.compile(rf"\b{re.escape(stem)}\w*\s+(?:област|кра[йяею]|автономн)\w*{extra}")))
    return out


def find_regions(text: str) -> list:
    t = str(text).replace("ё", "е")
    return [cov for cov, rx in region_gazetteer() if rx.search(t)]
