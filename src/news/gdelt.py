"""GDELT 2.0, поток translation: события из неанглоязычных СМИ (в т. ч.
русскоязычных – ТАСС, РИА, Regnum, ura.news…), машинно переведённые и
закодированные по схеме CAMEO, с геопривязкой места действия.

Файл выгрузки – 15 минут, tab-separated, без заголовка, 61 столбец
(http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf).
Храним только события в России (ActionGeo_CountryCode = RS) и столбцы,
нужные для признаков: время добавления, код и класс события, тон, число
статей, место действия, ссылку. Текстов статей в GDELT нет.
"""

import io
import re
import zipfile

import pandas as pd

MASTER = "https://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"

# номер столбца выгрузки -> имя
COLS = {0: "event_id", 1: "event_date", 26: "event_code", 28: "root_code", 29: "quad_class",
        30: "goldstein", 31: "num_mentions", 32: "num_sources", 33: "num_articles", 34: "avg_tone",
        51: "geo_type", 52: "geo_name", 53: "geo_country", 54: "geo_adm1", 56: "lat", 57: "lon",
        58: "geo_feature", 59: "date_added", 60: "url"}
NUM = ["quad_class", "goldstein", "num_mentions", "num_sources", "num_articles", "avg_tone",
       "geo_type", "lat", "lon"]


def parse_export(raw: bytes) -> pd.DataFrame:
    """zip выгрузки -> события в России, столбцы COLS."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        data = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(data), sep="\t", header=None, dtype=str, quoting=3,
                    usecols=list(COLS), on_bad_lines="skip")
    d = d.rename(columns=COLS)
    d = d[d["geo_country"] == "RS"].drop(columns="geo_country")
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date_added"] = pd.to_datetime(d["date_added"], format="%Y%m%d%H%M%S", errors="coerce")
    return d.reset_index(drop=True)


def export_urls(master: str, d0: str, d1: str) -> list[tuple[str, str]]:
    """(метка времени ГГГГММДДччммсс, url) выгрузок событий за [d0, d1] (ГГГГ-ММ-ДД)."""
    lo, hi = d0.replace("-", ""), d1.replace("-", "") + "235959"
    out = []
    for line in master.splitlines():
        m = re.search(r"(https?://\S+/(\d{14})\.translation\.export\.CSV\.zip)", line)
        if m and lo <= m.group(2) <= hi:
            out.append((m.group(2), m.group(1).replace("http://", "https://")))
    return out


# --- привязка к МО и регионам -------------------------------------------------
#
# Ближайшая точка не годится: GDELT геокодирует украинские города в российских
# тёзок (Artemovsk, Makeyevka, Gorlovka), а фамилии – в деревни (Zelenskiy,
# Volodin). Поэтому событие идёт МО, только если название места в GDELT –
# транслитерация названия МО (из справочника топонимов новостей), а субъект
# (ADM1) – регион этого МО (субъект сопоставляется по названию).

TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r",
                     "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))


def latin_key(s: str) -> str:
    """Сравнимая латиница: «Екатеринбург» и «Yekaterinburg», «Тюмень» и
    «Tyumen'» дают один ключ."""
    s = str(s).lower().replace("ё", "е")
    s = "".join(TRANSLIT.get(ch, ch) for ch in s)
    s = re.sub(r"[^a-z]", "", s)
    for a, b in (("yy", "y"), ("iy", "y"), ("ye", "e"), ("yo", "e"), ("kh", "h")):
        s = s.replace(a, b)
    return s


def place_key(geo_name: str) -> str:
    """«Orsk, Orenburgskaya Oblast', Russia» -> ключ города."""
    return latin_key(str(geo_name).split(",")[0])


def mo_keys() -> pd.DataFrame:
    """oktmo, cov_region, key, kind: город – ключ названия целиком; район
    («Абатский район») – основа прилагательного, событие в райцентре
    «Abatskoye» совпадает по началу."""
    from src.data.oktmo import cov_region_of_oktmo
    from src.news.gazetteer import gazetteer
    g = gazetteer()
    g = g[~g["ambiguous"]]
    out = g.assign(cov_region=g["oktmo"].map(cov_region_of_oktmo),
                   key=[latin_key(w if k == "city" else re.sub(r"(ий|ый|ой)$", "", w))
                        for w, k in zip(g["word"], g["kind"])])
    return out[out["key"].str.len() >= 4][["oktmo", "cov_region", "key", "kind"]]


GENERIC = {"respublika", "respubliki", "oblast", "oblasti", "kray", "kraya", "avtonomnaya", "avtonomnay",
           "avtonomny", "avtonomnogo", "okrug", "okruga", "gorod", "goroda", "federalnogo", "znacheny",
           "stolicy", "rossyskoy", "federacii", "sankt", "sovetskaya", "socialistichesk", "and", "general"}
ENDING = re.compile(r"(?:aya|oy|ogo|ey|ii|iya|ya|ia|y|a|i|e|u)$")


def _roots(name: str) -> list[str]:
    """Корни значимых слов названия субъекта латиницей, без падежного окончания:
    «Омской области» и «Omskaya Oblast'» -> omsk; «Москвы» -> moskv,
    «Московской» -> moskovsk."""
    out = []
    for w in re.split(r"[\s()\u2014–,-]+", str(name)):
        k = latin_key(w)
        if len(k) < 3 or k in GENERIC:
            continue
        r = ENDING.sub("", k)
        out.append(r if len(r) >= 4 else k)
    return out


def _same(a: str, b: str) -> bool:
    return min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))


def adm1_regions(ev: pd.DataFrame) -> dict[str, str]:
    """Код субъекта GDELT (RS55…) -> cov_region по названию субъекта
    («Bryanskaya Oblast'» – «Брянской области»; «Krasnodarskiy» и
    «Krasnoyarskiy» различаются целиком). Субъект без однозначного
    совпадения не привязывается."""
    from src import store
    from src.data.oktmo import cov_region_of_oktmo
    p = store.load("spending_mo")[["oktmo", "region"]].drop_duplicates("region")
    ours = [(r, cov_region_of_oktmo(o)) for reg, o in zip(p["region"], p["oktmo"]) for r in _roots(reg)]
    out = {}
    lab = ev.dropna(subset=["geo_adm1"]).drop_duplicates("geo_adm1")
    for adm, name in zip(lab["geo_adm1"], lab["geo_name"]):
        parts = [x.strip() for x in str(name).split(",")]
        if len(parts) < 2:
            continue
        roots = _roots(parts[-2])                       # «Orsk, Orenburgskaya Oblast', Russia»
        # каждый корень должен найти субъект (точное совпадение важнее префикса:
        # «Sakhalinskaya» – Сахалин, а не Саха); «Chechnya and Ingushetiya» – отброс
        per = []
        for r in roots:
            exact = {c for o, c in ours if o == r}
            per.append(exact or {c for o, c in ours if _same(r, o)})
        cand = set.intersection(*per) if per and all(per) else set()
        if len(cand) == 1:
            out[adm] = cand.pop()
    return out


def assign(ev: pd.DataFrame) -> pd.DataFrame:
    """События -> level (mo|region), oktmo, cov_region. Город (geo_type 4) –
    МО по названию в своём субъекте; субъект целиком (5) – регион; прочее
    (страна, непривязанные города) отбрасывается."""
    keys = mo_keys()
    adm = adm1_regions(ev)
    ev = ev.assign(cov_region=ev["geo_adm1"].map(adm))
    city = ev[(ev["geo_type"] == 4) & ev["cov_region"].notna()]
    city = city.assign(k=city["geo_name"].map(place_key)).rename_axis("row").reset_index()
    exact = city.merge(keys[keys["kind"] == "city"][["key", "cov_region", "oktmo"]],
                       left_on=["k", "cov_region"], right_on=["key", "cov_region"]).drop(columns="key")
    rest = city[~city["row"].isin(exact["row"])]
    adj = keys[keys["kind"] == "adj"]
    hits = [exact]
    for reg, g in rest.groupby("cov_region"):
        for key, oktmo in zip(adj.loc[adj["cov_region"] == reg, "key"], adj.loc[adj["cov_region"] == reg, "oktmo"]):
            m = g[g["k"].str.startswith(key)]
            if len(m):
                hits.append(m.assign(oktmo=oktmo))
    mo = pd.concat(hits).drop_duplicates("row").set_index("row").drop(columns="k").assign(level="mo")
    reg = ev[(ev["geo_type"] == 5) & ev["cov_region"].notna()].assign(level="region", oktmo=None)
    return pd.concat([mo, reg])
