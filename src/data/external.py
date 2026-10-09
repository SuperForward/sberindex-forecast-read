"""Внешние открытые источники: курсы ЦБ РФ, координаты МО (Wikidata).

Загрузка – scripts/download_external.py; здесь чтение и приведение к
помесячным признакам.
"""

import xml.etree.ElementTree as ET

import pandas as pd

from src.config import RAW_DIR

EXT_DIR = RAW_DIR / "external"
CBR_CODES = {"USD": "R01235", "EUR": "R01239", "CNY": "R01375"}
CBR_URL = ("https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={d1}&date_req2={d2}"
           "&VAL_NM_RQ={code}")
WIKIDATA_QUERY = """
SELECT ?item ?oktmo ?lat ?lon WHERE {
  ?item wdt:P764 ?oktmo ; p:P625/psv:P625 [ wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon ] .
  FILTER(STRLEN(?oktmo) = 8)
}"""
# Административные центры: 11-значный ОКТМО населённого пункта на «001»,
# первые 8 знаков – код МО. Закрывает МО, у которых в Wikidata нет своего кода.
WIKIDATA_CENTRES_QUERY = """
SELECT ?oktmo ?lat ?lon WHERE {
  ?item wdt:P764 ?oktmo ; p:P625/psv:P625 [ wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon ] .
  FILTER(STRLEN(?oktmo) = 11 && STRENDS(?oktmo, "001"))
}"""
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def load_fx_monthly() -> pd.DataFrame:
    """Средний за месяц курс ЦБ и его изменение м/м: date, usd, eur, cny, *_mom."""
    frames = []
    for cur in CBR_CODES:
        path = EXT_DIR / f"cbr_{cur.lower()}.xml"
        root = ET.fromstring(path.read_bytes())
        rows = [(r.get("Date"), float(r.find("Value").text.replace(",", ".")) /
                 float(r.find("Nominal").text)) for r in root.findall("Record")]
        s = pd.DataFrame(rows, columns=["date", cur.lower()])
        s["date"] = pd.to_datetime(s["date"], format="%d.%m.%Y")
        frames.append(s.set_index("date"))
    d = pd.concat(frames, axis=1).sort_index()
    m = d.resample("MS").mean()
    for c in list(m.columns):
        m[f"{c}_mom"] = m[c].pct_change()
    return m.reset_index()


def load_mo_coords() -> pd.DataFrame:
    """Координаты МО: oktmo (8 знаков), lat, lon, coord_source.

    Порядок: wikidata (объект с кодом МО) -> wikidata_centre (адм. центр,
    11-значный код на «001») -> nominatim (геокодер OSM по названию МО).
    """
    df = pd.read_csv(EXT_DIR / "wikidata_oktmo_coords.csv", dtype={"oktmo": str})
    df = df.assign(coord_source="wikidata")[["oktmo", "lat", "lon", "coord_source"]].drop_duplicates("oktmo")
    parts = [df]
    centres = EXT_DIR / "wikidata_oktmo_centres.csv"
    if centres.exists():
        c = pd.read_csv(centres, dtype={"oktmo": str})
        c = c.assign(oktmo=c["oktmo"].str[:8], coord_source="wikidata_centre").drop_duplicates("oktmo")
        parts.append(c[["oktmo", "lat", "lon", "coord_source"]])
    geo = EXT_DIR / "nominatim_mo.csv"
    if geo.exists():
        g = pd.read_csv(geo, dtype={"oktmo": str}).dropna(subset=["lat"])
        parts.append(g.assign(coord_source="nominatim")[["oktmo", "lat", "lon", "coord_source"]])
    return pd.concat(parts).drop_duplicates("oktmo").reset_index(drop=True)


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["temperature_2m_mean", "precipitation_sum", "snowfall_sum"]


def region_points() -> pd.DataFrame:
    """Точка погоды на субъект: средние координаты МО панели в субъекте."""
    from src.config import PROCESSED_DIR
    panel = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet").drop_duplicates("oktmo")
    pts = panel.merge(load_mo_coords(), on="oktmo")
    # дальний восток за 180° (Чукотка) – lon отрицательный, среднее исказится
    pts["lon"] = pts["lon"].where(pts["lon"] > 0, pts["lon"] + 360)
    g = pts.groupby("region_code").agg(lat=("lat", "median"), lon=("lon", "median"),
                                       n_mo=("oktmo", "size")).reset_index()
    g["lon"] = g["lon"].where(g["lon"] <= 180, g["lon"] - 360)
    return g


def load_weather_monthly() -> pd.DataFrame:
    """Погода по субъектам помесячно: date, region_code, t_mean, precip, snow и
    аномалии к среднему этого месяца за 2019–2025 (t_anom, precip_anom)."""
    frames = []
    for path in sorted(EXT_DIR.glob("weather/region_*.json")):
        import json
        d = json.loads(path.read_text(encoding="utf-8"))["daily"]
        df = pd.DataFrame(d).rename(columns={"time": "day"})
        df["day"] = pd.to_datetime(df["day"])
        df["region_code"] = path.stem.split("_")[1]
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = daily["day"].dt.to_period("M").dt.to_timestamp()
    m = daily.groupby(["region_code", "date"]).agg(
        t_mean=("temperature_2m_mean", "mean"), precip=("precipitation_sum", "sum"),
        snow=("snowfall_sum", "sum")).reset_index()
    clim = m.groupby(["region_code", m["date"].dt.month])[["t_mean", "precip"]].transform("mean")
    m["t_anom"] = m["t_mean"] - clim["t_mean"]
    m["precip_anom"] = m["precip"] - clim["precip"]
    return m
