"""ЕМИСС (fedstat.ru): выгрузка показателя и разбор файлов.

- download() – полная выгрузка формой «Скачать» страницы показателя (срез
  по умолчанию, часто 1–2 года);
- parse_sdmx() / parse_xls() – выгрузки в SDMX и Excel (в том числе ручные
  со страницы показателя: годы, субъекты, периоды) -> длинная таблица.
"""

from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import pandas as pd

from src.config import DATA_DIR, NO_WINDOW, RAW_DIR

EMISS_DIR = RAW_DIR / "emiss"
PAGE = "https://fedstat.ru/indicator/{id}"


# Cookie сессии – не в data/raw: иначе служебный файл меняет «исходные
# данные» (воркер пересчитывает зря) и попадает в архив исходников.
JAR = DATA_DIR / ".cache" / "emiss.cookies"


def _curl(url: str, body: bytes | None = None, timeout: int = 300, referer: str | None = None,
          max_bytes: int | None = None) -> bytes:
    """Запросы в одной сессии: экспорт отдаёт данные только с cookie страницы.
    max_bytes – не качать больше (curl прерывает загрузку)."""
    JAR.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "-f", "-A", "Mozilla/5.0", "--max-time", str(timeout),
           "-c", str(JAR), "-b", str(JAR)]
    if max_bytes:
        cmd += ["--max-filesize", str(max_bytes)]
    if referer:
        cmd += ["-e", referer]
    if body is not None:
        cmd += ["-H", "Content-Type: application/x-www-form-urlencoded", "--data-binary", "@-"]
    r = subprocess.run(cmd + [url], input=body, capture_output=True, creationflags=NO_WINDOW)
    if r.returncode:
        raise RuntimeError(f"{url}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def download(indicator: int, fmt: str = "sdmx", max_bytes: int | None = None, timeout: int = 1800) -> bytes:
    """Полная выгрузка показателя формой «Скачать» страницы: токен struts из
    той же сессии, POST /indicator/<id>/download."""
    page = _curl(PAGE.format(id=indicator)).decode("utf-8")
    token = re.search(r'name="token"\s+value="([^"]+)"', page) or re.search(r'value="([^"]+)"\s+name="token"', page)
    if not token:
        raise RuntimeError(f"ЕМИСС {indicator}: не найден токен формы")
    body = urlencode({"struts.token.name": "token", "token": token.group(1),
                      "id": indicator, "format": fmt}).encode()
    raw = _curl(f"https://fedstat.ru/indicator/{indicator}/download", body, timeout=timeout,
                referer=PAGE.format(id=indicator), max_bytes=max_bytes)
    if raw.lstrip()[:15].lower().startswith(b"<!doctype html"):
        raise RuntimeError(f"ЕМИСС {indicator}: вместо данных страница ошибки")
    return raw


def parse_sdmx(raw: bytes) -> pd.DataFrame:
    """SDMX GenericData (1.0/2.0) -> длинная таблица: измерения серии + time, value.

    Коды измерений раскрываются подписями из CodeLists, если они есть.
    """
    root = ET.fromstring(raw)
    labels = {}
    for cl in root.iter():
        if cl.tag.endswith("CodeList"):
            cid = cl.get("id")
            for c in cl:
                if c.tag.endswith("Code"):
                    desc = next((d.text for d in c if d.tag.endswith("Description")), None)
                    labels[(cid, c.get("value"))] = desc
    rows = []
    for s in root.iter():
        if not s.tag.endswith("}Series"):
            continue
        key, attrs, obs = {}, {}, []
        for part in s:
            name = part.tag.split("}")[-1]
            if name == "SeriesKey":
                key = {v.get("concept"): v.get("value") for v in part}
            elif name == "Attributes":
                attrs = {v.get("concept"): v.get("value") for v in part}
            elif name == "Obs":
                t = next((x.text for x in part if x.tag.endswith("Time")), None)
                v = next((x.get("value") for x in part if x.tag.endswith("ObsValue")), None)
                obs.append((t, v))
        for t, v in obs:
            rows.append({**key, **attrs, "time": t, "value": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(",", "."), errors="coerce")
    for col in df.columns:
        lab = {v: labels[(c, v)] for (c, v) in labels if c.lower().startswith(col.lower())}
        if lab:
            df[col + "_name"] = df[col].map(lab)
    return df


MONTH_NAMES = {"январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
               "октябрь", "ноябрь", "декабрь"}


def _is_period(v) -> bool:
    s = str(v).strip()
    return s in MONTH_NAMES or "квартал" in s or "за год" in s or s.startswith("январь-")


def parse_xls(path) -> pd.DataFrame:
    """Ручная выгрузка ЕМИСС в Excel -> та же длинная таблица, что parse_sdmx:
    измерения dim<i>_name, PERIOD, time, value.

    Сайт отдаёт две раскладки:
      - широкая: строка 3 – годы (объединённые ячейки), строка 4 – периоды
        (если период не вынесен в столбец строк), дальше могут идти ещё
        строки заголовка (измерения dimh<k>_name), слева – столбцы измерений;
      - длинная: измерения, период, год и значение – по столбцам в строках.
    Подписи измерений берутся как есть, без отступов иерархии.
    Формат .xls ограничен 256 столбцами и 65 536 строками: длинную выгрузку
    сайт обрезает молча – проверяйте годы в логе build_extra."""
    raw = pd.read_excel(path, header=None, dtype=str)
    if raw.shape[1] >= 255 or len(raw) >= 65535:
        print(f"ЕМИСС {Path(path).name}: {len(raw)} строк × {raw.shape[1]} столбцов – упёрлись в предел .xls, "
              f"выгрузка обрезана; скачайте меньше лет за раз или в SDMX")
    year_row = next((i for i in range(min(10, len(raw)))
                     if raw.iloc[i].astype(str).str.fullmatch(r"\d{4}").sum() >= 1
                     and raw.iloc[i].astype(str).str.fullmatch(r"\d{4}").sum() == raw.iloc[i].notna().sum()), None)
    if year_row is not None:
        years = raw.iloc[year_row].where(raw.iloc[year_row].astype(str).str.fullmatch(r"\d{4}"))
        first = years.first_valid_index()
        years = years.ffill()
        nxt = raw.iloc[year_row + 1]
        has_period_row = nxt.iloc[first:].dropna().map(_is_period).all() and nxt.iloc[first:].notna().any()
        periods = nxt.ffill() if has_period_row else None
        start = year_row + (2 if has_period_row else 1)
        labels = list(range(first))
        # ещё строки заголовка (показатель, валюта, заёмщик…): столбцы подписей
        # в них пусты; объединённые ячейки -> ffill по столбцам
        headers = []
        while start < len(raw) and raw.iloc[start, :first].isna().all() and raw.iloc[start, first:].notna().any():
            headers.append(raw.iloc[start].ffill())
            start += 1
        body = raw.iloc[start:]
        long = body.melt(id_vars=labels, value_vars=list(range(first, raw.shape[1])), var_name="col")
        for k, h in enumerate(headers):
            long[f"h{k}"] = long["col"].map(h)
        long["time"] = long["col"].map(years)
        if periods is not None:
            long["PERIOD"] = long["col"].map(periods)
        else:                           # период – один из столбцов строк (месяц, квартал)
            pcol = next((c for c in labels if body[c].dropna().map(_is_period).all()), None)
            if pcol is not None:
                long["PERIOD"] = long[pcol]
                labels = [c for c in labels if c != pcol]
            elif "за год" in str(raw.iloc[0, 0]):   # «Уровень бедности (процент, значение показателя за год)»
                long["PERIOD"] = "значение показателя за год"
            else:
                raise ValueError(f"{path}: не найден период ни в заголовке, ни в столбцах")
        dims = labels + [f"h{k}" for k in range(len(headers))]
    else:                               # длинная: ищем столбцы года и значения
        body = raw.iloc[3:]
        ycol = next(c for c in body.columns if body[c].dropna().astype(str).str.fullmatch(r"\d{4}").all())
        pcol = next(c for c in body.columns if c != ycol and body[c].dropna().map(_is_period).all())
        vcol = body.columns[-1]
        dims = [c for c in body.columns if c not in (ycol, pcol, vcol)]
        long = body.rename(columns={ycol: "time", pcol: "PERIOD", vcol: "value"})
    long = long.rename(columns={c: f"dim{c}_name" for c in dims})
    for c in [f"dim{c}_name" for c in dims]:
        long[c] = long[c].astype(str).str.strip()
    long["PERIOD"] = long["PERIOD"].astype(str).str.strip()
    long["value"] = pd.to_numeric(long["value"].astype(str).str.replace(",", ".").str.replace(" ", ""),
                                  errors="coerce")
    long = long.dropna(subset=["value", "time"])
    long["time"] = long["time"].astype(int)
    return long[[f"dim{c}_name" for c in dims] + ["PERIOD", "time", "value"]].reset_index(drop=True)
