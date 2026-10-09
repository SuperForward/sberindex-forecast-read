"""БД показателей муниципальных образований Росстата (БД ПМО).

https://rosstat.gov.ru/dbscripts/munst/munstXX/DBInet.cgi – отдельная база на
каждый субъект. Протокол (восстановлен по JS страницы):

1. POST pl=<код показателя> -> страница выбора: массивы p_<измерение>[i]
   с кодами и <SELECT NAME=<измерение>> с подписями в том же порядке.
2. POST Qry=<изм>:<коды>;... QryGm=<раскладка> Format=CSV YearFrom/YearTo
   -> CSV: блоки «заголовок / строка лет / строки МО».

Один показатель на запрос: у показателей свои доп. измерения (возраст,
ОКВЭД, вид миграции), их перекрёстное произведение роняет сервер (500/504).
По доп. измерениям берётся «Всего», по периоду – годовое значение.

Запросы идут через системный curl: сертификат rosstat.gov.ru выдан
российским УЦ, которого нет в certifi; проверка сертификата не отключается.
"""

import re
import subprocess
from urllib.parse import urlencode

import pandas as pd

from src.config import NO_WINDOW, RAW_DIR

PMO_DIR = RAW_DIR / "rosstat" / "pmo"
URL = "https://rosstat.gov.ru/dbscripts/munst/{base}/DBInet.cgi"
ROOT_URL = "https://rosstat.gov.ru/dbscripts/munst/"
KEEP_ALL = {"Pokazateli", "munr", "tippos", "oktmo", "god"}


def _curl(url: str, data: dict | None = None, timeout: int = 300) -> bytes:
    cmd = ["curl", "-sS", "-f", "-A", "Mozilla/5.0", "--max-time", str(timeout)]
    inp = None
    if data is not None:
        inp = urlencode({k: str(v).encode("cp1251") for k, v in data.items()}).encode()
        cmd += ["-H", "Content-Type: application/x-www-form-urlencoded", "--data-binary", "@-"]
    r = subprocess.run(cmd + [url], input=inp, capture_output=True, creationflags=NO_WINDOW)
    if r.returncode:
        raise RuntimeError(f"{url}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def list_bases(timeout: int = 300) -> list[str]:
    """munst01, munst03, … – базы субъектов из листинга каталога."""
    html = _curl(ROOT_URL, timeout=timeout).decode("utf-8", errors="replace")
    return sorted(set(re.findall(r"(munst\d{2})", html)))


def available(timeout: int = 30) -> tuple[bool, str]:
    """Быстрая проверка, что БД ПМО отвечает: каталог и одна база.
    Без неё при упавшем сервере загрузка висит часами (1700 запросов × таймаут)."""
    try:
        bases = list_bases(timeout=timeout)
        if not bases:
            return False, "каталог баз пуст"
        page = _curl(URL.format(base=bases[0]), timeout=timeout).decode("cp1251", errors="replace")
        if "недоступна" in page or "SQL-сервер" in page:
            return False, "сервер БД ПМО отвечает «База данных недоступна»"
        return True, f"баз: {len(bases)}"
    except RuntimeError as e:
        return False, str(e)[:200]


def _selection(base: str, code: str) -> tuple[dict, dict]:
    """Коды и подписи всех измерений для показателя."""
    page = _curl(URL.format(base=base), {"pl": code}).decode("cp1251")
    codes: dict[str, list[str]] = {}
    for name, val in re.findall(r'p_(\w+)\[\d+\]="([^"]*)"', page):
        codes.setdefault(name, []).append(val)
    labels = {}
    for name in codes:
        m = re.search(r'<SELECT NAME="?%s"?[^>]*>(.*?)</SELECT>' % re.escape(name), page, re.S | re.I)
        labels[name] = [t.strip() for t in re.findall(r"<OPTION[^>]*>([^<\r\n]*)", m.group(1), re.I)] if m else []
    return codes, labels


def _pick_total(codes: list[str], labels: list[str], dim: str) -> list[str]:
    """Одно значение доп. измерения: «Всего»/годовое, иначе первое."""
    pats = [r"январь-декабрь|за год|^год"] if dim == "period" else [r"^всего", r"всего", r"итого"]
    for pat in pats:
        for c, t in zip(codes, labels):
            if re.search(pat, t, re.I):
                return [c]
    return codes[:1]


def fetch_indicator(base: str, code: str, year_from: int, year_to: int) -> tuple[bytes, pd.DataFrame]:
    """CSV показателя по всем МО базы + справочник код ОКТМО -> подпись."""
    codes, labels = _selection(base, code)
    if "oktmo" not in codes:
        raise RuntimeError(f"{base}/{code}: нет измерения oktmo")
    years = sorted(int(y) for y in labels.get("god", []) if y.strip().isdigit())
    if years:  # диапазон вне имеющихся лет сервер отвергает с 500
        year_from, year_to = max(year_from, years[0]), min(year_to, years[-1])
        if year_from > year_to:
            raise RuntimeError(f"{base}/{code}: нет лет в диапазоне (есть {years[0]}–{years[-1]})")
    sel = {d: (v if d in KEEP_ALL else _pick_total(v, labels.get(d, []), d)) for d, v in codes.items()}
    # только МО верхнего уровня: 8-значный ОКТМО с нулевой поселенческой
    # частью; поселения (11 знаков) раздувают запрос в десятки раз
    top = [c for c in codes["oktmo"] if len(c.zfill(8)) == 8 and c.zfill(8)[5:8] == "000"]
    sel["oktmo"] = top or codes["oktmo"]
    qry = "".join(f"{d}:{','.join(v)};" for d, v in sel.items())
    head = [d for d in sel if d not in ("oktmo", "god")]
    gm = "".join(f"{d}_z:{i + 1};" for i, d in enumerate(head)) + "oktmo_b:1;god_s:1;"
    data = {"Format": "CSV", "YearFrom": year_from, "YearTo": year_to, "DiagSz": "800x600",
            "Qry": qry, "QryGm": gm, "QryFootNotes": ";",
            "YearsList": ";".join(codes.get("god", [])) + ";", "tbl": "Показать таблицу"}
    raw = _curl(URL.format(base=base), data, timeout=600)
    ref = pd.DataFrame({"oktmo_code": codes["oktmo"],
                        "oktmo_label": labels.get("oktmo", [""] * len(codes["oktmo"]))[:len(codes["oktmo"])]})
    return raw, ref


def parse_csv(raw: bytes) -> pd.DataFrame:
    """CSV БД ПМО -> длинная таблица: header, label, year, value.

    Название показателя из заголовка не берём: в нём бывают запятые, а
    поля заголовка тоже разделены запятыми. Название – из конфига по коду.
    """
    text = raw.decode("cp1251", errors="replace").replace("\r", "")
    rows = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines) < 3:
            continue
        head = lines[0].strip()
        years = [y for y in lines[1].split(";")[1:] if y.strip()]
        for ln in lines[2:]:
            cells = ln.split(";")
            label = cells[0].strip()
            for y, v in zip(years, cells[1:]):
                v = v.strip().replace(",", ".").replace(" ", "")
                if v and v not in ("-", "…", "..."):
                    rows.append((head, label, int(y), v))
    df = pd.DataFrame(rows, columns=["header", "label", "year", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def load_pmo(names: dict[str, str] | None = None) -> pd.DataFrame:
    """Все скачанные показатели: oktmo (8 знаков), base, code, indicator, year, value.

    names: код -> название (из configs/pmo_indicators.yaml).

    Коды БД ПМО – ОКТМО без ведущего нуля; оставляем только МО верхнего
    уровня (поселенческая часть кода нулевая).
    """
    frames = []
    for csv in sorted(PMO_DIR.glob("munst*/*.csv")):
        if csv.name.endswith("_ref.csv"):
            continue
        ref = pd.read_csv(csv.with_name(csv.stem + "_ref.csv"), dtype=str)
        df = parse_csv(csv.read_bytes())
        if df.empty:
            continue
        # подпись строки CSV = подпись МО в справочнике базы; при повторах
        # подписей (тёзки внутри субъекта) код не восстановить – такие строки выпадают
        uniq = ref.drop_duplicates("oktmo_label", keep=False)
        df = df.merge(uniq, left_on="label", right_on="oktmo_label", how="inner")
        df["code"] = csv.stem
        df["base"] = csv.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["oktmo"] = out["oktmo_code"].str.zfill(8)
    out = out[(out["oktmo"].str.len() == 8) & (out["oktmo"].str[5:8] == "000")]
    out["indicator"] = out["code"].map(names or {}).fillna(out["code"])
    return out[["oktmo", "base", "code", "indicator", "year", "value"]]

