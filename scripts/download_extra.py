"""Дополнительные источники по configs/extra_sources.yaml.

Запуск: python -m scripts.download_extra [--only emiss,cbr,trudvsem,fns,treasury] [--ids 57039,59577]

- emiss:    региональные показатели ЕМИСС -> data/raw/emiss/<id>_<name>.xml.gz
- cbr:      файлы ЦБ -> data/raw/cbr/<name>.xlsx
- fns:      форма ФНС 1-НМ по субъектам, помесячно -> data/raw/fns/1nm_<ГГГГ-ММ-01>.zip
            (дата – «по состоянию на», данные нарастающим итогом с начала года)
- treasury: отчёт Казначейства 0503317 (консолидированные бюджеты субъектов,
            с разбивкой по типам местных бюджетов) -> data/raw/treasury/0503317_<ГГГГ-ММ-01>.parquet
- trudvsem: срез всех вакансий «Работы России» (официальная ежедневная выгрузка,
            читается потоком) -> data/raw/trudvsem/vacancies_<дата>.parquet
            Хранится только обезличенное: регион, адрес и координаты места работы,
            зарплата, сфера, отрасль, размер работодателя, даты. Контакты и
            реквизиты работодателя не читаются.

Уже скачанное не перекачивается (кроме вакансий: срез за сегодняшнюю дату).

ЕМИСС автоматически отдаёт только срез по умолчанию – часто один-два года.
Длинный ряд сайт выдаёт лишь на выгрузку с выбранными фильтрами, а её
делают со страницы показателя. Такой ряд скачивается вручную на https://fedstat.ru/indicator/<id>
(фильтры – годы и регионы, формат SDMX или Excel) и кладётся как
data/raw/emiss/<id>_<name>.xml (.xls): ручной файл не перезаписывается,
build_extra берёт его вместо автоматического.
"""

import argparse
import gzip
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import RAW_DIR, load_config  # noqa: E402


def _curl(url: str, timeout: int = 120) -> bytes:
    r = subprocess.run(["curl", "-sS", "-f", "-L", "-A", "Mozilla/5.0", "--max-time", str(timeout), url],
                       capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace").strip())
    return r.stdout


# ------------------------------------------------------------------ ЕМИСС

MAX_BYTES = 400_000_000      # полная выгрузка больше – не качаем
REFRESH_DAYS = 25            # файл старше – перекачать (ЕМИСС обновляется помесячно)
MIN_YEARS = 3                # ряд короче – предупреждение: нужна ручная выгрузка


def download_emiss(specs: list[dict], only_ids: set[int] | None) -> None:
    from src.data import emiss
    out_dir = RAW_DIR / "emiss"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if only_ids and spec["id"] not in only_ids:
            continue
        out = out_dir / f"{spec['id']}_{spec['name']}.xml.gz"
        manual = out_dir / f"{spec['id']}_{spec['name']}.xml"
        found = next((m for m in (manual, manual.with_suffix(".xls")) if m.exists()), None)
        if found:
            print(f"ЕМИСС {spec['id']} {spec['name']}: есть ручная выгрузка {found.name} – автоматическую "
                  f"не качаю", flush=True)
            continue
        if out.exists() and time.time() - out.stat().st_mtime < REFRESH_DAYS * 86400:
            print(f"ЕМИСС {spec['id']} {spec['name']}: свежий ({out.stat().st_size / 1e6:.1f} МБ)")
            continue
        t0 = time.time()
        # Полная выгрузка кнопкой «Скачать»: выгрузку со срезами сайт отдаёт
        # только со страницы показателя. Срезы
        # (keep, year_from) применяются уже при разборе, в build_extra.
        try:
            raw = emiss.download(spec["id"], max_bytes=MAX_BYTES)
        except RuntimeError as e:
            if "Maximum file size exceeded" in str(e) or "(63)" in str(e):
                print(f"ЕМИСС {spec['id']} {spec['name']}: полная выгрузка больше {MAX_BYTES / 1e6:.0f} МБ – "
                      f"пропускаю", flush=True)
                continue
            raise
        df = emiss.parse_sdmx(raw)          # сразу проверяем, что это данные, а не пустышка
        if df.empty:
            raise RuntimeError(f"ЕМИСС {spec['id']}: выгрузка пустая")
        from src.io_util import write_if_changed
        write_if_changed(out, gzip.compress(raw, mtime=0))   # mtime=0: одинаковые данные -> одинаковый архив
        print(f"ЕМИСС {spec['id']} {spec['name']}: {len(raw) / 1e6:.1f} МБ, наблюдений {len(df)}, "
              f"{df['time'].min()}–{df['time'].max()} -> {out.name} ({out.stat().st_size / 1e6:.1f} МБ), "
              f"{time.time() - t0:.0f} с", flush=True)
        years = df["time"].astype(int)
        if years.max() - years.min() + 1 < MIN_YEARS:
            print(f"ЕМИСС {spec['id']} {spec['name']}: ВНИМАНИЕ – только {years.min()}–{years.max()}, "
                  f"автоматическая выгрузка отдаёт срез по умолчанию. Длинный ряд – вручную: "
                  f"https://fedstat.ru/indicator/{spec['id']} -> {manual.relative_to(RAW_DIR.parent.parent)}",
                  flush=True)


# ------------------------------------------------------------------ ЦБ

def download_cbr(specs: list[dict]) -> None:
    out_dir = RAW_DIR / "cbr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        out = out_dir / f"{spec['name']}.xlsx"
        raw = _curl(spec["url"])
        if raw[:2] != b"PK":
            raise RuntimeError(f"ЦБ {spec['name']}: пришёл не xlsx")
        from src.io_util import write_if_changed
        changed = write_if_changed(out, raw)
        print(f"ЦБ {spec['name']}: {len(raw) / 1e6:.2f} МБ -> {out.name}{'' if changed else ' (без изменений)'}")


# ------------------------------------------------------------------ ФНС

def _report_date(name: str) -> str | None:
    """1nm010126reg.zip -> 2026-01-01; 1nm01122025reg.zip -> 2025-12-01."""
    import re
    m = re.search(r"1nm_?01(\d{2})(\d{4}|\d{2})", name)
    if not m:
        return None
    y = int(m.group(2)) if len(m.group(2)) == 4 else 2000 + int(m.group(2))
    return f"{y}-{m.group(1)}-01"


def download_fns(cfg: dict) -> None:
    import re
    out_dir = RAW_DIR / "fns"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = _curl(cfg["index_url"]).decode("utf-8")
    anchor = index.find("7707329152-nmpn")
    if anchor < 0:
        raise RuntimeError("ФНС: на странице форм не найдена строка 1-НМ (7707329152-nmpn)")
    from urllib.parse import urljoin
    # ссылки на годы относительные: href="16031004/">2025
    found = re.findall(r'href="\s*([^"]*?\d+/)"[^>]*>\s*(\d{4})', index[anchor:anchor + 6000])
    pages = {}
    for u, y in found:              # дальше идут строки других форм с теми же годами – берём первую
        if int(y) in pages:
            break
        pages[int(y)] = urljoin(cfg["index_url"], u.strip())
    pages = {y: u for y, u in pages.items() if y >= cfg.get("year_from", 2021)}
    print(f"ФНС 1-НМ: страницы годов {sorted(pages)}", flush=True)
    have, new = 0, 0
    for year, url in sorted(pages.items()):
        page = _curl(url).decode("utf-8")
        # архив по субъектам: *reg*.zip; в 10.2023 он назван 1nm011023ut.zip
        links = re.findall(r'href="([^"]+\.zip)"', page, re.I)
        dates = {}
        for lk in links:
            d = _report_date(lk.rsplit("/", 1)[-1])
            if d and (d not in dates or "reg" in lk.lower()):
                dates[d] = lk
        got = sorted(dates)
        expected = [f"{year + (m == 13)}-{(m - 1) % 12 + 1:02d}-01" for m in range(2, 14)]
        missing = [d for d in expected if d not in dates and d <= f"{date.today():%Y-%m}-01"]
        print(f"ФНС 1-НМ {year}: отчётов {len(got)}" + (f", нет на даты {missing}" if missing else ""), flush=True)
        for d, lk in dates.items():
            out = out_dir / f"1nm_{d}.zip"
            if out.exists():
                have += 1
                continue
            raw = _curl(lk if lk.startswith("http") else "https://www.nalog.gov.ru" + lk)
            if raw[:2] != b"PK":
                raise RuntimeError(f"ФНС 1-НМ {d}: пришёл не zip ({lk})")
            out.write_bytes(raw)
            new += 1
            print(f"  1-НМ на {d}: {len(raw) / 1e6:.1f} МБ -> {out.name}", flush=True)
    print(f"ФНС 1-НМ: новых {new}, уже было {have}", flush=True)


# ------------------------------------------------------------------ вакансии

# Поля ежедневной выгрузки «Работы России», которые сохраняем. Контакты
# (contactPerson, contactList), реквизиты и почта работодателя (company,
# fullCompanyName) – не читаются вовсе.
DUMP_FIELDS = ["id", "stateRegionCode", "regionName", "vacancyAddress", "geo", "salaryMin", "salaryMax",
               "professionalSphereName", "industryBranchName", "companyBusinessSize", "busyType",
               "scheduleType", "workPlaces", "datePublished", "creationDate", "dateModify", "deleted",
               "status"]


def download_trudvsem(cfg: dict) -> None:
    """Официальная ежедневная выгрузка всех вакансий (~2 ГБ) читается потоком:
    целиком на диск не пишется, сохраняется только обезличенная выжимка.
    Через API так нельзя: он отдаёт не больше 10 000 вакансий на запрос и
    ~9 с на страницу."""
    out_dir = RAW_DIR / "trudvsem"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"vacancies_{date.today():%Y-%m-%d}.parquet"
    t0 = time.time()
    proc = subprocess.Popen(["curl", "-sS", "-f", "-L", "-A", "Mozilla/5.0", "--max-time", "7200", cfg["dump_url"]],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parts, n = [], 0
    try:
        for chunk in pd.read_csv(proc.stdout, sep="|", usecols=DUMP_FIELDS, dtype=str, chunksize=100_000,
                                 on_bad_lines="skip", encoding="utf-8", encoding_errors="replace"):
            parts.append(chunk)
            n += len(chunk)
            print(f"  прочитано {n} вакансий, {time.time() - t0:.0f} с", flush=True)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        code = proc.wait()
    if code:
        raise RuntimeError(f"выгрузка вакансий оборвалась (curl {code}): {err}")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    geo = df["geo"].str.extract(r'"latitude"\s*:\s*"?([\d.]+).*?"longitude"\s*:\s*"?([\d.]+)')
    df["lat"], df["lng"] = pd.to_numeric(geo[0], errors="coerce"), pd.to_numeric(geo[1], errors="coerce")
    for c in ("salaryMin", "salaryMax", "workPlaces"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["geo"])
    df.to_parquet(out, index=False)
    print(f"вакансии: {len(df)} -> {out.name} ({out.stat().st_size / 1e6:.1f} МБ на диске, {time.time() - t0:.0f} с)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="emiss,cbr,trudvsem,fns,treasury")
    ap.add_argument("--ids", default="", help="только эти показатели ЕМИСС")
    args = ap.parse_args()
    cfg = load_config("extra_sources")
    only = set(args.only.split(","))
    ids = {int(x) for x in args.ids.split(",") if x} or None
    if "emiss" in only:
        download_emiss(cfg["emiss"], ids)
    if "cbr" in only:
        download_cbr(cfg["cbr"])
    if "trudvsem" in only:
        download_trudvsem(cfg["trudvsem"])
    if "fns" in only:
        download_fns(cfg["fns"])


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("download_extra", "data", main)
