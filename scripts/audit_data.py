"""Аудит скачанных данных: полнота и целостность.

Запуск: python -m scripts.audit_data

- data/raw/MANIFEST.csv – каждый файл: путь, размер, sha256 (фиксирует версию
  данных для воспроизводимости).
- reports/data_audit.md – проверки по источникам.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402
from src.data import rosstat  # noqa: E402
from src.data.oktmo import load_oktmo  # noqa: E402
from src.data.sberindex import load_spending  # noqa: E402

out: list[str] = []
issues: list[str] = []


def say(line: str = "") -> None:
    print(line)
    out.append(line)


def issue(text: str) -> None:
    issues.append(text)
    say(f"- ⚠ {text}")


def month_gaps(dates: pd.Series) -> list[str]:
    # к началу месяца: часть рядов датирована концом месяца (31.01)
    d = pd.DatetimeIndex(sorted(pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp().unique()))
    full = pd.date_range(d.min(), d.max(), freq="MS")
    return [f"{x:%Y-%m}" for x in full.difference(d)]


def manifest() -> None:
    say("## 0. Файлы и манифест")
    rows = []
    for f in sorted(RAW_DIR.rglob("*")):
        if not f.is_file() or f.name == "MANIFEST.csv":
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append({"path": f.relative_to(RAW_DIR).as_posix(), "bytes": f.stat().st_size, "sha256": h})
    m = pd.DataFrame(rows)
    m.to_csv(RAW_DIR / "MANIFEST.csv", index=False)
    say(f"- файлов: {len(m)}, объём {m['bytes'].sum() / 1e6:.1f} МБ → data/raw/MANIFEST.csv")
    empty = m[m["bytes"] < 1024]
    if len(empty):
        issue(f"подозрительно маленькие файлы (<1 КБ): {empty['path'].tolist()}")
    dup = m[m["sha256"].duplicated(keep=False)]
    if len(dup):
        say(f"- одинаковые по содержимому: {dup['path'].tolist()}")
    bad = []
    for z in RAW_DIR.rglob("*.zip"):
        try:
            if zipfile.ZipFile(z).testzip() is not None:
                bad.append(z.name)
        except zipfile.BadZipFile:
            bad.append(z.name)
    if bad:
        issue(f"битые zip: {bad}")
    else:
        say("- все zip-архивы целые")
    say()


def audit_sberindex_mo() -> None:
    say("## 1. СберИндекс: расходы по МО")
    raw = pd.read_csv(sorted(RAW_DIR.glob("potrebitelskie-*.csv"))[-1], sep=";")
    pq = pd.read_parquet(sorted((RAW_DIR / "sberindex").glob("potrebitelskie-*.parquet"))[-1])
    same = len(raw) == len(pq) and (raw["value"].to_numpy() == pq["value"].to_numpy()).all()
    say(f"- CSV {len(raw)} строк, parquet {len(pq)} строк, значения совпадают: {same}")
    if not same:
        issue("CSV и parquet выгрузки по МО расходятся")
    say(f"- пропуски value: {raw['value'].isna().sum()}, value ≤ 0: {(raw['value'] <= 0).sum()}, "
        f"obs_status: {raw['obs_status'].unique().tolist()}")
    s = load_spending()
    say(f"- месяцы {s['date'].min():%Y-%m}..{s['date'].max():%Y-%m}, пропусков в календаре: "
        f"{month_gaps(s['date']) or 'нет'}")
    n = s.groupby(["mo_name", "category"])["date"].nunique().unstack()
    incons = (n.nunique(axis=1) > 1).sum()
    say(f"- у МО разное число месяцев в разных категориях: {incons} МО")
    g = s.groupby(["mo_name", "category"])["date"].agg(["min", "max", "nunique"])
    span = ((g["max"].dt.year - g["min"].dt.year) * 12 + g["max"].dt.month - g["min"].dt.month + 1)
    holes = g[(span > g["nunique"])]
    say(f"- ряды с дырами внутри (не только обрезанные с краёв): {len(holes)} из {len(g)}")
    if len(holes):
        issue(f"{len(holes)} рядов по МО с пропущенными месяцами внутри периода")
    m = pd.read_csv(INTERIM_DIR / "mo_oktmo_map.csv", dtype=str)
    vc = m["match_type"].value_counts()
    bad = vc.drop(["territory_id", "exact", "normalized", "stem", "fuzzy", "by_neighbors"], errors="ignore")
    say(f"- сопоставление с ОКТМО: {vc.to_dict()}")
    if bad.sum():
        issue(f"без кода ОКТМО {bad.sum()} названий МО ({bad.to_dict()}); тёзки не разделимы без кода МО")
    say()


def audit_sberindex_national() -> None:
    say("## 2. СберИндекс: общероссийские ряды")
    say()
    say("| датасет | строк | период | частота | пропуски в календаре | дубли | NaN |")
    say("|---|---|---|---|---|---|---|")
    for f in sorted((RAW_DIR / "sberindex").glob("*_ru_*.csv")):
        df = pd.read_csv(f, sep=";")
        d = pd.to_datetime(df["period"])
        dims = [c for c in df.columns if c not in ("period", "value") and df[c].nunique() > 1]
        dup = df.duplicated(["period"] + dims).sum()
        freq = df["freq"].iloc[0] if "freq" in df else "?"
        if freq == "Месяц":
            gaps = month_gaps(d)
        else:
            step = d.drop_duplicates().sort_values().diff().dt.days.value_counts()
            gaps = [f"шаги {step.to_dict()}"] if len(step) > 1 else []
        name = f.name.split("_ru_")[0]
        say(f"| {name} | {len(df)} | {d.min():%Y-%m-%d}..{d.max():%Y-%m-%d} | {freq} | "
            f"{len(gaps) if isinstance(gaps, list) and gaps and not str(gaps[0]).startswith('шаги') else (gaps[0] if gaps else 0)} | "
            f"{dup} | {df['value'].isna().sum()} |")
        if dup:
            issue(f"{name}: {dup} дублей")
    say()
    say("- «Инфляция с сезонной корректировкой» не скачана (API 404); ИПЦ взят у Росстата.")
    say()


def audit_rosstat() -> None:
    say("## 3. Росстат")
    ok = {v: load_oktmo(v) for v in ["20230112", "20241227", "20260901"]}
    say("- ОКТМО: " + ", ".join(f"{v}: {len(d)} МО, {d['region_code'].nunique()} регионов"
                                for v, d in ok.items()))
    for y in (2023, 2024):
        try:
            p = rosstat.load_population_mo(y) if y == 2024 else rosstat.load_population_mo_by_name(y)
            say(f"- население МО {y}: {len(p)} МО, всего {int(p['pop'].sum()):,} чел.")
        except Exception as e:  # noqa: BLE001
            issue(f"население {y} не читается: {e}")
    cpi = rosstat.load_cpi_regions()
    r = cpi[cpi["level"] == "region"]
    per = r.groupby("date")["terr_code"].nunique()
    say(f"- ИПЦ по регионам: {r['date'].min():%Y-%m}..{r['date'].max():%Y-%m}, пропусков месяцев: "
        f"{month_gaps(r['date']) or 'нет'}, субъектов в месяц {per.min()}..{per.max()}, "
        f"NaN ИПЦ: {r['cpi'].isna().sum()}")
    out_of_range = r[(r["cpi"] < 95) | (r["cpi"] > 110)]
    if len(out_of_range):
        say(f"- ИПЦ вне 95..110% м/м: {len(out_of_range)} значений (проверить вручную)")
    ret = rosstat.load_retail_monthly()
    say(f"- розница РФ: {ret['date'].min():%Y-%m}..{ret['date'].max():%Y-%m}, пропусков: "
        f"{month_gaps(ret['date']) or 'нет'}; food+nonfood=total: "
        f"{((ret['food'] + ret['nonfood'] - ret['retail']).abs() < 1).mean():.1%}")
    cat = rosstat.load_catering_monthly()
    say(f"- общепит РФ: {cat['date'].min():%Y-%m}..{cat['date'].max():%Y-%m}, пропусков: "
        f"{month_gaps(cat['date']) or 'нет'}")
    cr = rosstat.load_catering_regions_monthly()
    say(f"- общепит регионы: {cr['terr_name'].nunique()} территорий, {cr['date'].min():%Y-%m}.."
        f"{cr['date'].max():%Y-%m}, NaN {cr['catering'].isna().sum()}, отрицательных {(cr['catering'] < 0).sum()}")
    sv = rosstat.load_paid_services_monthly()
    per = sv.groupby("date")["terr_name"].nunique()
    say(f"- платные услуги: {sv['date'].nunique()} мес. {sv['date'].min():%Y-%m}..{sv['date'].max():%Y-%m}, "
        f"пропусков: {month_gaps(sv['date']) or 'нет'}, территорий в месяц {per.min()}..{per.max()}")
    rf = sv[sv["terr_name"] == "Россия"].set_index("date")["services"]
    parts = sv[sv["terr_name"].str.contains("федеральный округ")].groupby("date")["services"].sum()
    d = ((parts / rf - 1).abs() * 100).dropna()
    say(f"- услуги: сумма ФО vs Россия, макс. расхождение {d.max():.2f}%")
    if d.max() > 1:
        issue("платные услуги: сумма федеральных округов ≠ Россия более чем на 1%")
    say()


def audit_processed() -> None:
    say("## 4. Обработанные данные")
    p = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet")
    pop = pd.read_parquet(PROCESSED_DIR / "mo_population.parquet")
    say(f"- панель: {len(p)} строк, {p['oktmo'].nunique()} МО, дублей "
        f"{p.duplicated(['oktmo', 'date', 'category']).sum()}")
    say(f"- население: {pop['pop'].notna().sum()} из {len(pop)} МО")
    miss = pop[pop["pop"].isna()]["mo_name"].tolist()
    if miss:
        issue(f"нет населения: {miss}")
    say()


def main() -> None:
    say("# Аудит данных")
    say()
    manifest()
    audit_sberindex_mo()
    audit_sberindex_national()
    audit_rosstat()
    audit_processed()
    say("## Итого замечаний")
    say()
    for i in issues:
        say(f"- {i}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_audit.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("audit_data", "data", main)
