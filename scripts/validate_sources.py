"""Сверка данных СберИндекса с Росстатом и проверка качества панели по МО.

Запуск: python -m scripts.validate_sources

Печатает отчёт и сохраняет его в reports/validation.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402
from src.data import rosstat  # noqa: E402

SI_DIR = RAW_DIR / "sberindex"
out: list[str] = []


def say(line: str = "") -> None:
    print(line)
    out.append(line)


def si(name: str) -> pd.DataFrame:
    df = pd.read_csv(sorted(SI_DIR.glob(f"{name}_ru_*.csv"))[-1], sep=";")
    df["date"] = pd.to_datetime(df["period"])
    return df


def check_spending_vs_rosstat() -> None:
    say("## 1. «Потребительские расходы» СберИндекса vs Росстат (РФ, млрд руб., помесячно)")
    retail = pd.concat([rosstat.load_retail_monthly(), rosstat.load_retail_current_year()])
    retail = retail.drop_duplicates("date", keep="last").set_index("date")
    catering = pd.concat([rosstat.load_catering_monthly(), rosstat.load_catering_current_year()])
    catering = catering.drop_duplicates("date", keep="last").set_index("date")["catering"]
    services = rosstat.load_paid_services_monthly()
    services = services[services["terr_name"] == "Россия"].set_index("date")["services"]
    ros = pd.DataFrame({"food": retail["food"], "nonfood": retail["nonfood"],
                        "catering": catering, "services": services})

    s = si("consumer-spending").pivot(index="date", columns="type", values="value")
    s = s.rename(columns={"Продовольственные товары": "food", "Непродовольственные товары": "nonfood",
                          "Общественное питание": "catering", "Услуги": "services", "Всего": "total"})
    say("| компонент | месяцев | период | медиана Δ, % | макс Δ, % | мес. с Δ > 1% |")
    say("|---|---|---|---|---|---|")
    for c in ["food", "nonfood", "catering", "services"]:
        m = pd.concat([s[c], ros[c]], axis=1, keys=["si", "ros"], sort=True).dropna()
        d = ((m["si"] / m["ros"] - 1) * 100).abs()
        say(f"| {c} | {len(m)} | {m.index.min():%Y-%m}..{m.index.max():%Y-%m} | {d.median():.3f} | "
            f"{d.max():.2f} | {(d > 1).sum()} |")
    parts = s[["food", "nonfood", "catering", "services"]].sum(axis=1)
    say(f"- «Всего» = сумма 4 компонентов: макс. расхождение {(s['total'] - parts).abs().max():.2f} млрд")
    say("- Последний месяц у СберИндекса – собственная оценка, Росстат его ещё не опубликовал.")
    say()


def check_real_vs_cpi() -> None:
    say("## 2. Реальные vs номинальные приросты СберИндекса vs ИПЦ Росстата (РФ, % г/г)")
    g = si("consumer-spending-growth")
    g = g[g["type"] == "Всего"].pivot(index="date", columns="value_type", values="value")
    deflator = ((1 + g["Номинальное"] / 100) / (1 + g["Реальное"] / 100) - 1) * 100
    cpi = rosstat.load_cpi_regions()
    rf = cpi[cpi["level"] == "rf"].set_index("date")["cpi"].sort_index()
    cpi_yoy = ((rf / 100).rolling(12).apply(lambda x: x.prod(), raw=True) - 1) * 100
    m = pd.concat({"deflator_si": deflator, "cpi_yoy": cpi_yoy}, axis=1).dropna()
    d = m["deflator_si"] - m["cpi_yoy"]
    say(f"- месяцев: {len(m)} ({m.index.min():%Y-%m}..{m.index.max():%Y-%m})")
    say(f"- корреляция: {m.corr().iloc[0, 1]:.3f}; средняя разница {d.mean():+.2f} п.п., "
        f"медиана |разницы| {d.abs().median():.2f} п.п.")
    say("- Дефлятор расходов ≠ ИПЦ по определению (другие веса), ждём близость, не равенство.")
    say()


def check_mo_panel() -> None:
    say("## 3. Панель расходов по МО (СберИндекс) + население Росстата")
    p = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet")
    pop = pd.read_parquet(PROCESSED_DIR / "mo_population.parquet")
    a = p[p["category"] == "all"]
    say(f"- МО в панели: {a['oktmo'].nunique()}; с населением: {pop['pop'].notna().sum()} "
        f"({pop['pop_source'].value_counts().to_dict()})")
    say(f"- дубли (МО, дата, категория): {p.duplicated(['oktmo', 'date', 'category']).sum()}")
    n = a.groupby("oktmo").size()
    say(f"- МО с полными 24 мес.: {(n == 24).sum()} из {len(n)}")
    w = p.pivot_table(index=["oktmo", "date"], columns="category", values="value", aggfunc="first").dropna()
    sub = w[["food", "health", "horeca", "transport", "marketplaces"]].sum(axis=1)
    say(f"- сумма 5 категорий > «все»: {(sub > w['all']).mean():.2%} строк; "
        f"медианная доля 5 категорий: {(sub / w['all']).median():.1%}")
    a = a.sort_values(["oktmo", "date"])
    r = a.groupby("oktmo")["value"].pct_change().abs()
    say(f"- скачки м/м > 50%: {(r > 0.5).mean():.2%}, > 100%: {(r > 1).mean():.2%}")

    # Правдоподобие уровня: если value – расходы на жителя в месяц, то
    # Σ value·население по МО даёт безналичные траты этих МО, которые должны
    # быть заметной, но меньшей долей всех потребительских расходов РФ.
    j = a.merge(pop[["oktmo", "pop"]].dropna(), on="oktmo")
    j = j[j["date"].dt.year == 2024]
    tot = (j["value"] * j["pop"].astype(float)).groupby(j["date"]).sum() / 1e9
    s = si("consumer-spending")
    s = s[(s["type"] == "Всего") & (s["date"].dt.year == 2024)].set_index("date")["value"]
    share = (tot / s).dropna()
    pop_share = j.drop_duplicates("oktmo")["pop"].astype(float).sum() / 146_150_789
    say(f"- население МО панели: {pop_share:.1%} населения РФ на 01.01.2024")
    say(f"- Σ(value × население) / потребительские расходы РФ, 2024: "
        f"{share.min():.1%}..{share.max():.1%} (медиана {share.median():.1%})")
    say("  Гипотеза «value = безналичные траты на жителя в месяц» правдоподобна, если доля "
        "ниже доли населения, но того же порядка (часть трат – наличные и другие банки).")
    say()


def main() -> None:
    say("# Сверка источников")
    say()
    check_spending_vs_rosstat()
    check_real_vs_cpi()
    check_mo_panel()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "validation.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("validate_sources", "data", main)
