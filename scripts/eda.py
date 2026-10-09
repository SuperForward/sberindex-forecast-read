"""EDA панели расходов по МО: отчёт reports/eda.md и графики reports/figures/.

Запуск: python -m scripts.eda
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from src.data import rosstat  # noqa: E402

FIG = REPORTS_DIR / "figures"
CATS = ["all", "food", "health", "horeca", "transport", "marketplaces"]
CAT_RU = {"all": "все", "food": "продовольствие", "health": "здоровье", "horeca": "общепит",
          "transport": "транспорт", "marketplaces": "маркетплейсы"}
TYPE_RU = {"go": "городской округ", "mr": "мун. район", "mo": "мун. округ", "vt": "внутригор. терр."}
out: list[str] = []


def say(line: str = "") -> None:
    print(line)
    out.append(line)


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=110)
    plt.close()
    say(f"![{name}](figures/{name})")
    say()


def load() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet")
    pop = pd.read_parquet(PROCESSED_DIR / "mo_population.parquet")[["oktmo", "pop", "pop_urban"]]
    mp = pd.read_csv(INTERIM_DIR / "mo_oktmo_map.csv", dtype=str)[["mo_name", "mo_type"]]
    p = p.merge(pop, on="oktmo", how="left").merge(mp, on="mo_name", how="left")
    p["pop"] = p["pop"].astype(float)
    p["urban_share"] = p["pop_urban"].astype(float) / p["pop"]
    return p


def section_coverage(p: pd.DataFrame) -> None:
    say("## 1. Покрытие")
    a = p[p["category"] == "all"]
    n = a.groupby("oktmo").size()
    say(f"- МО: {a['oktmo'].nunique()}, регионов: {a['region_code'].nunique()}, "
        f"месяцев: {a['date'].nunique()} ({a['date'].min():%Y-%m}..{a['date'].max():%Y-%m})")
    say(f"- полные 24 мес.: {(n == 24).sum()}; только 2023: "
        f"{a.groupby('oktmo')['date'].max().lt('2024-01-01').sum()}; только 2024: "
        f"{a.groupby('oktmo')['date'].min().ge('2024-01-01').sum()}")
    t = a.drop_duplicates("oktmo")["mo_type"].map(TYPE_RU).value_counts()
    say("- виды МО: " + ", ".join(f"{k} {v}" for k, v in t.items()))
    say()


def section_aggregate(p: pd.DataFrame) -> None:
    say("## 2. Динамика: средние расходы на жителя (взвешено населением)")
    full = p.groupby("oktmo").filter(lambda g: g["date"].nunique() == 24).dropna(subset=["pop"])
    agg = full.groupby(["date", "category"]).apply(
        lambda g: np.average(g["value"], weights=g["pop"]), include_groups=False).unstack()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    agg["all"].plot(ax=ax[0], marker="o", color="#21a038")
    ax[0].set_title("Все категории, руб./жителя/мес.")
    for c in CATS[1:]:
        (agg[c] / agg[c].iloc[:12].mean()).plot(ax=ax[1], label=CAT_RU[c])
    ax[1].set_title("Категории, к среднему 2023")
    ax[1].legend(fontsize=8)
    savefig("aggregate.png")
    g = agg.loc[agg.index.year == 2024].mean() / agg.loc[agg.index.year == 2023].mean() - 1
    say("- рост 2024 к 2023 (среднее за год): " + ", ".join(f"{CAT_RU[c]} {g[c]:+.1%}" for c in CATS))
    dec = agg["all"] / agg["all"].shift(1) - 1
    say(f"- декабрь к ноябрю: 2023 {dec.loc['2023-12-01']:+.1%}, 2024 {dec.loc['2024-12-01']:+.1%}; "
        f"январь к декабрю: 2024 {dec.loc['2024-01-01']:+.1%}")
    # сверка агрегата с Росстатом (вся РФ): совпадение динамики м/м
    ros = pd.concat([rosstat.load_retail_monthly()]).set_index("date")["retail"]
    both = pd.concat([agg["all"].pct_change(), ros.pct_change()], axis=1, keys=["si", "ros"], sort=True).dropna()
    say(f"- корреляция м/м изменений с оборотом розницы Росстата (РФ): {both.corr().iloc[0, 1]:.2f}")
    say()


def section_seasonality(p: pd.DataFrame) -> None:
    say("## 3. Сезонность")
    full = p[p["category"] == "all"].groupby("oktmo").filter(lambda g: len(g) == 24).copy()
    full["rel"] = full["value"] / full.groupby(["oktmo", full["date"].dt.year])["value"].transform("mean")
    prof = full.groupby([full["date"].dt.year, full["date"].dt.month])["rel"].median().unstack(0)
    prof.plot(marker="o", figsize=(7, 4), title="Медианный сезонный профиль (к среднему года)")
    plt.xlabel("месяц")
    savefig("seasonality.png")
    corr = prof.corr().iloc[0, 1]
    say(f"- профили 2023 и 2024 совпадают: корреляция {corr:.2f}. Пик – декабрь "
        f"({prof.loc[12].mean():.2f}), минимум – {prof.mean(axis=1).idxmin()}-й месяц "
        f"({prof.mean(axis=1).min():.2f}).")
    say("- Годовая сезонность видна и устойчива, но в каждом ряду всего 2 цикла: "
        "оценивать её надо по всем МО сразу, а не по одному ряду.")
    say()


def section_cross_section(p: pd.DataFrame) -> None:
    say("## 4. Различия между МО")
    a = p[(p["category"] == "all") & (p["date"].dt.year == 2024)]
    lvl = a.groupby(["oktmo", "mo_type"])["value"].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    lvl["value"].plot.hist(bins=60, ax=ax[0], color="#21a038")
    ax[0].set_title("Средние расходы на жителя в 2024, руб./мес.")
    lvl.boxplot(column="value", by="mo_type", ax=ax[1])
    ax[1].set_title("По видам МО")
    plt.suptitle("")
    savefig("cross_section.png")
    q = lvl["value"].quantile([0.05, 0.5, 0.95])
    say(f"- разброс уровней: 5% {q[0.05]:,.0f}, медиана {q[0.5]:,.0f}, 95% {q[0.95]:,.0f} руб. "
        f"(в {q[0.95] / q[0.05]:.1f} раза) – нужна нормировка при общей модели по всем МО")
    say("- медиана по видам: " + ", ".join(
        f"{TYPE_RU.get(k, k)} {v:,.0f}" for k, v in lvl.groupby("mo_type")["value"].median().items()))
    j = a.groupby("oktmo").agg(v=("value", "mean"), u=("urban_share", "first"), pop=("pop", "first")).dropna()
    say(f"- корреляция уровня с долей городского населения: {j['v'].corr(j['u']):.2f}, "
        f"с log(население): {j['v'].corr(np.log(j['pop'])):.2f}")
    sh = p[p["date"].dt.year == 2024].pivot_table(index="oktmo", columns="category", values="value", aggfunc="mean")
    shares = sh[CATS[1:]].div(sh["all"], axis=0).median()
    say("- медианные доли категорий: " + ", ".join(f"{CAT_RU[c]} {v:.0%}" for c, v in shares.items()))
    say()


def section_growth_and_shocks(p: pd.DataFrame) -> None:
    say("## 5. Рост и кандидаты в шоки")
    a = p[p["category"] == "all"].groupby("oktmo").filter(lambda g: len(g) == 24).sort_values(["oktmo", "date"])
    a["yoy"] = a.groupby("oktmo")["value"].pct_change(12)
    y = a[a["date"].dt.year == 2024]
    avg = y.groupby("oktmo")["yoy"].mean()
    avg.plot.hist(bins=60, figsize=(7, 4), title="Средний рост г/г в 2024 по МО", color="#21a038")
    savefig("yoy_growth.png")
    q = avg.quantile([0.05, 0.5, 0.95])
    say(f"- рост г/г в 2024: медиана {q[0.5]:+.1%}, 5% {q[0.05]:+.1%}, 95% {q[0.95]:+.1%}")
    # Общая по стране составляющая: медианный рост г/г по всем МО в месяце.
    common = y.groupby("date")["yoy"].median()
    say("- медианный рост г/г по всем МО, 2024: " + ", ".join(
        f"{d:%m} {v:+.1%}" for d, v in common.items()))
    say(f"  Выпадает {common.idxmin():%Y-%m} ({common.min():+.1%} при типичных "
        f"{common.median():+.1%}) – сдвиг общий для всех МО, а не локальный шок "
        "(причину – календарь, база января 2023 или методика – ещё надо проверить); "
        "детектор должен отделять общий фактор от локального.")
    # Локальные кандидаты: рост г/г за вычетом общего, в единицах разброса
    # самого МО (MAD).
    y = y.copy()
    y["idio"] = y["yoy"] - y["date"].map(common)
    med = y.groupby("oktmo")["idio"].transform("median")
    mad = y.groupby("oktmo")["idio"].transform(lambda s: (s - s.median()).abs().median() * 1.4826 + 1e-9)
    y["z"] = (y["idio"] - med) / mad
    top = y.loc[y["z"].abs().nlargest(12).index, ["mo_name", "region", "date", "yoy", "idio", "z"]]
    say("- локальные отклонения (рост г/г минус общий, |z| по MAD), топ-12:")
    say()
    say("| МО | регион | месяц | г/г | за вычетом общего | z |")
    say("|---|---|---|---|---|---|")
    for r in top.itertuples():
        say(f"| {r.mo_name} | {r.region} | {r.date:%Y-%m} | {r.yoy:+.1%} | {r.idio:+.1%} | {r.z:+.1f} |")
    say()
    share = y.groupby("date")["z"].apply(lambda s: (s.abs() > 3).mean())
    say(f"- доля МО с локальным |z| > 3: медиана {share.median():.1%} в месяц, макс {share.max():.1%} "
        f"({share.idxmax():%Y-%m})")
    say()


def main() -> None:
    p = load()
    say("# EDA: безналичные потребительские расходы по МО (СберИндекс, 2023–2024)")
    say()
    section_coverage(p)
    section_aggregate(p)
    section_seasonality(p)
    section_cross_section(p)
    section_growth_and_shocks(p)
    (REPORTS_DIR / "eda.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("eda", "data", main)
