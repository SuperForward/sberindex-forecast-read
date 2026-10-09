"""Графики методологического отчёта: docs/figures/*.png (+ схема архитектуры .svg).

Запуск: python -m scripts.build_report_figures

Данные – те же функции src/app_service, что у приложения и лендинга, и
результаты null-теста (reports/null_test/models/runs.csv), так что цифры на
графиках совпадают с таблицами отчёта. Перезапускать после пересчёта моделей.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import app_service as svc  # noqa: E402
from src.config import REPORTS_DIR, ROOT  # noqa: E402

OUT = ROOT / "docs" / "figures"
GREEN, RED, BLUE, VIOLET, AMBER, GREY = "#21a038", "#d6334a", "#2f7fd8", "#7a64e8", "#c98a00", "#8a93a0"
plt.rcParams.update({"font.family": "Segoe UI" if sys.platform == "win32" else "DejaVu Sans",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8e4", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})


def rub(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def short(name: str) -> str:
    return name.replace(" (Google)", "").replace(" (Amazon)", "").replace("Ансамбль (веса): ", "Ансамбль: ")


def mo_short(name: str) -> str:
    """Короткое имя МО, как в лендинге: «городской округ город Орск» → «Орск»."""
    import re
    name = re.sub(r"^городской округ (город )?", "", name)
    name = re.sub(r" муниципальный (район|округ)$", " р-н", name)
    return re.sub(r"^внутригородская территория города федерального значения ", "", name)


def fig_models(M: dict) -> None:
    rows = sorted(M["models"], key=lambda m: m["MAE"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    col = [GREEN if m["model"] == M["best"] else RED if m["model"].startswith("prophet") else
           AMBER if m["tuned"] else VIOLET if m["model"].startswith(("chronos", "timesfm")) else BLUE for m in rows]
    y = range(len(rows))
    ax.barh(y, [m["MAE"] for m in rows], color=col)
    ax.set_yticks(list(y), [short(m["name"]) for m in rows])
    ax.invert_yaxis()
    for i, m in enumerate(rows):
        ax.text(m["MAE"] + 40, i, rub(m["MAE"]), va="center", fontsize=8.5)
    ax.set_xlabel(f"MAE, ₽ на жителя в месяц ({len(M['origins'])} окон × 1–{M['horizon']} мес. × "
                  f"{rub(M['n_series'])} МО)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Сравнение моделей прогноза", loc="left", fontweight="bold")
    fig.text(0.01, -0.02, "зелёный: итоговый прогноз, красный: Prophet, фиолетовый: нейросети, "
             "жёлтый: ансамбли (состав выбран по этим окнам)", fontsize=8, color=GREY)
    fig.savefig(OUT / "models_mae.png")
    plt.close(fig)


def fig_horizons(H: dict, best: str) -> None:
    pick = {"catboost": "#1b7a8c", "random_forest": "#5fb36b", "chronos": VIOLET,
            "timesfm": "#b58cf0", "baseline": BLUE, "prophet_default": RED, "naive": GREY,
            "national_growth": AMBER, best: GREEN}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = [str(h) for h in H["horizons"]]
    for m in H["models"]:
        if m["model"] not in pick:
            continue
        ys = [m["mae"].get(h, m["mae"].get(int(h))) for h in xs]
        ax.plot(xs, [float("nan") if v is None else v for v in ys], marker="o", color=pick[m["model"]],
                lw=2.6 if m["model"] in (best, "prophet_default") else 1.6, label=short(m["name"]))
    ax.set_xlabel("горизонт, месяцев (общие целевые месяцы 07–12.2024)")
    ax.set_ylabel("MAE, ₽")
    ax.legend(ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("Точность по горизонтам", loc="left", fontweight="bold")
    fig.savefig(OUT / "horizons_mae.png")
    plt.close(fig)


# Цвет детектора: выбранный – зелёный, Chronos – фиолетовые, офлайн PELT –
# янтарный, остальные – свой цвет каждому, чтобы линии различались по легенде.
CPD_OTHER = ["#2f7fd8", "#0f9fb0", "#e0702a", "#b5498f", "#5c6bc0", "#6d8f1f", "#8a93a0", "#c2410c", "#0e7490"]


def _cpd_colors(C: dict) -> dict:
    rest = iter(CPD_OTHER)
    return {m["method"]: GREEN if m["method"] == C["best"] else VIOLET if m["method"] == "chronos_median"
            else "#c4a8f5" if m["method"].startswith("chronos") else AMBER if not m["online"]
            else next(rest, BLUE) for m in C["methods"]}


def fig_cpd(C: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    col = _cpd_colors(C)
    far = [str(f) for f in C["far_targets"]]
    for m in C["methods"]:
        c = col[m["method"]]
        ls = "--" if m["method"] == "zscore_no_common" else "-"
        a1.plot([float(f) * 100 for f in far], [m["recall"][f] * 100 for f in far], marker="o", ls=ls, color=c,
                lw=2.6 if m["method"] == C["best"] else 1.3, label=m["name"])
    a1.set_xlabel("доля ложных тревог, %")
    a1.set_ylabel("зафиксировано сдвигов, %")
    a1.set_title("Полнота при одинаковых ложных тревогах", loc="left", fontweight="bold", fontsize=10)
    # легенда под графиками: внутри поля она закрывала линии
    fig.legend(*a1.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    sizes = sorted(C["methods"][0]["by_size"], key=float)
    w = 0.8 / len(C["methods"])
    for i, m in enumerate(C["methods"]):
        c = col[m["method"]]
        a2.bar([j + i * w for j in range(len(sizes))], [m["by_size"][s] * 100 for s in sizes], w,
               color=c, alpha=1 if m["method"] == C["best"] else 0.55)
    a2.set_xticks([j + 0.4 - w / 2 for j in range(len(sizes))], [f"провал {abs(float(s)) * 100:.0f}%" for s in sizes])
    a2.set_title("По размеру сдвига (5% ложных тревог; цвета как слева)", loc="left", fontweight="bold", fontsize=10)
    a2.grid(axis="x", visible=False)
    fig.savefig(OUT / "cpd_recall.png")
    plt.close(fig)


def fig_null() -> bool:
    f = REPORTS_DIR / "null_test" / "models" / "runs.csv"
    if not f.exists():
        return False
    r = pd.read_csv(f)
    names = {"random_forest": "Случайный лес", "xgboost": "XGBoost", "lightgbm": "LightGBM",
             "catboost": "CatBoost", "hist_gbm": "HistGB", "ridge": "Ridge"}
    sc = {"real": ("настоящие данные", GREEN), "noise_features": ("признаки заменены шумом", GREY),
          "shuffled_target": ("цель перемешана", AMBER), "shuffled_within_month": ("перемешана внутри месяца", RED)}
    g = r.groupby(["model", "scenario"])["MAE"].mean().unstack()
    models = [m for m in names if m in g.index]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for i, (s, (lab, c)) in enumerate(sc.items()):
        if s in g:
            ax.bar([j + i * w for j in range(len(models))], [g.loc[m, s] for m in models], w, color=c, label=lab)
    ax.set_xticks([j + 1.5 * w for j in range(len(models))], [names[m] for m in models])
    ax.axhline(1239, color=BLUE, ls="--", lw=1)
    ax.text(-0.45, 1239, "baseline", color=BLUE, va="bottom", fontsize=8.5)
    ax.set_ylabel("MAE, ₽")
    ax.set_ylim(0, g.max().max() * 1.2)          # место под легенду
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    n = int(r.loc[r.scenario != "real", "seed"].nunique())
    ax.set_title(f"Null-тест: модели на настоящих и случайных данных ({n} повторов)", loc="left", fontweight="bold")
    fig.savefig(OUT / "null_test.png")
    plt.close(fig)
    return True


def fig_mo(oktmo: str, fname: str, best: str) -> None:
    D, F, Cm = svc.mo_detail(oktmo), svc.mo_forecast(oktmo), svc.cpd_mo(oktmo)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.8), gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.42},
                                 sharex=False)
    s = pd.DataFrame(D["series"]["all"]).assign(time=lambda d: pd.to_datetime(d["time"]))
    a1.plot(s["time"], s["value"], color="#15191e", lw=2.2, marker="o", ms=3, label="факт")
    # для сравнения – другая сильная модель: CatBoost, а если лучший он сам, то Chronos на факторе
    other = "chronos_base_rel_reg" if best == "catboost" else "catboost"
    for m, c, ls in ((best, GREEN, "-"), ("prophet_default", RED, "--"), (other, VIOLET, "--")):
        if m not in F["models"]:
            continue
        # окна идут каждый месяц и перекрываются – берём прогноз на 1 месяц из каждого окна
        pts = [w["points"][0] for w in F["models"][m]["windows"] if w["points"]]
        p = pd.DataFrame(pts).assign(time=lambda d: pd.to_datetime(d["time"])).sort_values("time")
        mae = sum(w["mae"] for w in F["models"][m]["windows"]) / len(F["models"][m]["windows"])
        a1.plot(p["time"], p["value"], color=c, ls=ls, lw=1.8, label=f"{short(F['models'][m]['name'])} (MAE {rub(mae)})")
    for e in D.get("events", []):
        a1.axvline(pd.Timestamp(e["date"][:7] + "-01"), color=AMBER, ls=":", lw=2)
    a1.set_ylabel("₽ на жителя в месяц")
    fig.text(0.01, -0.01, "Прогнозы на 1 месяц вперёд из каждого окна; MAE по всем окнам и горизонтам.",
             fontsize=8, color=GREY)
    a1.legend(fontsize=8.5, frameon=False, ncol=2)
    a1.set_title(f"{mo_short(D['info']['mo_name'])}, муниципалитет {D['info']['region_short']}", loc="left",
                 fontweight="bold")
    if not Cm.get("error"):
        c = pd.DataFrame(Cm["score"]).assign(time=lambda d: pd.to_datetime(d["time"]))
        a2.bar(c["time"], c["value"], width=20, color=[RED if v > Cm["threshold"] else BLUE for v in c["value"]])
        a2.axhline(Cm["threshold"], color=RED, ls="--", lw=1)
        a2.set_ylim(0, max(c["value"].max(), Cm["threshold"]) * 1.15)
        a2.set_ylabel("сигнал детектора")
        a2.set_title(f"Сигнал детектора: {Cm['name']}", loc='left', fontsize=10, color='#555', pad=6)
    a1.set_xlim(s["time"].min() - pd.Timedelta(days=15), s["time"].max() + pd.Timedelta(days=15))
    a2.set_xlim(a1.get_xlim())
    fig.savefig(OUT / fname)
    plt.close(fig)


ARCH_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 430" font-family="Segoe UI, DejaVu Sans, sans-serif">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6b7480"/></marker></defs>
<rect width="1160" height="430" fill="#ffffff"/>
<g fill="#ffffff" stroke="#d5d9d3" stroke-width="1.5">
<rect x="10" y="20" width="220" height="390" rx="10"/><rect x="285" y="20" width="400" height="390" rx="10" stroke="#21a038"/>
<rect x="305" y="68" width="170" height="70" rx="8"/><rect x="495" y="68" width="170" height="70" rx="8"/>
<rect x="305" y="158" width="170" height="70" rx="8"/><rect x="495" y="158" width="170" height="70" rx="8"/>
<rect x="740" y="140" width="170" height="150" rx="10"/>
<rect x="978" y="40" width="172" height="80" rx="8"/><rect x="978" y="175" width="172" height="80" rx="8"/><rect x="978" y="310" width="172" height="80" rx="8"/>
</g>
<g stroke="#6b7480" stroke-width="1.5" fill="none" marker-end="url(#ah)">
<path d="M232 215 H282"/><path d="M687 215 H737"/><path d="M912 180 C945 180 945 80 975 80"/><path d="M912 215 H975"/><path d="M912 250 C945 250 945 350 975 350"/>
</g>
<g font-size="14" font-weight="700" fill="#15191e">
<text x="30" y="50">Источники</text><text x="305" y="50">Воркер: расписание и пересчёт</text>
<text x="318" y="94">Загрузка</text><text x="508" y="94">Данные</text><text x="318" y="184">Модели</text><text x="508" y="184">Сдвиги и нейросети</text>
<text x="758" y="172">DuckDB</text><text x="994" y="72">Приложение</text><text x="994" y="207">REST API</text><text x="994" y="342">Веб-версия</text>
</g>
<g font-size="12" fill="#6b7480">
<text x="30" y="80">СберИндекс</text><text x="30" y="104">Росстат БД ПМО, ЕМИСС</text><text x="30" y="128">Банк России</text>
<text x="30" y="152">Календарь, погода</text><text x="30" y="176">Вакансии</text><text x="30" y="200">Новости Lenta.ru, GDELT</text>
<text x="30" y="240">проверка каждые 6 ч,</text><text x="30" y="260">качаем только новое</text>
<text x="318" y="116">исходные файлы</text><text x="508" y="116">таблица МО, признаки</text>
<text x="318" y="206">проверка, горизонты</text><text x="508" y="206">детекторы, Chronos, TimesFM</text>
<text x="305" y="262">• пересчитывается только то, что изменилось</text><text x="305" y="286">• сначала проверка качества, потом публикация</text>
<text x="305" y="310">• при ошибке остаются прошлые результаты</text><text x="305" y="334">• все настройки моделей лежат в YAML</text>
<text x="305" y="358">• запускается сам, по расписанию</text>
<text x="758" y="196">готовые</text><text x="758" y="216">результаты</text><text x="758" y="248">и журнал запусков</text>
<text x="994" y="96">PySide6 / Windows</text><text x="994" y="231">FastAPI, /docs</text><text x="994" y="366">GitHub Pages</text>
</g></svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    M, H, C = svc.model_metrics(), svc.horizons_overview(), svc.cpd_overview()
    fig_models(M)
    fig_horizons(H, M["best"])
    fig_cpd(C)
    made = ["models_mae", "horizons_mae", "cpd_recall"]
    if fig_null():
        made.append("null_test")
    fig_mo("53723000", "case_orsk.png", M["best"])
    made.append("case_orsk")
    if C.get("flagged"):
        fig_mo(C["flagged"][0]["oktmo"], "case_flagged.png", M["best"])
        made.append("case_flagged")
    (OUT / "architecture.svg").write_text(ARCH_SVG, encoding="utf-8")
    print("готово:", ", ".join(made), "+ architecture.svg ->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_report_figures", "data", main)
