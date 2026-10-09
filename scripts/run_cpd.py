"""Сравнение детекторов структурных сдвигов по конфигу.

Запуск: python -m scripts.run_cpd --config cpd

1. Синтетика: в данные СберИндекса вставляются сдвиги уровня с известной
   датой (src/cpd/series.py::inject_shifts), repeats раз с разной разметкой.
2. Порог каждого метода – из целевой доли ложных тревог на «чистых» МО
   (per-series FAR: доля МО без сдвига, у которых за 2024 г. была хоть одна
   тревога). Так методы сравниваются честно: при одинаковом числе ложных
   тревог – кто ловит больше и быстрее.
3. Лучший онлайн-метод (recall при FAR 5%, при равенстве – задержка)
   прогоняется на реальных данных: тревоги по МО и проверка паводков 2024 г.

Результаты в reports/cpd/: comparison.csv, by_size.csv, real_events.csv,
alarms.parquet (лучший метод, реальные данные), summary.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR, load_config  # noqa: E402
from src.cpd.methods import (ONLINE, SCORES, START, pelt_alarms, pelt_ruptures,  # noqa: E402
                             segmentations)
from src.cpd.series import (apply_shifts, category_z, inject_shifts, load_levels,  # noqa: E402
                            local_growth)

ONLINE = {**ONLINE, "zscore_no_common": True, "ewma_cat": True, "zscore_cat": True,
          "ewma_mix": True, "zscore_mix": True}


def category_levels(Y: pd.DataFrame, categories: list[str]) -> dict[str, pd.DataFrame]:
    """Уровни категорий для тех же МО, что и Y; МО без полной истории или с
    нулями в категории – без пропусков: берётся уровень всех расходов (сигнал
    этой категории тогда совпадает с общим)."""
    out = {}
    for c in categories:
        Yc = load_levels(c).reindex(Y.index)
        bad = ~(Yc > 0).all(axis=1)
        Yc.loc[bad] = Y.loc[bad].to_numpy()
        out[c] = Yc
    return out


def method_input(Y: pd.DataFrame, spec: dict, cats: dict | None = None) -> np.ndarray:
    """z – вход детектора. categories в spec – сигнал по категориям
    (src/cpd/series.py::category_z); cats – их уровни (со вставленными
    сдвигами на синтетике), иначе берутся из данных."""
    if spec.get("categories"):
        zc = category_z(cats if cats is not None else category_levels(Y, spec["categories"]))
        if not spec.get("with_total"):
            return zc
        # два сигнала: все расходы и категории (score берёт больший из нормированных)
        return np.stack([method_input(Y, {}), zc])
    X, sigma = local_growth(Y, remove_common=spec.get("remove_common", True))
    return (X.to_numpy() / sigma.to_numpy()[:, None])


def input_key(spec: dict) -> tuple:
    return (spec.get("remove_common", True), tuple(spec.get("categories") or ()), bool(spec.get("with_total")))


def score(name: str, spec: dict, z: np.ndarray):
    """Сила сигнала (МО × месяц) или таблица разбиений для PELT."""
    base = spec.get("base", name)
    params = {k: v for k, v in spec.items() if k not in ("base", "remove_common", "categories", "with_total")}
    if base == "pelt":
        return segmentations(z)
    if z.ndim == 3:
        # сила каждого сигнала делится на свою медиану по всем рядо-месяцам
        # мониторинга – шкалы сравнимы, тревога – по большему. Медиана, а не
        # верхний перцентиль: вставленные сдвиги (10% МО) поднимают хвост, и
        # порог с синтетики на реальных данных давал бы в 2,5 раза больше тревог
        parts = [SCORES[base](zi, **params) for zi in z]
        return np.maximum.reduce([s / np.median(s[:, START:]) for s in parts])
    return SCORES[base](z, **params)


def alarms(base: str, s, thr: float, T: int) -> np.ndarray:
    if base == "pelt":
        # у PELT порог – штраф: больше штраф – меньше тревог
        return pelt_alarms(s, thr, T)
    return s > thr


def calibrate(base: str, runs: list, far: float, T: int) -> float:
    """Порог, при котором доля «чистых» МО с тревогой равна far."""
    if base != "pelt":
        mx = np.concatenate([r["s"][r["clean"]][:, START:].max(axis=1) for r in runs])
        return float(np.quantile(mx, 1 - far))
    lo, hi = np.log(1e-3), np.log(1e4)          # бинарный поиск по log штрафа
    for _ in range(40):
        mid = (lo + hi) / 2
        f = np.mean(np.concatenate([alarms(base, r["s"], np.exp(mid), T)[r["clean"]].any(axis=1)
                                    for r in runs]))
        lo, hi = (mid, hi) if f > far else (lo, mid)
    return float(np.exp(hi))


def calibrate_rate(base: str, runs: list, rate: float, T: int) -> float:
    """Порог, при котором на «чистых» МО тревожных месяцев – rate на 100
    рядо-месяцев."""
    q = rate / 100
    if base != "pelt":
        v = np.concatenate([r["s"][r["clean"]][:, START:].ravel() for r in runs])
        return float(np.quantile(v, 1 - q))
    lo, hi = np.log(1e-3), np.log(1e4)
    for _ in range(40):
        mid = (lo + hi) / 2
        f = np.mean(np.concatenate([alarms(base, r["s"], np.exp(mid), T)[r["clean"]][:, START:].ravel()
                                    for r in runs]))
        lo, hi = (mid, hi) if f > q else (lo, mid)
    return float(np.exp(hi))


def evaluate(base: str, runs: list, thr: float, w: int, T: int) -> pd.DataFrame:
    """По каждому вставленному сдвигу: зафиксирован ли в [τ, τ+w], задержка, ранняя тревога."""
    rows = []
    for r in runs:
        a = alarms(base, r["s"], thr, T)
        for _, e in r["lab"].iterrows():
            i, t0 = e["row"], e["tau"] - 12
            win = a[i, t0:t0 + w + 1]
            rows.append({"size": e["size"], "hit": bool(win.any()),
                         "delay": int(np.argmax(win)) if win.any() else np.nan,
                         "early": bool(a[i, START:t0].any())})
    flags = np.concatenate([alarms(base, r["s"], thr, T)[r["clean"]].any(axis=1) for r in runs])
    return pd.DataFrame(rows).assign(far=flags.mean(), fp=int(flags.sum()))


def precision_f1(ev: pd.DataFrame) -> tuple[float, float]:
    """Точность по МО: зафиксированные сдвиги / (зафиксированные + чистые МО с тревогой).
    Зависит от доли МО со сдвигом (synthetic.share), поэтому главная шкала
    сравнения – полнота при одинаковой доле ложных тревог."""
    tp = ev["hit"].sum()
    prec = tp / (tp + ev["fp"].iloc[0]) if tp + ev["fp"].iloc[0] else np.nan
    rec = ev["hit"].mean()
    return float(prec), float(2 * prec * rec / (prec + rec)) if prec + rec else np.nan


def simulate(Y, cfg, months, cat_names, cat_levels, sizes, duration=None, seed0=None) -> dict:
    """Повторы синтетики: одни и те же разметки для всех методов."""
    sc = cfg["synthetic"]
    seed0 = cfg["seed"] if seed0 is None else seed0
    runs = {m: [] for m in cfg["methods"]}
    for rep in range(sc["repeats"]):
        Y2, lab = inject_shifts(Y, sc["share"], sizes, months, seed=seed0 + rep, duration=duration)
        lab["row"] = Y.index.get_indexer(lab["oktmo"])
        clean = np.ones(len(Y), dtype=bool)
        clean[lab["row"]] = False
        # как сдвиг всех расходов проходит в категории (synthetic.category_shift):
        # same – так же; random – в каждой категории s·U(0, 2), в среднем так же;
        # total_only – только вне пяти категорий, категории не сдвигаются
        mode = sc.get("category_shift", "same")
        rng = np.random.default_rng(seed0 + 1000 + rep)
        cat_factor = {c: (np.ones(len(lab)) if mode == "same" else np.zeros(len(lab)) if mode == "total_only"
                          else rng.uniform(0, 2, len(lab))) for c in cat_names}
        cache = {}
        for name, spec in cfg["methods"].items():
            spec = spec or {}
            key = input_key(spec)
            if key not in cache:
                cats = {c: apply_shifts(cat_levels[c], lab, cat_factor[c]) for c in spec.get("categories") or ()}
                cache[key] = method_input(Y2, spec, cats or None)
            runs[name].append({"s": score(name, spec, cache[key]), "lab": lab, "clean": clean})
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpd")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from src.forecast import foundation
    ok, why = foundation.available("chronos")
    for fm in ("chronos_interval", "chronos_median"):
        if not ok and fm in cfg["methods"]:
            print(f"{fm} пропущен: {why}")
            cfg["methods"].pop(fm)
    sc = cfg["synthetic"]
    out = REPORTS_DIR / "cpd"
    out.mkdir(parents=True, exist_ok=True)

    Y = load_levels(cfg["category"])
    cols = list(Y.columns.strftime("%Y-%m"))
    months = [cols.index(m) for m in sc["months"]]
    T = len(cols) - 12
    print(f"МО: {len(Y)}, месяцев мониторинга: {T}, повторов: {sc['repeats']}")

    cat_names = sorted({c for spec in cfg["methods"].values() for c in (spec or {}).get("categories") or ()})
    cat_levels = category_levels(Y, cat_names)
    runs = simulate(Y, cfg, months, cat_names, cat_levels, sc["sizes"])

    rows, by_size, thresholds = [], [], {}
    for name, spec in cfg["methods"].items():
        base = (spec or {}).get("base", name)
        for far in cfg["far_targets"]:
            thr = calibrate(base, runs[name], far, T)
            ev = evaluate(base, runs[name], thr, sc["detect_window"], T)
            thresholds[(name, far)] = thr
            prec, f1 = precision_f1(ev)
            rows.append({"method": name, "online": ONLINE.get(name, False), "far_target": far,
                         "threshold": round(thr, 4), "far_clean": round(ev["far"].iloc[0], 3),
                         "recall": round(ev["hit"].mean(), 3), "precision": round(prec, 3), "f1": round(f1, 3),
                         "delay_months": round(ev["delay"].mean(), 2),
                         "early_alarm": round(ev["early"].mean(), 3), "n_shifts": len(ev)})
            by_size.append(ev.groupby("size")["hit"].mean().rename("recall").reset_index()
                           .assign(method=name, far_target=far))
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparison.csv", index=False)

    # Вторая шкала: ложные тревоги на 100 рядо-месяцев чистых МО
    rate_rows = []
    for name, spec in cfg["methods"].items():
        base = (spec or {}).get("base", name)
        for rate in cfg.get("far_rate_targets", []):
            thr = calibrate_rate(base, runs[name], rate, T)
            ev = evaluate(base, runs[name], thr, sc["detect_window"], T)
            rate_rows.append({"method": name, "online": ONLINE.get(name, False),
                              "false_per_100_series_months": rate, "far_clean_series": round(ev["far"].iloc[0], 3),
                              "recall": round(ev["hit"].mean(), 3), "delay_months": round(ev["delay"].mean(), 2)})
    comp_rate = pd.DataFrame(rate_rows)
    comp_rate.to_csv(out / "comparison_rate.csv", index=False)
    by_size = pd.concat(by_size)
    by_size.to_csv(out / "by_size.csv", index=False)

    # Устойчивость выбора к форме сдвига: рост вместо падения, временный шок.
    # Порог калибруется заново на каждом сценарии при той же доле ложных тревог.
    scen_rows = []
    for k, scn in enumerate(cfg.get("scenarios", []), start=1):
        sruns = simulate(Y, cfg, months, cat_names, cat_levels, scn["sizes"], scn.get("duration"),
                         seed0=cfg["seed"] + 100 * k)
        for name, spec in cfg["methods"].items():
            base = (spec or {}).get("base", name)
            thr = calibrate(base, sruns[name], cfg["main_far"], T)
            ev = evaluate(base, sruns[name], thr, sc["detect_window"], T)
            prec, f1 = precision_f1(ev)
            scen_rows.append({"scenario": scn["name"], "title": scn["title"], "method": name,
                              "online": ONLINE.get(name, False), "recall": round(ev["hit"].mean(), 3),
                              "precision": round(prec, 3), "f1": round(f1, 3),
                              "delay_months": round(ev["delay"].mean(), 2)})
        print(f"сценарий {scn['name']}: готов", flush=True)
    scen = pd.DataFrame(scen_rows)
    if len(scen):
        scen.to_csv(out / "scenarios.csv", index=False)

    main_far = cfg["main_far"]
    cand = comp[(comp["far_target"] == main_far) & comp["online"]
                & (comp["method"] != "zscore_no_common")]
    best = cand.sort_values(["recall", "delay_months"], ascending=[False, True]).iloc[0]["method"]
    print(f"лучший онлайн-метод при FAR {main_far:.0%}: {best}")

    # Реальные данные: тревоги лучшего метода и всех методов на паводках
    real_rows = []
    X_real, sigma = local_growth(Y)
    z_real = X_real.to_numpy() / sigma.to_numpy()[:, None]
    real_alarms = {}
    for name, spec in cfg["methods"].items():
        spec = spec or {}
        base = spec.get("base", name)
        z = z_real if input_key(spec) == input_key({}) else method_input(
            Y, spec, {c: cat_levels[c] for c in spec.get("categories") or ()} or None)
        s = score(name, spec, z)
        for far in cfg["far_targets"]:
            a = alarms(base, s, thresholds[(name, far)], T)
            if far == main_far:
                real_alarms[name] = (s, a)
            for e in cfg["real_events"]:
                i = Y.index.get_loc(e["oktmo"])
                t0 = cols[12:].index(e["month"])
                first = np.flatnonzero(a[i, t0:])
                real_rows.append({"method": name, "far_target": far, "oktmo": e["oktmo"], "name": e["name"],
                                  "event": e["month"],
                                  "first_alarm": cols[12 + t0 + first[0]] if len(first) else None,
                                  "caught_3m": bool(len(first) and first[0] <= sc["detect_window"]),
                                  "alarm_before": bool(a[i, START:t0].any())})
    real = pd.DataFrame(real_rows)
    real.to_csv(out / "real_events.csv", index=False)
    real_sum = real[real["far_target"] == main_far].groupby("method")["caught_3m"].sum()         .rename("real_caught").reset_index()
    flagged = pd.DataFrame([{"method": m, "mo_flagged_2024": int(a.any(axis=1).sum())}
                            for m, (_, a) in real_alarms.items()])

    s, a = real_alarms[best]
    # x – локальный рост всех расходов (в долях, понятен без пояснений);
    # z – сигнал, который видел лучший метод (у методов по категориям – их сигнал)
    spec_best = cfg["methods"][best] or {}
    z_best = z_real if input_key(spec_best) == input_key({}) else method_input(
        Y, spec_best, {c: cat_levels[c] for c in spec_best.get("categories") or ()} or None)
    if z_best.ndim == 3:
        z_best = z_best[0]            # два сигнала: в таблицу – сигнал всех расходов
    long = pd.DataFrame({
        "oktmo": np.repeat(Y.index.to_numpy(), T),
        "date": np.tile(pd.to_datetime(cols[12:]).to_numpy(), len(Y)),
        "x": X_real.to_numpy().ravel(), "z": z_best.ravel(),
        "score": s.ravel(), "alarm": a.ravel()})
    long["method"] = best
    long["threshold"] = thresholds[(best, main_far)]
    long.to_parquet(out / "alarms.parquet", index=False)

    # Сверка точного PELT с ruptures на части реальных рядов
    seg = segmentations(z_real[:200])
    pen = thresholds[("pelt", main_far)]
    k = np.argmin(seg[0] + pen * np.arange(seg[0].shape[1]), axis=1)
    same = np.mean([sorted(int(b) for b in seg[1][i, k[i]] if b >= 0) == pelt_ruptures(z_real[i], pen)
                    for i in range(200)])

    main_tab = comp[comp["far_target"] == main_far].merge(real_sum, on="method").merge(flagged, on="method")
    main_tab = main_tab.sort_values(["online", "recall"], ascending=[False, False])
    md = ["# Сравнение детекторов структурных сдвигов", "",
          f"МО с полной историей: {len(Y)}; мониторинг: {cols[12]}…{cols[-1]}; "
          f"синтетических сдвигов: {int(main_tab['n_shifts'].iloc[0])} "
          f"({sc['repeats']} разметок × {sc['share']:.0%} МО, размеры {sc['sizes']}).", "",
          f"## Главная таблица (ложные тревоги на чистых МО: {main_far:.0%})", "",
          main_tab.drop(columns=["far_target"]).to_markdown(index=False), "",
          f"**Лучший онлайн-метод: {best}.**", "",
          "## Другая шкала: ложные тревоги на 100 рядо-месяцев", "",
          "far_clean_series – доля чистых МО, "
          "у которых при этом пороге за год была хотя бы одна тревога.", "",
          (comp_rate.pivot_table(index="method", columns="false_per_100_series_months", values="recall")
           .to_markdown(floatfmt=".2f") if len(comp_rate) else "–"), "",
          f"## Другие формы сдвига (ложные тревоги {main_far:.0%}, порог калибруется заново)", "",
          (pd.concat([comp[comp["far_target"] == main_far].assign(title="падение (основной)")
                      [["title", "method", "recall", "f1"]],
                      scen[["title", "method", "recall", "f1"]] if len(scen) else None])
           .pivot_table(index="method", columns="title", values=["recall", "f1"], sort=False)
           .to_markdown(floatfmt=".2f") if len(scen) else "–"), "",
          "precision – зафиксированные сдвиги / (зафиксированные + чистые МО с тревогой) при 10% МО со сдвигом; "
          "f1 – среднее гармоническое precision и recall.", "",
          "## Доля зафиксированных сдвигов по размеру", "",
          by_size.pivot_table(index=["method", "far_target"], columns="size", values="recall")
                 .to_markdown(floatfmt=".2f"), "",
          "## Реальные паводки апреля 2024 г.", "",
          real.assign(first_alarm=real["first_alarm"].fillna("–"))
              .pivot_table(index=["far_target", "name"], columns="method", values="first_alarm",
                           aggfunc="first").to_markdown(), "",
          f"Точный PELT (перебор разбиений) совпал с ruptures на {same:.0%} из 200 реальных рядов. "
          "На случайных рядах изредка бывают расхождения, и в них у перебора целевая функция "
          "ниже, т. е. его решение точнее (отсечение в ruptures при min_size не всегда точно)."]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(main_tab.to_string(index=False))
    print(real.groupby(["method", "far_target"])["caught_3m"].sum().unstack().to_string())


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("run_cpd", "cpd", main)
