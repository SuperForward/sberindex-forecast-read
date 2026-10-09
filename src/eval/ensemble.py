"""Сравнение моделей между собой и ансамбли.

Ансамбль помогает тем сильнее, чем меньше похожи ошибки моделей, поэтому
считаем и корреляцию ошибок, и MAE самих ансамблей.

Веса подбираются на ранних окнах, а метрика считается на последнем: окон
всего три, и веса, подобранные на всех, «подсмотрят» ответ.
"""

import itertools

import numpy as np
import pandas as pd

from src.eval.metrics import mae


def error_matrix(pred: pd.DataFrame) -> pd.DataFrame:
    """Ошибки (yhat − y) по одинаковому набору точек: строка – точка, столбец – модель."""
    e = pred.assign(err=pred["yhat"] - pred["y"])
    wide = e.pivot_table(index=["series_id", "origin", "date"], columns="model", values="err")
    return wide.dropna()


KEYS = ["series_id", "origin", "date", "h"]


def holdout_selection(pred: pd.DataFrame, known_until: str, eval_from: str,
                      max_members: int = 12) -> tuple[pd.DataFrame, list[str]]:
    """Честная проверка ансамбля: состав (равные веса, модель можно взять
    повторно) выбирается жадно только по точкам с датой ≤ known_until – это всё,
    что известно к прогнозу из eval_from; качество – на окнах с origin ≥ eval_from.
    Возвращает MAE всех одиночных моделей и ансамбля на отложенных окнах."""
    solo = pred[~pred["model"].str.startswith("ens_")]
    wide = solo.pivot_table(index=["series_id", "origin", "date"], columns="model", values="yhat").dropna()
    y = solo.drop_duplicates(["series_id", "origin", "date"]).set_index(["series_id", "origin", "date"])["y"]
    y = y.reindex(wide.index).to_numpy()
    dates = wide.index.get_level_values("date")
    sel_m = np.asarray(dates <= pd.Timestamp(known_until))
    late = np.asarray(wide.index.get_level_values("origin") >= pd.Timestamp(eval_from))
    err = lambda yhat, m: float(np.mean(np.abs(yhat[m] - y[m])))
    cols = list(wide.columns)
    X = wide.to_numpy()
    chosen, best, total = [], np.inf, np.zeros(len(wide))
    for _ in range(max_members):
        cand = [(c, err((total + X[:, i]) / (len(chosen) + 1), sel_m)) for i, c in enumerate(cols)]
        c, sc = min(cand, key=lambda t: t[1])
        if sc >= best - 0.5:
            break
        chosen.append(c)
        total += X[:, cols.index(c)]
        best = sc
    ens = total / len(chosen)
    rows = [{"model": c, "MAE_holdout": err(X[:, i], late), "MAE_select": err(X[:, i], sel_m)}
            for i, c in enumerate(cols)]
    rows.append({"model": "holdout_ensemble", "MAE_holdout": err(ens, late), "MAE_select": best})
    out = pd.DataFrame(rows).sort_values("MAE_holdout")
    out["n_holdout"], out["n_select"] = int(late.sum()), int(sel_m.sum())
    return out, chosen


def _greedy(X: np.ndarray, y: np.ndarray, m: np.ndarray, max_members: int = 12) -> list[int]:
    """Жадный состав с равными весами (как в holdout_selection) по точкам m."""
    chosen, best, total = [], np.inf, np.zeros(len(X))
    for _ in range(max_members):
        cand = [(i, float(np.mean(np.abs((total[m] + X[m, i]) / (len(chosen) + 1) - y[m]))))
                for i in range(X.shape[1])]
        i, sc = min(cand, key=lambda t: t[1])
        if sc >= best - 0.5:
            break
        chosen.append(i)
        total += X[:, i]
        best = sc
    return chosen


def rolling_selection(pred: pd.DataFrame, first_eval: int = 2) -> pd.DataFrame:
    """Скользящая проверка способов собрать прогноз из моделей.

    Для каждого окна, начиная с first_eval-го, способ видит только точки с
    датой ≤ origin этого окна (то, что известно к прогнозу), и оценивается
    на прогнозах из этого origin. Так сравниваются не составы, а процедуры:
    жадный выбор, лучшая модель по прошлому, top-k, взвешивание по прошлой
    MAE и среднее/медиана всех моделей без выбора. holdout_selection делает
    то же с одной точкой разреза; одна точка – одно случайное окно.
    Возвращает MAE по окнам (столбцы) и в среднем (mean) для каждого способа
    и одиночной модели."""
    solo = pred[~pred["model"].str.startswith("ens_")]
    wide = solo.pivot_table(index=KEYS, columns="model", values="yhat").dropna()
    y = solo.drop_duplicates(KEYS).set_index(KEYS)["y"].reindex(wide.index).to_numpy()
    cols, X = list(wide.columns), wide.to_numpy()
    org = wide.index.get_level_values("origin")
    dates = wide.index.get_level_values("date")
    rows = []
    for o in sorted(org.unique())[first_eval:]:
        known, test = np.asarray(dates <= o), np.asarray(org == o)
        past = np.array([np.mean(np.abs(X[known, i] - y[known])) for i in range(len(cols))])
        rank = np.argsort(past)
        inv = 1 / past[rank[:5]]
        ways = {"среднее всех моделей": X.mean(axis=1),
                "медиана всех моделей": np.median(X, axis=1),
                "жадный выбор по прошлому": X[:, _greedy(X, y, known)].mean(axis=1),
                "лучшая модель по прошлому": X[:, rank[0]],
                "среднее 3 лучших по прошлому": X[:, rank[:3]].mean(axis=1),
                "среднее 5 лучших по прошлому": X[:, rank[:5]].mean(axis=1),
                "5 лучших, веса 1/MAE": (X[:, rank[:5]] * inv).sum(axis=1) / inv.sum()}
        ways |= {f"модель {c}": X[:, i] for i, c in enumerate(cols)}
        rows += [{"way": k, "origin": o, "MAE": float(np.mean(np.abs(v[test] - y[test])))}
                 for k, v in ways.items()]
    out = pd.DataFrame(rows).pivot_table(index="way", columns="origin", values="MAE")
    out.columns = [f"{c:%Y-%m}" for c in out.columns]
    out["mean"] = out.mean(axis=1)
    return out.sort_values("mean").reset_index()


def _wide(pred: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Прогнозы моделей по общим точкам + факт. Ключ – наблюдение, не значение
    факта: у глобальных моделей оно проходит через логарифм и обратно."""
    sub = pred[pred["model"].isin(models)]
    wide = sub.pivot_table(index=KEYS, columns="model", values="yhat").dropna()
    y = sub.drop_duplicates(KEYS).set_index(KEYS)["y"].reindex(wide.index)
    return wide, y


def blend(pred: pd.DataFrame, models: list[str], weights: np.ndarray, name: str) -> pd.DataFrame:
    """Взвешенное среднее прогнозов в виде обычной таблицы прогнозов."""
    wide, y = _wide(pred, models)
    out = wide.assign(model=name, y=y, yhat=(wide[models] * weights).sum(axis=1))
    return out.reset_index()[KEYS + ["y", "model", "yhat"]]


def fit_weights(pred: pd.DataFrame, models: list[str], fit_origins: list) -> np.ndarray:
    """Веса из сетки с шагом 0.1 (для пары) по MAE на ранних окнах.

    Сетка, а не аналитическая формула: минимизируем MAE, а не квадрат
    ошибки, и веса держим неотрицательными с суммой 1.
    """
    wide, y_s = _wide(pred[pred["origin"].isin(fit_origins)], models)
    if wide.empty:
        return np.full(len(models), 1 / len(models))
    y = y_s.to_numpy(float)
    best, best_mae = None, np.inf
    steps = np.arange(0, 1.01, 0.1)
    grid = (itertools.product(steps, repeat=len(models) - 1) if len(models) > 1 else [()])
    for head in grid:
        tail = 1 - sum(head)
        if tail < -1e-9:
            continue
        w = np.array([*head, max(tail, 0.0)])
        m = mae(y, (wide[models].to_numpy() * w).sum(axis=1))
        if m < best_mae:
            best, best_mae = w, m
    return best


def ensemble_report(pred: pd.DataFrame, out_dir) -> pd.DataFrame | None:
    """Корреляция ошибок + ансамбли. Возвращает прогнозы ансамблей."""
    models = sorted(pred["model"].unique())
    if len(models) < 2:
        return None
    err = error_matrix(pred)
    if err.empty:
        return None
    err.corr().round(3).to_csv(out_dir / "error_correlation.csv")

    origins = sorted(pred["origin"].unique())
    fit_origins, test_origin = origins[:-1], origins[-1]
    # Ранжируем по ранним окнам: выбрать «лучшую» модель по последнему окну
    # значило бы подсмотреть ответ там, где ансамбль потом проверяется.
    early = pred[pred["origin"].isin(fit_origins)]
    single = {m: mae(*early[early["model"] == m][["y", "yhat"]].to_numpy().T) for m in models}
    ranked = sorted(models, key=lambda m: single[m])

    rows, blends = [], {}
    for a, b in itertools.combinations(models, 2):
        w = fit_weights(pred, [a, b], fit_origins)
        eq = blend(pred, [a, b], np.array([0.5, 0.5]), f"ens_eq[{a}+{b}]")
        wt = blend(pred, [a, b], w, f"ens_w[{a}+{b}]")
        # MAE одиночных моделей – на тех же точках, что и ансамбль, иначе
        # сравнение идёт по разным выборкам
        wide, y = _wide(pred, [a, b])
        last = wide.index.get_level_values("origin") == test_origin
        on_test = wt[wt["origin"] == test_origin]
        rows.append({
            "models": f"{a} + {b}", "corr_err": round(float(err[a].corr(err[b])), 3),
            "MAE_a": round(mae(y.to_numpy(), wide[a].to_numpy())),
            "MAE_b": round(mae(y.to_numpy(), wide[b].to_numpy())),
            "MAE_ens_50_50": round(mae(*eq[["y", "yhat"]].to_numpy().T)),
            "weight_a": round(float(w[0]), 2),
            "MAE_ens_weighted_last_origin": round(mae(*on_test[["y", "yhat"]].to_numpy().T))
            if len(on_test) else None,
            "MAE_best_single_last_origin": round(min(
                mae(y[last].to_numpy(), wide.loc[last, m].to_numpy()) for m in (a, b)))
            if last.any() else None,
        })
        blends[(a, b)] = (eq, wt, w)

    pd.DataFrame(rows).sort_values("MAE_ens_50_50").to_csv(out_dir / "ensembles.csv", index=False)

    # В таблицу метрик кладём не все 40+ комбинаций, а две: лучшую пару и
    # тройку лучших моделей. И пара, и тройка выбраны по ранним окнам.
    keep = []
    if blends:
        best_pair = min(blends, key=lambda k: mae(
            *blends[k][1][blends[k][1]["origin"].isin(fit_origins)][["y", "yhat"]].to_numpy().T))
        eq, wt, w = blends[best_pair]
        a, b = best_pair
        keep.append(wt.assign(model=f"ens_w[{a}+{b}]"))
    if len(ranked) >= 3:
        keep.append(blend(pred, ranked[:3], np.full(3, 1 / 3), "ens_eq[top3]"))
    if not keep:
        return None
    res = pd.concat(keep, ignore_index=True)
    res["category"] = res["series_id"].str.split("|").str[1]
    return res
