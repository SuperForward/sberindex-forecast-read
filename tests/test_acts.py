"""Разбор официальных актов о ЧС (src/news/acts.py)."""

import pandas as pd

from src.news.acts import kind, parse, region_monthly


def test_kind():
    assert kind("О введении на территории Оренбургской области режима чрезвычайной ситуации") == "intro"
    assert kind("Об отмене на территории Курганской области режима чрезвычайной ситуации") == "cancel"
    assert kind("О предоставлении выплаты гражданам, пострадавшим в результате чрезвычайной ситуации") == "support"
    assert kind("О внесении изменений в постановление о введении режима чрезвычайной ситуации") != "intro"


def test_parse_region_and_cause():
    items = [{"id": "1", "documentDate": "2024-04-04T00:00:00", "publishDateShort": "2024-04-05T00:00:00",
              "name": "О введении режима чрезвычайной ситуации в лесах",
              "complexName": "Указ Губернатора Оренбургской области от 04.04.2024 № 103-ук\n\"О введении\""}]
    a = parse(items)
    assert a.loc[0, "kind"] == "intro"
    assert a.loc[0, "cov_region"] == "53"
    assert "fire" in a.loc[0, "cause"]


def test_region_monthly_active_until_cancel():
    acts = pd.DataFrame({"cov_region": ["53", "53"], "kind": ["intro", "cancel"],
                         "date": pd.to_datetime(["2024-04-04", "2024-06-10"]), "id": ["a", "b"]})
    months = pd.date_range("2024-03-01", "2024-08-01", freq="MS")
    rm = region_monthly(acts, months).set_index("month")
    assert rm.loc["2024-03-01", "chs_active"] == 0
    assert rm.loc["2024-04-01", "chs_intro"] == 1
    assert rm.loc[["2024-04-01", "2024-05-01", "2024-06-01"], "chs_active"].tolist() == [1, 1, 1]
    assert rm.loc["2024-07-01", "chs_active"] == 0


def test_acts_known_only_after_publication():
    """Признак модели: акт виден с месяца публикации, режим – до отмены,
    опубликованной к t, или default_len мес. – без будущей отмены."""
    from src.forecast.global_model import acts_region_known
    acts = pd.DataFrame({"cov_region": ["53", "53"], "kind": ["intro", "cancel"],
                         "date": pd.to_datetime(["2024-03-29", "2024-08-10"]),
                         "published": pd.to_datetime(["2024-04-02", "2024-08-12"]), "cause": ["", ""]})
    months = pd.date_range("2024-03-01", "2024-09-01", freq="MS")
    k = acts_region_known(acts, months, default_len=3).set_index("t")
    assert k.loc["2024-03-01", "chs_intro"] == 0          # подписан в марте, вышел в апреле
    assert k.loc["2024-04-01", "chs_intro"] == 1
    assert list(k["chs_active"]) == [0, 1, 1, 1, 0, 0, 0]  # отмена в августе к июлю ещё не известна


def test_event_forms_use_only_past():
    from src.forecast.global_model import event_forms
    n = pd.DataFrame({"oktmo": "1", "t": pd.date_range("2024-01-01", periods=6, freq="MS"),
                      "news_x": [0, 2, 0, 0, 4, 0]})
    f = event_forms(n, "oktmo", ["news_x"], "flag").set_index("t")
    assert list(f["news_x"]) == [0, 1, 0, 0, 1, 0]
    assert list(f["news_x_prev"]) == [0, 0, 1, 1, 0, 1]
    d = event_forms(n, "oktmo", ["news_x"], "dev").set_index("t")
    assert pd.isna(d.loc["2024-03-01", "news_x_dev"])           # норма – минимум 3 прошлых месяца
    assert d.loc["2024-05-01", "news_x_dev"] == 4 - 2 / 4      # среднее за январь–апрель
    l6 = event_forms(n, "oktmo", ["news_x"], "lag6").set_index("t")
    assert l6.loc["2024-05-01", "news_x_prev3_5"] == 2         # февраль


def test_apply_shifts_factor_per_shift():
    """Сдвиг в категории: s·f с τ, до τ ряд не меняется."""
    from src.cpd.series import apply_shifts
    Y = pd.DataFrame([[100.0] * 4, [50.0] * 4], index=["a", "b"],
                     columns=pd.date_range("2024-01-01", periods=4, freq="MS"))
    lab = pd.DataFrame({"oktmo": ["a", "b"], "tau": [2, 1], "size": [-0.1, -0.2]})
    out = apply_shifts(Y, lab, factor=[1.0, 0.5])
    assert list(out.loc["a"]) == [100, 100, 90, 90]
    assert list(out.loc["b"]) == [50, 45, 45, 45]


def test_error_band_includes_common_miss():
    """Общий промах всех МО в одном окне расширяет интервал."""
    import numpy as np
    from src.eval.intervals import error_band
    rng = np.random.default_rng(0)
    rows = []
    for o, common in [("2024-01-01", 0.0), ("2024-02-01", 0.0), ("2024-03-01", -0.10)]:
        for i in range(200):
            rows.append({"origin": pd.Timestamp(o), "h": 1, "yhat": 100.0,
                         "y": 100 * np.exp(common + rng.normal(0, 0.02))})
    b = error_band(pd.DataFrame(rows))
    assert b.at[1, "lo"] < -0.05            # квантиль 10% ушёл к общему промаху −10%
    assert b.at[1, "hi"] > 0.0
