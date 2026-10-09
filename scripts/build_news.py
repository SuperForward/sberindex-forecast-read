"""Новости -> тип события, МО/регион, агрегаты по месяцам.

Запуск: python -m scripts.build_news

- data/processed/news_matches.parquet – каждая значимая новость с местом:
  published, month, level (mo|region), oktmo, cov_region, types (тип), p_type
  (уверенность модели), p_severe (вероятность массового события), title, url.
  Для интерфейса и проверки глазами.
- data/processed/news_mo_monthly.parquet – МО × месяц: взвешенное число
  новостей каждого типа, news_<тип>. Вес новости = p_type × (1 + p_severe);
  новость о регионе в целом идёт каждому его МО с весом region_weight.
- reports/news/model_cv.md – качество классификатора на разметке против словарей.

Тип определяет модель src/news/model.py (эмбеддинги e5 + логрегрессия на
ручной разметке); без весов или torch – словари src/news/classify.py, как
раньше. Классифицируются только новости с местом (МО или регион): остальные
в признаки всё равно не попадают.

Время: месяц – месяц публикации. Признак за месяц t известен к концу месяца
t, поэтому для прогноза из месяца t его можно использовать без заглядывания
в будущее.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import store  # noqa: E402
from src.config import DATA_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, load_config  # noqa: E402
from src.news import classify, gazetteer  # noqa: E402

# новость о регионе в целом – слабее сигнал для конкретного МО (configs/news.yaml)
REGION_WEIGHT = float(load_config("news").get("region_weight", 0.3))
MIN_P = float(load_config("news").get("min_p", 0.5))       # порог уверенности модели в типе
PLACES = DATA_DIR / ".cache" / "news_places.parquet"


def places(news: pd.DataFrame) -> pd.DataFrame:
    """url -> mos, regions (через запятую). Поиск ~2 тыс. шаблонов МО по 100+
    тыс. заголовков – минуты, поэтому с кэшем; кэш сбрасывается, если
    изменился справочник топонимов (src/news/gazetteer.py)."""
    version = hashlib.sha1(Path(gazetteer.__file__).read_bytes()).hexdigest()[:12]
    old = pd.read_parquet(PLACES) if PLACES.exists() else pd.DataFrame(columns=["url", "mos", "regions", "version"])
    old = old[old["version"] == version]
    new = news[~news["url"].isin(old["url"])][["url", "title"]]
    if len(new):
        print(f"места: {len(new)} новых заголовков", flush=True)
        new = new.assign(mos=new["title"].map(lambda t: ",".join(gazetteer.find_mo(t))),
                         regions=new["title"].map(lambda t: ",".join(gazetteer.find_regions(t))),
                         version=version).drop(columns="title")
        old = pd.concat([old, new], ignore_index=True)
        PLACES.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(PLACES, index=False)
    return news[["url"]].merge(old[["url", "mos", "regions"]], on="url", how="left")


def classify_news(news: pd.DataFrame) -> pd.DataFrame:
    """type, p_type, p_severe – моделью, а без неё словарями (p = 1, p_severe = 0)."""
    from src.news import model
    ok, why = model.available()
    if ok:
        models = model.fit(news)
        cv = model.cross_validate(news)
        (REPORTS_DIR / "news").mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "news" / "model_cv.md").write_text(
            "# Классификатор новостей: 5-кратная кросс-проверка на разметке\n\n"
            + cv.to_markdown(index=False) + "\n", encoding="utf-8")
        print(cv.to_string(index=False))
        return model.predict(news, models)
    print(f"модель новостей недоступна ({why}) – словари", flush=True)
    lex = {"emergency": "emergency", "production": "production_neg", "fiscal": "fiscal",
           "social": "fiscal", "prices": "prices"}
    t = news["title"].map(lambda s: lex.get((classify.classify(s) or ["none"])[0], "none"))
    return pd.DataFrame({"type": t, "p_type": 1.0, "p_severe": 0.0}, index=news.index)


def main() -> None:
    from scripts.build_covariates import cov_region_of_oktmo
    files = sorted((RAW_DIR / "news").glob("lenta_*.parquet"))
    if not files:
        raise SystemExit("нет новостей в data/raw/news (python -m scripts.download_news)")
    news = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("url")
    news["published"] = pd.to_datetime(news["published"])
    news = news.reset_index(drop=True)
    pl = places(news)
    news["mos"], news["regions"] = pl["mos"].fillna("").to_numpy(), pl["regions"].fillna("").to_numpy()
    placed = news[(news["mos"] != "") | (news["regions"] != "")].copy()
    placed = placed.join(classify_news(placed))
    shock = placed[(placed["type"] != "none") & (placed["p_type"] >= MIN_P)]
    print(f"новостей: {len(news)}, с местом: {len(placed)}, значимых: {len(shock)} "
          f"({shock['type'].value_counts().to_dict()})")

    rows = []
    for r in shock.itertuples():
        mos = [o for o in r.mos.split(",") if o]
        regions = [c for c in r.regions.split(",") if c]
        base = {"published": r.published, "month": r.published.to_period("M").to_timestamp(),
                "types": r.type, "p_type": r.p_type, "p_severe": r.p_severe, "title": r.title, "url": r.url}
        for o in mos:
            rows.append({**base, "level": "mo", "oktmo": o, "cov_region": cov_region_of_oktmo(o)})
        mo_regions = {cov_region_of_oktmo(o) for o in mos}
        for c in regions:
            if c not in mo_regions:           # регион уже покрыт упоминанием его МО
                rows.append({**base, "level": "region", "oktmo": None, "cov_region": c})
    m = pd.DataFrame(rows, columns=["published", "month", "level", "oktmo", "cov_region", "types", "p_type",
                                    "p_severe", "title", "url"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m.to_parquet(PROCESSED_DIR / "news_matches.parquet", index=False)
    print(f"привязано: к МО {int((m['level'] == 'mo').sum())}, к региону {int((m['level'] == 'region').sum())}; "
          f"МО с упоминаниями: {m['oktmo'].nunique()}")

    # МО × месяц × тип: своё упоминание (вес 1) + новость о регионе (вес REGION_WEIGHT),
    # каждая – с уверенностью модели и поправкой на силу события
    panel = store.load("spending_mo")[["oktmo"]].drop_duplicates()
    panel["cov_region"] = panel["oktmo"].map(cov_region_of_oktmo)
    m = m.assign(type=m["types"], w=m["p_type"] * (1 + m["p_severe"]))
    own = m[m["level"] == "mo"].groupby(["oktmo", "month", "type"])["w"].sum()
    reg = m[m["level"] == "region"].groupby(["cov_region", "month", "type"])["w"].sum() * REGION_WEIGHT
    reg = panel.merge(reg.reset_index(), on="cov_region").set_index(["oktmo", "month", "type"])["w"]
    agg = pd.concat([own, reg]).groupby(level=[0, 1, 2]).sum().unstack("type", fill_value=0)
    agg.columns = [f"news_{c}" for c in agg.columns]
    agg = agg.reset_index()
    agg.to_parquet(PROCESSED_DIR / "news_mo_monthly.parquet", index=False)
    print(f"news_mo_monthly: {len(agg)} строк (МО × месяц), столбцы {[c for c in agg if c.startswith('news_')]}")


if __name__ == "__main__":
    from src.runlog import run_logged  # noqa: E402
    run_logged("build_news", "news", main)
