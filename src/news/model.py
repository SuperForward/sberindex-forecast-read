"""Тип и сила события по заголовку: эмбеддинги multilingual-e5-base +
логистическая регрессия, обученная на ручной разметке.

Словари (src/news/classify.py) путали «сбили беспилотник» с ущербом и почти
не находили экономику: на отложенной разметке значимые новости – F1 0,50,
типы – macro-F1 0,33. Эта модель – 0,80 и 0,72 (5-кратная кросс-проверка,
reports/news/model_cv.md). Работает офлайн: веса в data/models (python -m
worker models), обучение – секунды на каждом запуске, результат
воспроизводим.

Разметка – reference/news_title_labels.csv: ссылка, тип, сила (1 – местное
событие, 2 – массовое: эвакуация, режим ЧС, разрушения); заголовки берутся
из архива data/raw/news по ссылке. Типы:
  none            – на расходы жителей не влияет (политика, криминал, сбитый
                    БПЛА без последствий, бои без ущерба для жителей);
  emergency       – бедствие, авария, пожар, отключение света/тепла/воды;
  attack          – удар с разрушениями, жертвами, эвакуацией;
  production_neg  – закрытие, остановка, увольнения, банкротство;
  production_pos  – открытие, запуск, инвестиции, рабочие места;
  fiscal          – зарплаты, выплаты, пособия, бюджет;
  prices          – цены, тарифы, стоимость жилья, дефицит;
  demand          – транспорт, аэропорты, ограничения, туризм, торговля.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import DATA_DIR, ROOT, cpu_budget

WEIGHTS = DATA_DIR / "models" / "multilingual-e5-base"
LABELS = ROOT / "reference" / "news_title_labels.csv"
CACHE = DATA_DIR / ".cache" / "news_emb_e5base"
TYPES = ["emergency", "attack", "production_neg", "production_pos", "fiscal", "prices", "demand"]
C_TYPE, C_SEV = 64.0, 4.0                         # подобраны кросс-проверкой


def available() -> tuple[bool, str]:
    if not (WEIGHTS / "model.safetensors").exists():
        return False, f"нет весов {WEIGHTS} (скачать: python -m worker models)"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"не установлена библиотека: {e.name}"
    return True, "ok"


@lru_cache(maxsize=1)
def _encoder():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # веса локальные, в сеть не ходим
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(cpu_budget())
    return AutoTokenizer.from_pretrained(str(WEIGHTS)), AutoModel.from_pretrained(str(WEIGHTS)).eval()


def embed(titles: list[str], batch: int = 64) -> np.ndarray:
    """Нормированные эмбеддинги (среднее по токенам), префикс «query: » – как у e5."""
    import torch
    tok, m = _encoder()
    out = []
    with torch.no_grad():
        for k in range(0, len(titles), batch):
            b = tok(["query: " + str(t) for t in titles[k:k + batch]], padding=True, truncation=True,
                    max_length=64, return_tensors="pt")
            h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            out.append(torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1).numpy())
    return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def embed_cached(news: pd.DataFrame) -> np.ndarray:
    """Эмбеддинги по url с кэшем в data/.cache: заголовок по ссылке не меняется,
    считаются только новые (e5-base на CPU – десятки заголовков в секунду)."""
    urls_f, emb_f = CACHE.with_suffix(".urls.parquet"), CACHE.with_suffix(".npy")
    if urls_f.exists() and emb_f.exists():
        urls, emb = pd.read_parquet(urls_f)["url"].tolist(), np.load(emb_f)
    else:
        urls, emb = [], np.zeros((0, 768), dtype=np.float32)
    pos = {u: i for i, u in enumerate(urls)}
    new = news[~news["url"].isin(pos)].drop_duplicates("url")
    if len(new):
        print(f"эмбеддинги: {len(new)} новых заголовков", flush=True)
        emb = np.vstack([emb, embed(new["title"].tolist())]).astype(np.float32)
        urls = urls + new["url"].tolist()
        pos = {u: i for i, u in enumerate(urls)}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_f, emb)
        pd.DataFrame({"url": urls}).to_parquet(urls_f, index=False)
    return emb[[pos[u] for u in news["url"]]]


def labels(news: pd.DataFrame) -> pd.DataFrame:
    """Разметка с заголовками из архива (ссылки, которых нет в архиве, отбрасываются)."""
    lab = pd.read_csv(LABELS, dtype={"url": str})
    return lab.merge(news[["url", "title"]].drop_duplicates("url"), on="url")


def fit(news: pd.DataFrame):
    """(классификатор типа, классификатор силы 2 против 1) на всей разметке."""
    from sklearn.linear_model import LogisticRegression
    lab = labels(news)
    x = embed_cached(lab)
    clf = LogisticRegression(C=C_TYPE, max_iter=5000, class_weight="balanced").fit(x, lab["type"])
    rel = (lab["type"] != "none").to_numpy()
    sev = LogisticRegression(C=C_SEV, max_iter=5000, class_weight="balanced").fit(
        x[rel], (lab.loc[rel, "severity"] >= 2).astype(int))
    return clf, sev


def predict(news: pd.DataFrame, models=None) -> pd.DataFrame:
    """news (url, title) -> type (argmax), p_type, p_severe (вероятность силы 2)."""
    clf, sev = models or fit(news)
    x = embed_cached(news)
    p = clf.predict_proba(x)
    k = p.argmax(1)
    return pd.DataFrame({"type": clf.classes_[k], "p_type": p[np.arange(len(k)), k],
                         "p_severe": sev.predict_proba(x)[:, 1]}, index=news.index)


def cross_validate(news: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """Качество на разметке (k-кратная кросс-проверка) против словарей:
    выделение значимых новостей (P, R, F1) и различение типов (macro-F1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from src.news import classify
    lab = labels(news)
    y = lab["type"].to_numpy(dtype=object)
    x = embed_cached(lab)
    pred = cross_val_predict(LogisticRegression(C=C_TYPE, max_iter=5000, class_weight="balanced"), x, y,
                             cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    # словари: ЧС-словарь не отличает бедствие от удара – засчитываем ему оба
    lex_map = {"emergency": "emergency", "production": "production_pos", "fiscal": "fiscal",
               "social": "fiscal", "prices": "prices"}
    lex = np.array([lex_map.get((classify.classify(t) or ["none"])[0], "none") for t in lab["title"]], dtype=object)
    lex = np.where((lex == "emergency") & (y == "attack"), "attack", lex)
    rows = []
    for name, p in (("словари", lex), ("e5-base + логрегрессия", pred)):
        pr, rc, f1, _ = precision_recall_fscore_support(y != "none", p != "none", average="binary")
        rows.append({"method": name, "labels": len(y), "relevant_P": round(pr, 2), "relevant_R": round(rc, 2),
                     "relevant_F1": round(f1, 2), "types_macroF1": round(f1_score(y, p, average="macro"), 2)})
    return pd.DataFrame(rows)
