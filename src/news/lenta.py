"""Архив Lenta.ru по дням: разбор страниц рубрик.

Страница https://lenta.ru/rubrics/<рубрика>/ГГГГ/ММ/ДД/[page/N/] – список
новостей дня: заголовок, время публикации, ссылка. Храним только это
(без текстов статей): для привязки к МО и типу шока хватает заголовка, а
полные тексты – чужая интеллектуальная собственность.
"""

import html
import re

import pandas as pd

ITEM = re.compile(r'<a[^>]+href="(/news/(\d{4})/(\d{2})/(\d{2})/[^"]+/)"[^>]*>(.*?)</a>', re.S)
TIME = re.compile(r"(\d{1,2}):(\d{2}),\s*\d{1,2}\s+\S+\s+\d{4}")


def parse(page: str, rubric: str) -> pd.DataFrame:
    rows, seen = [], set()
    for url, y, m, d, body in ITEM.findall(page):
        if url in seen:
            continue
        text = html.unescape(re.sub(r"<[^>]+>", " ", body))
        text = re.sub(r"\s+", " ", text).strip()
        tm = TIME.search(text)
        title = TIME.split(text)[0].strip() if tm else text
        if len(title) < 15:
            continue
        seen.add(url)
        hh, mm = (int(tm.group(1)), int(tm.group(2))) if tm else (0, 0)
        rows.append({"published": pd.Timestamp(int(y), int(m), int(d), hh, mm), "rubric": rubric,
                     "title": title, "url": "https://lenta.ru" + url})
    df = pd.DataFrame(rows, columns=["published", "rubric", "title", "url"])
    # пустая страница – тоже таблица с датой нужного типа, иначе фильтр по дате падает
    df["published"] = pd.to_datetime(df["published"])
    return df


def has_next(page: str, n: int) -> bool:
    return f"/page/{n + 1}/" in page
