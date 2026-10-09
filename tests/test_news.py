"""Новости: тип шока и привязка к МО/региону по заголовку."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import store  # noqa: E402
from src.news import classify, lenta  # noqa: E402

HAS_DATA = not store.load("spending_mo").empty


def test_classify_types():
    assert classify.classify("В Орске прорвало дамбу, эвакуированы тысячи жителей") == ["emergency"]
    assert "production" in classify.classify("В Новосибирске открыли новый завод")
    assert "fiscal" in classify.classify("Учителям задержали выплату зарплаты")
    assert classify.classify("Футбольный клуб сменил тренера") == []


def test_lenta_parse_keeps_only_title_time_url():
    page = ('<a href="/news/2024/04/09/test-news/" class="card"><span>В Кургане ввели режим ЧС</span>'
            '<time>12:30, 9 апреля 2024</time></a>')
    d = lenta.parse(page, "russia")
    assert list(d.columns) == ["published", "rubric", "title", "url"]
    assert d.iloc[0]["title"] == "В Кургане ввели режим ЧС"
    assert str(d.iloc[0]["published"]) == "2024-04-09 12:30:00"


@pytest.mark.skipif(not HAS_DATA, reason="нужна панель МО")
@pytest.mark.parametrize("text, mo, region", [
    ("Жители затопленного Орска пришли к губернатору", "53723000", None),
    ("Паводок в Ишимском районе: подтоплены дома", "71626000", None),
    ("В Абатском объявили эвакуацию", "71603000", None),
    ("Паводок в Курганской области", None, "37"),          # регион, а не город Курган
    ("В Кургане ввели режим ЧС", "37701000", None),
    ("В Советском районе прошли выборы", None, None),      # тёзки без региона – не угадываем
    ("Мирный протест прошёл спокойно", None, None),        # прилагательное, а не город Мирный
])
def test_places(text, mo, region):
    from src.news import gazetteer
    mos, regs = gazetteer.find_mo(text), gazetteer.find_regions(text)
    assert (mo in mos) if mo else not mos, (text, mos)
    assert (region in regs) if region else True, (text, regs)


def test_lenta_parse_empty_page_has_datetime_column():
    """Страница без новостей (выходной, сбой сайта) не должна ронять загрузку."""
    d = lenta.parse("<html><body>нет новостей</body></html>", "russia")
    assert d.empty
    assert (d[d["published"].dt.date == __import__("datetime").date(2024, 3, 1)]).empty


def test_gdelt_latin_keys_match_transliteration():
    from src.news import gdelt
    assert gdelt.latin_key("Екатеринбург") == gdelt.place_key("Yekaterinburg, Sverdlovskaya Oblast', Russia")
    assert gdelt.latin_key("Тюмень") == gdelt.place_key("Tyumen, Tyumenskaya Oblast', Russia")
    assert gdelt.latin_key("Орск") == gdelt.place_key("Orsk, Orenburgskaya Oblast', Russia")
    assert gdelt.latin_key("Ишим") != gdelt.latin_key("Ишимский")


def test_gdelt_region_roots():
    from src.news import gdelt
    same = lambda a, b: any(gdelt._same(x, y) for x in gdelt._roots(a) for y in gdelt._roots(b))  # noqa: E731
    assert same("Omskaya Oblast'", "Омской области")
    assert same("Udmurtiya", "Удмуртской Республики")
    assert not same("Krasnodarskiy Kray", "Красноярского края")
    assert not same("Moskva", "Московской области")


def test_gdelt_parse_keeps_only_russia():
    import io
    import zipfile
    from src.news import gdelt
    row = ["0"] * 61
    row[0], row[1], row[26], row[28], row[29], row[53] = "1", "20240410", "042", "04", "1", "RS"
    row[52], row[54], row[59], row[60] = "Orsk, Orenburgskaya Oblast', Russia", "RS55", "20240410001500", "https://x"
    other = list(row)
    other[0], other[53] = "2", "US"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.CSV", "\n".join("\t".join(r) for r in (row, other)))
    d = gdelt.parse_export(buf.getvalue())
    assert list(d["event_id"]) == ["1"]
    assert str(d.iloc[0]["date_added"]) == "2024-04-10 00:15:00"


def test_news_labels_file():
    import pandas as pd
    from src.news import model
    lab = pd.read_csv(model.LABELS)
    assert list(lab.columns) == ["url", "type", "severity", "sample"]
    assert set(lab["type"]) <= set(model.TYPES) | {"none"}
    assert lab["url"].is_unique
    assert "title" not in lab                      # тексты чужих заголовков в git не храним
