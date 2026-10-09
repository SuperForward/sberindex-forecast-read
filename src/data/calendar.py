"""Производственный календарь РФ (xmlcalendar.ru, data/raw/calendar/calendar_<год>.json).

В JSON для каждого месяца перечислены нерабочие дни; «*» – сокращённый
предпраздничный рабочий день (он рабочий), «+» – перенесённый выходной.
Будний день, не попавший в список, – рабочий; суббота/воскресенье, не
попавшие в список, – рабочие дни переноса.
"""

import json

import pandas as pd

from src.config import RAW_DIR

CAL_DIR = RAW_DIR / "calendar"


def load_days() -> pd.DataFrame:
    """По дням: date, is_workday, is_short, is_holiday (нерабочий будний день)."""
    rows = []
    for path in sorted(CAL_DIR.glob("calendar_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        year = data["year"]
        for m in data["months"]:
            off, short = set(), set()
            for tok in m["days"].split(","):
                day = int(tok.rstrip("*+"))
                (short if tok.endswith("*") else off).add(day)
            for d in pd.date_range(f"{year}-{m['month']:02d}-01", periods=31, freq="D"):
                if d.month != m["month"]:
                    break
                rows.append((d, d.day not in off, d.day in short, d.day in off and d.weekday() < 5))
    return pd.DataFrame(rows, columns=["date", "is_workday", "is_short", "is_holiday"])


def load_monthly() -> pd.DataFrame:
    """Помесячные признаки календаря.

    days, workdays, nonworkdays, short_days, holidays_on_weekdays,
    weekend_workdays (субботы-переносы), workdays_yoy (разница рабочих дней
    с тем же месяцем год назад), long_weekend_max (самая длинная серия
    нерабочих дней, начинающаяся в месяце).
    """
    d = load_days().sort_values("date")
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["weekend"] = d["date"].dt.weekday >= 5
    run = (d["is_workday"] != d["is_workday"].shift()).cumsum()
    d["off_run"] = d.groupby(run)["is_workday"].transform("size").where(~d["is_workday"], 0)
    g = d.groupby("month")
    m = pd.DataFrame({
        "days": g.size(),
        "workdays": g["is_workday"].sum(),
        "short_days": g["is_short"].sum(),
        "holidays_on_weekdays": g["is_holiday"].sum(),
        "weekend_workdays": g.apply(lambda x: (x["is_workday"] & x["weekend"]).sum(), include_groups=False),
        "long_weekend_max": g["off_run"].max(),
    })
    m["nonworkdays"] = m["days"] - m["workdays"]
    m["workdays_yoy"] = m["workdays"] - m["workdays"].shift(12)
    return m.reset_index().rename(columns={"month": "date"})
