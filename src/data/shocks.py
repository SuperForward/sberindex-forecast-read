"""Справочник задокументированных шоков (reference/shocks_events.csv).

Каждое событие: дата начала, субъект, название МО (mo_query), тип шока,
источник. Здесь события привязываются к ОКТМО МО панели: по основе
названия внутри субъекта; пустой mo_query – событие на весь субъект.

Типы (shock_type): socio_demographic, production_local_market, fiscal_budget.
"""

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.data.mo_match import norm_key, stem_key

EVENTS_FILE = ROOT / "reference" / "shocks_events.csv"


def load_events() -> pd.DataFrame:
    """События × МО панели: event_id, start_date, oktmo, mo_name, shock_type, … in_panel."""
    ev = pd.read_csv(EVENTS_FILE, dtype=str)
    ev["start_date"] = pd.to_datetime(ev["start_date"])
    panel = pd.read_parquet(PROCESSED_DIR / "spending_mo.parquet").drop_duplicates("oktmo")
    panel = panel[["oktmo", "mo_name", "region"]].copy()
    panel["k"] = panel["mo_name"].map(lambda n: stem_key(norm_key(n)))
    rows = []
    for r in ev.itertuples(index=False):
        # «Курская область» -> «Курск»: регион в панели записан в родительном падеже
        root = r.region.split()[0][:-2]
        reg = panel[panel["region"].str.contains(root, na=False)]
        if isinstance(r.mo_query, str) and r.mo_query:
            reg = reg[reg["k"].str.startswith(stem_key(norm_key(r.mo_query)))]
        base = r._asdict()
        if reg.empty:
            rows.append({**base, "oktmo": None, "mo_name": None, "in_panel": False})
        for m in reg.itertuples():
            rows.append({**base, "oktmo": m.oktmo, "mo_name": m.mo_name, "in_panel": True})
    return pd.DataFrame(rows)
