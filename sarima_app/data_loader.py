"""Utilitaires de chargement et normalisation de CSV français."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class LoadResult:
    data: pd.DataFrame
    message: str


MONTH_MAP = {
    "janv": 1,
    "janvier": 1,
    "fév": 2,
    "fev": 2,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avr": 4,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "sept": 9,
    "oct": 10,
    "octobre": 10,
    "nov": 11,
    "novembre": 11,
    "déc": 12,
    "dec": 12,
    "décembre": 12,
    "decembre": 12,
}


def _parse_date(value: str) -> pd.Timestamp | None:
    s = str(value).strip().lower()
    patterns = [
        r"^(\d{1,2})/(\d{4})$",                # MM/AAAA
        r"^(\d{1,2})/(\d{1,2})/(\d{4})$",      # JJ/MM/AAAA
        r"^(\d{4})[-/](\d{1,2})$",              # AAAA-MM
        r"^(\d{4})-(\d{1,2})-(\d{1,2})$",      # AAAA-MM-JJ
        r"^(\d{4})/(\d{1,2})/(\d{1,2})$",      # AAAA/MM/JJ
    ]
    m = re.match(patterns[0], s)
    if m:
        return pd.Timestamp(year=int(m.group(2)), month=int(m.group(1)), day=1)
    m = re.match(patterns[1], s)
    if m:
        return pd.Timestamp(year=int(m.group(3)), month=int(m.group(2)), day=1)
    m = re.match(patterns[2], s)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    m = re.match(patterns[3], s)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    m = re.match(patterns[4], s)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    m = re.match(r"^([a-zéûôîç\.]+)[-\s](\d{4})$", s)
    if m:
        month = MONTH_MAP.get(m.group(1).replace(".", ""))
        if month:
            return pd.Timestamp(year=int(m.group(2)), month=month, day=1)
    # Fallback permissif: dayfirst=True pour formats français hors ISO explicites.
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)


def _parse_amount(value: str) -> float | None:
    s = str(value).strip().replace("\u00a0", " ").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def load_csv(uploaded_file) -> LoadResult:
    if uploaded_file is None:
        return LoadResult(pd.DataFrame(columns=["date", "value"]), "Aucun fichier.")

    content = uploaded_file.read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        return LoadResult(pd.DataFrame(columns=["date", "value"]), "Encodage non supporté.")

    first_line = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first_line.count(";") >= first_line.count(",") else ","
    raw = pd.read_csv(io.StringIO(text), sep=sep)

    if raw.empty or raw.shape[1] < 2:
        return LoadResult(pd.DataFrame(columns=["date", "value"]), "CSV vide ou invalide.")

    rows = []
    for _, row in raw.iloc[:, :2].iterrows():
        date = _parse_date(row.iloc[0])
        value = _parse_amount(row.iloc[1])
        if date is not None and value is not None:
            rows.append((date, value))

    if not rows:
        return LoadResult(pd.DataFrame(columns=["date", "value"]), "Aucune ligne exploitable.")

    df = pd.DataFrame(rows, columns=["date", "value"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df = df.groupby("date", as_index=False)["value"].sum()
    return LoadResult(df, f"{len(df)} mois chargés.")
