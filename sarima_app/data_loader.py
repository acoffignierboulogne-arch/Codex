"""Utilitaires de chargement et normalisation de CSV français."""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LoadResult:
    data: pd.DataFrame
    message: str


def _fix_mojibake(text: str) -> str:
    """Corrige quelques séquences d'encodage mal décodées fréquentes."""
    replacements = {
        "Ã©": "é",
        "Ã¨": "è",
        "Ãª": "ê",
        "Ã«": "ë",
        "Ã ": "à",
        "Ã¹": "ù",
        "Ã»": "û",
        "Ã´": "ô",
        "Ã®": "î",
        "Ã¯": "ï",
        "Ã§": "ç",
        "Ã‰": "É",
    }
    out = text
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    return out


def _slug(value: str) -> str:
    s = _fix_mojibake(str(value)).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _parse_amount(value) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace("\u00a0", " ").replace(" ", "")
    if not s:
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _detect_separator(first_line: str) -> str:
    return ";" if first_line.count(";") >= first_line.count(",") else ","


def _canonical_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes vers des noms canoniques robustes."""
    alias_map = {
        "exercice": "exercice",
        "mois": "mois",
        "date": "date",
        "titre_depenses": "titre_depenses",
        "titre_recettes": "titre_recettes",
        "chapitre": "chapitre",
        "compte_execution": "compte_execution",
        "sous_compte": "sous_compte",
        "sous_compte_classe_6": "sous_compte_classe_6",
        "libelle_du_type": "libelle_type",
        "libelle_type": "libelle_type",
        "prevision_cumulee": "prevision_cumulee",
        "prevision_depenses_prevision_cumulee": "prevision_cumulee_depenses",
        "prevision_recettes_prevision_cumulee": "prevision_cumulee_recettes",
        "budget_primitif_depense": "budget_primitif_depense",
        "budget_primitif_recette": "budget_primitif_recette",
        "realise_depenses_cumul_realise_date_comptable": "montant_depenses",
        "realise_recettes_cumul_realise_date_comptable": "montant_recettes",
        # variantes sans accents / espaces
        "realis_depenses_cumul_realise_date_comptable": "montant_depenses",
        "realise_depenses_cumul_rea lise_date_comptable": "montant_depenses",
    }
    rename = {}
    for col in raw.columns:
        slug = _slug(col)
        # correction spécifique de quelques slugs issus d'accents mal lus
        slug = slug.replace("d_penses", "depenses").replace("r_alis", "realis")
        slug = slug.replace("compte_ex_cution", "compte_execution")
        slug = slug.replace("libell_du_type", "libelle_du_type")
        canon = alias_map.get(slug)
        if canon:
            rename[col] = canon
    return raw.rename(columns=rename)


def _build_date_from_year_month(df: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(df.get("exercice"), errors="coerce")
    months = pd.to_numeric(df.get("mois"), errors="coerce")
    return pd.to_datetime(
        {"year": years, "month": months, "day": 1}, errors="coerce"
    )


def _build_flat_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Transforme le gros fichier budgétaire en format filtrable unique."""
    df = _canonical_columns(raw.copy())

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        dates = pd.to_datetime(
            {"year": dates.dt.year, "month": dates.dt.month, "day": 1}, errors="coerce"
        )
    elif {"exercice", "mois"}.issubset(df.columns):
        dates = _build_date_from_year_month(df)
    else:
        return pd.DataFrame(columns=["date", "flux", "titre", "chapitre", "compte_execution", "sous_compte", "sous_compte_classe_6", "libelle_type", "value"])

    base_cols = {
        "chapitre": df.get("chapitre"),
        "compte_execution": df.get("compte_execution"),
        "sous_compte": df.get("sous_compte"),
        "sous_compte_classe_6": df.get("sous_compte_classe_6"),
        "libelle_type": df.get("libelle_type"),
    }

    dep = pd.DataFrame({
        "date": dates,
        "flux": "Dépenses",
        "titre": df.get("titre_depenses"),
        "value": df.get("montant_depenses"),
        "prevision_cumulee": df.get("prevision_cumulee_depenses", df.get("prevision_cumulee")),
        "budget_primitif": df.get("budget_primitif_depense"),
        **base_cols,
    })
    rec = pd.DataFrame({
        "date": dates,
        "flux": "Recettes",
        "titre": df.get("titre_recettes"),
        "value": df.get("montant_recettes"),
        "prevision_cumulee": df.get("prevision_cumulee_recettes", df.get("prevision_cumulee")),
        "budget_primitif": df.get("budget_primitif_recette"),
        **base_cols,
    })

    flat = pd.concat([dep, rec], ignore_index=True)
    flat["value"] = flat["value"].map(_parse_amount)
    flat["prevision_cumulee"] = flat["prevision_cumulee"].map(_parse_amount)
    flat["budget_primitif"] = flat["budget_primitif"].map(_parse_amount)
    for c in ["titre", "chapitre", "compte_execution", "sous_compte", "sous_compte_classe_6", "libelle_type"]:
        flat[c] = flat[c].fillna("(vide)").astype(str).map(_fix_mojibake)
    flat = flat.dropna(subset=["date", "value"])  # garder zéros
    flat["date"] = pd.to_datetime(flat["date"], errors="coerce")
    flat = flat.dropna(subset=["date"])
    flat["date"] = pd.to_datetime({"year": flat["date"].dt.year, "month": flat["date"].dt.month, "day": 1})
    return flat.sort_values("date").reset_index(drop=True)


def load_csv(uploaded_file) -> LoadResult:
    if uploaded_file is None:
        return LoadResult(pd.DataFrame(), "Aucun fichier.")

    content = uploaded_file.read()
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return LoadResult(pd.DataFrame(), "Encodage non supporté.")

    text = _fix_mojibake(text)
    lines = text.splitlines()
    first_line = lines[0] if lines else ""
    sep = _detect_separator(first_line)
    raw = pd.read_csv(io.StringIO(text), sep=sep)

    if raw.empty:
        return LoadResult(pd.DataFrame(), "CSV vide.")

    flat = _build_flat_dataset(raw)
    if flat.empty:
        # fallback ancien format 2 colonnes date/montant
        rows = []
        for _, row in raw.iloc[:, :2].iterrows():
            date = pd.to_datetime(row.iloc[0], errors="coerce", dayfirst=True)
            value = _parse_amount(row.iloc[1])
            if pd.notna(date) and value is not None:
                rows.append((pd.Timestamp(year=date.year, month=date.month, day=1), value))
        if not rows:
            return LoadResult(pd.DataFrame(), "Aucune ligne exploitable.")
        df = pd.DataFrame(rows, columns=["date", "value"])
        df = df.groupby("date", as_index=False)["value"].sum().sort_values("date")
        df["flux"] = "Dépenses"
        df["titre"] = "(global)"
        df["chapitre"] = "(global)"
        df["compte_execution"] = "(global)"
        df["sous_compte"] = "(global)"
        df["sous_compte_classe_6"] = "(global)"
        df["libelle_type"] = "(global)"
        df["prevision_cumulee"] = np.nan
        df["budget_primitif"] = np.nan
        return LoadResult(df, f"{len(df)} mois chargés (format simple).")

    mois = flat["date"].dt.to_period("M").nunique()
    return LoadResult(flat, f"{len(flat):,} lignes chargées, {mois} mois disponibles après normalisation.")
