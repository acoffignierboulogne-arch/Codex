"""
Module de chargement et parsing des données CSV au format français.
Gère les dates françaises et les formats numériques avec virgule décimale.
"""
import pandas as pd
import numpy as np
import re
from io import StringIO

# Mapping des mois français vers leur numéro
MOIS_FR = {
    'janv': 1, 'jan': 1, 'janvier': 1,
    'févr': 2, 'fev': 2, 'fevr': 2, 'février': 2, 'fevrier': 2,
    'mars': 3, 'mar': 3,
    'avr': 4, 'avril': 4,
    'mai': 5,
    'juin': 6, 'jun': 6,
    'juil': 7, 'juillet': 7,
    'août': 8, 'aout': 8, 'aou': 8,
    'sept': 9, 'sep': 9, 'septembre': 9,
    'oct': 10, 'octobre': 10,
    'nov': 11, 'novembre': 11,
    'déc': 12, 'dec': 12, 'décembre': 12, 'decembre': 12
}


def parse_french_date(s: str):
    """
    Parse une date française dans différents formats vers un Timestamp mensuel.
    Formats acceptés :
    - "01/2019" ou "01/01/2019" (JJ/MM/AAAA ou MM/AAAA)
    - "janv-2019", "janv. 2019", "janvier 2019"
    - "2019-01" (ISO partiel)
    Retourne un pd.Timestamp ou None si non reconnu.
    """
    s = str(s).strip()

    # Format ISO : 2019-01
    m = re.match(r'^(\d{4})-(\d{1,2})$', s)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)

    # Format JJ/MM/AAAA
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        return pd.Timestamp(year=int(m.group(3)), month=int(m.group(2)), day=1)

    # Format MM/AAAA
    m = re.match(r'^(\d{1,2})/(\d{4})$', s)
    if m:
        return pd.Timestamp(year=int(m.group(2)), month=int(m.group(1)), day=1)

    # Format texte français : janv-2019, janv. 2019, janvier 2019
    m = re.match(r'^([a-zéûùôâîèàê]+)[\s.\-]+(\d{4})$', s, re.IGNORECASE)
    if m:
        mois_str = m.group(1).lower().rstrip('.')
        annee = int(m.group(2))
        # Chercher le mois dans le mapping (correspondance partielle)
        for key, val in MOIS_FR.items():
            if mois_str.startswith(key) or key.startswith(mois_str):
                return pd.Timestamp(year=annee, month=val, day=1)

    return None


def parse_french_number(s):
    """
    Parse un nombre au format français (espaces milliers, virgule décimale).
    Exemples : "1 234 567,89" -> 1234567.89, "1234567.89" -> 1234567.89
    Retourne un float ou None si invalide.
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Supprimer les espaces insécables et normaux (séparateur de milliers)
    s = s.replace('\xa0', '').replace(' ', '')
    # Remplacer la virgule décimale par un point
    s = s.replace(',', '.')
    # Supprimer les symboles monétaires
    s = re.sub(r'[€$£]', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def detect_separator(content: str) -> str:
    """Détecte automatiquement le séparateur CSV (point-virgule ou virgule)."""
    first_line = content.split('\n')[0] if '\n' in content else content
    count_semicolon = first_line.count(';')
    count_comma = first_line.count(',')
    return ';' if count_semicolon >= count_comma else ','


def load_csv(file_content) -> pd.DataFrame:
    """
    Charge un CSV avec dates françaises et montants.
    Retourne un DataFrame avec colonne 'valeur' et un DatetimeIndex nommé 'date'.
    Trie par date et vérifie l'absence de doublons.
    Lève ValueError si le fichier est invalide ou trop court (<24 mois).
    """
    if isinstance(file_content, bytes):
        # Essayer UTF-8 avec BOM, puis latin-1
        try:
            content = file_content.decode('utf-8-sig')
        except UnicodeDecodeError:
            content = file_content.decode('latin-1')
    else:
        content = file_content

    if not content.strip():
        raise ValueError("Le fichier CSV est vide.")

    sep = detect_separator(content)

    # Lecture brute sans interprétation des types
    df_raw = pd.read_csv(StringIO(content), sep=sep, header=None, dtype=str)

    # Garder seulement les 2 premières colonnes
    if df_raw.shape[1] < 2:
        raise ValueError("Le CSV doit avoir au moins 2 colonnes (date, montant).")

    df_raw = df_raw.iloc[:, :2].copy()
    df_raw.columns = ['date_str', 'valeur_str']

    # Supprimer la ligne d'en-tête si la première ligne n'est pas une date valide
    premiere_date = parse_french_date(df_raw.iloc[0]['date_str'])
    if premiere_date is None:
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    # Parser chaque ligne
    dates = []
    valeurs = []
    erreurs = []

    for idx, row in df_raw.iterrows():
        d = parse_french_date(row['date_str'])
        v = parse_french_number(row['valeur_str'])

        if d is None:
            erreurs.append(f"Ligne {idx+1}: date non reconnue '{row['date_str']}'")
            continue
        if v is None:
            erreurs.append(f"Ligne {idx+1}: montant invalide '{row['valeur_str']}'")
            continue

        dates.append(d)
        valeurs.append(v)

    if len(dates) == 0:
        raise ValueError(f"Aucune donnée valide trouvée. Erreurs: {'; '.join(erreurs[:5])}")

    df = pd.DataFrame({'valeur': valeurs}, index=pd.DatetimeIndex(dates))
    df.index.name = 'date'

    # Trier par date
    df = df.sort_index()

    # Vérifier les doublons (garder la dernière valeur)
    doublons = df.index[df.index.duplicated()].unique()
    if len(doublons) > 0:
        df = df[~df.index.duplicated(keep='last')]

    # Vérifier la longueur minimale pour SARIMA saisonnier
    if len(df) < 24:
        raise ValueError(
            f"Série trop courte : {len(df)} mois. Minimum requis : 24 mois."
        )

    return df


def aggregate_series(series: pd.Series, mode: str) -> pd.Series:
    """
    Agrège une série mensuelle en périodes plus larges.

    Modes :
    - 'mensuel'       : pas d'agrégation
    - 'trimestriel'   : T1=janv-mars, T2=avr-juin, T3=juil-sept, T4=oct-déc
    - 'quadrimestriel': Q1=janv-avr, Q2=mai-août, Q3=sept-déc
    - 'semestriel'    : S1=janv-juin, S2=juil-déc
    """
    if mode == 'mensuel':
        return series

    def get_period_label(date, mode):
        mois = date.month
        annee = date.year
        if mode == 'trimestriel':
            q = (mois - 1) // 3 + 1
            return f"T{q} {annee}"
        elif mode == 'quadrimestriel':
            q = (mois - 1) // 4 + 1
            return f"Q{q} {annee}"
        elif mode == 'semestriel':
            s = 1 if mois <= 6 else 2
            return f"S{s} {annee}"
        return str(date)

    labels = series.index.map(lambda d: get_period_label(d, mode))
    aggregated = series.groupby(labels).sum()

    # Trier par ordre chronologique
    def sort_key(label):
        parts = label.split()
        num = int(parts[0][1:])
        annee = int(parts[1])
        return (annee, num)

    try:
        aggregated = aggregated.reindex(
            sorted(aggregated.index, key=sort_key)
        )
    except Exception:
        pass

    return aggregated
