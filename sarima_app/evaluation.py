"""
Module d'évaluation des performances de prévision SARIMA.
Calcule les écarts annuels cumulés entre réalité et prévision.
"""
import pandas as pd
import numpy as np


def annual_comparison(
    actual: pd.Series,
    forecast: pd.Series,
    cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Construit le tableau de comparaison annuelle cumulée.

    Paramètres :
    - actual      : série des valeurs réelles (index DatetimeIndex mensuel)
    - forecast    : série des prévisions (index DatetimeIndex mensuel)
    - cutoff_date : dernier mois de calibration (exclu du calcul)

    Retourne un DataFrame avec colonnes :
    Année | Réel (€) | Prévision (€) | Écart (€) | Écart (%)

    N'affiche que les années post-cutoff disposant de 12 mois réels ET 12 mois prévisionnels.
    """
    if actual is None or forecast is None:
        return pd.DataFrame()

    # Données réelles uniquement après le cutoff
    actual_post = actual[actual.index > cutoff_date]

    if actual_post.empty:
        return pd.DataFrame()

    results = []

    for year in sorted(actual_post.index.year.unique()):
        actual_year = actual_post[actual_post.index.year == year]
        forecast_year = forecast[forecast.index.year == year]

        # L'année doit avoir 12 mois de données réelles ET de prévisions
        if len(actual_year) < 12 or len(forecast_year) < 12:
            continue

        # Gérer le cas où actual est un DataFrame ou une Series
        if hasattr(actual_year, 'columns'):
            somme_reel = actual_year['valeur'].sum()
        else:
            somme_reel = actual_year.sum()

        somme_prev = forecast_year.sum()
        ecart_abs = somme_prev - somme_reel
        ecart_rel = (ecart_abs / somme_reel * 100) if somme_reel != 0 else np.nan

        results.append({
            'Année': year,
            'Réel (€)': somme_reel,
            'Prévision (€)': somme_prev,
            'Écart (€)': ecart_abs,
            'Écart (%)': ecart_rel
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index('Année')


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """
    Calcule les métriques d'erreur classiques entre la série réelle et la prévision.

    Retourne un dict avec :
    - MAE  : erreur absolue moyenne
    - RMSE : racine de l'erreur quadratique moyenne
    - MAPE : erreur absolue pourcentuelle moyenne (%)
    - BIAS : biais moyen (prévision - réel)
    """
    # Aligner les deux séries sur les dates communes
    common_idx = actual.index.intersection(forecast.index)
    if len(common_idx) == 0:
        return {}

    y_true = actual.loc[common_idx]
    y_pred = forecast.loc[common_idx]

    erreurs = y_pred - y_true
    mae = np.mean(np.abs(erreurs))
    rmse = np.sqrt(np.mean(erreurs ** 2))
    mape = np.mean(np.abs(erreurs / y_true.replace(0, np.nan))) * 100
    biais = np.mean(erreurs)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'Biais': biais
    }
