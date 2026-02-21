"""
Module SARIMA — implémentation du modèle SARIMA(p,d,q)(P,D,Q)[12]
via statsmodels.tsa.statespace.SARIMAX.

SARIMA = Seasonal AutoRegressive Integrated Moving Average
- (p,d,q) : composantes non-saisonnières (AR, différenciation, MA)
- (P,D,Q) : composantes saisonnières
- m=12    : période saisonnière (données mensuelles)
"""
import numpy as np
import pandas as pd
import warnings
from itertools import product
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class SARIMAModel:
    """
    Classe encapsulant le modèle SARIMA pour la prévision de séries temporelles
    de dépenses hospitalières mensuelles.
    """

    def __init__(
        self,
        p: int = 1, d: int = 1, q: int = 1,
        P: int = 1, D: int = 1, Q: int = 1,
        m: int = 12
    ):
        """
        Initialise le modèle SARIMA avec ses paramètres.

        p, d, q : ordres non-saisonniers (AR, différenciation, MA)
        P, D, Q : ordres saisonniers (AR, différenciation, MA)
        m       : période saisonnière (12 pour données mensuelles)
        """
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.results_ = None
        self.aic_ = None
        self.bic_ = None
        self.llf_ = None
        self.error_message_ = None

    def fit(self, train_data: pd.Series) -> bool:
        """
        Ajuste le modèle SARIMA sur les données d'entraînement.

        Paramètres :
        - train_data : pd.Series avec un DatetimeIndex de fréquence mensuelle

        Retourne True si l'estimation a convergé, False sinon.
        """
        if not STATSMODELS_AVAILABLE:
            self.error_message_ = "statsmodels non disponible. Installez avec : pip install statsmodels"
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = SARIMAX(
                    train_data,
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.m),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization='approximate_diffuse'
                )

                self.results_ = model.fit(
                    disp=False,
                    maxiter=200,
                    method='lbfgs'
                )

                self.aic_ = self.results_.aic
                self.bic_ = self.results_.bic
                self.llf_ = self.results_.llf
                self.error_message_ = None
                return True

        except Exception as e:
            self.error_message_ = str(e)
            self.results_ = None
            return False

    def predict(self, horizon: int) -> Optional[pd.DataFrame]:
        """
        Génère les prévisions pour les 'horizon' mois suivant la fin des données.

        Retourne un DataFrame avec colonnes :
        - forecast  : prévision ponctuelle
        - lower_ci  : borne inférieure IC 95%
        - upper_ci  : borne supérieure IC 95%

        Retourne None si le modèle n'a pas été ajusté ou en cas d'erreur.
        """
        if self.results_ is None:
            return None

        try:
            forecast_obj = self.results_.get_forecast(steps=horizon)
            forecast_mean = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.05)

            result = pd.DataFrame({
                'forecast': forecast_mean,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            })
            return result

        except Exception as e:
            logger.error(f"Erreur de prévision : {e}")
            return None

    def get_fitted_values(self) -> Optional[pd.Series]:
        """Retourne les valeurs ajustées (prédictions in-sample)."""
        if self.results_ is None:
            return None
        return self.results_.fittedvalues

    def get_residuals(self) -> Optional[pd.Series]:
        """Retourne les résidus du modèle ajusté."""
        if self.results_ is None:
            return None
        return self.results_.resid

    def ljung_box_test(self, lags: int = 12) -> Optional[pd.DataFrame]:
        """
        Test de Ljung-Box pour vérifier l'autocorrélation des résidus.

        H0 : les résidus sont indépendants (pas d'autocorrélation résiduelle).
        p-value > 0.05 → le modèle est correctement spécifié.

        Retourne un DataFrame avec lb_stat et lb_pvalue pour chaque lag.
        """
        if self.results_ is None:
            return None
        try:
            resid = self.get_residuals().dropna()
            result = acorr_ljungbox(resid, lags=lags, return_df=True)
            return result
        except Exception as e:
            logger.error(f"Erreur test Ljung-Box : {e}")
            return None

    def get_model_info(self) -> dict:
        """
        Retourne un dictionnaire d'informations sur le modèle ajusté :
        ordre, AIC, BIC, log-vraisemblance et message d'erreur éventuel.
        """
        return {
            'ordre': (
                f"SARIMA({self.p},{self.d},{self.q})"
                f"({self.P},{self.D},{self.Q})[{self.m}]"
            ),
            'AIC': self.aic_,
            'BIC': self.bic_,
            'Log-vraisemblance': self.llf_,
            'erreur': self.error_message_
        }


def grid_search_sarima(
    data: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    p_range: range,
    d_range: range,
    q_range: range,
    P_range: range,
    D_range: range,
    Q_range: range,
    target_years: list,
    criterion: str = 'ecart_annuel',
    max_combinations: int = 200,
    progress_callback=None
) -> list:
    """
    Grid search sur les paramètres SARIMA pour trouver la meilleure combinaison.

    Paramètres :
    - data            : DataFrame avec colonne 'valeur' et DatetimeIndex
    - cutoff_date     : dernier mois de calibration
    - p/d/q_range     : plages pour les ordres non-saisonniers
    - P/D/Q_range     : plages pour les ordres saisonniers
    - target_years    : années sur lesquelles évaluer les écarts
    - criterion       : 'ecart_annuel' ou 'aic'
    - max_combinations: limite du nombre de combinaisons testées
    - progress_callback: fonction(ratio, message) appelée à chaque itération

    Retourne une liste triée (meilleur score en premier) de dicts avec les résultats.
    """
    if not STATSMODELS_AVAILABLE:
        return []

    # Série d'entraînement jusqu'au cutoff inclus
    train_data = data[data.index <= cutoff_date]['valeur']
    actual_post = data[data.index > cutoff_date]['valeur']

    # Générer toutes les combinaisons possibles
    all_combinations = list(
        product(p_range, d_range, q_range, P_range, D_range, Q_range)
    )

    # Limiter si trop grand
    if len(all_combinations) > max_combinations:
        all_combinations = all_combinations[:max_combinations]

    results = []
    total = len(all_combinations)

    for i, (p, d, q, P, D, Q) in enumerate(all_combinations):
        if progress_callback:
            progress_callback(
                i / total,
                f"Test {i+1}/{total} : SARIMA({p},{d},{q})({P},{D},{Q})[12]"
            )

        model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q)

        if not model.fit(train_data):
            continue

        # Calculer l'horizon de prévision nécessaire
        if not actual_post.empty:
            last_actual = actual_post.index[-1]
            horizon = len(pd.date_range(
                start=cutoff_date + pd.DateOffset(months=1),
                end=last_actual,
                freq='MS'
            ))
        else:
            horizon = 12

        forecast_df = model.predict(horizon)
        if forecast_df is None:
            continue

        score = None

        if criterion == 'aic' and model.aic_ is not None:
            score = model.aic_

        elif criterion == 'ecart_annuel' and target_years:
            # Calculer l'écart annuel moyen sur les années cibles
            total_ecart = 0.0
            valid_years = 0

            for year in target_years:
                actual_year = actual_post[actual_post.index.year == year]
                forecast_year = forecast_df[
                    forecast_df.index.year == year
                ]['forecast']

                if len(actual_year) == 12 and len(forecast_year) == 12:
                    somme_reel = actual_year.sum()
                    somme_prev = forecast_year.sum()
                    if somme_reel != 0:
                        ecart = abs((somme_prev - somme_reel) / somme_reel * 100)
                        total_ecart += ecart
                        valid_years += 1

            if valid_years > 0:
                score = total_ecart / valid_years

        if score is not None:
            results.append({
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q,
                'score': score,
                'AIC': model.aic_,
                'BIC': model.bic_,
                'ordre': f"SARIMA({p},{d},{q})({P},{D},{Q})[12]"
            })

    # Trier par score croissant (plus petit = meilleur)
    results.sort(key=lambda x: x['score'])
    return results
