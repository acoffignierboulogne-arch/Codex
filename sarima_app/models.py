"""Couche modèle SARIMA: fit, prévision et grid search."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from evaluation import annual_comparison


@dataclass
class FitResult:
    success: bool
    message: str
    aic: float | None = None
    bic: float | None = None
    llf: float | None = None
    model_fit: object | None = None


class SarimaForecaster:
    def __init__(self, series: pd.Series):
        self.series = series.asfreq("MS")

    def fit(self, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> FitResult:
        try:
            model = SARIMAX(
                self.series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            return FitResult(True, "OK", aic=fit.aic, bic=fit.bic, llf=fit.llf, model_fit=fit)
        except Exception as exc:  # noqa: BLE001
            return FitResult(False, f"Échec de convergence SARIMA: {exc}")

    @staticmethod
    def forecast(fit, start_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
        pred = fit.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "date": pd.date_range(start_date, periods=horizon, freq="MS"),
                "value": mean.values,
                "lower": ci.iloc[:, 0].values,
                "upper": ci.iloc[:, 1].values,
            }
        )

    @staticmethod
    def diagnostics(fit) -> dict:
        resid = pd.Series(fit.resid).dropna()
        lb = acorr_ljungbox(resid, lags=[12], return_df=True)
        return {
            "residuals": resid,
            "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]) if not lb.empty else np.nan,
        }

    def grid_search(
        self,
        train_series: pd.Series,
        reference_series: pd.Series,
        search_space: dict,
        target_years: Iterable[int],
        criterion: str,
        max_combinations: int,
        progress_callback,
    ) -> list[dict]:
        p_vals = range(search_space["p_min"], search_space["p_max"] + 1)
        d_vals = range(search_space["d_min"], search_space["d_max"] + 1)
        q_vals = range(search_space["q_min"], search_space["q_max"] + 1)
        P_vals = range(search_space["P_min"], search_space["P_max"] + 1)
        D_vals = range(search_space["D_min"], search_space["D_max"] + 1)
        Q_vals = range(search_space["Q_min"], search_space["Q_max"] + 1)
        combos = list(product(p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals))[:max_combinations]

        results: list[dict] = []
        real_df = reference_series.dropna().reset_index()
        real_df.columns = ["date", "value"]

        for i, (p, d, q, P, D, Q) in enumerate(combos, start=1):
            progress_callback(i / max(len(combos), 1))
            fit = SarimaForecaster(train_series).fit((p, d, q), (P, D, Q, 12))
            if not fit.success:
                continue

            score = float(fit.aic)
            if criterion == "Écart annuel cumulé":
                in_sample = fit.model_fit.get_prediction(start=train_series.index[0], end=train_series.index[-1]).predicted_mean
                max_ref_date = reference_series.dropna().index.max()
                horizon = max(1, (max_ref_date.to_period("M") - train_series.index[-1].to_period("M")).n)
                forecast_df = SarimaForecaster.forecast(
                    fit.model_fit,
                    start_date=train_series.index[-1] + pd.offsets.MonthBegin(1),
                    horizon=horizon,
                )[["date", "value"]]
                pred_df = pd.concat(
                    [
                        pd.DataFrame({"date": in_sample.index, "value": in_sample.values}),
                        forecast_df,
                    ],
                    ignore_index=True,
                )
                comp = annual_comparison(real_df, pred_df)
                comp = comp[comp["Année"].isin(target_years)]
                if comp.empty:
                    continue
                score = float(comp["Écart relatif %"].abs().sum())

            results.append({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "score": score, "aic": float(fit.aic)})

        return sorted(results, key=lambda x: x["score"])[:10]
