"""Sélection robuste de modèles SARIMA avec backtest rolling-origin."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


CANDIDATES = [
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 0), (0, 1, 1, 12)),
    ((0, 1, 1), (1, 1, 0, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 0), (0, 1, 1, 12)),
    ((0, 1, 2), (0, 1, 1, 12)),
    ((1, 1, 0), (1, 1, 0, 12)),
    ((0, 1, 1), (1, 1, 1, 12)),
    ((0, 1, 1), (0, 0, 1, 12)),
    ((1, 1, 0), (0, 0, 1, 12)),
    ((0, 1, 1), (1, 0, 0, 12)),
    ((1, 1, 1), (0, 0, 1, 12)),
    ((2, 1, 0), (0, 0, 1, 12)),
    ((0, 1, 2), (0, 0, 1, 12)),
    ((0, 0, 1), (0, 1, 1, 12)),
    ((1, 0, 0), (0, 1, 1, 12)),
    ((1, 0, 1), (0, 1, 1, 12)),
    ((0, 0, 2), (0, 1, 1, 12)),
]


@dataclass
class SarimaSelectionConfig:
    horizon: int = 12
    max_origins: int = 36
    penalty_lb_p: float = 0.05
    penalty_factor: float = 1.10
    residual_lags: list[int] = field(default_factory=lambda: [6, 12, 18, 24])
    seasonal_period: int = 12


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, str]:
    denom = np.abs(y_true)
    if np.any(denom <= 1e-9):
        smape_vals = 2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)
        return float(np.mean(smape_vals) * 100), "sMAPE"
    mape_vals = np.abs((y_true - y_pred) / denom)
    return float(np.mean(mape_vals) * 100), "MAPE"


def _compute_error_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float | str]:
    if not y_true:
        return {"mae": np.nan, "rmse": np.nan, "mape_or_smape": np.nan, "pct_metric": "MAPE"}
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mape_value, pct_name = _safe_mape(yt, yp)
    return {
        "mae": float(np.mean(np.abs(yp - yt))),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "mape_or_smape": mape_value,
        "pct_metric": pct_name,
    }


def _candidate_key(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> str:
    return f"{order}x{seasonal_order}"


def fit_predict_sarima(
    train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    horizon: int,
) -> np.ndarray:
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=horizon).predicted_mean
    return np.asarray(pred.values, dtype=float)


def compute_backtest_scores(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    cfg: SarimaSelectionConfig,
    logs: list[str],
) -> dict[str, Any]:
    s = series.dropna().sort_index().asfreq("MS")
    n = len(s)
    min_train = max(24, cfg.seasonal_period * 2)
    if n < min_train + 1:
        return {"ok": False, "reason": "série trop courte", "folds": 0}

    last_origin = n - 2
    first_origin = min_train - 1
    origins = list(range(first_origin, last_origin + 1))
    origins = origins[-max(12, min(cfg.max_origins, len(origins))):]

    monthly_true: list[float] = []
    monthly_pred: list[float] = []
    monthly_by_h: dict[int, dict[str, list[float]]] = {h: {"true": [], "pred": []} for h in range(1, cfg.horizon + 1)}
    r12_true: list[float] = []
    r12_pred: list[float] = []
    annual_abs: list[float] = []
    annual_pct: list[float] = []
    annual_true: list[float] = []
    annual_pred: list[float] = []
    annual_rows: list[dict[str, Any]] = []

    folds = 0
    for origin_idx in origins:
        train = s.iloc[: origin_idx + 1]
        fc_h = min(cfg.horizon, n - origin_idx - 1)
        if fc_h <= 0:
            continue
        try:
            yhat = fit_predict_sarima(train, order, seasonal_order, cfg.horizon)
        except Exception as exc:  # noqa: BLE001
            logs.append(f"Backtest fold ignoré {order}x{seasonal_order} origin={train.index[-1].date()} : {exc}")
            continue
        if np.any(~np.isfinite(yhat)) or np.any(yhat < 0):
            return {"ok": False, "reason": "prévisions négatives", "folds": folds}

        origin_date = train.index[-1]
        folds += 1
        for h in range(1, fc_h + 1):
            u = origin_date + pd.DateOffset(months=h)
            if u not in s.index:
                continue
            yt = float(s.loc[u])
            yp = float(yhat[h - 1])
            monthly_true.append(yt)
            monthly_pred.append(yp)
            monthly_by_h[h]["true"].append(yt)
            monthly_by_h[h]["pred"].append(yp)

            start = u - pd.DateOffset(months=11)
            span = pd.date_range(start, u, freq="MS")
            r12_r = 0.0
            r12_p = 0.0
            ok = True
            for m in span:
                if m not in s.index:
                    ok = False
                    break
                rv = float(s.loc[m])
                if m <= origin_date:
                    pv = rv
                else:
                    step = (m.to_period("M") - origin_date.to_period("M")).n
                    if step <= 0 or step > len(yhat):
                        ok = False
                        break
                    pv = float(yhat[step - 1])
                r12_r += rv
                r12_p += pv
            if ok:
                r12_true.append(r12_r)
                r12_pred.append(r12_p)

        y = origin_date.year
        year_months = pd.date_range(f"{y}-01-01", f"{y}-12-01", freq="MS")
        if not set(year_months).issubset(set(s.index)):
            continue
        ytd_real = float(s[(s.index.year == y) & (s.index.month <= origin_date.month)].sum())
        remain_pred = 0.0
        rem_ok = True
        for m in year_months:
            if m.month <= origin_date.month:
                continue
            step = (m.to_period("M") - origin_date.to_period("M")).n
            if step <= 0 or step > len(yhat):
                rem_ok = False
                break
            remain_pred += float(yhat[step - 1])
        if rem_ok:
            annual_real_total = float(s[s.index.year == y].sum())
            annual_pred_total = ytd_real + remain_pred
            abs_err = abs(annual_pred_total - annual_real_total)
            pct_err = 0.0 if abs(annual_real_total) <= 1e-9 else abs_err / abs(annual_real_total) * 100
            annual_abs.append(abs_err)
            annual_pct.append(pct_err)
            annual_true.append(annual_real_total)
            annual_pred.append(annual_pred_total)
            annual_rows.append(
                {
                    "origin": origin_date,
                    "origin_month": origin_date.month,
                    "year": y,
                    "annual_real": annual_real_total,
                    "annual_pred": annual_pred_total,
                    "err_abs": abs_err,
                    "err_pct": pct_err,
                }
            )

    if folds == 0:
        return {"ok": False, "reason": "aucun fold valide", "folds": 0}

    monthly = _compute_error_metrics(monthly_true, monthly_pred)
    monthly_h_rows = []
    for h in range(1, cfg.horizon + 1):
        mh = _compute_error_metrics(monthly_by_h[h]["true"], monthly_by_h[h]["pred"])
        monthly_h_rows.append({"h": h, "n": len(monthly_by_h[h]["true"]), **mh})
    r12 = _compute_error_metrics(r12_true, r12_pred)
    annual_total = _compute_error_metrics(annual_true, annual_pred)

    annual_vs_month = pd.DataFrame(annual_rows)
    if not annual_vs_month.empty:
        annual_vs_month = (
            annual_vs_month.groupby("origin_month", as_index=False)
            .agg(
                n=("err_abs", "count"),
                err_abs_mean=("err_abs", "mean"),
                err_pct_mean=("err_pct", "mean"),
            )
            .sort_values("origin_month")
        )

    return {
        "ok": True,
        "folds": folds,
        "monthly": monthly,
        "monthly_by_horizon": pd.DataFrame(monthly_h_rows),
        "r12": r12,
        "annual": {
            "err_abs_mean": float(np.mean(annual_abs)) if annual_abs else np.nan,
            "err_pct_mean": float(np.mean(annual_pct)) if annual_pct else np.nan,
            **annual_total,
        },
        "annual_origin_table": annual_vs_month,
    }


def _stationarity_tests(series: pd.Series, seasonal_period: int) -> pd.DataFrame:
    s = series.dropna().sort_index().asfreq("MS")
    transforms = {
        "niveau": s,
        "diff_1": s.diff(1),
        f"diff_saison_{seasonal_period}": s.diff(seasonal_period),
        f"diff_1_plus_saison_{seasonal_period}": s.diff(1).diff(seasonal_period),
    }
    rows: list[dict[str, Any]] = []
    for name, ts in transforms.items():
        x = ts.dropna()
        if len(x) < 24:
            rows.append({"serie": name, "adf_pvalue": np.nan, "kpss_pvalue": np.nan, "adf_stat": np.nan, "kpss_stat": np.nan})
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
        except Exception:  # noqa: BLE001
            adf_stat, adf_p = np.nan, np.nan
        try:
            kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
        except Exception:  # noqa: BLE001
            kpss_stat, kpss_p = np.nan, np.nan
        rows.append({"serie": name, "adf_pvalue": adf_p, "kpss_pvalue": kpss_p, "adf_stat": adf_stat, "kpss_stat": kpss_stat})
    return pd.DataFrame(rows)


def _residual_diagnostics(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    cfg: SarimaSelectionConfig,
    logs: list[str],
) -> dict[str, Any]:
    s = series.dropna().sort_index().asfreq("MS")
    if len(s) > 36:
        ref = s.iloc[:-12]
    else:
        ref = s
    if len(ref) < 24:
        ref = s

    model = SARIMAX(
        ref,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    resid = pd.Series(fit.resid).dropna()

    lb = acorr_ljungbox(resid, lags=cfg.residual_lags, return_df=True)
    lb_dict = {int(lag): float(pv) for lag, pv in zip(lb.index.tolist(), lb["lb_pvalue"].tolist(), strict=False)}
    white_noise_ok = all((pv > cfg.penalty_lb_p) for pv in lb_dict.values()) if lb_dict else False

    max_lag = int(min(24, max(1, len(resid) // 2 - 1)))
    acf_vals = acf(resid, nlags=max_lag, fft=True)
    acf_df = pd.DataFrame({"lag": np.arange(len(acf_vals)), "acf": acf_vals})

    pacf_df = None
    try:
        pacf_vals = pacf(resid, nlags=max_lag, method="ywm")
        pacf_df = pd.DataFrame({"lag": np.arange(len(pacf_vals)), "pacf": pacf_vals})
    except Exception as exc:  # noqa: BLE001
        logs.append(f"PACF indisponible {order}x{seasonal_order}: {exc}")

    try:
        _jb_stat, jb_pvalue, _skew, _kurt = jarque_bera(resid)
        jb_p = float(jb_pvalue)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Jarque-Bera indisponible {order}x{seasonal_order}: {exc}")
        jb_p = np.nan

    return {
        "lb_pvalues": lb_dict,
        "white_noise_ok": bool(white_noise_ok),
        "acf_resid": acf_df,
        "pacf_resid": pacf_df,
        "jb_pvalue": jb_p,
    }




def _series_feature_diagnostics(series: pd.Series, seasonal_period: int) -> pd.DataFrame:
    s = series.dropna().sort_index().asfreq("MS")
    if len(s) < 24:
        return pd.DataFrame([
            {"feature": "history_months", "value": len(s), "interpretation": "Historique court: privilégier des modèles simples (p,q,P,Q faibles)."}
        ])

    trend_strength = float(abs(s.rolling(12, min_periods=12).mean().diff().dropna().mean() or 0) / max(abs(float(s.mean())), 1e-9))
    seasonal_profile = s.groupby(s.index.month).mean()
    seasonal_strength = float(seasonal_profile.std(ddof=0) / max(abs(float(s.mean())), 1e-9))
    volatility_ratio = float(s.diff().abs().mean() / max(abs(float(s.mean())), 1e-9))
    zero_share = float((s <= 1e-9).mean())

    return pd.DataFrame([
        {"feature": "history_months", "value": len(s), "interpretation": "Plus l'historique est long, plus on peut tolérer des ordres modérés."},
        {"feature": "trend_strength", "value": trend_strength, "interpretation": "Tendance élevée: d=1 est souvent pertinent; tendance faible: d=0 possible."},
        {"feature": "seasonality_strength", "value": seasonal_strength, "interpretation": "Saisonnalité forte: tester P/Q saisonniers; faible: rester parcimonieux."},
        {"feature": "volatility_ratio", "value": volatility_ratio, "interpretation": "Volatilité élevée: éviter les ordres trop hauts pour limiter l'instabilité."},
        {"feature": "zero_or_quasi_zero_share", "value": zero_share, "interpretation": "Présence de zéros: privilégier sMAPE côté lecture d'erreur et prudence sur MA."},
    ])

def _build_ui_text() -> dict[str, str]:
    return {
        "explain_stationarity": "Les tests de stationnarité servent de repère: on vérifie si la série brute ou différenciée devient plus stable dans le temps.",
        "explain_adf_kpss": "ADF teste H0=non-stationnaire (racine unitaire) alors que KPSS teste H0=stationnaire. Les deux sont complémentaires et indicatifs, pas un verdict automatique.",
        "explain_ljung_box": "Le Ljung–Box vérifie s'il reste de l'autocorrélation dans les résidus. p-value élevée: résidus proches d'un bruit blanc; p-value faible: structure non captée.",
        "explain_why_penalty": "Un modèle avec bons scores mais résidus autocorrélés peut être fragile. On applique une pénalisation douce pour privilégier les modèles plus robustes.",
        "explain_monthly_vs_r12_vs_annual": "Un modèle peut être excellent au mois mais se tromper sur les cumuls: R12 mesure la trajectoire glissante, annuel mesure l'atterrissage budgétaire de l'année civile.",
        "explain_why_annual_moves_in_S1": "En S1, le total annuel est sensible car beaucoup de mois restent à prévoir. En S2, le YTD réel pèse davantage, donc l'estimation se stabilise mécaniquement.",
        "explain_param_guide": "Guide de calibration: tendance forte => d=1; saisonnalité forte => activer P/Q saisonniers; volatilité forte => rester parcimonieux sur p/q pour éviter des prévisions instables.",
    }


def select_sarima_params(series: pd.Series, cfg: SarimaSelectionConfig) -> dict[str, Any]:
    logs: list[str] = []
    s = series.dropna().sort_index().asfreq("MS")

    stationarity_df = _stationarity_tests(s, cfg.seasonal_period)
    feature_diagnostics_df = _series_feature_diagnostics(s, cfg.seasonal_period)

    candidate_rows: list[dict[str, Any]] = []
    residual_by_candidate: dict[str, dict[str, Any]] = {}

    for order, seasonal_order in CANDIDATES:
        key = _candidate_key(order, seasonal_order)
        try:
            bt = compute_backtest_scores(s, order, seasonal_order, cfg, logs)
            if not bt.get("ok"):
                logs.append(f"Candidat ignoré {key}: {bt.get('reason', 'invalide')}")
                continue

            resid = _residual_diagnostics(s, order, seasonal_order, cfg, logs)
            residual_by_candidate[key] = resid

            lb12 = resid["lb_pvalues"].get(12, np.nan)
            lb24 = resid["lb_pvalues"].get(24, np.nan)
            white_noise_ok = bool(resid["white_noise_ok"])
            penalty = cfg.penalty_factor if not white_noise_ok else 1.0

            score_monthly = float(bt["monthly"]["rmse"])
            score_r12 = float(bt["r12"]["rmse"])
            score_annual = float(bt["annual"]["err_abs_mean"])

            candidate_rows.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "candidate": key,
                    "folds": int(bt["folds"]),
                    "lb_pvalue_lag12": lb12,
                    "lb_pvalue_lag24": lb24,
                    "white_noise_ok": white_noise_ok,
                    "penalty_applied": penalty != 1.0,
                    "score_monthly": score_monthly,
                    "score_monthly_adj": score_monthly * penalty,
                    "score_r12": score_r12,
                    "score_r12_adj": score_r12 * penalty,
                    "score_annual": score_annual,
                    "score_annual_adj": score_annual * penalty,
                    "monthly_mae": float(bt["monthly"]["mae"]),
                    "monthly_pct": float(bt["monthly"]["mape_or_smape"]),
                    "monthly_pct_metric": bt["monthly"]["pct_metric"],
                    "r12_mae": float(bt["r12"]["mae"]),
                    "r12_pct": float(bt["r12"]["mape_or_smape"]),
                    "r12_pct_metric": bt["r12"]["pct_metric"],
                    "annual_err_pct": float(bt["annual"]["err_pct_mean"]),
                    "annual_origin_table": bt["annual_origin_table"],
                    "monthly_by_horizon": bt["monthly_by_horizon"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logs.append(f"Candidat ignoré {key}: {exc}")

    if not candidate_rows:
        return {
            "best": {"annual": None, "monthly": None, "r12": None},
            "candidates_table": pd.DataFrame(),
            "stationarity_tests": stationarity_df,
            "series_feature_diagnostics": feature_diagnostics_df,
            "residual_diagnostics": {"by_candidate": residual_by_candidate},
            "ui_text": _build_ui_text(),
            "logs": logs,
        }

    candidates_df = pd.DataFrame(candidate_rows)
    best_monthly_row = candidates_df.sort_values("score_monthly_adj", ascending=True).iloc[0]
    best_r12_row = candidates_df.sort_values("score_r12_adj", ascending=True).iloc[0]
    best_annual_row = candidates_df.sort_values("score_annual_adj", ascending=True).iloc[0]

    def build_best(row: pd.Series) -> dict[str, Any]:
        return {
            "order": tuple(row["order"]),
            "seasonal_order": tuple(row["seasonal_order"]),
            "scores": {
                "score_monthly": float(row["score_monthly"]),
                "score_monthly_adj": float(row["score_monthly_adj"]),
                "score_r12": float(row["score_r12"]),
                "score_r12_adj": float(row["score_r12_adj"]),
                "score_annual": float(row["score_annual"]),
                "score_annual_adj": float(row["score_annual_adj"]),
            },
        }

    out_df = candidates_df.drop(columns=["annual_origin_table", "monthly_by_horizon"])

    return {
        "best": {
            "annual": build_best(best_annual_row),
            "monthly": build_best(best_monthly_row),
            "r12": build_best(best_r12_row),
        },
        "candidates_table": out_df,
        "stationarity_tests": stationarity_df,
        "series_feature_diagnostics": feature_diagnostics_df,
        "residual_diagnostics": {"by_candidate": residual_by_candidate},
        "ui_text": _build_ui_text(),
        "logs": logs,
    }
