from __future__ import annotations

from io import StringIO
from itertools import product
import threading
import unicodedata
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, session
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
app.secret_key = "local-dev-secret-key"

MAX_GRID_COMBINATIONS = 250
CSV_CACHE: dict[str, str] = {}
HW_GRID_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8]


def _format_fr(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.casefold()


def _parse_csv(file_content: str) -> pd.DataFrame:
    attempts = [{"sep": ";", "decimal": ","}, {"sep": None, "engine": "python"}, {}]
    for kwargs in attempts:
        try:
            df = pd.read_csv(StringIO(file_content), **kwargs)
            if not df.empty and len(df.columns) >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(StringIO(file_content))


def _prepare_dataframe(file_content: str, date_column: str | None, value_column: str | None) -> pd.Series:
    df = _parse_csv(file_content)
    if df.empty:
        raise ValueError("Le fichier CSV est vide.")

    if date_column is None or value_column is None:
        if len(df.columns) < 2:
            raise ValueError("Le CSV doit contenir au moins 2 colonnes : date et dépense.")
        date_column, value_column = df.columns[0], df.columns[1]

    if date_column not in df.columns or value_column not in df.columns:
        normalized_map = {_normalize_label(col): col for col in df.columns}
        date_column = normalized_map.get(_normalize_label(date_column), date_column)
        value_column = normalized_map.get(_normalize_label(value_column), value_column)

    if date_column not in df.columns or value_column not in df.columns:
        raise ValueError(f"Colonnes date/valeur introuvables. Colonnes détectées: {list(df.columns)}")

    data = df[[date_column, value_column]].copy()
    data.columns = ["date", "value"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    value_as_str = data["value"].astype(str).str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)

    def _fr_to_float_text(v: str) -> str:
        txt = str(v)
        if "," in txt:
            txt = txt.replace(".", "")
            txt = txt.replace(",", ".")
        return txt

    value_as_str = value_as_str.map(_fr_to_float_text).str.replace(r"[^0-9+\-\.]", "", regex=True)
    data["value"] = pd.to_numeric(value_as_str, errors="coerce")
    data = data.dropna()

    monthly = data.sort_values("date").set_index("date")["value"].resample("MS").sum()
    if len(monthly) < 24:
        raise ValueError("Il faut au moins 24 mois de données pour SARIMA/Holt-Winters.")
    return monthly


def _apply_cutoff_index(series: pd.Series, cutoff_index: str | None) -> tuple[pd.Series, pd.Timestamp, int]:
    idx = len(series) - 1 if not cutoff_index else int(cutoff_index)
    idx = max(0, min(idx, len(series) - 1))
    return series.iloc[: idx + 1], series.index[idx], idx


def _sarima_forecast(series: pd.Series, periods: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fobj = fit.get_forecast(steps=periods)
    return fit, fobj.predicted_mean, fobj.conf_int()


def _hw_fit_forecast(
    series: pd.Series,
    periods: int,
    seasonal_periods: int,
    trend: str | None,
    seasonal: str,
    alpha: float,
    beta: float,
    gamma: float,
):
    model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method="estimated")
    fit = model.fit(
        smoothing_level=alpha,
        smoothing_trend=beta if trend else None,
        smoothing_seasonal=gamma,
        optimized=False,
    )
    forecast = fit.forecast(periods)
    return fit, forecast


def _resolve_target_year(series: pd.Series, cutoff_year: int) -> int:
    counts = series.groupby(series.index.year).size()
    full_years = sorted(int(y) for y, c in counts.items() if c == 12 and int(y) <= cutoff_year - 1)
    if not full_years:
        raise ValueError("Aucune année civile complète disponible avant le cutoff pour le grid search.")
    return full_years[-1]


def _rolling_cumulative_mape_sarima(full_series: pd.Series, target_year: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> float:
    year_data = full_series[full_series.index.year == target_year]
    actual_total = float(year_data.sum())
    apes = []
    for month_dt in sorted(year_data.index):
        train = full_series[full_series.index < month_dt]
        if len(train) < 24:
            continue
        remaining = 12 - month_dt.month
        _, fc, _ = _sarima_forecast(train, max(remaining, 1), order, seasonal_order)
        projected_total = float(year_data[year_data.index <= month_dt].sum()) + (float(fc.iloc[:remaining].sum()) if remaining > 0 else 0.0)
        apes.append(abs(projected_total - actual_total) / abs(actual_total) * 100)
    if not apes:
        raise ValueError("Pas assez d'historique pour la MAPE cumulée.")
    return float(sum(apes) / len(apes))


def _annual_mape_sarima(full_series: pd.Series, target_year: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> float:
    year_data = full_series[full_series.index.year == target_year]
    train = full_series[full_series.index < pd.Timestamp(f"{target_year}-01-01")]
    _, fc, _ = _sarima_forecast(train, 12, order, seasonal_order)
    actual_total = float(year_data.sum())
    return abs(float(fc.sum()) - actual_total) / abs(actual_total) * 100


def _grid_values(max_value: int, fast_mode: bool) -> list[int]:
    if max_value <= 0:
        return [0]
    if not fast_mode:
        return list(range(max_value + 1))
    vals = {0, max_value}
    if max_value >= 2:
        vals.add(max_value // 2)
    return sorted(vals)


def _grid_search_sarima(series: pd.Series, target_year: int, s: int, max_p: int, max_d: int, max_q: int, max_P: int, max_D: int, max_Q: int, fast_mode: bool):
    combos = list(product(_grid_values(max_p, fast_mode), _grid_values(max_d, fast_mode), _grid_values(max_q, fast_mode), _grid_values(max_P, fast_mode), _grid_values(max_D, fast_mode), _grid_values(max_Q, fast_mode)))
    if len(combos) > MAX_GRID_COMBINATIONS:
        raise ValueError("Trop de combinaisons SARIMA, réduisez les bornes.")

    results = []
    for p, d, q, P, D, Q in combos:
        order = (p, d, q)
        seasonal = (P, D, Q, s)
        try:
            mape_roll = _rolling_cumulative_mape_sarima(series, target_year, order, seasonal)
            mape_annual = _annual_mape_sarima(series, target_year, order, seasonal)
            results.append({"order": order, "seasonal": seasonal, "mape": mape_roll, "annual_mape": mape_annual})
        except Exception:
            continue

    results.sort(key=lambda r: (r["mape"], r["annual_mape"], sum(r["order"]) + sum(r["seasonal"][:3])))
    return results, len(combos)


def _hw_rolling_cumulative_mape(full_series: pd.Series, target_year: int, s: int, seasonal: str, alpha: float, beta: float, gamma: float) -> float:
    year_data = full_series[full_series.index.year == target_year]
    actual_total = float(year_data.sum())
    apes = []
    for month_dt in sorted(year_data.index):
        train = full_series[full_series.index < month_dt]
        if len(train) < max(24, s * 2):
            continue
        remaining = 12 - month_dt.month
        _, fc = _hw_fit_forecast(train, max(remaining, 1), s, "add", seasonal, alpha, beta, gamma)
        projected_total = float(year_data[year_data.index <= month_dt].sum()) + (float(fc.iloc[:remaining].sum()) if remaining > 0 else 0.0)
        apes.append(abs(projected_total - actual_total) / abs(actual_total) * 100)
    if not apes:
        raise ValueError("Pas assez d'historique pour la MAPE HW.")
    return float(sum(apes) / len(apes))


def _annual_mape_hw(full_series: pd.Series, target_year: int, s: int, seasonal: str, alpha: float, beta: float, gamma: float) -> float:
    year_data = full_series[full_series.index.year == target_year]
    train = full_series[full_series.index < pd.Timestamp(f"{target_year}-01-01")]
    _, fc = _hw_fit_forecast(train, 12, s, "add", seasonal, alpha, beta, gamma)
    actual_total = float(year_data.sum())
    return abs(float(fc.sum()) - actual_total) / abs(actual_total) * 100


def _grid_search_hw(series: pd.Series, target_year: int, s: int, fast_mode: bool):
    vals = HW_GRID_VALUES[::2] if fast_mode else HW_GRID_VALUES
    combos = list(product(["add", "mul"], vals, vals, vals))
    results = []
    for seasonal, alpha, beta, gamma in combos:
        try:
            mape_roll = _hw_rolling_cumulative_mape(series, target_year, s, seasonal, alpha, beta, gamma)
            mape_annual = _annual_mape_hw(series, target_year, s, seasonal, alpha, beta, gamma)
            results.append({"seasonal_type": seasonal, "alpha": alpha, "beta": beta, "gamma": gamma, "mape": mape_roll, "annual_mape": mape_annual})
        except Exception:
            continue
    results.sort(key=lambda r: (r["mape"], r["annual_mape"]))
    return results, len(combos)


def _build_forecast_figure(history: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame, fitted: pd.Series, cutoff: pd.Timestamp) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines+markers", name="Réel"))
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode="lines", name="Ajusté (in-sample)", line=dict(color="#16a34a")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Prévision (horizon)", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int.iloc[:, 1], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int.iloc[:, 0], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(99,110,250,0.2)", name="Intervalle"))
    fig.add_vline(x=cutoff, line_width=2, line_dash="dot", line_color="#f97316")
    fig.update_layout(title="Prévision SARIMA", xaxis_title="Date", yaxis_title="Montant", template="plotly_white", xaxis=dict(rangeslider=dict(visible=True)))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_hw_figure(history: pd.Series, fitted: pd.Series, forecast: pd.Series, cutoff: pd.Timestamp, seasonal_type: str) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines+markers", name="Réel"))
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode="lines", name="Ajusté HW", line=dict(color="#9333ea")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Prévision HW", line=dict(dash="dash")))
    fig.add_vline(x=cutoff, line_width=2, line_dash="dot", line_color="#f97316")
    fig.update_layout(title=f"Prévision Holt-Winters ({seasonal_type})", xaxis_title="Date", yaxis_title="Montant", template="plotly_white", xaxis=dict(rangeslider=dict(visible=True)))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_budget_projection(history: pd.Series, forecast: pd.Series, year: int) -> tuple[pd.DataFrame, float, float, float]:
    labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    real_by_m = history[history.index.year == year].groupby(history[history.index.year == year].index.month).sum().to_dict()
    pred_by_m = forecast[forecast.index.year == year].groupby(forecast[forecast.index.year == year].index.month).sum().to_dict()

    cr = cp = cf = 0.0
    rows = []
    for m in range(1, 13):
        real = float(real_by_m.get(m, 0.0))
        pred = float(pred_by_m.get(m, 0.0))
        proj = real if real != 0 else pred
        cr += real
        cf += pred
        cp += proj
        rows.append({"Mois": labels[m - 1], "Réel": real, "Prédit": pred, "Projeté": proj, "Cumul réel": cr, "Cumul prédit": cf, "Cumul projeté": cp})

    return pd.DataFrame(rows), cp, cr, cf


def _build_budget_figure(df: pd.DataFrame, year: int) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Mois"], y=df["Cumul réel"], name="Cumul réel"))
    fig.add_trace(go.Scatter(x=df["Mois"], y=df["Cumul prédit"], mode="lines+markers", name="Cumul prédit"))
    fig.add_trace(go.Scatter(x=df["Mois"], y=df["Cumul projeté"], mode="lines+markers", name="Cumul projeté"))
    fig.update_layout(title=f"Budget annuel projeté - {year}", xaxis_title="Mois", yaxis_title="Montant cumulé", template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_decomposition_figure(series: pd.Series, s: int) -> str | None:
    if len(series) < max(24, s * 2):
        return None
    try:
        dec = seasonal_decompose(series, model="additive", period=s, extrapolate_trend="freq")
    except Exception:
        return None
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Série", "Tendance", "Saisonnalité", "Résidus"))
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dec.trend.index, y=dec.trend.values, mode="lines", line=dict(color="#16a34a")), row=2, col=1)
    fig.add_trace(go.Scatter(x=dec.seasonal.index, y=dec.seasonal.values, mode="lines", line=dict(color="#9333ea")), row=3, col=1)
    fig.add_trace(go.Scatter(x=dec.resid.index, y=dec.resid.values, mode="lines", line=dict(color="#ea580c")), row=4, col=1)
    fig.update_layout(height=900, title="Décomposition de la série", template="plotly_white", showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _ensure_client_id() -> str:
    client_id = session.get("client_id")
    if not client_id:
        client_id = str(pd.Timestamp.utcnow().value)
        session["client_id"] = client_id
    return client_id


@app.route("/", methods=["GET", "POST"])
def index():
    forecast_graph_html = budget_graph_html = decomposition_graph_html = hw_graph_html = None
    budget_table = top_models = hw_top_models = metrics = hw_metrics = None
    error = warning = None

    defaults = {
        "forecast_periods": "6", "cutoff_index": "", "cutoff_month": "",
        "p": "1", "d": "1", "q": "1", "P": "1", "D": "1", "Q": "1", "s": "12",
        "max_p": "2", "max_d": "1", "max_q": "2", "max_P": "1", "max_D": "1", "max_Q": "1",
        "date_column": "", "value_column": "", "run_grid_search": "on", "grid_fast": "on", "csv_cached": "",
    }
    cutoff_slider = None

    if request.method == "POST":
        form_data = {k: request.form.get(k, v) for k, v in defaults.items()}
        try:
            uploaded = request.files.get("csv_file")
            cid = _ensure_client_id()
            cached = CSV_CACHE.get(cid)
            if uploaded and uploaded.filename:
                raw = uploaded.read()
                try:
                    file_content = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    try:
                        file_content = raw.decode("cp1252")
                    except UnicodeDecodeError:
                        file_content = raw.decode("latin-1")
                CSV_CACHE[cid] = file_content
            elif cached:
                file_content = cached
            else:
                raise ValueError("Veuillez importer un fichier CSV au moins une fois.")

            series = _prepare_dataframe(file_content, form_data["date_column"] or None, form_data["value_column"] or None)
            history, cutoff, cutoff_idx = _apply_cutoff_index(series, form_data["cutoff_index"])
            cutoff_slider = {"min": 0, "max": len(series) - 1, "value": cutoff_idx, "labels": [d.strftime("%Y-%m") for d in series.index], "selected": cutoff.strftime("%Y-%m")}

            periods = max(1, int(form_data["forecast_periods"]))
            order = (int(form_data["p"]), int(form_data["d"]), int(form_data["q"]))
            s_period = max(2, int(form_data["s"]))
            seasonal = (int(form_data["P"]), int(form_data["D"]), int(form_data["Q"]), s_period)
            run_grid = request.form.get("run_grid_search") == "on"
            fast_mode = request.form.get("grid_fast") == "on"

            target_year = _resolve_target_year(history, cutoff.year)

            if run_grid:
                gs_results, tested = _grid_search_sarima(history, target_year, s_period, int(form_data["max_p"]), int(form_data["max_d"]), int(form_data["max_q"]), int(form_data["max_P"]), int(form_data["max_D"]), int(form_data["max_Q"]), fast_mode)
                if gs_results:
                    best = gs_results[0]
                    order, seasonal = best["order"], best["seasonal"]
                    top_models = [{"rank": i + 1, "order": str(r["order"]), "seasonal": str(r["seasonal"]), "mape": _format_fr(r["mape"], 3), "annual_mape": _format_fr(r["annual_mape"], 3)} for i, r in enumerate(gs_results[:10])]
                    metrics = {"target_year": target_year, "best_mape": _format_fr(best["mape"], 3), "best_annual_mape": _format_fr(best["annual_mape"], 3), "selected_order": str(order), "selected_seasonal": str(seasonal)}
                else:
                    warning = f"SARIMA grid: aucun modèle convergent ({tested} combinaisons)."

            months_to_year_end = max(0, 12 - cutoff.month)
            needed_steps = max(periods, months_to_year_end)

            fit, forecast, conf_int = _sarima_forecast(history, needed_steps, order, seasonal)
            forecast_graph_html = _build_forecast_figure(history, forecast.iloc[:periods], conf_int.iloc[:periods], fit.fittedvalues.reindex(history.index), cutoff)
            decomposition_graph_html = _build_decomposition_figure(history, s_period)

            # Holt-Winters comparison
            hw_results, hw_tested = _grid_search_hw(history, target_year, s_period, fast_mode)
            if hw_results:
                hw_best = hw_results[0]
                hw_fit, hw_fc = _hw_fit_forecast(history, needed_steps, s_period, "add", hw_best["seasonal_type"], hw_best["alpha"], hw_best["beta"], hw_best["gamma"])
                hw_graph_html = _build_hw_figure(history, hw_fit.fittedvalues.reindex(history.index), hw_fc.iloc[:periods], cutoff, hw_best["seasonal_type"])
                hw_top_models = [
                    {
                        "rank": i + 1,
                        "type": r["seasonal_type"],
                        "alpha": _format_fr(r["alpha"], 2),
                        "beta": _format_fr(r["beta"], 2),
                        "gamma": _format_fr(r["gamma"], 2),
                        "mape": _format_fr(r["mape"], 3),
                        "annual_mape": _format_fr(r["annual_mape"], 3),
                    }
                    for i, r in enumerate(hw_results[:10])
                ]
                hw_metrics = {
                    "target_year": target_year,
                    "type": hw_best["seasonal_type"],
                    "alpha": _format_fr(hw_best["alpha"], 2),
                    "beta": _format_fr(hw_best["beta"], 2),
                    "gamma": _format_fr(hw_best["gamma"], 2),
                    "best_mape": _format_fr(hw_best["mape"], 3),
                    "best_annual_mape": _format_fr(hw_best["annual_mape"], 3),
                    "tested": hw_tested,
                }

            budget_year = cutoff.year
            budget_df, projected_total, actual_total, predicted_total = _build_budget_projection(history, forecast, budget_year)
            budget_graph_html = _build_budget_figure(budget_df, budget_year)
            budget_table = [{k: (_format_fr(v, 2) if isinstance(v, (int, float)) else v) for k, v in row.items()} for row in budget_df.to_dict(orient="records")]

            if metrics is None:
                annual_mape = _annual_mape_sarima(history, target_year, order, seasonal)
                roll_mape = _rolling_cumulative_mape_sarima(history, target_year, order, seasonal)
                metrics = {
                    "target_year": target_year,
                    "best_mape": _format_fr(roll_mape, 3),
                    "best_annual_mape": _format_fr(annual_mape, 3),
                    "selected_order": str(order),
                    "selected_seasonal": str(seasonal),
                }

            metrics.update({"budget_year": budget_year, "projected_total": _format_fr(projected_total, 2), "actual_total": _format_fr(actual_total, 2), "predicted_total": _format_fr(predicted_total, 2)})

            form_data["csv_cached"] = "1"
            form_data["cutoff_index"] = str(cutoff_idx)
            form_data["cutoff_month"] = cutoff.strftime("%Y-%m")
            defaults.update(form_data)
            defaults["run_grid_search"] = "on" if run_grid else ""
            defaults["grid_fast"] = "on" if fast_mode else ""
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            defaults.update(form_data)
            if session.get("client_id") in CSV_CACHE:
                defaults["csv_cached"] = "1"

    if session.get("client_id") in CSV_CACHE:
        defaults["csv_cached"] = "1"

    return render_template(
        "index.html",
        forecast_graph_html=forecast_graph_html,
        budget_graph_html=budget_graph_html,
        decomposition_graph_html=decomposition_graph_html,
        hw_graph_html=hw_graph_html,
        budget_table=budget_table,
        top_models=top_models,
        hw_top_models=hw_top_models,
        metrics=metrics,
        hw_metrics=hw_metrics,
        error=error,
        warning=warning,
        defaults=defaults,
        cutoff_slider=cutoff_slider,
    )


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"

    def _open_browser() -> None:
        try:
            webbrowser.open_new(url)
        except Exception:
            pass

    threading.Timer(0.8, _open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)
