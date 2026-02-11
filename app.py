from __future__ import annotations

from io import StringIO
from itertools import product
import threading
import unicodedata
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request, session
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
app.secret_key = "local-dev-secret-key"

MAX_GRID_COMBINATIONS = 250
CSV_CACHE: dict[str, str] = {}


def _format_fr(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.casefold()


def _parse_csv(file_content: str) -> pd.DataFrame:
    attempts = [
        {"sep": ";", "decimal": ","},
        {"sep": None, "engine": "python"},
        {},
    ]
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
        date_column = df.columns[0]
        value_column = df.columns[1]

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

    value_as_str = value_as_str.map(_fr_to_float_text)
    value_as_str = value_as_str.str.replace(r"[^0-9+\-\.]", "", regex=True)
    data["value"] = pd.to_numeric(value_as_str, errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("Aucune donnée exploitable après nettoyage.")

    monthly = data.sort_values("date").set_index("date")["value"].resample("MS").sum()
    if len(monthly) < 24:
        raise ValueError("Il faut au moins 24 mois de données pour SARIMA.")

    return monthly


def _apply_cutoff_index(series: pd.Series, cutoff_index: str | None) -> tuple[pd.Series, pd.Timestamp, int]:
    idx = len(series) - 1 if not cutoff_index else int(cutoff_index)
    idx = max(0, min(idx, len(series) - 1))
    return series.iloc[: idx + 1], series.index[idx], idx


def _sarima_forecast(series: pd.Series, periods: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    forecast_obj = fit.get_forecast(steps=periods)
    forecast = forecast_obj.predicted_mean
    conf = forecast_obj.conf_int()
    if conf.shape[1] < 2:
        raise ValueError("Intervalle de confiance inattendu (conf_int).")
    return fit, forecast, conf


def _resolve_target_year(series: pd.Series, cutoff_year: int) -> int:
    counts = series.groupby(series.index.year).size()
    full_years = sorted(int(y) for y, c in counts.items() if c == 12 and int(y) <= cutoff_year - 1)
    if not full_years:
        raise ValueError("Aucune année civile complète disponible avant le cutoff pour le grid search.")
    return full_years[-1]


def _rolling_cumulative_mape(full_series: pd.Series, target_year: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> float:
    year_data = full_series[full_series.index.year == target_year]
    actual_total = float(year_data.sum())
    if len(year_data) != 12 or actual_total == 0:
        raise ValueError("Année cible invalide pour la MAPE.")

    apes: list[float] = []
    for month_dt in sorted(year_data.index):
        train = full_series[full_series.index < month_dt]
        if len(train) < 24:
            continue
        remaining = 12 - month_dt.month
        _, fc, _ = _sarima_forecast(train, max(remaining, 1), order, seasonal_order)
        observed_ytd = float(year_data[year_data.index <= month_dt].sum())
        projected_total = observed_ytd + (float(fc.iloc[:remaining].sum()) if remaining > 0 else 0.0)
        apes.append(abs(projected_total - actual_total) / abs(actual_total) * 100)

    if not apes:
        raise ValueError("Pas assez d'historique pour calculer la MAPE cumulée.")
    return float(sum(apes) / len(apes))


def _grid_values(max_value: int, fast_mode: bool) -> list[int]:
    if max_value <= 0:
        return [0]
    if not fast_mode:
        return list(range(max_value + 1))
    vals = {0, max_value}
    if max_value >= 2:
        vals.add(max_value // 2)
    return sorted(vals)


def _grid_search(series: pd.Series, target_year: int, seasonal_period: int, max_p: int, max_d: int, max_q: int, max_P: int, max_D: int, max_Q: int, fast_mode: bool):
    p_vals = _grid_values(max_p, fast_mode)
    d_vals = _grid_values(max_d, fast_mode)
    q_vals = _grid_values(max_q, fast_mode)
    P_vals = _grid_values(max_P, fast_mode)
    D_vals = _grid_values(max_D, fast_mode)
    Q_vals = _grid_values(max_Q, fast_mode)

    combos = list(product(p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals))
    if len(combos) > MAX_GRID_COMBINATIONS:
        raise ValueError(f"Trop de combinaisons ({len(combos)}). Réduisez les bornes.")

    results: list[dict] = []
    for p, d, q, P, D, Q in combos:
        order = (p, d, q)
        seasonal = (P, D, Q, seasonal_period)
        try:
            score = _rolling_cumulative_mape(series, target_year, order, seasonal)
            results.append({"order": order, "seasonal": seasonal, "mape": score})
        except Exception:
            continue

    results.sort(key=lambda item: (item["mape"], sum(item["order"]) + sum(item["seasonal"][:3]), item["order"], item["seasonal"]))
    return results, len(combos)


def _build_forecast_figure(history: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame, fitted: pd.Series, cutoff: pd.Timestamp) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines+markers", name="Réel"))
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode="lines", name="Ajusté (in-sample)", line=dict(color="#16a34a")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Prévision (horizon)", line=dict(dash="dash")))

    lower_col = conf_int.columns[0]
    upper_col = conf_int.columns[1]
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int[upper_col], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int[lower_col], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(99,110,250,0.2)", name="Intervalle"))

    fig.add_vline(x=cutoff, line_width=2, line_dash="dot", line_color="#f97316")
    fig.update_layout(title="Prévision SARIMA", xaxis_title="Date", yaxis_title="Montant", template="plotly_white", xaxis=dict(rangeslider=dict(visible=True)))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_budget_projection(history: pd.Series, forecast: pd.Series, year: int) -> tuple[pd.DataFrame, float, float, float]:
    labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    real_year = history[history.index.year == year]
    real_by_month = real_year.groupby(real_year.index.month).sum().to_dict()
    fc_year = forecast[forecast.index.year == year]
    fc_by_month = fc_year.groupby(fc_year.index.month).sum().to_dict()

    cumul_real = cumul_projected = cumul_predicted = 0.0
    rows = []
    for month in range(1, 13):
        real_val = float(real_by_month.get(month, 0.0))
        pure_pred = float(fc_by_month.get(month, 0.0))
        projected_val = real_val if real_val != 0 else pure_pred

        cumul_real += real_val
        cumul_projected += projected_val
        cumul_predicted += pure_pred

        rows.append({
            "Mois": labels[month - 1],
            "Réel": real_val,
            "Prédit": pure_pred,
            "Projeté": projected_val,
            "Cumul réel": cumul_real,
            "Cumul prédit": cumul_predicted,
            "Cumul projeté": cumul_projected,
        })

    return pd.DataFrame(rows), round(cumul_projected, 2), round(cumul_real, 2), round(cumul_predicted, 2)


def _build_budget_figure(yearly_df: pd.DataFrame, year: int) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_df["Mois"], y=yearly_df["Cumul réel"], name="Cumul réel"))
    fig.add_trace(go.Scatter(x=yearly_df["Mois"], y=yearly_df["Cumul prédit"], mode="lines+markers", name="Cumul prédit"))
    fig.add_trace(go.Scatter(x=yearly_df["Mois"], y=yearly_df["Cumul projeté"], mode="lines+markers", name="Cumul projeté"))
    fig.update_layout(title=f"Budget annuel projeté - {year}", xaxis_title="Mois", yaxis_title="Montant cumulé", template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _ensure_client_id() -> str:
    client_id = session.get("client_id")
    if not client_id:
        client_id = str(pd.Timestamp.utcnow().value)
        session["client_id"] = client_id
    return client_id


def _get_cached_csv(client_id: str) -> str | None:
    return CSV_CACHE.get(client_id)


def _set_cached_csv(client_id: str, content: str) -> None:
    CSV_CACHE[client_id] = content


@app.route("/", methods=["GET", "POST"])
def index():
    forecast_graph_html = budget_graph_html = None
    budget_table = top_models = metrics = None
    error = warning = None

    defaults = {
        "forecast_periods": "6",
        "cutoff_index": "",
        "cutoff_month": "",
        "p": "1", "d": "1", "q": "1", "P": "1", "D": "1", "Q": "1", "s": "12",
        "max_p": "2", "max_d": "1", "max_q": "2", "max_P": "1", "max_D": "1", "max_Q": "1",
        "date_column": "", "value_column": "", "run_grid_search": "on", "grid_fast": "on", "csv_cached": "",
    }
    cutoff_slider = None

    if request.method == "POST":
        form_data = {key: request.form.get(key, value) for key, value in defaults.items()}
        try:
            uploaded = request.files.get("csv_file")
            client_id = _ensure_client_id()
            cached_csv = _get_cached_csv(client_id)

            if uploaded and uploaded.filename:
                raw_bytes = uploaded.read()
                try:
                    file_content = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    try:
                        file_content = raw_bytes.decode("cp1252")
                    except UnicodeDecodeError:
                        file_content = raw_bytes.decode("latin-1")
                _set_cached_csv(client_id, file_content)
            elif cached_csv:
                file_content = cached_csv
            else:
                raise ValueError("Veuillez importer un fichier CSV au moins une fois.")

            series = _prepare_dataframe(file_content, form_data["date_column"] or None, form_data["value_column"] or None)
            history, cutoff, cutoff_idx = _apply_cutoff_index(series, form_data["cutoff_index"])
            cutoff_slider = {
                "min": 0,
                "max": len(series) - 1,
                "value": cutoff_idx,
                "labels": [dt.strftime("%Y-%m") for dt in series.index],
                "selected": cutoff.strftime("%Y-%m"),
            }

            periods = max(1, int(form_data["forecast_periods"]))
            order = (int(form_data["p"]), int(form_data["d"]), int(form_data["q"]))
            seasonal_period = max(2, int(form_data["s"]))
            seasonal = (int(form_data["P"]), int(form_data["D"]), int(form_data["Q"]), seasonal_period)

            run_grid = request.form.get("run_grid_search") == "on"
            fast_mode = request.form.get("grid_fast") == "on"

            if run_grid:
                target_year = _resolve_target_year(history, cutoff.year)
                gs_results, tested = _grid_search(
                    history,
                    target_year,
                    seasonal_period,
                    int(form_data["max_p"]),
                    int(form_data["max_d"]),
                    int(form_data["max_q"]),
                    int(form_data["max_P"]),
                    int(form_data["max_D"]),
                    int(form_data["max_Q"]),
                    fast_mode,
                )
                if gs_results:
                    best = gs_results[0]
                    order, seasonal = best["order"], best["seasonal"]
                    top_models = [
                        {
                            "rank": i + 1,
                            "order": str(item["order"]),
                            "seasonal": str(item["seasonal"]),
                            "mape": _format_fr(item["mape"], 3),
                        }
                        for i, item in enumerate(gs_results[:10])
                    ]
                    metrics = {
                        "target_year": target_year,
                        "best_mape": _format_fr(best["mape"], 3),
                        "selected_order": str(order),
                        "selected_seasonal": str(seasonal),
                    }
                else:
                    warning = f"Grid search: aucun modèle n'a convergé sur {tested} combinaisons. Paramètres manuels conservés."

            months_to_year_end = max(0, 12 - cutoff.month)
            needed_steps = max(periods, months_to_year_end)

            fit, forecast, conf_int = _sarima_forecast(history, needed_steps, order, seasonal)
            shown_forecast = forecast.iloc[:periods]
            shown_conf = conf_int.iloc[:periods]
            fitted = fit.fittedvalues.reindex(history.index)
            forecast_graph_html = _build_forecast_figure(history, shown_forecast, shown_conf, fitted, cutoff)

            budget_year = cutoff.year
            budget_df, projected_total, actual_total, predicted_total = _build_budget_projection(history, forecast, budget_year)
            budget_graph_html = _build_budget_figure(budget_df, budget_year)

            budget_table = []
            for row in budget_df.to_dict(orient="records"):
                budget_table.append({k: (_format_fr(v, 2) if isinstance(v, (int, float)) else v) for k, v in row.items()})

            if metrics is None:
                metrics = {
                    "target_year": budget_year,
                    "best_mape": None,
                    "selected_order": str(order),
                    "selected_seasonal": str(seasonal),
                }

            metrics.update(
                {
                    "budget_year": budget_year,
                    "projected_total": _format_fr(projected_total, 2),
                    "actual_total": _format_fr(actual_total, 2),
                    "predicted_total": _format_fr(predicted_total, 2),
                }
            )

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
        budget_table=budget_table,
        top_models=top_models,
        metrics=metrics,
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
