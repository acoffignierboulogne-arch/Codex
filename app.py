from __future__ import annotations

from io import StringIO
from itertools import product

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request, session
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
app.secret_key = "local-dev-secret-key"

MAX_GRID_COMBINATIONS = 250
CSV_CACHE: dict[str, str] = {}


def _parse_csv(file_content: str) -> pd.DataFrame:
    """Parse robuste pour CSV FR/EN (BOM, ;, ,)."""
    try:
        return pd.read_csv(StringIO(file_content), sep=None, engine="python")
    except Exception:
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
        raise ValueError("Colonnes date/valeur introuvables dans le CSV.")

    data = df[[date_column, value_column]].copy()
    data.columns = ["date", "value"]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    value_as_str = data["value"].astype(str).str.replace(" ", "", regex=False)
    value_as_str = value_as_str.str.replace(",", ".", regex=False)
    data["value"] = pd.to_numeric(value_as_str, errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("Aucune donnée exploitable après nettoyage (dates/valeurs invalides).")

    data = data.sort_values("date").set_index("date")
    monthly = data["value"].resample("MS").sum()

    if len(monthly) < 24:
        raise ValueError("Il faut au moins 24 mois de données pour une prévision SARIMA fiable.")

    return monthly


def _apply_cutoff_index(series: pd.Series, cutoff_index: str | None) -> tuple[pd.Series, pd.Timestamp, int]:
    if cutoff_index is None or cutoff_index == "":
        idx = len(series) - 1
    else:
        idx = int(cutoff_index)

    idx = max(0, min(idx, len(series) - 1))
    cutoff = series.index[idx]
    filtered = series.iloc[: idx + 1]

    if filtered.empty:
        raise ValueError("Aucune donnée <= cutoff. Ajustez le curseur.")

    return filtered, cutoff, idx


def _sarima_forecast(series: pd.Series, periods: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    forecast_obj = fit.get_forecast(steps=periods)
    forecast = forecast_obj.predicted_mean
    conf = forecast_obj.conf_int()

    if conf.shape[1] < 2:
        raise ValueError("Intervalle de confiance inattendu (conf_int).")

    return fit, forecast, conf


def _last_complete_calendar_year(series: pd.Series) -> int:
    counts = series.groupby(series.index.year).size()
    full_years = counts[counts == 12]
    if full_years.empty:
        raise ValueError("Aucune année civile complète (12 mois) trouvée pour le scoring MAPE.")
    return int(full_years.index.max())


def _rolling_cumulative_mape(full_series: pd.Series, target_year: int, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> float:
    year_data = full_series[full_series.index.year == target_year]
    if len(year_data) != 12:
        raise ValueError("L'année cible doit contenir 12 mois réels.")

    actual_total = float(year_data.sum())
    if actual_total == 0:
        raise ValueError("Le total annuel réel est nul, la MAPE n'est pas définie.")

    apes: list[float] = []
    for month_dt in sorted(year_data.index):
        train = full_series[full_series.index < month_dt]
        if len(train) < 24:
            continue

        remaining = 12 - month_dt.month
        _, fc, _ = _sarima_forecast(train, max(remaining, 1), order, seasonal_order)

        observed_ytd = float(year_data[year_data.index <= month_dt].sum())
        predicted_rest = float(fc.iloc[:remaining].sum()) if remaining > 0 else 0.0
        projected_total = observed_ytd + predicted_rest

        ape = abs(projected_total - actual_total) / abs(actual_total) * 100
        apes.append(ape)

    if not apes:
        raise ValueError("Pas assez d'historique pour calculer la MAPE cumulée rolling.")

    return float(sum(apes) / len(apes))


def _grid_search(series: pd.Series, seasonal_period: int, max_p: int, max_d: int, max_q: int, max_P: int, max_D: int, max_Q: int):
    target_year = _last_complete_calendar_year(series)

    combos = list(product(range(max_p + 1), range(max_d + 1), range(max_q + 1), range(max_P + 1), range(max_D + 1), range(max_Q + 1)))
    if len(combos) > MAX_GRID_COMBINATIONS:
        raise ValueError(f"Trop de combinaisons ({len(combos)}). Réduisez les bornes (max {MAX_GRID_COMBINATIONS}).")

    results: list[dict] = []
    for p, d, q, P, D, Q in combos:
        order = (p, d, q)
        seasonal = (P, D, Q, seasonal_period)
        try:
            score = _rolling_cumulative_mape(series, target_year, order, seasonal)
            results.append({"order": order, "seasonal": seasonal, "mape": score})
        except Exception:
            continue

    results.sort(key=lambda item: item["mape"])
    return target_year, results, len(combos)


def _build_forecast_figure(history: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines+markers", name="Historique (jusqu'au cutoff)"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Prévision", line=dict(dash="dash")))

    lower_col = conf_int.columns[0]
    upper_col = conf_int.columns[1]
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int[upper_col], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=conf_int[lower_col],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)",
            name="Intervalle de confiance",
        )
    )

    fig.add_vline(x=cutoff, line_width=2, line_dash="dot", line_color="#f97316")
    fig.update_layout(
        title="Prévision des dépenses mensuelles (SARIMA)",
        xaxis_title="Date",
        yaxis_title="Montant",
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_budget_projection(history: pd.Series, forecast: pd.Series, year: int) -> tuple[pd.DataFrame, float, float]:
    labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    real_year = history[history.index.year == year]
    real_by_month = real_year.groupby(real_year.index.month).sum().to_dict()
    fc_year = forecast[forecast.index.year == year]
    fc_by_month = fc_year.groupby(fc_year.index.month).sum().to_dict()

    cumul_real = 0.0
    cumul_projected = 0.0
    rows = []

    for month in range(1, 13):
        real_val = float(real_by_month.get(month, 0.0))
        projected_val = real_val if real_val != 0 else float(fc_by_month.get(month, 0.0))
        cumul_real += real_val
        cumul_projected += projected_val

        rows.append(
            {
                "Mois": labels[month - 1],
                "Réel": round(real_val, 2),
                "Projeté": round(projected_val, 2),
                "Cumul réel": round(cumul_real, 2),
                "Cumul projeté": round(cumul_projected, 2),
            }
        )

    return pd.DataFrame(rows), round(cumul_projected, 2), round(cumul_real, 2)


def _build_budget_figure(yearly_df: pd.DataFrame, year: int) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_df["Mois"], y=yearly_df["Cumul réel"], name="Cumul réel"))
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
    forecast_graph_html = None
    budget_graph_html = None
    budget_table = None
    top_models = None
    metrics = None
    error = None
    warning = None

    defaults = {
        "forecast_periods": "6",
        "cutoff_index": "",
        "cutoff_month": "",
        "p": "1",
        "d": "1",
        "q": "1",
        "P": "1",
        "D": "1",
        "Q": "1",
        "s": "12",
        "max_p": "2",
        "max_d": "1",
        "max_q": "2",
        "max_P": "1",
        "max_D": "1",
        "max_Q": "1",
        "date_column": "",
        "value_column": "",
        "run_grid_search": "on",
        "csv_cached": "",
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
                    file_content = raw_bytes.decode("latin-1")
                _set_cached_csv(client_id, file_content)
            elif cached_csv:
                file_content = cached_csv
            else:
                raise ValueError("Veuillez importer un fichier CSV au moins une fois (ensuite il sera conservé en mémoire).")

            series = _prepare_dataframe(file_content, form_data["date_column"] or None, form_data["value_column"] or None)
            history, cutoff, cutoff_idx = _apply_cutoff_index(series, form_data["cutoff_index"])
            cutoff_labels = [dt.strftime("%Y-%m") for dt in series.index]
            cutoff_slider = {
                "min": 0,
                "max": len(series) - 1,
                "value": cutoff_idx,
                "labels": cutoff_labels,
                "selected": cutoff.strftime("%Y-%m"),
            }

            periods = max(1, int(form_data["forecast_periods"]))
            order = (int(form_data["p"]), int(form_data["d"]), int(form_data["q"]))
            seasonal_period = max(2, int(form_data["s"]))
            seasonal = (int(form_data["P"]), int(form_data["D"]), int(form_data["Q"]), seasonal_period)
            run_grid = request.form.get("run_grid_search") == "on"

            if run_grid:
                target_year, gs_results, tested = _grid_search(
                    history,
                    seasonal_period,
                    int(form_data["max_p"]),
                    int(form_data["max_d"]),
                    int(form_data["max_q"]),
                    int(form_data["max_P"]),
                    int(form_data["max_D"]),
                    int(form_data["max_Q"]),
                )
                if gs_results:
                    best = gs_results[0]
                    order = best["order"]
                    seasonal = best["seasonal"]
                    top_models = [
                        {
                            "rank": i + 1,
                            "order": str(item["order"]),
                            "seasonal": str(item["seasonal"]),
                            "mape": round(item["mape"], 3),
                        }
                        for i, item in enumerate(gs_results[:10])
                    ]
                    metrics = {
                        "target_year": target_year,
                        "best_mape": round(best["mape"], 3),
                        "selected_order": str(order),
                        "selected_seasonal": str(seasonal),
                    }
                else:
                    warning = (
                        f"Grid search: aucun modèle n'a convergé sur {tested} combinaisons. "
                        "On conserve les paramètres manuels saisis."
                    )

            months_to_year_end = max(0, 12 - cutoff.month)
            needed_steps = max(periods, months_to_year_end)

            _, forecast, conf_int = _sarima_forecast(history, needed_steps, order, seasonal)
            shown_forecast = forecast.iloc[:periods]
            shown_conf = conf_int.iloc[:periods]
            forecast_graph_html = _build_forecast_figure(history, shown_forecast, shown_conf, cutoff)

            budget_year = cutoff.year
            budget_df, projected_total, actual_total = _build_budget_projection(history, forecast, budget_year)
            budget_graph_html = _build_budget_figure(budget_df, budget_year)
            budget_table = budget_df.to_dict(orient="records")

            if metrics is None:
                metrics = {
                    "target_year": budget_year,
                    "best_mape": None,
                    "selected_order": str(order),
                    "selected_seasonal": str(seasonal),
                }

            metrics.update({"budget_year": budget_year, "projected_total": projected_total, "actual_total": actual_total})

            form_data["csv_cached"] = "1"
            form_data["cutoff_index"] = str(cutoff_idx)
            form_data["cutoff_month"] = cutoff.strftime("%Y-%m")
            defaults.update(form_data)
            defaults["run_grid_search"] = "on" if run_grid else ""
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
    # Spyder/%runfile: éviter SystemExit lié au watchdog du reloader.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)
