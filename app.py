from __future__ import annotations

from io import StringIO
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)


def _prepare_dataframe(file_content: str, date_column: str | None, value_column: str | None) -> pd.DataFrame:
    df = pd.read_csv(StringIO(file_content))

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
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("Aucune donnée exploitable après nettoyage (dates/valeurs invalides).")

    data = data.sort_values("date").set_index("date")
    monthly = data["value"].resample("MS").sum()

    if len(monthly) < 24:
        raise ValueError("Il faut au moins 24 mois de données pour une prévision SARIMA fiable.")

    return monthly.to_frame(name="value")


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
    return forecast, conf


def _build_figure(history: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame) -> str:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines+markers",
            name="Historique dépenses",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Prévision",
            line=dict(dash="dash"),
        )
    )

    lower_name = conf_int.columns[0]
    upper_name = conf_int.columns[1]

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=conf_int[upper_name],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Borne haute",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=conf_int[lower_name],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)",
            name="Intervalle de confiance",
        )
    )

    fig.update_layout(
        title="Prévision des dépenses mensuelles (SARIMA)",
        xaxis_title="Date",
        yaxis_title="Montant",
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/", methods=["GET", "POST"])
def index():
    graph_html = None
    error = None

    defaults = {
        "forecast_periods": "6",
        "p": "1",
        "d": "1",
        "q": "1",
        "P": "1",
        "D": "1",
        "Q": "1",
        "s": "12",
        "date_column": "",
        "value_column": "",
    }

    if request.method == "POST":
        form_data = {key: request.form.get(key, value) for key, value in defaults.items()}
        try:
            uploaded = request.files.get("csv_file")
            if not uploaded or uploaded.filename == "":
                raise ValueError("Veuillez importer un fichier CSV.")

            file_content = uploaded.read().decode("utf-8")
            df = _prepare_dataframe(file_content, form_data["date_column"] or None, form_data["value_column"] or None)

            periods = max(1, int(form_data["forecast_periods"]))
            order = (int(form_data["p"]), int(form_data["d"]), int(form_data["q"]))
            seasonal_order = (int(form_data["P"]), int(form_data["D"]), int(form_data["Q"]), int(form_data["s"]))

            forecast, conf_int = _sarima_forecast(df["value"], periods, order, seasonal_order)
            graph_html = _build_figure(df["value"], forecast, conf_int)

            defaults.update(form_data)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            defaults.update(form_data)

    return render_template("index.html", graph_html=graph_html, error=error, defaults=defaults)


if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port, debug=True)
