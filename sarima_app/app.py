"""Application Streamlit SARIMA avec UI orientée dashboard."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import webbrowser


def _is_probably_spyder() -> bool:
    """Détecte un lancement depuis Spyder même si --wdir est absent de sys.argv."""
    if "--wdir" in sys.argv:
        return True
    if any(k.startswith("SPYDER") for k in os.environ):
        return True
    return "spyder_kernels" in sys.modules


def _is_streamlit_child_process() -> bool:
    """Évite la récursion: ne pas relancer Streamlit depuis un process déjà lancé pour Streamlit."""
    if os.environ.get("SARIMA_STREAMLIT_CHILD") == "1":
        return True
    # Sécurité supplémentaire selon l'entrypoint courant
    argv0 = os.path.basename(sys.argv[0]).lower() if sys.argv else ""
    return "streamlit" in argv0


# Bootstrap Spyder: démarre Streamlit automatiquement pour ouvrir une URL locale.
if __name__ == "__main__" and _is_probably_spyder() and not _is_streamlit_child_process():
    url = "http://127.0.0.1:8501"
    if importlib.util.find_spec("streamlit") is None:
        print("[ERREUR] Le module streamlit n'est pas installé dans cet environnement Python.")
        print("[INFO] Installez les dépendances puis relancez:")
        print("       pip install -r requirements.txt")
        print("[INFO] Depuis la console IPython de Spyder, utilisez aussi: !streamlit run app.py")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.abspath(__file__),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    print("[INFO] Lancement automatique Streamlit depuis Spyder...")
    print("[INFO] URL:", url)
    try:
        child_env = os.environ.copy()
        child_env["SARIMA_STREAMLIT_CHILD"] = "1"
        subprocess.Popen(cmd, env=child_env)
        time.sleep(1.2)
        webbrowser.open(url)
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERREUR] Impossible de lancer Streamlit automatiquement: {exc}")
        print("[INFO] Lancez manuellement (terminal): streamlit run app.py")
        print("[INFO] Ou depuis la console Spyder: !streamlit run app.py")
        sys.exit(1)

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from streamlit.runtime.scriptrunner import get_script_run_ctx

from data_loader import load_csv
from evaluation import aggregate_by_period, annual_comparison
from models import SarimaForecaster

st.set_page_config(page_title="SARIMA Dépenses", layout="wide")

# Exécution sécurisée hors runtime Streamlit (ex: python app.py).
if get_script_run_ctx() is None:
    print("[INFO] Cette application doit être lancée via Streamlit.")
    print("[INFO] Terminal: streamlit run app.py")
    print("[INFO] Console Spyder/IPython: !streamlit run app.py")
    sys.exit(0)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; max-width: 1300px;}
      .stMetric {background: white; border:1px solid #d8e0ea; border-radius: 10px; padding:8px;}
      .title {color:#0f2744; font-weight:700; margin: 0.2rem 0 0.4rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Prévision SARIMA des dépenses mensuelles")
st.caption("Le graphique se recalcule automatiquement à chaque changement de paramètres ou de cutoff.")


@st.cache_data(show_spinner=False)
def cached_load(uploaded_file):
    return load_csv(uploaded_file)


def fmt_euro(v: float) -> str:
    return f"{v:,.2f} €".replace(",", " ").replace(".", ",")


with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Importer un CSV", type=["csv"])
    aggregation_mode = st.selectbox("Agrégation", ["Mensuel", "Trimestriel", "Quadrimestriel", "Semestriel"])
    agg_map = {"Mensuel": 1, "Trimestriel": 3, "Quadrimestriel": 4, "Semestriel": 6}

loaded = cached_load(uploaded)
st.info(loaded.message)
if loaded.data.empty:
    st.warning("Aucune donnée chargée. Importez un CSV dans la barre latérale.")
    st.stop()

# Correction du crash Windows: forcer explicitement le type datetime avant asfreq.
df = loaded.data.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "value"]).sort_values("date")
series = pd.Series(df["value"].values, index=pd.DatetimeIndex(df["date"]), name="value")
series = series[~series.index.duplicated(keep="last")].asfreq("MS")
valid_series = series.dropna()
if valid_series.empty:
    st.warning("Aucune observation mensuelle valide après parsing. Vérifiez le format des colonnes.")
    st.stop()

if len(valid_series) < 24:
    st.warning("Série trop courte: minimum 24 mois requis.")
    st.stop()

with st.sidebar:
    st.subheader("Paramètres SARIMA")
    p = st.slider("p", 0, 5, 1)
    d = st.slider("d", 0, 2, 1)
    q = st.slider("q", 0, 5, 1)
    P = st.slider("P", 0, 3, 1)
    D = st.slider("D", 0, 2, 1)
    Q = st.slider("Q", 0, 3, 1)

    min_cut = valid_series.index[23]
    max_cut = valid_series.index[-1]
    cutoff_value = st.slider(
        "Cutoff (dernier mois connu)",
        min_value=min_cut.to_pydatetime(),
        max_value=max_cut.to_pydatetime(),
        value=max_cut.to_pydatetime(),
        format="MM/YYYY",
    )
    cutoff = pd.Timestamp(cutoff_value).replace(day=1)
    horizon = st.slider("Horizon (mois)", 1, 36, 12)

train = series[series.index <= cutoff].dropna()
post_real = series[series.index > cutoff].dropna()
forecaster = SarimaForecaster(train)
fit = forecaster.fit((p, d, q), (P, D, Q, 12))

if not fit.success:
    st.error(fit.message)
    st.stop()

pred_df = forecaster.forecast(
    fit.model_fit,
    start_date=cutoff + pd.offsets.MonthBegin(1),
    horizon=horizon,
)
in_sample = fit.model_fit.get_prediction(start=train.index[0], end=train.index[-1]).predicted_mean
insample_df = pd.DataFrame({"date": in_sample.index, "value": in_sample.values})
all_pred = pd.concat([insample_df, pred_df[["date", "value"]]], ignore_index=True)

agg = agg_map[aggregation_mode]
real_pre_ag = aggregate_by_period(pd.DataFrame({"date": train.index, "value": train.values}), agg)
real_post_ag = aggregate_by_period(pd.DataFrame({"date": post_real.index, "value": post_real.values}), agg)
pred_ag = aggregate_by_period(all_pred, agg)
ci_low_ag = aggregate_by_period(pred_df[["date", "lower"]].rename(columns={"lower": "value"}), agg)
ci_high_ag = aggregate_by_period(pred_df[["date", "upper"]].rename(columns={"upper": "value"}), agg)

st.markdown("<p class='title'>Courbes réel vs prévision</p>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=real_pre_ag["date"], y=real_pre_ag["value"], mode="lines", name="Réel pré-cutoff", line=dict(color="#2f6db3", width=2)))
fig.add_trace(go.Scatter(x=real_post_ag["date"], y=real_post_ag["value"], mode="lines", name="Réel post-cutoff", line=dict(color="#8eb6dd", width=2, dash="dash")))
fig.add_trace(go.Scatter(x=ci_low_ag["date"], y=ci_low_ag["value"], mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=ci_high_ag["date"], y=ci_high_ag["value"], mode="lines", fill="tonexty", fillcolor="rgba(120,120,120,0.2)", line=dict(width=0), name="IC 95%"))
fig.add_trace(go.Scatter(x=pred_ag["date"], y=pred_ag["value"], mode="lines", name="Prévision SARIMA", line=dict(color="#f07a24", width=2)))
fig.update_layout(height=520, margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Date", yaxis_title="Montant (€)")
st.plotly_chart(fig, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("AIC", f"{fit.aic:.2f}")
m2.metric("BIC", f"{fit.bic:.2f}")
m3.metric("Log-likelihood", f"{fit.llf:.2f}")

st.markdown("<p class='title'>Comparaison annuelle cumulée</p>", unsafe_allow_html=True)
annual = annual_comparison(pd.DataFrame({"date": valid_series.index, "value": valid_series.values}), all_pred)
if not annual.empty:
    styled = annual.style.format({"Σ Réel": fmt_euro, "Σ Prévu": fmt_euro, "Écart absolu": fmt_euro, "Écart relatif %": "{:.2f}%"}).apply(
        lambda row: ["background-color:#fff0f2" if row["Écart relatif %"] > 5 else "" for _ in row], axis=1
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.warning("Aucune année complète commune entre réel et prévision.")

st.markdown("<p class='title'>Grid Search SARIMA (AIC uniquement, optionnel)</p>", unsafe_allow_html=True)
with st.expander("Paramétrer le grid search"):
    c = st.columns(3)
    p_min, p_max = c[0].number_input("p min", 0, 5, 0), c[0].number_input("p max", 0, 5, 2)
    d_min, d_max = c[1].number_input("d min", 0, 2, 0), c[1].number_input("d max", 0, 2, 1)
    q_min, q_max = c[2].number_input("q min", 0, 5, 0), c[2].number_input("q max", 0, 5, 2)
    c2 = st.columns(3)
    P_min, P_max = c2[0].number_input("P min", 0, 3, 0), c2[0].number_input("P max", 0, 3, 1)
    D_min, D_max = c2[1].number_input("D min", 0, 2, 0), c2[1].number_input("D max", 0, 2, 1)
    Q_min, Q_max = c2[2].number_input("Q min", 0, 3, 0), c2[2].number_input("Q max", 0, 3, 1)
    total = (p_max-p_min+1)*(d_max-d_min+1)*(q_max-q_min+1)*(P_max-P_min+1)*(D_max-D_min+1)*(Q_max-Q_min+1)
    st.write(f"Combinaisons: **{int(total)}**")
    max_combo = st.number_input("Limiter à N combinaisons", 1, 1000, min(int(total), 120))

    if st.button("Lancer le Grid Search"):
        bar = st.progress(0)
        search_space = {
            "p_min": int(p_min), "p_max": int(p_max), "d_min": int(d_min), "d_max": int(d_max),
            "q_min": int(q_min), "q_max": int(q_max), "P_min": int(P_min), "P_max": int(P_max),
            "D_min": int(D_min), "D_max": int(D_max), "Q_min": int(Q_min), "Q_max": int(Q_max),
        }
        top = forecaster.grid_search(
            train,
            search_space=search_space,
            target_years=[],
            criterion="AIC",
            max_combinations=int(max_combo),
            progress_callback=lambda v: bar.progress(min(1.0, v)),
        )
        st.dataframe(pd.DataFrame(top) if top else pd.DataFrame(columns=["p", "d", "q", "P", "D", "Q", "score", "aic"]), use_container_width=True)

st.markdown("<p class='title'>Diagnostics SARIMA</p>", unsafe_allow_html=True)
diag = forecaster.diagnostics(fit.model_fit)
st.write(f"p-value Ljung-Box (lag 12): **{diag['ljung_box_pvalue']:.4f}**")
fig_m, axes = plt.subplots(2, 2, figsize=(10, 6))
axes[0, 0].hist(diag["residuals"], bins=20)
axes[0, 0].set_title("Histogramme résidus")
pd.Series(diag["residuals"]).plot(kind="kde", ax=axes[0, 1], title="Densité résidus")
plot_acf(diag["residuals"], ax=axes[1, 0], lags=24)
plot_pacf(diag["residuals"], ax=axes[1, 1], lags=24, method="ywm")
plt.tight_layout()
st.pyplot(fig_m)
