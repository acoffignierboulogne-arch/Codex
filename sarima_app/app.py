"""Application Streamlit SARIMA avec UI orientée dashboard."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import webbrowser


def _is_probably_spyder() -> bool:
    if "--wdir" in sys.argv:
        return True
    if any(k.startswith("SPYDER") for k in os.environ):
        return True
    return "spyder_kernels" in sys.modules


def _is_streamlit_child_process() -> bool:
    if os.environ.get("SARIMA_STREAMLIT_CHILD") == "1":
        return True
    argv0 = os.path.basename(sys.argv[0]).lower() if sys.argv else ""
    return "streamlit" in argv0


if __name__ == "__main__" and _is_probably_spyder() and not _is_streamlit_child_process():
    url = "http://127.0.0.1:8501"
    if importlib.util.find_spec("streamlit") is None:
        print("[ERREUR] Le module streamlit n'est pas installé dans cet environnement Python.")
        print("[INFO] Installez les dépendances puis relancez:")
        print("       pip install -r requirements.txt")
        print("[INFO] Depuis la console IPython de Spyder, utilisez aussi: !streamlit run app.py")
        sys.exit(1)
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), "--server.address", "127.0.0.1", "--server.port", "8501"]
    print("[INFO] Lancement automatique Streamlit depuis Spyder...")
    print("[INFO] URL:", url)
    child_env = os.environ.copy()
    child_env["SARIMA_STREAMLIT_CHILD"] = "1"
    subprocess.Popen(cmd, env=child_env)
    time.sleep(1.2)
    webbrowser.open(url)
    sys.exit(0)

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from streamlit.runtime.scriptrunner import get_script_run_ctx

from data_loader import load_csv
from evaluation import aggregate_by_period, annual_comparison
from models import SarimaForecaster

st.set_page_config(page_title="SARIMA Dépenses", layout="wide")
if get_script_run_ctx() is None:
    print("[INFO] Cette application doit être lancée via Streamlit.")
    print("[INFO] Terminal: streamlit run app.py")
    print("[INFO] Console Spyder/IPython: !streamlit run app.py")
    sys.exit(0)

st.markdown("""
<style>
.block-container {padding-top: 1rem; max-width: 1300px;}
.title {color:#0f2744; font-weight:700; margin: 0.2rem 0 0.4rem 0;}
.note {color:#5b6b7b; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("Prévision SARIMA des dépenses mensuelles")
st.caption("Graphique en priorité, réglages en sidebar, recalcul automatique à chaque changement.")

for k, v in {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1}.items():
    st.session_state.setdefault(k, v)

# Applique les meilleurs paramètres AVANT l'instanciation des widgets (contrainte Streamlit).
pending_best = st.session_state.pop("pending_best_params", None)
if pending_best is not None:
    for key in ["p", "d", "q", "P", "D", "Q"]:
        st.session_state[key] = int(pending_best[key])

@st.cache_data(show_spinner=False)
def cached_load(uploaded_file):
    return load_csv(uploaded_file)


def fmt_euro(v: float) -> str:
    return f"{v:,.2f} €".replace(",", " ").replace(".", ",")


def compute_projection_yearly(real_series: pd.Series, pred_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    real_map = {d.strftime("%Y-%m"): float(v) for d, v in real_series.dropna().items()}
    pred_map = {pd.Timestamp(r.date).strftime("%Y-%m"): float(r.value) for r in pred_df.itertuples()}
    start_year = real_series.dropna().index.min().year
    end_year = max(real_series.dropna().index.max().year, pred_df["date"].max().year)
    rows = []
    for year in range(start_year, end_year + 1):
        real_cum = 0.0
        proj_cum = 0.0
        covered = 0
        for month in range(1, 13):
            d = pd.Timestamp(year=year, month=month, day=1)
            key = d.strftime("%Y-%m")
            if key in real_map:
                real_cum += real_map[key]
            proj_val = None
            if d <= cutoff and key in real_map:
                proj_val = real_map[key]
            elif key in pred_map:
                proj_val = pred_map[key]
            if proj_val is not None:
                proj_cum += proj_val
                covered += 1
        if covered > 0:
            abs_gap = abs(real_cum - proj_cum)
            rel_gap = 0 if real_cum == 0 else abs_gap / abs(real_cum) * 100
            rows.append({"Année": year, "Réel cumulé": real_cum, "Prévision cumulée": proj_cum, "Écart abs": abs_gap, "Écart rel %": rel_gap, "Mois couverts": covered})
    return pd.DataFrame(rows)

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

df = loaded.data.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "value"]).sort_values("date")
series = pd.Series(df["value"].values, index=pd.DatetimeIndex(df["date"]), name="value")
series = series[~series.index.duplicated(keep="last")].asfreq("MS")
valid_series = series.dropna()
if len(valid_series) < 24:
    st.warning("Série trop courte: minimum 24 mois requis.")
    st.stop()

with st.sidebar:
    st.subheader("Paramètres SARIMA")
    p = st.slider("p", 0, 5, key="p")
    st.caption("p (AR): mémoire des valeurs passées. ↑p = modèle plus sensible aux dépendances temporelles, mais plus complexe.")
    d = st.slider("d", 0, 2, key="d")
    st.caption("d (différenciation): retire la tendance de fond. d=0 si série déjà stable, d=1/2 si tendance marquée.")
    q = st.slider("q", 0, 5, key="q")
    st.caption("q (MA): corrige les chocs/erreurs récents. ↑q aide si la série subit des à-coups court terme.")
    P = st.slider("P", 0, 3, key="P")
    st.caption("P (AR saisonnier): mémoire des mêmes mois des années passées.")
    D = st.slider("D", 0, 2, key="D")
    st.caption("D (diff saisonnière): enlève une saisonnalité structurelle annuelle persistante.")
    Q = st.slider("Q", 0, 3, key="Q")
    st.caption("Q (MA saisonnier): absorbe les chocs saisonniers récurrents.")

    min_cut = valid_series.index[23]
    max_cut = valid_series.index[-1]
    cutoff = pd.Timestamp(st.slider("Cutoff", min_value=min_cut.to_pydatetime(), max_value=max_cut.to_pydatetime(), value=max_cut.to_pydatetime(), format="MM/YYYY")).replace(day=1)
    horizon = st.slider("Horizon (mois)", 1, 36, 12)

train = series[series.index <= cutoff].dropna()
post_real = series[series.index > cutoff].dropna()
forecaster = SarimaForecaster(train)
fit = forecaster.fit((p, d, q), (P, D, Q, 12))
if not fit.success:
    st.error(fit.message)
    st.stop()

pred_df = forecaster.forecast(fit.model_fit, start_date=cutoff + pd.offsets.MonthBegin(1), horizon=horizon)
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
st.markdown("<p class='note'>AIC/BIC: critères de qualité pénalisant la complexité (plus bas = meilleur compromis). Log-likelihood: qualité d'ajustement brut (plus haut = mieux).</p>", unsafe_allow_html=True)

seasonality_ratio = float(valid_series.groupby(valid_series.index.month).mean().std() / max(valid_series.std(), 1e-9))
volatility_ratio = float(valid_series.std() / max(abs(valid_series.mean()), 1e-9))
ll_note = "Un log-likelihood négatif est fréquent sur des séries monétaires: ce n'est PAS, à lui seul, un mauvais signe."
if fit.llf is not None and fit.llf > 0:
    ll_note = "Log-likelihood positif: ajustement statistique global plutôt favorable."

st.markdown(
    f"""
<div class='note'>
<b>Lecture rapide de vos données :</b><br>
- Saisonnière estimée : <b>{'forte' if seasonality_ratio > 0.35 else 'modérée' if seasonality_ratio > 0.15 else 'faible'}</b> (ratio saisonnalité ≈ {seasonality_ratio:.2f}).<br>
- Volatilité relative : <b>{'élevée' if volatility_ratio > 0.6 else 'moyenne' if volatility_ratio > 0.25 else 'faible'}</b> (écart-type / moyenne ≈ {volatility_ratio:.2f}).<br>
- {ll_note}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<p class='title'>Comparaison annuelle cumulée (années complètes)</p>", unsafe_allow_html=True)
annual = annual_comparison(pd.DataFrame({"date": valid_series.index, "value": valid_series.values}), all_pred)
if not annual.empty:
    styled = annual.style.format({"Σ Réel": fmt_euro, "Σ Prévu": fmt_euro, "Écart absolu": fmt_euro, "Écart relatif %": "{:.2f}%"})
    st.dataframe(styled, use_container_width=True)

st.markdown("<p class='title'>Projection annuelle (inclut années incomplètes, ex: 2025)</p>", unsafe_allow_html=True)
projection = compute_projection_yearly(valid_series, all_pred, cutoff)
if not projection.empty:
    st.dataframe(
        projection.style.format({"Réel cumulé": fmt_euro, "Prévision cumulée": fmt_euro, "Écart abs": fmt_euro, "Écart rel %": "{:.2f}%"}),
        use_container_width=True,
    )

st.markdown("<p class='title'>Grid Search SARIMA</p>", unsafe_allow_html=True)
with st.expander("Configurer et lancer"):
    c = st.columns(3)
    p_min, p_max = c[0].number_input("p min", 0, 5, 0), c[0].number_input("p max", 0, 5, 2)
    d_min, d_max = c[1].number_input("d min", 0, 2, 0), c[1].number_input("d max", 0, 2, 1)
    q_min, q_max = c[2].number_input("q min", 0, 5, 0), c[2].number_input("q max", 0, 5, 2)
    c2 = st.columns(3)
    P_min, P_max = c2[0].number_input("P min", 0, 3, 0), c2[0].number_input("P max", 0, 3, 1)
    D_min, D_max = c2[1].number_input("D min", 0, 2, 0), c2[1].number_input("D max", 0, 2, 1)
    Q_min, Q_max = c2[2].number_input("Q min", 0, 3, 0), c2[2].number_input("Q max", 0, 3, 1)

    years = sorted({int(d.year) for d in valid_series.index if d.year >= 2019})
    target_years = st.multiselect("Années cibles", years, default=years[-2:] if len(years) >= 2 else years)
    criterion = st.selectbox("Critère", ["Écart annuel cumulé", "AIC"])

    total = (p_max-p_min+1)*(d_max-d_min+1)*(q_max-q_min+1)*(P_max-P_min+1)*(D_max-D_min+1)*(Q_max-Q_min+1)
    st.write(f"Combinaisons: **{int(total)}**")
    max_combo = st.number_input("Limiter à N combinaisons", 1, 1000, min(int(total), 120))

    if st.button("Lancer le Grid Search"):
        bar = st.progress(0)
        top = forecaster.grid_search(
            train_series=train,
            reference_series=valid_series,
            search_space={"p_min":int(p_min),"p_max":int(p_max),"d_min":int(d_min),"d_max":int(d_max),"q_min":int(q_min),"q_max":int(q_max),"P_min":int(P_min),"P_max":int(P_max),"D_min":int(D_min),"D_max":int(D_max),"Q_min":int(Q_min),"Q_max":int(Q_max)},
            target_years=target_years,
            criterion=criterion,
            max_combinations=int(max_combo),
            progress_callback=lambda v: bar.progress(min(1.0, v)),
        )
        st.session_state["grid_top"] = top

if st.session_state.get("grid_top"):
    top_df = pd.DataFrame(st.session_state["grid_top"])
    st.dataframe(top_df, use_container_width=True)
    best = st.session_state["grid_top"][0]
    st.success(f"Meilleur modèle: (p,d,q)=({best['p']},{best['d']},{best['q']}), (P,D,Q)=({best['P']},{best['D']},{best['Q']})")
    if st.button("Appliquer les meilleurs paramètres"):
        st.session_state["pending_best_params"] = best
        st.rerun()

st.markdown("<p class='title'>Diagnostics SARIMA</p>", unsafe_allow_html=True)
st.markdown("<p class='note'>Histogramme/Densité: distribution des résidus (idéalement centrée proche de 0). ACF/PACF: auto-corrélation restante (idéalement faible). Ljung-Box: p-value élevée => résidus proches d'un bruit blanc.</p>", unsafe_allow_html=True)
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

st.markdown("<p class='title'>Décomposition de la série (tendance / saisonnalité / résidus)</p>", unsafe_allow_html=True)
st.markdown("<p class='note'>Tendance: mouvement de fond long terme. Saisonnalité: motif périodique annuel. Résidus: part non expliquée.</p>", unsafe_allow_html=True)
decomp = seasonal_decompose(valid_series, model="additive", period=12, extrapolate_trend="freq")
fig_d, axd = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axd[0].plot(decomp.observed); axd[0].set_title("Série observée")
axd[1].plot(decomp.trend); axd[1].set_title("Tendance")
axd[2].plot(decomp.seasonal); axd[2].set_title("Saisonnalité")
axd[3].plot(decomp.resid); axd[3].set_title("Résidus")
plt.tight_layout()
st.pyplot(fig_d)
