"""
Application SARIMA — Prévision de dépenses hospitalières
Interface Streamlit pour le modèle SARIMA(p,d,q)(P,D,Q)[12]

Usage :
    streamlit run app.py
"""

import warnings
import hashlib
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from data_loader import load_csv, aggregate_series
from models import SARIMAModel, grid_search_sarima
from evaluation import annual_comparison, compute_metrics

# ---------------------------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prévision SARIMA — Hôpital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour l'apparence hospitalière
st.markdown("""
<style>
.metric-card {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #1e3a5f;
    margin-bottom: 0.5rem;
}
.warning-text  { color: #f97316; font-weight: 600; }
.success-text  { color: #22c55e; font-weight: 600; }
.section-title { color: #1e3a5f; font-size: 1.1rem; font-weight: 700; margin-top: 1rem; }
div[data-testid="stSidebar"] { background-color: #f0f4f8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MOIS_ABR = [
    'janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
    'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.'
]

AGREGATION_MODES = ['mensuel', 'trimestriel', 'quadrimestriel', 'semestriel']

CSV_EXEMPLE = """date;montant
01/2019;1850000
02/2019;1720000
03/2019;1980000
04/2019;1900000
05/2019;1870000
06/2019;1750000
07/2019;1620000
08/2019;1580000
09/2019;1890000
10/2019;2010000
11/2019;1950000
12/2019;2100000
01/2020;1920000
02/2020;1800000
03/2020;2050000
04/2020;1960000
05/2020;1930000
06/2020;1820000
07/2020;1690000
08/2020;1650000
09/2020;1960000
10/2020;2080000
11/2020;2020000
12/2020;2180000
01/2021;2000000
02/2021;1870000
03/2021;2130000
04/2021;2040000
05/2021;2010000
06/2021;1890000
07/2021;1760000
08/2021;1720000
09/2021;2040000
10/2021;2160000
11/2021;2100000
12/2021;2260000
01/2022;2080000
02/2022;1950000
03/2022;2210000
04/2022;2120000
05/2022;2090000
06/2022;1970000
07/2022;1840000
08/2022;1800000
09/2022;2120000
10/2022;2240000
11/2022;2180000
12/2022;2340000
"""


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def format_fr_money(n) -> str:
    """Formate un nombre en euros au format français : 1 234 567,89 €"""
    if pd.isna(n):
        return "N/A"
    try:
        # Séparateur de milliers = espace, décimale = virgule
        formatted = f"{n:,.2f}".replace(',', ' ').replace('.', ',')
        # Corriger : les virgules utilisées comme milliers deviennent espaces
        # mais on a utilisé , pour les milliers et . pour la décimale
        # format: 1,234,567.89 -> remplacer , par ' ' puis . par ','
        return formatted + " €"
    except (TypeError, ValueError):
        return "N/A"


def format_fr_pct(n) -> str:
    """Formate un pourcentage au format français : 1,23 %"""
    if pd.isna(n):
        return "N/A"
    try:
        return f"{n:.2f}".replace('.', ',') + " %"
    except (TypeError, ValueError):
        return "N/A"


def data_hash(df: pd.DataFrame) -> str:
    """Calcule un hash SHA-256 pour identifier un DataFrame de manière unique."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]


def get_month_label(ts: pd.Timestamp) -> str:
    """Retourne l'étiquette de mois français pour un Timestamp."""
    return f"{MOIS_ABR[ts.month - 1]} {ts.year}"


def make_tickvals_labels(index: pd.DatetimeIndex):
    """
    Génère les valeurs et étiquettes pour l'axe temporel Plotly.
    Affiche uniquement les labels de janvier de chaque année pour la lisibilité.
    """
    tickvals = []
    ticktext = []
    for ts in index:
        if ts.month == 1:
            tickvals.append(ts)
            ticktext.append(str(ts.year))
    return tickvals, ticktext


# ---------------------------------------------------------------------------
# Mise en cache
# ---------------------------------------------------------------------------

@st.cache_data
def cached_load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Charge et met en cache le CSV uploadé."""
    return load_csv(file_bytes)


@st.cache_resource
def cached_fit_sarima(
    df_hash: str,
    p: int, d: int, q: int,
    P: int, D: int, Q: int,
    cutoff_str: str
) -> SARIMAModel:
    """
    Ajuste et met en cache le modèle SARIMA pour un jeu de paramètres donné.
    La clé de cache combine le hash du DataFrame et tous les paramètres.
    """
    # Récupérer les données depuis le session_state (non sérialisables par cache_resource)
    df = st.session_state.get('df')
    if df is None:
        return None

    cutoff_date = pd.Timestamp(cutoff_str)
    train_data = df[df.index <= cutoff_date]['valeur']

    model = SARIMAModel(p=p, d=d, q=q, P=P, D=D, Q=Q)
    success = model.fit(train_data)

    if not success:
        return model  # On retourne quand même pour accéder au message d'erreur

    return model


# ---------------------------------------------------------------------------
# Graphique Plotly principal
# ---------------------------------------------------------------------------

def build_forecast_chart(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    mode: str = 'mensuel'
) -> go.Figure:
    """
    Construit le graphique Plotly avec 4 traces :
    1. Valeurs réelles pré-cutoff (ligne bleue pleine)
    2. Valeurs réelles post-cutoff (ligne bleue pointillée)
    3. Prévision SARIMA (ligne orange pleine)
    4. Intervalle de confiance 95% (zone grise)
    """
    fig = go.Figure()

    # --- Séparation pré/post cutoff ---
    real_pre = df[df.index <= cutoff_date]['valeur']
    real_post = df[df.index > cutoff_date]['valeur']

    # --- Agrégation si demandée ---
    if mode != 'mensuel':
        real_pre_agg = aggregate_series(real_pre, mode)
        real_post_agg = aggregate_series(real_post, mode)
        forecast_agg = aggregate_series(forecast_df['forecast'], mode)
        lower_agg = aggregate_series(forecast_df['lower_ci'], mode)
        upper_agg = aggregate_series(forecast_df['upper_ci'], mode)

        x_pre = list(real_pre_agg.index)
        x_post = list(real_post_agg.index)
        x_fc = list(forecast_agg.index)
        x_lower = list(lower_agg.index)
        x_upper = list(upper_agg.index)

        y_pre = real_pre_agg.values
        y_post = real_post_agg.values
        y_fc = forecast_agg.values
        y_lower = lower_agg.values
        y_upper = upper_agg.values
    else:
        x_pre = real_pre.index.tolist()
        x_post = real_post.index.tolist()
        x_fc = forecast_df.index.tolist()
        x_lower = forecast_df.index.tolist()
        x_upper = forecast_df.index.tolist()

        y_pre = real_pre.values
        y_post = real_post.values
        y_fc = forecast_df['forecast'].values
        y_lower = forecast_df['lower_ci'].values
        y_upper = forecast_df['upper_ci'].values

    # Trace 1 : Réelles pré-cutoff (bleu foncé, trait plein)
    fig.add_trace(go.Scatter(
        x=x_pre, y=y_pre,
        name="Réalisé (calibration)",
        mode='lines',
        line=dict(color='#1e3a5f', width=2),
        hovertemplate='%{x}<br>%{y:,.0f} €<extra>Réalisé</extra>'
    ))

    # Trace 2 : Réelles post-cutoff (bleu clair, pointillé)
    if len(x_post) > 0:
        fig.add_trace(go.Scatter(
            x=x_post, y=y_post,
            name="Réalisé (validation)",
            mode='lines',
            line=dict(color='#5b9bd5', width=2, dash='dash'),
            hovertemplate='%{x}<br>%{y:,.0f} €<extra>Réalisé validation</extra>'
        ))

    # Trace 3 : IC 95% borne inférieure (invisible, sert de base pour fill)
    fig.add_trace(go.Scatter(
        x=x_lower, y=y_lower,
        name="IC 95% inf.",
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Trace 4 : IC 95% borne supérieure + remplissage vers la borne inférieure
    fig.add_trace(go.Scatter(
        x=x_upper, y=y_upper,
        name="IC 95%",
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(200,200,200,0.4)',
        hoverinfo='skip'
    ))

    # Trace 5 : Prévision SARIMA (orange, trait plein)
    fig.add_trace(go.Scatter(
        x=x_fc, y=y_fc,
        name="Prévision SARIMA",
        mode='lines',
        line=dict(color='#f97316', width=2.5),
        hovertemplate='%{x}<br>%{y:,.0f} €<extra>Prévision</extra>'
    ))

    # Ligne verticale au cutoff
    fig.add_vline(
        x=cutoff_date,
        line_dash="dot",
        line_color="#6b7280",
        annotation_text=f"Cutoff : {get_month_label(cutoff_date)}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#6b7280")
    )

    # Mise en forme générale
    fig.update_layout(
        title=dict(
            text="Prévision des dépenses hospitalières — SARIMA",
            font=dict(size=16, color='#1e3a5f'),
            x=0.02
        ),
        xaxis=dict(
            title="Période",
            showgrid=True,
            gridcolor='#e5e7eb',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Montant (€)",
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat=',.0f',
            tickfont=dict(size=11)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='left', x=0
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=480,
        margin=dict(l=60, r=30, t=80, b=60)
    )

    return fig


# ---------------------------------------------------------------------------
# Section Diagnostics
# ---------------------------------------------------------------------------

def render_diagnostics(model: SARIMAModel, df: pd.DataFrame, cutoff_date: pd.Timestamp):
    """
    Affiche les graphiques de diagnostic du modèle SARIMA :
    - Histogramme des résidus
    - QQ-plot (test de normalité)
    - ACF et PACF des résidus
    - Tableau du test de Ljung-Box
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        TSAPLOTS_OK = True
    except ImportError:
        TSAPLOTS_OK = False

    resid = model.get_residuals()
    if resid is None or resid.dropna().empty:
        st.warning("Impossible de calculer les résidus du modèle.")
        return

    resid_clean = resid.dropna()

    # --- Graphique des résidus dans le temps ---
    st.markdown("#### Résidus dans le temps")
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=resid_clean.index,
        y=resid_clean.values,
        mode='lines',
        name='Résidus',
        line=dict(color='#1e3a5f', width=1.5)
    ))
    fig_resid.add_hline(y=0, line_dash='dash', line_color='#ef4444')
    fig_resid.update_layout(
        height=280,
        yaxis_title="Résidu",
        xaxis_title="Date",
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=20, t=30, b=40)
    )
    st.plotly_chart(fig_resid, width='stretch')

    # --- Histogramme + QQ-plot côte à côte ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution des résidus")
        fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
        ax_hist.hist(
            resid_clean.values, bins=20,
            color='#5b9bd5', edgecolor='white', alpha=0.85
        )
        # Courbe normale théorique
        mu, sigma = resid_clean.mean(), resid_clean.std()
        x_range = np.linspace(resid_clean.min(), resid_clean.max(), 100)
        pdf = stats.norm.pdf(x_range, mu, sigma)
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(x_range, pdf, color='#f97316', linewidth=2, label='Normale théorique')
        ax_hist2.set_ylabel('Densité', fontsize=9)
        ax_hist.set_xlabel('Résidu', fontsize=9)
        ax_hist.set_ylabel('Fréquence', fontsize=9)
        ax_hist.set_title('Histogramme des résidus', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_hist, width='stretch')
        plt.close(fig_hist)

    with col2:
        st.markdown("#### QQ-plot (normalité)")
        fig_qq, ax_qq = plt.subplots(figsize=(5, 3.5))
        (osm, osr), (slope, intercept, r_val) = stats.probplot(
            resid_clean.values, dist='norm'
        )
        ax_qq.scatter(osm, osr, color='#1e3a5f', s=15, alpha=0.7, label='Résidus')
        x_line = np.linspace(min(osm), max(osm), 100)
        ax_qq.plot(x_line, slope * x_line + intercept,
                   color='#ef4444', linewidth=2, label='Droite théorique')
        ax_qq.set_xlabel('Quantiles théoriques', fontsize=9)
        ax_qq.set_ylabel('Quantiles observés', fontsize=9)
        ax_qq.set_title('QQ-plot des résidus', fontsize=10)
        ax_qq.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_qq, width='stretch')
        plt.close(fig_qq)

    # --- ACF et PACF ---
    if TSAPLOTS_OK:
        st.markdown("#### ACF et PACF des résidus")
        col3, col4 = st.columns(2)

        with col3:
            try:
                fig_acf, ax_acf = plt.subplots(figsize=(5, 3.5))
                plot_acf(resid_clean, lags=24, ax=ax_acf, color='#1e3a5f',
                         vlines_kwargs={'colors': '#1e3a5f'})
                ax_acf.set_title('ACF des résidus', fontsize=10)
                ax_acf.set_xlabel('Lag (mois)', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_acf, width='stretch')
                plt.close(fig_acf)
            except Exception as e:
                st.warning(f"Impossible d'afficher l'ACF : {e}")

        with col4:
            try:
                fig_pacf, ax_pacf = plt.subplots(figsize=(5, 3.5))
                plot_pacf(resid_clean, lags=min(24, len(resid_clean) // 2 - 1),
                          ax=ax_pacf, color='#1e3a5f',
                          vlines_kwargs={'colors': '#1e3a5f'})
                ax_pacf.set_title('PACF des résidus', fontsize=10)
                ax_pacf.set_xlabel('Lag (mois)', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_pacf, width='stretch')
                plt.close(fig_pacf)
            except Exception as e:
                st.warning(f"Impossible d'afficher le PACF : {e}")

    # --- Test de Ljung-Box ---
    st.markdown("#### Test de Ljung-Box")
    st.caption(
        "H₀ : les résidus sont indépendants. "
        "p-value > 0,05 → pas d'autocorrélation résiduelle (modèle bien spécifié)."
    )
    lb_result = model.ljung_box_test(lags=12)
    if lb_result is not None:
        lb_display = lb_result.copy()
        lb_display.index.name = 'Lag'
        lb_display.columns = ['Statistique LB', 'p-value']
        lb_display['Interprétation'] = lb_display['p-value'].apply(
            lambda p: "✓ OK (p > 0,05)" if p > 0.05 else "⚠ Autocorrélation (p ≤ 0,05)"
        )
        # Formatage des colonnes numériques
        lb_display['Statistique LB'] = lb_display['Statistique LB'].map('{:.4f}'.format)
        lb_display['p-value'] = lb_display['p-value'].map('{:.4f}'.format)
        st.dataframe(lb_display, width='stretch')
    else:
        st.warning("Impossible de calculer le test de Ljung-Box.")

    # --- Test de normalité (Shapiro-Wilk si n < 5000) ---
    st.markdown("#### Test de normalité (Shapiro-Wilk)")
    try:
        if len(resid_clean) <= 5000:
            stat_sw, p_sw = stats.shapiro(resid_clean.values)
            interp_sw = "Les résidus semblent normaux (p > 0,05)." \
                if p_sw > 0.05 else "Les résidus ne suivent pas une loi normale (p ≤ 0,05)."
            col_sw1, col_sw2, col_sw3 = st.columns(3)
            col_sw1.metric("Statistique W", f"{stat_sw:.4f}")
            col_sw2.metric("p-value", f"{p_sw:.4f}")
            col_sw3.metric("Conclusion", "Normal" if p_sw > 0.05 else "Non normal")
            st.caption(interp_sw)
        else:
            st.info("Série trop longue pour le test de Shapiro-Wilk (n > 5000).")
    except Exception as e:
        st.warning(f"Test de Shapiro-Wilk indisponible : {e}")


# ---------------------------------------------------------------------------
# Section Grid Search
# ---------------------------------------------------------------------------

def render_grid_search(df: pd.DataFrame, cutoff_date: pd.Timestamp):
    """
    Affiche l'interface de grid search pour l'optimisation automatique
    des paramètres SARIMA.
    """
    st.markdown("### Optimisation automatique des paramètres SARIMA")
    st.caption(
        "Le grid search teste toutes les combinaisons de paramètres dans les "
        "plages définies et sélectionne le modèle minimisant le critère choisi."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Plages de paramètres")
        gs_col1, gs_col2, gs_col3 = st.columns(3)

        with gs_col1:
            st.markdown("**Non-saisonnier**")
            p_max = st.slider("p max", 0, 3, 2, key='gs_p')
            d_vals = st.multiselect(
                "d", [0, 1, 2], default=[1], key='gs_d',
                help="Ordre de différenciation"
            )
            q_max = st.slider("q max", 0, 3, 2, key='gs_q')

        with gs_col2:
            st.markdown("**Saisonnier**")
            P_max = st.slider("P max", 0, 2, 1, key='gs_P')
            D_vals = st.multiselect(
                "D", [0, 1, 2], default=[1], key='gs_D',
                help="Ordre de différenciation saisonnière"
            )
            Q_max = st.slider("Q max", 0, 2, 1, key='gs_Q')

        with gs_col3:
            st.markdown("**Critère & Options**")
            criterion = st.selectbox(
                "Critère d'optimisation",
                options=['ecart_annuel', 'aic'],
                format_func=lambda x: {
                    'ecart_annuel': 'Écart annuel (%)',
                    'aic': 'AIC (Akaike)'
                }[x],
                key='gs_criterion'
            )
            max_combs = st.number_input(
                "Limite de combinaisons",
                min_value=10, max_value=1000, value=200, step=10,
                key='gs_max'
            )

    with col2:
        st.markdown("#### Années cibles")
        actual_post = df[df.index > cutoff_date]
        # Années avec données complètes (12 mois)
        years_available = []
        for y in actual_post.index.year.unique():
            if len(actual_post[actual_post.index.year == y]) == 12:
                years_available.append(y)

        if years_available:
            target_years = st.multiselect(
                "Années d'évaluation",
                options=sorted(years_available),
                default=sorted(years_available),
                key='gs_years',
                help="Sélectionner les années avec 12 mois complets post-cutoff"
            )
        else:
            target_years = []
            st.info("Aucune année complète disponible après le cutoff.")

    # Calcul du nombre de combinaisons
    if d_vals and D_vals:
        n_p = p_max + 1
        n_d = len(d_vals)
        n_q = q_max + 1
        n_P = P_max + 1
        n_D = len(D_vals)
        n_Q = Q_max + 1
        total_combs = n_p * n_d * n_q * n_P * n_D * n_Q
    else:
        total_combs = 0

    if total_combs > max_combs:
        st.warning(
            f"Nombre de combinaisons théoriques ({total_combs:,}) supérieur "
            f"à la limite ({max_combs:,}). Seules les {max_combs:,} premières "
            "seront testées."
        )
    else:
        st.info(f"Nombre de combinaisons à tester : **{total_combs}**")

    # Validation et lancement
    can_launch = (
        total_combs > 0
        and (criterion != 'ecart_annuel' or len(target_years) > 0)
    )

    if not can_launch:
        if criterion == 'ecart_annuel' and len(target_years) == 0:
            st.warning(
                "Avec le critère 'Écart annuel', vous devez sélectionner "
                "au moins une année cible."
            )

    if st.button(
        "Lancer le Grid Search",
        type="primary",
        disabled=not can_launch,
        key='gs_launch'
    ):
        # Construire les plages
        p_range = range(0, p_max + 1)
        d_range = sorted(d_vals)
        q_range = range(0, q_max + 1)
        P_range = range(0, P_max + 1)
        D_range = sorted(D_vals)
        Q_range = range(0, Q_max + 1)

        # Barre de progression
        progress_bar = st.progress(0, text="Initialisation...")
        status_text = st.empty()

        def update_progress(ratio, message):
            progress_bar.progress(min(ratio, 1.0), text=message)
            status_text.caption(message)

        with st.spinner("Grid search en cours..."):
            gs_results = grid_search_sarima(
                data=df,
                cutoff_date=cutoff_date,
                p_range=p_range,
                d_range=d_range,
                q_range=q_range,
                P_range=P_range,
                D_range=D_range,
                Q_range=Q_range,
                target_years=list(target_years),
                criterion=criterion,
                max_combinations=int(max_combs),
                progress_callback=update_progress
            )

        progress_bar.progress(1.0, text="Terminé !")
        status_text.empty()

        # Sauvegarder les résultats dans le session_state
        st.session_state['gs_results'] = gs_results
        st.session_state['gs_criterion'] = criterion

    # Affichage des résultats
    if 'gs_results' in st.session_state and st.session_state['gs_results']:
        gs_results = st.session_state['gs_results']
        criterion_used = st.session_state.get('gs_criterion', criterion)

        st.markdown("---")
        st.markdown(f"### Top 10 des meilleurs modèles (critère : {criterion_used})")

        top10 = gs_results[:10]
        df_top10 = pd.DataFrame(top10)

        # Colonnes à afficher
        cols_display = ['ordre', 'score', 'AIC', 'BIC']
        cols_available = [c for c in cols_display if c in df_top10.columns]

        rename_map = {
            'ordre': 'Modèle',
            'score': 'Score (critère)',
            'AIC': 'AIC',
            'BIC': 'BIC'
        }

        df_display = df_top10[cols_available].rename(columns=rename_map)

        # Formatage numérique
        for col in ['Score (critère)', 'AIC', 'BIC']:
            if col in df_display.columns:
                df_display[col] = df_display[col].map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )

        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = "Rang"

        st.dataframe(df_display, width='stretch')

        # Bouton pour appliquer le meilleur modèle
        if gs_results:
            best = gs_results[0]
            st.markdown(
                f"**Meilleur modèle** : `{best['ordre']}` "
                f"— Score : {best['score']:.4f}"
            )
            if st.button("Appliquer le meilleur modèle aux paramètres", key='gs_apply'):
                st.session_state['apply_gs'] = best
                st.success(
                    f"Paramètres appliqués : p={best['p']}, d={best['d']}, "
                    f"q={best['q']}, P={best['P']}, D={best['D']}, Q={best['Q']}"
                )
                st.rerun()

    elif 'gs_results' in st.session_state and not st.session_state['gs_results']:
        st.error(
            "Aucun modèle n'a convergé. Essayez d'élargir les plages de paramètres "
            "ou de modifier le cutoff."
        )


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    """Fonction principale de l'application Streamlit."""

    # Titre principal
    st.title("Prévision des dépenses hospitalières")
    st.caption(
        "Modèle SARIMA(p,d,q)(P,D,Q)[12] — Outil d'analyse financière pour "
        "établissements publics de santé"
    )

    # Initialisation du session_state
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'apply_gs' not in st.session_state:
        st.session_state['apply_gs'] = None

    # Récupérer les paramètres appliqués depuis le grid search
    applied_gs = st.session_state.get('apply_gs', None)

    # -----------------------------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/"
            "Blason_france_official.svg/50px-Blason_france_official.svg.png",
            width=40
        )
        st.markdown("## Configuration SARIMA")
        st.markdown("---")

        # --- Section 1 : Import des données ---
        st.markdown("### 1. Données")
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=['csv', 'txt'],
            help=(
                "Format attendu : 2 colonnes (date, montant). "
                "Séparateur : ; ou ,. "
                "Dates : MM/AAAA, JJ/MM/AAAA, janv-2019, 2019-01"
            )
        )

        # Bouton pour charger les données d'exemple
        if st.button("Charger les données d'exemple", key='load_example'):
            try:
                df_example = load_csv(CSV_EXEMPLE)
                st.session_state['df'] = df_example
                st.success(f"Données d'exemple chargées : {len(df_example)} mois")
            except Exception as e:
                st.error(f"Erreur : {e}")

        # Chargement du fichier uploadé
        if uploaded_file is not None:
            try:
                df_loaded = cached_load_csv(uploaded_file.read(), uploaded_file.name)
                st.session_state['df'] = df_loaded
                st.success(
                    f"Fichier chargé : {len(df_loaded)} mois "
                    f"({df_loaded.index.min().strftime('%m/%Y')} — "
                    f"{df_loaded.index.max().strftime('%m/%Y')})"
                )
            except Exception as e:
                st.error(f"Erreur de chargement : {e}")

        df = st.session_state.get('df')

        if df is not None:
            st.markdown("---")

            # --- Section 2 : Agrégation ---
            st.markdown("### 2. Affichage")
            mode_agregation = st.selectbox(
                "Agrégation temporelle",
                options=AGREGATION_MODES,
                index=0,
                format_func=lambda x: {
                    'mensuel': 'Mensuel',
                    'trimestriel': 'Trimestriel',
                    'quadrimestriel': 'Quadrimestriel',
                    'semestriel': 'Semestriel'
                }[x],
                key='mode_agregation'
            )

            st.markdown("---")

            # --- Section 3 : Paramètres SARIMA ---
            st.markdown("### 3. Paramètres SARIMA")
            st.caption("(p,d,q)(P,D,Q)[12]")

            # Valeurs par défaut depuis le grid search si appliqué
            default_p = applied_gs['p'] if applied_gs else 1
            default_d = applied_gs['d'] if applied_gs else 1
            default_q = applied_gs['q'] if applied_gs else 1
            default_P = applied_gs['P'] if applied_gs else 1
            default_D = applied_gs['D'] if applied_gs else 1
            default_Q = applied_gs['Q'] if applied_gs else 1

            st.markdown("**Composante non-saisonnière**")
            p = st.slider("p (AR)", 0, 5, default_p, key='param_p',
                          help="Ordre autorégressif non-saisonnier")
            d = st.slider("d (Différenciation)", 0, 2, default_d, key='param_d',
                          help="Ordre de différenciation non-saisonnière")
            q = st.slider("q (MA)", 0, 5, default_q, key='param_q',
                          help="Ordre moyenne mobile non-saisonnière")

            st.markdown("**Composante saisonnière [m=12]**")
            P = st.slider("P (AR sais.)", 0, 3, default_P, key='param_P',
                          help="Ordre autorégressif saisonnier")
            D = st.slider("D (Diff. sais.)", 0, 2, default_D, key='param_D',
                          help="Ordre de différenciation saisonnière")
            Q = st.slider("Q (MA sais.)", 0, 3, default_Q, key='param_Q',
                          help="Ordre moyenne mobile saisonnière")

            st.markdown("---")

            # --- Section 4 : Cutoff ---
            st.markdown("### 4. Cutoff de calibration")

            date_min = df.index.min()
            date_max = df.index.max()
            # Cutoff par défaut : 80% de la série
            n_total = len(df)
            default_cutoff_idx = max(
                int(n_total * 0.8) - 1,
                min(23, n_total - 2)  # Au moins 24 mois de train si possible
            )
            default_cutoff = df.index[default_cutoff_idx]

            # Construire la liste des dates disponibles comme cutoff
            # (au moins 24 mois de train, au moins 1 mois de test)
            valid_cutoff_dates = df.index[23:-1]  # De janvier an+2 à avant-dernier

            if len(valid_cutoff_dates) == 0:
                st.warning("Série trop courte pour définir un cutoff.")
                cutoff_date = df.index[-2]
            else:
                # Slider sur l'index de la date
                cutoff_idx = st.slider(
                    "Dernier mois de calibration",
                    min_value=0,
                    max_value=len(valid_cutoff_dates) - 1,
                    value=min(
                        len(valid_cutoff_dates) - 1,
                        max(0, default_cutoff_idx - 23)
                    ),
                    format="%d",
                    key='cutoff_slider'
                )
                cutoff_date = valid_cutoff_dates[cutoff_idx]
                n_train = len(df[df.index <= cutoff_date])
                n_test = len(df[df.index > cutoff_date])
                st.caption(
                    f"Calibration : {get_month_label(date_min)} — "
                    f"{get_month_label(cutoff_date)} "
                    f"(**{n_train} mois**)\n\n"
                    f"Validation : {get_month_label(cutoff_date)} — "
                    f"{get_month_label(date_max)} "
                    f"(**{n_test} mois**)"
                )

            st.markdown("---")

            # --- Section 5 : Horizon de prévision ---
            st.markdown("### 5. Horizon de prévision")
            horizon = st.slider(
                "Nombre de mois à prévoir",
                min_value=1,
                max_value=36,
                value=12,
                key='horizon',
                help="Horizon au-delà des données disponibles"
            )

        else:
            # Valeurs par défaut si pas de données
            mode_agregation = 'mensuel'
            p, d, q = 1, 1, 1
            P, D, Q = 1, 1, 1
            cutoff_date = None
            horizon = 12

    # -----------------------------------------------------------------------
    # ZONE PRINCIPALE
    # -----------------------------------------------------------------------

    if df is None:
        # Page d'accueil sans données
        st.info(
            "Bienvenue dans l'outil de prévision SARIMA pour les dépenses hospitalières.\n\n"
            "**Pour commencer :**\n"
            "1. Importez votre fichier CSV via la barre latérale, ou\n"
            "2. Cliquez sur **Charger les données d'exemple**\n\n"
            "**Format CSV attendu :**\n"
            "- 2 colonnes : date (format français) et montant\n"
            "- Séparateur : `;` ou `,`\n"
            "- Minimum 24 mois de données"
        )

        st.markdown("#### Exemple de fichier CSV accepté")
        st.code(
            "date;montant\n"
            "01/2019;1 850 000\n"
            "02/2019;1 720 000,50\n"
            "janv-2020;1 920 000\n"
            "2020-03;2 050 000",
            language="text"
        )

        st.markdown("#### Formats de dates acceptés")
        df_formats = pd.DataFrame({
            'Format': ['MM/AAAA', 'JJ/MM/AAAA', 'mois-AAAA', 'mois. AAAA', 'AAAA-MM'],
            'Exemple': ['01/2019', '01/01/2019', 'janv-2019', 'janv. 2019', '2019-01'],
            'Description': [
                'Mois/Année', 'Jour/Mois/Année', 'Abréviation française',
                'Abréviation avec point', 'Format ISO partiel'
            ]
        })
        st.dataframe(df_formats, width='stretch', hide_index=True)
        return

    # -----------------------------------------------------------------------
    # Métriques synthétiques en haut
    # -----------------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        n_mois = len(df)
        st.metric("Mois de données", f"{n_mois}")

    with col_m2:
        moy_mensuelle = df['valeur'].mean()
        st.metric("Moyenne mensuelle", format_fr_money(moy_mensuelle))

    with col_m3:
        n_train = len(df[df.index <= cutoff_date])
        st.metric("Mois de calibration", f"{n_train}")

    with col_m4:
        n_test = len(df[df.index > cutoff_date])
        st.metric("Mois de validation", f"{n_test}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Ajustement du modèle SARIMA
    # -----------------------------------------------------------------------
    df_hash = data_hash(df)
    cutoff_str = cutoff_date.isoformat()

    # Stocker le df dans session_state pour le cache_resource
    st.session_state['df'] = df

    with st.spinner(
        f"Ajustement SARIMA({p},{d},{q})({P},{D},{Q})[12] en cours..."
    ):
        model = cached_fit_sarima(df_hash, p, d, q, P, D, Q, cutoff_str)

    model_fitted = model is not None and model.results_ is not None

    if not model_fitted:
        err = model.error_message_ if model else "Modèle non initialisé."
        st.error(
            f"Le modèle SARIMA({p},{d},{q})({P},{D},{Q})[12] n'a pas convergé.\n\n"
            f"Détail : {err}\n\n"
            "Essayez de modifier les paramètres p, d, q, P, D, Q ou le cutoff."
        )

    # Calcul des prévisions si le modèle a convergé
    forecast_df = None
    if model_fitted:
        # Horizon = données manquantes + horizon demandé par l'utilisateur
        last_data = df.index[-1]
        last_cutoff = cutoff_date
        n_existing = len(pd.date_range(
            start=last_cutoff + pd.DateOffset(months=1),
            end=last_data,
            freq='MS'
        ))
        total_horizon = n_existing + horizon
        forecast_df = model.predict(total_horizon)

        if forecast_df is None:
            st.error(
                "Les prévisions n'ont pas pu être générées. "
                "Vérifiez les paramètres du modèle."
            )

    # -----------------------------------------------------------------------
    # ONGLETS PRINCIPAUX
    # -----------------------------------------------------------------------
    tab_prevision, tab_diagnostics, tab_gridsearch = st.tabs([
        "Prévision",
        "Diagnostics du modèle",
        "Grid Search"
    ])

    # ===================================================================
    # ONGLET 1 : PREVISION
    # ===================================================================
    with tab_prevision:

        # Informations sur le modèle
        if model_fitted:
            info = model.get_model_info()
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            col_info1.metric(
                "Modèle",
                f"SARIMA({p},{d},{q})({P},{D},{Q})[12]"
            )
            col_info2.metric(
                "AIC",
                f"{info['AIC']:.2f}" if info['AIC'] is not None else "N/A"
            )
            col_info3.metric(
                "BIC",
                f"{info['BIC']:.2f}" if info['BIC'] is not None else "N/A"
            )
            col_info4.metric(
                "Log-vraisemblance",
                f"{info['Log-vraisemblance']:.2f}"
                if info['Log-vraisemblance'] is not None else "N/A"
            )
        else:
            st.warning(
                "Le modèle n'a pas convergé avec les paramètres actuels. "
                "Le graphique affiche uniquement les données historiques."
            )

        # Graphique principal
        if forecast_df is not None:
            fig = build_forecast_chart(
                df=df,
                forecast_df=forecast_df,
                cutoff_date=cutoff_date,
                mode=mode_agregation
            )
            st.plotly_chart(fig, width='stretch')
        else:
            # Graphique sans prévision
            fig_only = go.Figure()
            fig_only.add_trace(go.Scatter(
                x=df.index,
                y=df['valeur'].values,
                name="Données réelles",
                mode='lines',
                line=dict(color='#1e3a5f', width=2)
            ))
            fig_only.add_vline(
                x=cutoff_date,
                line_dash="dot",
                line_color="#6b7280",
                annotation_text=f"Cutoff : {get_month_label(cutoff_date)}"
            )
            fig_only.update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_only, width='stretch')

        # Table de données agrégées
        st.markdown("#### Données et prévisions")
        col_tbl1, col_tbl2 = st.columns([3, 1])
        with col_tbl2:
            show_raw = st.checkbox("Afficher toutes les lignes", value=False)

        if forecast_df is not None:
            # Construire un tableau combiné réel + prévision
            df_combined = df['valeur'].copy().rename("Réel")

            fc_series = forecast_df['forecast'].rename("Prévision")
            lower_series = forecast_df['lower_ci'].rename("IC inf. 95%")
            upper_series = forecast_df['upper_ci'].rename("IC sup. 95%")

            df_table = pd.concat([df_combined, fc_series, lower_series, upper_series], axis=1)

            if mode_agregation != 'mensuel':
                # Agréger
                df_table_agg = pd.DataFrame({
                    'Réel': aggregate_series(df_table['Réel'].dropna(), mode_agregation),
                    'Prévision': aggregate_series(
                        df_table['Prévision'].dropna(), mode_agregation
                    )
                })
                df_show = df_table_agg
            else:
                df_show = df_table
                df_show.index = df_show.index.strftime('%m/%Y')

            if not show_raw:
                df_show = df_show.tail(36)

            # Formatage pour affichage
            def fmt_col(col):
                return col.map(
                    lambda x: format_fr_money(x) if pd.notna(x) else "-"
                )

            df_fmt = df_show.copy()
            for col in df_fmt.columns:
                df_fmt[col] = fmt_col(df_show[col])

            st.dataframe(df_fmt, width='stretch')

        # --- Tableau de performance annuelle ---
        st.markdown("#### Performance annuelle (validation post-cutoff)")

        if model_fitted and forecast_df is not None:
            comp_df = annual_comparison(
                actual=df['valeur'],
                forecast=forecast_df['forecast'],
                cutoff_date=cutoff_date
            )

            if not comp_df.empty:
                # Formatage des colonnes monétaires
                comp_display = comp_df.copy()
                for col in ['Réel (€)', 'Prévision (€)', 'Écart (€)']:
                    if col in comp_display.columns:
                        comp_display[col] = comp_display[col].map(format_fr_money)
                if 'Écart (%)' in comp_display.columns:
                    comp_display['Écart (%)'] = comp_display['Écart (%)'].map(
                        lambda x: format_fr_pct(x) if pd.notna(x) else "N/A"
                    )

                st.dataframe(comp_display, width='stretch')

                # Métriques globales sur les données de validation
                actual_post_series = df[df.index > cutoff_date]['valeur']
                fc_post_series = forecast_df['forecast']
                metrics = compute_metrics(actual_post_series, fc_post_series)

                if metrics:
                    st.markdown("#### Métriques d'erreur sur la période de validation")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE", format_fr_money(metrics.get('MAE')))
                    m2.metric("RMSE", format_fr_money(metrics.get('RMSE')))
                    m3.metric(
                        "MAPE",
                        f"{metrics.get('MAPE (%)', 0):.2f} %".replace('.', ',')
                    )
                    m4.metric("Biais", format_fr_money(metrics.get('Biais')))

            else:
                st.info(
                    "Aucune année complète disponible en validation "
                    "(12 mois réels ET 12 mois prévisionnels requis)."
                )
        else:
            st.info("Ajustez les paramètres SARIMA pour voir les performances.")

    # ===================================================================
    # ONGLET 2 : DIAGNOSTICS
    # ===================================================================
    with tab_diagnostics:
        if not model_fitted:
            st.warning(
                "Le modèle SARIMA doit converger pour afficher les diagnostics. "
                "Modifiez les paramètres dans la barre latérale."
            )
        else:
            render_diagnostics(model, df, cutoff_date)

    # ===================================================================
    # ONGLET 3 : GRID SEARCH
    # ===================================================================
    with tab_gridsearch:
        if df is None:
            st.info("Chargez des données pour accéder au grid search.")
        else:
            render_grid_search(df, cutoff_date)


# ---------------------------------------------------------------------------
# Lancement
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
