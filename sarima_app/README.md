# Application SARIMA — Prévision de dépenses hospitalières

Outil d'analyse financière pour établissements publics de santé français.
Permet de prévoir les dépenses hospitalières mensuelles à l'aide du modèle
SARIMA(p,d,q)(P,D,Q)[12] via une interface Streamlit interactive.

## Prérequis

- Python 3.10+
- pip

## Installation et lancement

```bash
cd sarima_app
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse
`http://localhost:8501`.

## Utilisation

### 1. Importer les données

Upload votre fichier CSV via la barre latérale (bouton **Importer un fichier CSV**),
ou cliquez sur **Charger les données d'exemple** pour tester avec des données
synthétiques.

**Format du CSV :**
- 2 colonnes : date française et montant mensuel
- Séparateur : point-virgule (`;`) ou virgule (`,`)
- Encodage : UTF-8 ou Latin-1
- Minimum : **24 mois** de données

### 2. Configurer le modèle

Dans la barre latérale :
- **Agrégation** : choisir l'affichage mensuel, trimestriel, quadrimestriel ou semestriel
- **Paramètres SARIMA** : ajuster p, d, q (non-saisonnier) et P, D, Q (saisonnier)
- **Cutoff** : glissière pour définir le dernier mois de calibration
- **Horizon** : nombre de mois à prévoir au-delà des données existantes (1 à 36)

### 3. Interpréter les résultats

| Onglet | Contenu |
|--------|---------|
| **Prévision** | Graphique interactif + tableau de performance annuelle |
| **Diagnostics** | Résidus, histogramme, QQ-plot, ACF/PACF, test de Ljung-Box |
| **Grid Search** | Optimisation automatique des paramètres SARIMA |

## Formats de dates acceptés

| Format | Exemple | Description |
|--------|---------|-------------|
| `MM/AAAA` | `01/2019` | Mois/Année |
| `JJ/MM/AAAA` | `01/01/2019` | Jour/Mois/Année |
| `mois-AAAA` | `janv-2019` | Abréviation française |
| `mois. AAAA` | `janv. 2019` | Abréviation avec point |
| `mois AAAA` | `janvier 2019` | Mois complet |
| `AAAA-MM` | `2019-01` | Format ISO partiel |

## Modèle SARIMA

SARIMA = **S**easonal **A**uto**R**egressive **I**ntegrated **M**oving **A**verage

```
SARIMA(p, d, q)(P, D, Q)[m=12]
       ─────────────────
       │        │
       │        └── Composante saisonnière (période = 12 mois)
       └────────── Composante non-saisonnière
```

| Paramètre | Rôle |
|-----------|------|
| `p` | Ordre autorégressif (AR) non-saisonnier |
| `d` | Ordre de différenciation non-saisonnière |
| `q` | Ordre moyenne mobile (MA) non-saisonnière |
| `P` | Ordre autorégressif (AR) saisonnier |
| `D` | Ordre de différenciation saisonnière |
| `Q` | Ordre moyenne mobile (MA) saisonnière |
| `m=12` | Période saisonnière (mensuel) |

**Point de départ recommandé :** SARIMA(1,1,1)(1,1,1)[12]

## Grid Search

Le grid search teste automatiquement toutes les combinaisons de paramètres
dans les plages définies et classe les modèles selon deux critères :

- **Écart annuel (%)** : minimise l'écart entre dépenses réelles et prévues
  sur les années de validation sélectionnées
- **AIC** : minimise le critère d'information d'Akaike (compromis
  ajustement/parcimonie)

Le bouton **Appliquer le meilleur modèle** transfère automatiquement les
paramètres optimaux dans les curseurs de la barre latérale.

## Structure du projet

```
sarima_app/
├── app.py           # Interface Streamlit principale
├── models.py        # Classe SARIMAModel et fonction grid_search_sarima
├── data_loader.py   # Parsing CSV format français
├── evaluation.py    # Métriques de performance annuelle
├── requirements.txt # Dépendances Python
└── README.md        # Documentation
```

## Dépendances principales

| Bibliothèque | Usage |
|-------------|-------|
| `streamlit` | Interface web interactive |
| `statsmodels` | Modèle SARIMAX, tests statistiques |
| `plotly` | Graphiques interactifs |
| `pandas` | Manipulation des séries temporelles |
| `numpy` | Calculs numériques |
| `matplotlib` | Graphiques de diagnostic (ACF/PACF) |
| `scipy` | QQ-plot, test de Shapiro-Wilk |

## Exemple de fichier CSV

```csv
date;montant
01/2019;1 850 000
02/2019;1 720 000
03/2019;1 980 000
janv-2020;1 920 000
2020-02;1 800 000
janvier 2021;2 000 000
```

## Notes techniques

- Le modèle est estimé par maximum de vraisemblance (méthode L-BFGS-B, 200 itérations max)
- L'intervalle de confiance à 95% est affiché en zone grise sur le graphique
- Les doublons de dates sont automatiquement résolus (dernière valeur conservée)
- Les modèles ajustés sont mis en cache (évite les recalculs lors des interactions)
- Encodage CSV supporté : UTF-8 (avec ou sans BOM), Latin-1
