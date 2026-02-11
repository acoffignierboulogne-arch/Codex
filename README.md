# Application locale SARIMA pour dépenses mensuelles

Cette application Python fournit une interface HTML locale pour :
- importer un fichier `.csv` de dépenses mensuelles,
- ajuster les paramètres SARIMA,
- visualiser l'historique et les prévisions dans un graphique Plotly.

## Prérequis
- Python 3.10+

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement
```bash
python app.py
```

Puis ouvrir : [http://localhost:5000](http://localhost:5000)

## Format CSV attendu
Par défaut, l'application prend les deux premières colonnes du CSV :
1. date (exemple: `2024-01-01`)
2. montant de dépense

Vous pouvez aussi renseigner explicitement les noms des colonnes dans le formulaire.

Un exemple de fichier est fourni : `sample_depenses.csv`.
