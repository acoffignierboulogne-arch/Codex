# Application locale SARIMA pour dépenses mensuelles

Cette application Python (Flask + HTML) permet de tester la robustesse d'un modèle SARIMA sur des dépenses mensuelles, avec une interface interactive.

## Fonctionnalités
- Import d'un CSV de dépenses mensuelles (gestion BOM, séparateurs `;`/`,` et encodage UTF-8/latin-1).
- Le CSV est conservé en mémoire côté serveur pendant la session, pour éviter le réupload à chaque ajustement.
- Réglage de l'horizon via un curseur (1 à 24 mois) avec recalcul automatique.
- Cutoff via curseur (pas de saisie manuelle) pour simuler l'état du modèle à une date donnée, avec recalcul automatique à chaque mouvement.
- Grid search SARIMA (activé par défaut) pour minimiser la MAPE cumulée rolling sur l'année N-1 du cutoff.
- Option "mode rapide" pour tester une grille allégée (pas plus larges) et réduire le temps de calcul.
- Onglet **Prévision** (historique + projection + intervalle de confiance).
- Onglet **Décomposition** (Série, Tendance, Saisonnalité, Résidus).
- Onglet **Holt-Winters** (additif/multiplicatif) avec grid search sur alpha/beta/gamma.
- Onglet **Budget** (cumul réel / cumulé projeté sur l'année du cutoff).
- Les montants sont formatés en français avec séparateur de milliers (espace) pour la lisibilité.

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
Puis ouvrir : [http://127.0.0.1:5000](http://127.0.0.1:5000)

Au lancement, l'application tente aussi d'ouvrir automatiquement le navigateur par défaut (pratique sous Spyder).

## Note Spyder (SystemExit watchdog)
Le lancement Flask est configuré avec `use_reloader=False` pour éviter l'erreur `SystemExit: 1` observée depuis Spyder avec `%runfile`.

## Format CSV attendu
Le parsing gère les CSV français du type `Date;Dépenses` avec virgule décimale (ex: `1347528,18`) et noms de colonnes accentués.

Par défaut, l'application prend les deux premières colonnes du CSV :
1. date (exemple: `2024-01-01`)
2. montant de dépense

Vous pouvez aussi renseigner explicitement les noms des colonnes dans le formulaire.

Un exemple est fourni : `sample_depenses.csv`.

## Métrique du grid search
Pour chaque combinaison SARIMA candidate :
- l'application prend l'année N-1 par rapport au cutoff (ex: cutoff oct-2025 => année cible 2024),
- simule pour chaque mois la projection de fin d'année (réel cumulé + prévision des mois restants),
- compare au total réel annuel,
- et calcule la moyenne des erreurs absolues en % (MAPE cumulée rolling).

Le meilleur modèle est celui avec la MAPE la plus faible.

Si aucun modèle du grid search ne converge, l'application ne plante pas: elle affiche un avertissement et continue avec les paramètres manuels.


## Interprétation des MAPE
- **MAPE cumulée rolling** : moyenne des écarts de cumul projeté intra-annuel.
- **MAPE annuelle cumul total** : |cumul prédit annuel - cumul réel annuel| / cumul réel annuel.
