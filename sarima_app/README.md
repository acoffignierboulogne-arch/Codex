# Application SARIMA (Streamlit)

## Lancement
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fonctionnalités
- Import CSV français (`;` ou `,`, dates `MM/AAAA`, `JJ/MM/AAAA`, `janv-AAAA`, `AAAA-MM`).
- Paramètres SARIMA `(p,d,q)(P,D,Q)[12]` via sidebar.
- Cutoff dynamique + horizon (1 à 36 mois).
- Graphique Plotly avec réel pré/post cutoff, prévision, et bande IC95%.
- Tableau annuel cumulé (réel vs prévision) avec surlignage des écarts > 5%.
- Grid search configurable (critère AIC ou écart annuel cumulé), top 10.
- Diagnostics résidus: histogramme, densité, ACF, PACF, Ljung-Box.

## Structure
- `app.py` : interface Streamlit.
- `data_loader.py` : parsing robuste des fichiers CSV français.
- `models.py` : encapsulation fit/prévision/grid search SARIMA.
- `evaluation.py` : agrégation temporelle et évaluation annuelle.


## Où taper les commandes (important)
- **Commandes Git** (`git fetch`, `git checkout`, `git push`, `git tag`, etc.) : dans un **terminal système**
  (PowerShell, Invite de commandes, Git Bash, ou terminal intégré VS Code), placé dans le dossier du repo.
- **Commandes Streamlit** (`streamlit run app.py`) : dans ce même **terminal système**.
- **Console Python de Spyder** : n'accepte que du Python ; pour une commande shell, il faut préfixer par `!`
  (ex. `!streamlit run app.py`).
- **GitHub Desktop** : utile pour les actions visuelles (changer de branche, pull/push), mais les commandes texte
  de ce README sont à exécuter dans un terminal.

Exemple (PowerShell/CMD/Git Bash) :
```bash
cd C:\Users\User\Documents\Python\codex\...\Codex
git fetch --all --tags
git checkout -b backup/forecasting-full-rework reference-model-v1
git push -u origin backup/forecasting-full-rework
git push origin reference-model-v1
```

## Exécution depuis Spyder
Vous pouvez lancer directement avec `%runfile app.py --wdir`:
- le script détecte Spyder (même si `--wdir` n'est pas propagé),
- démarre automatiquement `streamlit run app.py` sur `http://127.0.0.1:8501`,
- et ouvre le navigateur.

⚠️ Dans la console IPython de Spyder, `streamlit run app.py` **sans** `!` provoque un `SyntaxError`.
Utilisez:
```python
!streamlit run app.py
```
ou un terminal classique:
```bash
streamlit run app.py
```


### Si des onglets s'ouvrent en boucle
Le bootstrap Spyder est protégé contre la récursion (`SARIMA_STREAMLIT_CHILD=1`).
Si vous aviez une ancienne session, fermez les anciens processus Python/Streamlit puis relancez une seule fois `%runfile app.py --wdir`.


## Fichier budgétaire multi-comptes
L'application accepte désormais un CSV "large" contenant en une seule fois recettes + dépenses, puis applique des filtres dans l'interface:
- `Titre (dépenses)` / `Titre (recettes)`
- `Chapitre`
- `Compte exécution`
- `Sous compte`
- `Sous-compte classe 6`
- `Libellé du type`

Les colonnes de montants utilisées sont:
- `Réalisé dépenses.Cumul réalisé date comptable`
- `Réalisé recettes.Cumul réalisé date comptable`
- `Prévision cumulée` (optionnel, pour ventiler un budget annuel/cumulé en profil mensuel)

Les colonnes avec accents mal décodés (ex: `RÃ©alisÃ©`) sont normalisées automatiquement pour éviter les bugs d'encodage.


## Mode prévision annuelle ventilée
Si la colonne `Prévision cumulée` est présente, l'application construit une courbe mensuelle ventilée selon:
- le profil historique (moyenne des poids mensuels), ou
- un profil saisonnier dérivé du modèle.

Cette courbe est affichée avec le réel et la prévision SARIMA pour comparaison.
