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

## Exécution depuis Spyder
Vous pouvez lancer directement avec `%runfile app.py --wdir`:
- le script détecte Spyder,
- démarre automatiquement `streamlit run app.py` sur `http://127.0.0.1:8501`,
- et ouvre le navigateur.

Si l'auto-lancement échoue, lancez manuellement:
```bash
streamlit run app.py
```
