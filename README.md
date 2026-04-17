## Analyse-de-correlation-dynamique-entre-actifs-financiers-Python-pour-la-data-science-

Étudier le comportement des grandes valeurs technologiques (GAFAM) au cours du temps afin de :

analyser leurs rendements
mesurer leur risque (volatilité)
étudier leurs interactions dynamiques (alpha, beta)


## comment lancer le code

Installer les bibliothèques nécessaires avec :

```bash
pip install yfinance pandas numpy matplotlib statsmodels
```


## Les données

# la source, fiabilité

Les données boursières proviennent de Yahoo Finance, via la bibliothèque Python yfinance.
Les cours utilisés sont les prix de clôture quotidiens (Close) pour les actifs : AAPL, MSFT, GOOGL, AMZN, META, sur la période à partir du 2018-01-01.

En termes de fiabilité :

Yahoo Finance est une source très utilisée pour des analyses académiques/prototypage.
Les données sont globalement fiables pour l’étude statistique des rendements.
Cependant, pour un usage réel (ex: trading), il faudrait des flux de marché professionnels plus stricts.

# Méthode de recup des données: api, webscrapping,...

Méthode de récupération des données
Les données sont récupérées par API Python avec yfinance (pas de web scraping manuel).

Exemple de logique utilisée :

téléchargement groupé des tickers ;
conservation de la colonne Close ;
suppression des colonnes entièrement vides ;
calcul des rendements logarithmiques :


## stat desc +visualisation

L’analyse descriptive repose sur les rendements (logarithmiques) des GAFAM :

Calcul des corrélations des rendements avec le marché (bêta)
Calcul de la prédiction CAPM des rendements (alpha)
Etude de la stationnarité des rendements 


Calcul des corrélations entre actifs 
Ici on a des corrélations positives significatives entre grandes valeurs technologiques, suggérant des co-mouvements sectoriels.

## choix modele + variable

Variables : rendements de AAPL, MSFT, GOOGL, AMZN, META pour les comparer avec ceux du S&P500 et Nasdaq

## optionel : biblio site atricle etudes


Documentation yfinance : https://pypi.org/project/yfinance/

Documentation statsmodels VAR : https://www.statsmodels.org/

Notions de causalité de Granger : Granger, C. W. J. (1969), Investigating Causal Relations by Econometric Models and Cross-spectral Methods.
