# 📊 Econometric & Portfolio Analysis – GAFAM

Ce projet explore l’analyse économétrique et la construction de portefeuilles financiers à partir des données des actions GAFAM (Google, Apple, Facebook/Meta, Amazon, Microsoft), complétées par des actifs diversifiants (or, obligations américaines).

Les données sont récupérées via l’API `yfinance`.

---

## 🧠 Objectifs du projet

Ce projet a deux objectifs principaux :

1. **Analyse économétrique des données financières**
2. **Construction et comparaison de stratégies de portefeuille**

---

## 📁 Structure du projet

Le projet contient principalement deux notebooks et un dossier src :

### 📓 1. `Modele.ipynb`
Notebook principal dédié à la construction de portefeuilles.

Deux stratégies sont implémentées :

- **Portefeuille de variance minimale (long-only)**
  - Univers : GAFAM
  - Estimation du risque via la covariance de Ledoit-Wolf
  - Objectif : minimiser le risque

- **Portefeuille dynamique max-Sharpe avec Machine Learning (long/short)**
  - Univers : GAFAM + Or + Bons du Trésor US
  - Prédiction des rendements via XGBoost
  - Optimisation du ratio de Sharpe sous contraintes
  - Backtest out-of-sample avec fenêtre glissante

---

### 📓 2. Notebook d’analyse économétrique (Analyse.ipynb)

Ce notebook propose une **analyse descriptive et économétrique** des données financières :

- statistiques descriptives
- analyse des rendements
- exploration des corrélations
- premières analyses de type économétrique

### 📁 3. Dossier src

Ce dossier contient toutes les fonctions utiles à la construction des portefeuilles. Ces fonctions sont réparties dans les fichiers suivants :

- `data_utils.py` qui contient toutes les fonctions utilitaires pour le téléchargement et prétraitement des données
- `features.py` qui contient les fonctions de feature engineering pour la prevision de rendements.
- `metrics.py` Metriques de performance et fonctions de visualisation pour le backtest.
- `ml_utils.py` Fonctions utilitaires pour l'apprentissage supervise avec XGBoost.
- `portfolio_optim.py` Les fonctions d'optimisation de portefeuille (modele de risque + allocations).
---

## ⚙️ Installation

Avant de lancer les notebooks, installez les dépendances :

```bash
<<<<<<< HEAD
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
=======
pip install -r requirements.txt
>>>>>>> 2167cf2cc7a1e72f86fd9680c206d102bbf5ca96


Documentation yfinance : https://pypi.org/project/yfinance/

Documentation statsmodels VAR : https://www.statsmodels.org/

Notions de causalité de Granger : Granger, C. W. J. (1969), Investigating Causal Relations by Econometric Models and Cross-spectral Methods.
