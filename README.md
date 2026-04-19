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
pip install -r requirements.txt


