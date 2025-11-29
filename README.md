````markdown
# SAINTv2 – Scalping BTCUSD M1 avec Agents Spécialisés (Long / Short)

Ce dépôt contient un pipeline complet pour entraîner, évaluer et faire tourner en **live** un système de trading RL basé sur **SAINTv2** (tabular transformer) pour le scalping **BTCUSD M1**, avec :

- Un **agent LONG** spécialisé dans les entrées acheteuses  
- Un **agent SHORT** spécialisé dans les entrées vendeuses    

Les données viennent de **MetaTrader 5** (MT5), et l’exécution live se fait directement via l’API MT5.

---

## 1. Fonctionnalités

- **Entraînement PPO** avec :
  - Reward Sharpe-like sur le log-return d’equity
  - Action masking (5 actions : `BUY1`, `SELL1`, `BUY1.8`, `SELL1.8`, `HOLD`)
  - SL/TP **automatiques** basés sur l’ATR, ATR figé à l’entrée
  - Position sizing fixe : `size = position_size * risk_scale` (1x / 1.8x)
  - Spécialisation par **side** :
    - `side="long"` : agent BUY-only
    - `side="short"` : agent SELL-only

- **Évaluation longue durée** (backtest offline) sur historique MT5
- **Script live** :
  - Agents LONG & SHORT pour **ouvrir** les positions
  - SL/TP gérés par le broker

---

## 2. Structure (suggestion)

```text
.
├── model saint sell & buy & close.py          # Entraînement des agents long / short
├── ia_live sell & buy & close.py       # Script live (LONG + SHORT)
├── norm_stats_ohlc_indics.npz    # Stats de normalisation (sauvé par le training)
├── bestprofit_saintv2_loup_long_long.pth   # Modèle agent LONG (live)
├── bestprofit_saintv2_loup_short_short.pth # Modèle agent SHORT (live)
└── README.md
````

---

## 3. Prérequis

### 3.1. Environnement Python

* Python 3.10+ recommandé
* Bibliothèques principales :

  * `MetaTrader5`
  * `numpy`
  * `pandas`
  * `torch`
  * `gymnasium`
  * `matplotlib` (optionnel pour plots)

Installation rapide (exemple) :

```bash
pip install MetaTrader5 numpy pandas torch gymnasium matplotlib
```

> ⚠️ Vérifie la version de PyTorch compatible avec ta carte GPU (ou CPU only).

### 3.2. MetaTrader 5

* MT5 **installé** sur la machine
* Compte (démo ou réel) connecté sur **BTCUSD** avec :

  * Historique M1 suffisant
  * Historique H1/M5 suffisant (selon le script, tu utilises M5 en HTF dans le training et éventuellement H1 en eval/live selon ta version)
* L’API Python `MetaTrader5` doit pouvoir se connecter (même utilisateur que ton MT5 ouvert).

---

## 4. Données & Features

Les features sont construites à partir de M1 + H1/M5 :

* **M1** :

  * OHLC
  * Returns : `ret_1`, `ret_3`, `ret_5`, `ret_15`, `ret_60`
  * Volatilité réalisée & régime de vol
  * EMAs 5,10,20
  * RSI 7,14
  * ATR(14)
  * Stochastique (K, D)
  * MACD + signal
  * Ichimoku (Tenkan, Kijun, Spans + distances au prix)
  * Moyenne mobile 100 + z-score
  * Encodage temporel (heure, jour de la semaine)
  * `tick_volume_log`
* **H1/M5** :

  * Close, EMA20, RSI14, MACD, z-score 100, Ichimoku, vol réalisée, etc. suffixés `_h1` (ou `_m5` selon le script)

Les colonnes finales sont listées dans :

```python
FEATURE_COLS_M1 = [...]
FEATURE_COLS_H1 = [...]
FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1
```

Les stats de normalisation sont sauvegardées dans :

```text
norm_stats_ohlc_indics.npz
```

et sont utilisées pour :

* Le training
* L’éval
* Le live

---

## 5. Entraînement des agents

Script : `model saint sell & buy & close.py` (selon ton nom de fichier)

Le cœur du training est la fonction :

```python
run_training(cfg: PPOConfig)
```

### 5.1. Config principale

La dataclass `PPOConfig` contient notamment :

* `symbol: "BTCUSD"`
* `timeframe: mt5.TIMEFRAME_M1`
* `htf_timeframe: mt5.TIMEFRAME_M5`
* `n_bars`: nombre de bougies pour le dataset
* `lookback`: longueur de la fenêtre temporelle
* Hyperparamètres PPO (epochs, batch_size, gamma, lambda_gae, clip, etc.)
* Hyperparamètres trading (initial_capital, leverage, fee_rate, position_size, SL/TP ATR, etc.)
* **Spécialisation agent** via `side` :

  * `"long"`  → agent BUY-only
  * `"short"` → agent SELL-only
  * 
* `model_prefix`: utilisé pour nommer les fichiers de modèle (best_ / last_)

### 5.2. Lancer l’entraînement

Exemple typique dans le `if __name__ == "__main__":` :

```python
if __name__ == "__main__":
    # Entraînement agent LONG
    cfg_long = PPOConfig(side="long", model_prefix="saintv2_loup_long")
    run_training(cfg_long)

    # Entraînement agent SHORT
    cfg_short = PPOConfig(side="short", model_prefix="saintv2_loup_short")
    run_training(cfg_short)
```

À la fin, tu obtiens des fichiers du type :

* `best_saintv2_loup_long_long.pth`
* `best_saintv2_loup_short_short.pth`

(selon ta logique de nommage dans `model_prefix`).

---

## 6. Backtest longue durée

Script : `backtest_saintv2_trio.py` (nom à adapter)

Ce script :

1. Télécharge un historique long M1/H1 via MT5
2. Reconstruit les features + normalisation
3. Simule les actions du modèle sur toute la période
4. Applique la même logique SL/TP ATR, position sizing fixe
5. Calcule :

   * Capital final
   * Profit / %
   * Max drawdown
   * Nombre de trades
   * Winrate
   * Gain moyen / perte moyenne
   * Expectancy par trade

La logique sera très proche du script live, mais en “mode simulation” sur historique.

---

## 7. Script Live (3 agents)

Script : `ia_live sell & buy & close.py`

### 7.1. Fichiers nécessaires

Dans le même dossier que le script, tu dois avoir :

* `norm_stats_ohlc_indics.npz`
* `bestprofit_saintv2_loup_long_long.pth`
* `bestprofit_saintv2_loup_short_short.pth`

### 7.2. Logique de décision (résumé)

À chaque nouvelle bougie M1 :

1. **Construction de l’obs** (fenêtre `lookback` x features normalisées)
2. Lecture de la **position actuelle** via `get_current_position()` → `pos ∈ {-1,0,1}`

#### Si `pos != 0` (déjà en position) :

* On interroge **l’agent CLOSE** :

  * masque CLOSE :

    * si pos == 0 → HOLD uniquement
    * sinon → `{CLOSE, HOLD}` mappés sur `{0, 4}`
  * si l’agent choisit `CLOSE` (0) → `close_position_market(cfg)` :

    * envoie un ordre inverse au marché avec le volume de la position
  * sinon → on laisse SL/TP broker gérer la sortie

#### Si `pos == 0` (flat) :

* On interroge **agent LONG** (side="long") et **agent SHORT** (side="short")

* Pour chaque agent :

  * masque :

    * LONG : `{BUY1 (0), BUY1.8 (2), HOLD (4)}`
    * SHORT : `{SELL1 (1), SELL1.8 (3), HOLD (4)}`
  * on récupère `probs_long`, `probs_short`

* On calcule un **score d’ouverture** :

  ```python
  score_long = max(prob(BUY1), prob(BUY1.8)) - prob(HOLD)
  score_short = max(prob(SELL1), prob(SELL1.8)) - prob(HOLD)
  ```

* Si `score_long <= 0` et `score_short <= 0` → **HOLD global**

* Sinon :

  * on choisit l’agent avec le plus grand score (`LONG` ou `SHORT`)
  * on compare BUY1 vs BUY1.8 (ou SELL1 vs SELL1.8)
  * on en déduit :

    * `env_action = 0` (BUY) ou `env_action = 1` (SELL)
    * `risk_scale = 1.0` ou `1.8`
  * on appelle `send_order(...)` qui :

    * calcule ATR d’entrée + SL/TP
    * envoie un `ORDER_TYPE_BUY` ou `ORDER_TYPE_SELL` avec `sl` / `tp` posés serveur

### 7.3. Lancement

```bash
python live_saintv2_3agents.py
```

Assure-toi que :

* MT5 est ouvert avec le bon compte
* Le symbole `BTCUSD` est disponible
* Les historiques M1/M5/H1 sont chargés

---

## 8. Avertissement

> **Attention :**
> Ce projet est à vocation **expérimentale** et **éducative**.
> Le trading algorithmique, en particulier avec effet de levier sur crypto, comporte un **risque élevé de perte en capital**.
>
> * Ne jamais utiliser ce code en réel sans :
>
>   * tests approfondis,
>   * validations indépendantes,
>   * compréhension complète du fonctionnement,
>   * gestion de risque stricte.
> * L’auteur du code n’est pas responsable des pertes éventuelles.

---

## 9. Pistes d’amélioration

* Ajout de métriques plus détaillées (equity curve, heatmaps de décisions)
* Early-stopping plus fin basé sur des backtests paramétrés
* Hyperparam tuning (Optuna, W&B, etc.)
* Ajout d’un logger ou d’une DB pour les décisions live (audit trail)
* Support multi-symboles / multi-timeframes

---

## 10. Contact / Contributions

* Ouvre une **issue** pour :

  * bugs
  * idées d’amélioration
  * questions sur l’architecture
* Tu peux aussi proposer des **PR** avec :

  * améliorations de code
  * nouveaux scripts d’analyse / visualisation
  * nouveaux setups d’agents

Bon scalping robotisé 🐺📉📈

```
```
