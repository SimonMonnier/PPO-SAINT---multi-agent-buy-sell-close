# 🐺 SAINTv2 — Trading RL BTCUSD M1

### *Backtest · Entraînement PPO · Exécution Live MetaTrader 5*

SAINTv2 “Loup Ω” est un agent de trading automatisé basé sur PPO + SAINT (Self-Attention across Interleaved Time-series).
Il est conçu pour du **scalping BTCUSD en M1**, avec fusion **M1 + H1**, gestion avancée du risque, SL/TP dynamiques, break-even intelligent et trailing basé ATR.

Ce dépôt contient :

* ⚡ Backtest complet
* 📡 Exécution Live sur MetaTrader 5
* 🧠 Entraînement PPO + architecture SAINTv2
* 📦 Modèles pré-entraînés (long & short)
* 📊 Normalisation globale des indicateurs

---

# 📁 Contenu du projet

## 🧪 Backtest

`backtest_saintv2_trio.py` 

* Fusion M1/H1 (`merge_asof`)
* Indicateurs identiques au training :

  * RSI14, ATR14, vol20, returns, range_norm
  * Momentum-Confirmed Entry Filter (mom_5, rsi_ok, high_vol_regime)
* Gestion de position et moteur de trade :

  * BUY1 / SELL1 / BUY1.8 / SELL1.8 / HOLD
  * SL/TP basés ATR
  * Break-even automatique
  * Trailing intelligent
* Action mask identique au training
* Simulation microstructure (spread, slippage)
* Structure d’observation `(25 × 20)` identique au modèle

---

## 📡 Exécution Live MT5

`ia_live sell & buy & close.py` 

* Récupération M1/H1 depuis MetaTrader 5
* Normalisation identique au training
* Action mask live (long only / short only / duel)
* Ouverture et gestion des ordres MT5 :

  * Volume intelligent basé sur le risk scale
  * SL/TP proposés à l’ouverture via ATR
* Break-even + trailing en conditions réelles
* Compatible avec multiples agents (long + short séparés)

---

## 🧠 Entraînement PPO + SAINTv2

`model saint sell & buy & close.py` 

* Implémentation complète PPO :

  * GAE(λ), clipping, entropy, KL-target
  * Training multi-epoch avec batchs 256
* Environnement Gym RL spécialisé trading :

  * Observation normalisée M1/H1
  * Embedding position :

    * pos, entry_price_scaled, current_price_scaled, last_risk_scale
  * Reward shaping optimisé :

    * Momentum bonus
    * Holding penalty
    * Latent PnL reward
    * TP/SL incentives
* SAINTv2 Single-Head :

  * Attention 2D : RowAttention + ColumnAttention
  * FFN gated
  * Actor/Critic intégré
* Curriculum de volatilité
* Split + Walk-Forward supporté

---

# 🎯 Actions disponibles

| ID | Action  | Description                        |
| -- | ------- | ---------------------------------- |
| 0  | BUY1    | Achat taille standard              |
| 1  | SELL1   | Vente taille standard              |
| 2  | BUY1.8  | Achat agressif                     |
| 3  | SELL1.8 | Vente agressive                    |
| 4  | HOLD    | Ne rien faire / rester en position |

Modes disponibles :

* **both** (complet : BUY + SELL)
* **long** (seulement BUY)
* **short** (seulement SELL)
* **duel** (backtest long vs short)
* **close** (agent dédié à la fermeture de positions)

---

# 📊 Normalisation des données

`norm_stats_ohlc_indics.npz` contient les **moyennes et écarts-types** utilisés sur toutes les features M1/H1.

⚠️ **Indispensable** :
Toutes les phases (training, backtest, live) utilisent exactement ces statistiques, sans quoi le modèle perd toute cohérence.

---

# 🤖 Modèles pré-entraînés inclus

* `bestprofit_saintv2_loup_long_wf1_long_wf1.pth`
* `bestprofit_saintv2_loup_short_wf1_short_wf1.pth`

Ces modèles sont directement exploitables en :

* Backtest
* Trading live
* Transfert learning

---

# 🛠 Installation

```bash
pip install torch numpy pandas MetaTrader5 gymnasium
```

MetaTrader 5 doit être installé (Windows uniquement).

---

# ▶️ Utilisation

## Backtest (offline)

```bash
python backtest_saintv2_trio.py
```

## Live Trading (MT5)

```bash
python "ia_live sell & buy & close.py"
```

## Entraînement complet PPO

```bash
python "model saint sell & buy & close.py"
```

---

# 🧩 Architecture SAINTv2 (résumé)

* Input `(T=25, F=20)`
* Projection linéaire 1→D
* Embedding temporel & d’indice de feature
* **RowAttention** : dépendances temporelles
* **ColumnAttention** : dépendances entre features
* Gated FFN
* MLP final
* **Head Actor (5 actions)**
* **Head Critic (valeur V)**

---

# ⚠️ Avertissement

Projet à but expérimental.
Aucune performance financière n’est garantie.
Utilisation en réel = **à vos risques**.

---
