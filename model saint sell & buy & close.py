# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup" avec 4 modes d’agent :
#   side = "both"  → agent symétrique (BUY + SELL)
#   side = "long"  → agent spécialisé BUY
#   side = "short" → agent spécialisé SELL
#   side = "close" → agent spécialisé CLÔTURE :
#                       - entrées générées par les agents long/short gelés
#                       - lui ne décide que CLOSE / HOLD en position
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import MetaTrader5 as mt5
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Categorical


# Optimisations PyTorch
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ============================================================
# SEED GLOBAL
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# CONSTANTES
# ============================================================

# 0:BUY1, 1:SELL1, 2:BUY1.8, 3:SELL1.8, 4:HOLD
N_ACTIONS = 5
MASK_VALUE = -1e4  # valeur de masquage compatible float16

# Stats de normalisation
NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

# Modèles pré-entraînés pour les ENTRÉES (utilisés uniquement quand side="close")
BEST_MODEL_LONG_PATH = "best_saintv2_loup_long_long.pth"
BEST_MODEL_SHORT_PATH = "best_saintv2_loup_short_short.pth"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class PPOConfig:
    # Données
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_M5
    n_bars: int = 161800
    lookback: int = 25

    # PPO Training
    epochs: int = 200
    episodes_per_epoch: int = 4
    episode_length: int = 2000
    updates_per_epoch: int = 4
    tp_shrink: float = 0.7

    batch_size: int = 256
    gamma: float = 0.97
    lambda_gae: float = 0.95
    clip_eps: float = 0.12
    lr: float = 3e-4
    target_kl: float = 0.03
    value_coef: float = 0.5
    entropy_coef: float = 0.08  # patch: plus d'exploration au début
    max_grad_norm: float = 1.0

    # SAINT
    d_model: int = 80

    # Trading
    initial_capital: float = 1000.0
    position_size: float = 0.06
    leverage: float = 6.0
    fee_rate: float = 0.0004
    min_capital_frac: float = 0.2
    max_drawdown: float = 0.8

    # Risk management / position sizing
    risk_per_trade: float = 0.012
    max_position_frac: float = 0.35
    position_vol_penalty: float = 1e-3

    # StopLoss / TakeProfit ATR
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    # Microstructure
    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    # Scalp (plus utilisé dans le reward, mais conservé dans la config)
    scalping_max_holding: int = 12
    scalping_holding_penalty: float = 2.5e-5
    scalping_flat_penalty: float = 5e-6
    scalping_flat_bonus: float = 5e-5

    # Curriculum vol
    use_vol_curriculum: bool = True

    # Device
    force_cpu: bool = False
    use_amp: bool = True

    # Spécialisation d'agent
    # "both"  -> BUY + SELL
    # "long"  -> seulement BUY1 / BUY1.8 / HOLD
    # "short" -> seulement SELL1 / SELL1.8 / HOLD
    # "close" -> agent de clôture : CLOSE / HOLD en position, entrées gérées par long/short
    side: str = "both"

    # Préfixe pour nommer les fichiers de modèle
    model_prefix: str = "saintv2_singlehead_scalping_ohlc_indics_h1_loup"


# ============================================================
# INDICATEURS (M1 & H1)
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    # returns
    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_5"] = c.pct_change(5)
    df["ret_15"] = c.pct_change(15)
    df["ret_60"] = c.pct_change(60)

    # realized vol
    ret = c.pct_change()
    df["realized_vol_20"] = ret.rolling(20).std()

    # Volatility regime
    roll_mean = df["realized_vol_20"].rolling(500).mean()
    roll_std = df["realized_vol_20"].rolling(500).std()
    df["vol_regime"] = (df["realized_vol_20"] - roll_mean) / (roll_std + 1e-8)

    # EMAs
    df["ema_5"] = c.ewm(span=5, adjust=False).mean()
    df["ema_10"] = c.ewm(span=10, adjust=False).mean()
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()

    # RSI
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)

    df["rsi_7"] = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)

    # ATR(14)
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Stoch
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14 + 1e-8) * 100
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal

    # Ichimoku
    conv_period = 9
    base_period = 26
    span_b_period = 52

    conv_line = (h.rolling(conv_period).max() + l.rolling(conv_period).min()) / 2
    base_line = (h.rolling(base_period).max() + l.rolling(base_period).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(base_period)
    span_b = ((h.rolling(span_b_period).max() + l.rolling(span_b_period).min()) / 2).shift(base_period)

    df["ichimoku_tenkan"] = conv_line
    df["ichimoku_kijun"] = base_line
    df["ichimoku_span_a"] = span_a
    df["ichimoku_span_b"] = span_b

    df["dist_tenkan"] = (c - conv_line) / (c + 1e-8)
    df["dist_kijun"] = (c - base_line) / (c + 1e-8)
    df["dist_span_a"] = (c - span_a) / (c + 1e-8)
    df["dist_span_b"] = (c - span_b) / (c + 1e-8)

    ma_100 = c.rolling(100).mean()
    std_100 = c.rolling(100).std()
    df["ma_100"] = ma_100
    df["zscore_100"] = (c - ma_100) / (std_100 + 1e-8)

    idx = df.index
    hours = idx.hour.values
    dows = idx.dayofweek.values

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7)

    if "tick_volume" in df.columns:
        df["tick_volume_log"] = np.log1p(df["tick_volume"])
    else:
        df["tick_volume_log"] = 0.0

    return df


# ============================================================
# CHARGEMENT M1 + H1
# ============================================================

def load_mt5_data(cfg: PPOConfig) -> pd.DataFrame:
    print("Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.init()")

    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe, 0, cfg.n_bars
    )

    n_h1 = max(cfg.n_bars // 30, 5000)
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.htf_timeframe, 0, n_h1
    )

    mt5.shutdown()

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1")

    # M1
    df_m1 = pd.DataFrame(rates_m1)
    df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
    df_m1.set_index("time", inplace=True)
    df_m1 = df_m1[["open", "high", "low", "close", "tick_volume"]]
    df_m1 = add_indicators(df_m1)

    # H1 (ou M5 selon cfg.htf_timeframe)
    df_h1 = pd.DataFrame(rates_h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_h1.set_index("time", inplace=True)
    df_h1 = df_h1[["open", "high", "low", "close", "tick_volume"]]
    df_h1 = add_indicators(df_h1)
    df_h1 = df_h1.add_suffix("_h1")

    df_m1_reset = df_m1.reset_index()
    df_h1_reset = df_h1.reset_index()
    df_h1_reset = df_h1_reset.rename(columns={"time_h1": "time"})

    merged = pd.merge_asof(
        df_m1_reset.sort_values("time"),
        df_h1_reset.sort_values("time"),
        on="time",
        direction="backward"
    )

    merged = merged.dropna().reset_index(drop=True)
    print(f"{len(merged)} bougies M1 alignées avec H1 après indicateurs.")
    return merged


# ============================================================
# FEATURES / NORMALISATION
# ============================================================

FEATURE_COLS_M1 = [
    "open", "high", "low", "close",
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_60",
    "realized_vol_20", "vol_regime",
    "ema_5", "ema_10", "ema_20",
    "rsi_7", "rsi_14",
    "atr_14",
    "stoch_k", "stoch_d",
    "macd", "macd_signal",
    "dist_tenkan", "dist_kijun", "dist_span_a", "dist_span_b",
    "ma_100", "zscore_100",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "tick_volume_log",
]

FEATURE_COLS_H1 = [
    "close_h1",
    "ema_20_h1",
    "rsi_14_h1",
    "macd_h1",
    "macd_signal_h1",
    "zscore_100_h1",
    "dist_tenkan_h1",
    "dist_kijun_h1",
    "dist_span_a_h1",
    "dist_span_b_h1",
    "realized_vol_20_h1",
]

FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1

N_BASE_FEATURES = len(FEATURE_COLS)
# 4 features supplémentaires pour l’embedding de position :
#   - position (-1,0,1)
#   - entry_price_scaled
#   - current_price_scaled
#   - last_risk_scale
N_POS_FEATURES = 4
OBS_N_FEATURES = N_BASE_FEATURES + N_POS_FEATURES


class MarketData:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 stats: Optional[Dict[str, np.ndarray]] = None):
        X = df[feature_cols].values.astype(np.float32)

        if stats is not None:
            mean, std = stats["mean"], stats["std"]
            std = np.where(std < 1e-8, 1.0, std)
            X = (X - mean) / std

        self.features = X
        self.close = df["close"].values.astype(np.float32)
        self.length = len(df)

        self.atr14 = df["atr_14"].values.astype(np.float32) if "atr_14" in df.columns else np.zeros(len(df), np.float32)
        self.ema20_h1 = df["ema_20_h1"].values.astype(np.float32) if "ema_20_h1" in df.columns else np.zeros(len(df), np.float32)
        self.high = df["high"].values.astype(np.float32) if "high" in df.columns else np.zeros(len(df), np.float32)
        self.low = df["low"].values.astype(np.float32) if "low" in df.columns else np.zeros(len(df), np.float32)

    def __len__(self):
        return self.length


def create_datasets(df: pd.DataFrame, feature_cols: List[str]):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = df[:train_end].reset_index(drop=True)
    df_val = df[train_end:val_end].reset_index(drop=True)
    df_test = df[val_end:].reset_index(drop=True)

    Xtrain = df_train[feature_cols].values.astype(np.float32)
    mean, std = Xtrain.mean(0), Xtrain.std(0)
    stats = {"mean": mean, "std": std}

    train_data = MarketData(df_train, feature_cols, stats)
    val_data = MarketData(df_val, feature_cols, stats)
    test_data = MarketData(df_test, feature_cols, stats)

    print(f"SPLIT : train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return train_data, val_data, test_data, stats


# ======================================================================
# ENVIRONNEMENT
# ======================================================================

class BTCTradingEnvDiscrete(gym.Env):
    """
    Actions env :
        0 = BUY (ou CLOSE si side="close" et position != 0)
        1 = SELL
        2 = HOLD

    L'agent RL single-head a 5 actions (0..4), mais ce mapping se fait
    dans le training via map_agent_action_to_env_action().
    L'env ne voit que 0/1/2.

    Observation :
        - fenêtre (lookback, OBS_N_FEATURES)
          = features M1/H1 normalisées
          + embedding de position :
              - position (-1 / 0 / +1)
              - prix d’entrée (scalé)
              - prix actuel (scalé)
              - last_risk_scale (scale appliqué lors de la dernière action)
    """

    metadata = {"render_modes": []}

    def __init__(self, data: MarketData, cfg: PPOConfig):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.lookback = cfg.lookback

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, OBS_N_FEATURES),
            dtype=np.float32
        )

        if self.cfg.use_vol_curriculum:
            self._init_vol_curriculum()
        else:
            self.low_vol_starts = None
            self.high_vol_starts = None

        self.risk_scale = 1.0
        self.last_risk_scale = 1.0
        self.reset()

    # ---------- Curriculum vol ----------

    def _init_vol_curriculum(self):
        close = self.data.close
        ret = np.diff(close) / (close[:-1] + 1e-8)
        vol20 = pd.Series(ret).rolling(20).std().to_numpy()
        vol20 = np.concatenate([[np.nan], vol20])

        valid = ~np.isnan(vol20)
        if valid.sum() < 30:
            self.low_vol_starts = None
            self.high_vol_starts = None
            return

        q_low, q_high = np.quantile(vol20[valid], [0.3, 0.7])

        candidate_low = np.where((vol20 <= q_low) & valid)[0]
        candidate_high = np.where((vol20 >= q_high) & valid)[0]

        max_start = self.data.length - self.cfg.episode_length - 2
        low = candidate_low[
            (candidate_low >= self.lookback) &
            (candidate_low <= max_start)
        ]
        high = candidate_high[
            (candidate_high >= self.lookback) &
            (candidate_high <= max_start)
        ]

        self.low_vol_starts = low if len(low) > 0 else None
        self.high_vol_starts = high if len(high) > 0 else None

    def set_risk_scale(self, scale: float):
        self.risk_scale = float(max(scale, 0.0))
        if self.risk_scale <= 0.0:
            self.risk_scale = 1.0
        # On expose ce scale à l'agent via l'observation suivante
        self.last_risk_scale = float(self.risk_scale)

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.data.length - self.cfg.episode_length - 2

        start_idx = None
        if self.cfg.use_vol_curriculum and self.low_vol_starts is not None and self.high_vol_starts is not None:
            if np.random.rand() < 0.5 and len(self.low_vol_starts) > 0:
                start_idx = int(np.random.choice(self.low_vol_starts))
            elif len(self.high_vol_starts) > 0:
                start_idx = int(np.random.choice(self.high_vol_starts))

        if start_idx is None:
            start_idx = np.random.randint(self.lookback, max_start)

        self.start_idx = start_idx
        self.end_idx = self.start_idx + self.cfg.episode_length
        self.idx = self.start_idx

        self.capital = self.cfg.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.current_size = 0.0

        self.sl_price = 0.0
        self.tp_price = 0.0
        self.entry_idx = -1
        self.entry_atr = 0.0

        self.last_realized_pnl = 0.0
        self.peak_capital = self.capital
        self.trades_pnl: List[float] = []

        self.bars_in_position = 0
        self.risk_scale = 1.0
        self.last_risk_scale = 1.0

        self.max_dd = 0.0

        obs = self._get_obs()
        return obs, {
            "capital": self.capital,
            "position": self.position,
            "drawdown": 0.0,
            "done_reason": None
        }

    def _get_obs(self):
        start = self.idx - self.lookback
        base = self.data.features[start:self.idx]  # (T, N_BASE_FEATURES)

        # Embedding de position + info de levier
        price_scale = 100000.0
        if self.idx > 0:
            current_price = float(self.data.close[self.idx - 1])
        else:
            current_price = 0.0

        if self.position != 0 and self.entry_price > 0.0:
            entry_scaled = float(self.entry_price / price_scale)
        else:
            entry_scaled = 0.0

        current_scaled = float(current_price / price_scale) if current_price > 0.0 else 0.0
        pos_feature = float(self.position)  # -1 / 0 / +1
        risk_feature = float(self.last_risk_scale)

        extra_vec = np.array(
            [pos_feature, entry_scaled, current_scaled, risk_feature],
            dtype=np.float32
        )  # (4,)

        extra_block = np.repeat(extra_vec[None, :], self.lookback, axis=0)  # (T, 4)

        obs = np.concatenate([base, extra_block], axis=-1).astype(np.float32)
        return obs

    def _apply_micro(self, price: float, side: int, is_entry: bool = True) -> float:
        """
        Applique spread + slippage.
        is_entry = True  → pire prix (adverse)
        is_entry = False → meilleur prix (favorable, pour la sortie)
        """
        spread = self.cfg.spread_bps / 10_000.0
        slip = self.cfg.slippage_bps / 10_000.0

        # Spread : toujours adverse à l'entrée, favorable à la sortie
        if is_entry:
            price *= (1 + side * spread * 0.5)   # long: +spread/2, short: -spread/2
        else:
            price *= (1 - side * spread * 0.5)   # inverse à la sortie

        # Slippage : toujours aléatoire
        price *= (1 + np.random.uniform(-slip, slip))
        return price

    def _compute_dynamic_size(self, price: float) -> float:
        base = float(self.cfg.position_size)
        scale = float(max(self.risk_scale, 0.0))
        if scale <= 0.0:
            scale = 1.0
        size = base * scale
        return float(size)

    def step(self, action: int):
        """
        action env :
          0 = BUY, 1 = SELL, 2 = HOLD
          (sauf quand side="close" et position != 0 : 0 = CLOSE)
        """
        price = self.data.close[self.idx]
        high_bar = self.data.high[self.idx]
        low_bar = self.data.low[self.idx]

        old_pos = self.position
        prev_capital = self.capital

        if self.idx > 0:
            prev_price = self.data.close[self.idx - 1]
        else:
            prev_price = price

        if old_pos != 0 and self.current_size > 0 and self.entry_price > 0:
            prev_latent = (
                old_pos *
                (prev_price - self.entry_price) *
                self.current_size *
                self.cfg.leverage
            )
        else:
            prev_latent = 0.0

        prev_equity = prev_capital + prev_latent

        realized = 0.0
        realized_trade = 0.0

        # temps en position
        if old_pos != 0:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

        # --------- FERMETURE MANUELLE (agent close) ---------
        manual_close = False
        if self.cfg.side == "close" and old_pos != 0 and action == 0:
            # CLOSE immédiat au prix courant (microstructure appliquée)
            exit_price = self._apply_micro(price, -old_pos)

            pnl = (
                old_pos *
                (exit_price - self.entry_price) *
                self.current_size *
                self.cfg.leverage
            )
            fee = self.cfg.fee_rate * exit_price * self.current_size
            realized = pnl - fee
            realized_trade = realized

            self.capital += realized
            self.last_realized_pnl = realized
            self.trades_pnl.append(realized)

            self.position = 0
            self.current_size = 0.0
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0
            self.entry_idx = -1
            self.entry_atr = 0.0
            self.risk_scale = 1.0
            self.last_risk_scale = 1.0
            manual_close = True

        # --------- Ouverture (si flat, et pas en train de faire un close) ---------
        if not manual_close and action in (0, 1) and old_pos == 0:
            side = 1 if action == 0 else -1
            if side != 0:
                size = self._compute_dynamic_size(price)
                if size > 0.0:
                    self.current_size = size
                    self.position = side

                    exec_price = self._apply_micro(price, side, is_entry=True)
                    self.entry_price = exec_price
                    self.entry_idx = self.idx

                    atr_raw = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
                    fallback = 0.0015 * exec_price
                    self.entry_atr = max(atr_raw, fallback, 1e-8)

                    sl_dist = self.cfg.atr_sl_mult * self.entry_atr
                    tp_dist = self.cfg.atr_tp_mult * self.entry_atr * self.cfg.tp_shrink

                    if side == 1:
                        self.sl_price = max(1e-8, exec_price - sl_dist)
                        self.tp_price = max(1e-8, exec_price + tp_dist)
                    else:
                        self.sl_price = max(1e-8, exec_price + sl_dist)
                        self.tp_price = max(1e-8, exec_price - tp_dist)

                    fee = self.cfg.fee_rate * exec_price * size
                    self.capital -= fee
                else:
                    self.position = 0
                    self.current_size = 0.0
                    self.entry_price = 0.0
                    self.sl_price = 0.0
                    self.tp_price = 0.0
                    self.entry_idx = -1
                    self.entry_atr = 0.0

        # --------- FERMETURE AUTOMATIQUE SL/TP ---------
        hit_sl = False
        hit_tp = False

        if (
            not manual_close and
            self.position != 0 and
            self.current_size > 0 and
            self.entry_price > 0 and
            self.entry_idx >= 0 and
            self.idx > self.entry_idx
        ):
            exit_price = None

            if self.position == 1:  # long
                if self.sl_price > 0 and low_bar <= self.sl_price:
                    exit_price = self.sl_price
                    hit_sl = True
                elif self.tp_price > 0 and high_bar >= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True

            elif self.position == -1:  # short
                if self.sl_price > 0 and high_bar >= self.sl_price:
                    exit_price = self.sl_price
                    hit_sl = True
                elif self.tp_price > 0 and low_bar <= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True

            if exit_price is not None:
                exit_price = self._apply_micro(exit_price, -self.position, is_entry=False)

                pnl = (self.position * (exit_price - self.entry_price) * self.current_size * self.cfg.leverage)
                fee = self.cfg.fee_rate * exit_price * self.current_size
                realized = pnl - fee
                realized_trade = realized

                self.capital += realized
                self.last_realized_pnl = realized
                self.trades_pnl.append(realized)

                self.position = 0
                self.current_size = 0.0
                self.entry_price = 0.0
                self.sl_price = 0.0
                self.tp_price = 0.0
                self.entry_idx = -1
                self.entry_atr = 0.0
                self.risk_scale = 1.0
                self.last_risk_scale = 1.0

                # --------- LATENT PNL CORRIGÉ (prix extrême dans la bougie) ---------
        latent = 0.0
        if self.position != 0 and self.current_size > 0 and self.entry_price > 0:
            current_price_for_pnl = price  # par défaut = close

            if self.position == 1:  # LONG
                # On prend le plus haut de la bougie (meilleur prix possible)
                current_price_for_pnl = high_bar
            elif self.position == -1:  # SHORT
                # On prend le plus bas de la bougie (meilleur prix possible)
                current_price_for_pnl = low_bar

            latent = (
                self.position *
                (current_price_for_pnl - self.entry_price) *
                self.current_size *
                self.cfg.leverage
            )

        equity = self.capital + latent
        equity_clamped = max(equity, 1e-8)
        prev_equity_clamped = max(prev_equity, 1e-8)

        # Reward de base = log-ret sur l’equity (pur)
        log_ret = math.log(equity_clamped / prev_equity_clamped)
        reward = log_ret

        # Peak equity / drawdown
        self.peak_capital = max(self.peak_capital, equity)
        dd = (self.peak_capital - equity) / (self.peak_capital + 1e-8)
        self.max_dd = max(self.max_dd, dd)

        # Bonus / malus surfine
        if hit_tp:
            reward += 0.002
        if hit_sl:
            reward -= 0.005
        if dd > 0.6:
            reward -= 0.01

        # Clip final (stabilisation)
        reward = float(np.clip(reward, -0.03, 0.03))

        # Conditions de fin
        self.idx += 1
        done = False
        done_reason = None

        if self.idx >= self.end_idx:
            done = True
            done_reason = "episode_end"

        if dd > self.cfg.max_drawdown:
            done = True
            done_reason = "max_drawdown"

        if self.capital <= self.cfg.initial_capital * self.cfg.min_capital_frac:
            done = True
            done_reason = "min_capital"

        obs = self._get_obs()

        return obs, float(reward), done, False, {
            "capital": self.capital,
            "drawdown": self.max_dd,
            "done_reason": done_reason,
            "position": self.position
        }


# ======================================================================
# SAINT v2 — SINGLE-HEAD (ACTOR + CRITIC)
# ======================================================================

class GatedFFN(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.05):
        super().__init__()
        inner = d * mult
        self.lin1 = nn.Linear(d, inner * 2)
        self.lin2 = nn.Linear(inner, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, gate = self.lin1(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(gate)
        h = self.lin2(self.dropout(h))
        return x + h


class ColumnAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.reshape(B * T, F, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        return h.reshape(B, T, F, D)


class RowAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        h = h.reshape(B, F, T, D).permute(0, 2, 1, 3)
        return h


class SAINTv2Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, mult: int):
        super().__init__()
        self.ra1 = RowAttention(d, heads, dropout)
        self.ff1 = GatedFFN(d, mult, dropout)

        self.ra2 = RowAttention(d, heads, dropout)
        self.ff2 = GatedFFN(d, mult, dropout)

        self.ca = ColumnAttention(d, heads, dropout)
        self.ff3 = GatedFFN(d, mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ra1(x)
        x = self.ff1(x)
        x = self.ra2(x)
        x = self.ff2(x)
        x = self.ca(x)
        x = self.ff3(x)
        return x


class SAINTPolicySingleHead(nn.Module):
    """
    SAINTv2 simplifié :
      - actor: logits (N_ACTIONS)
      - critic: V(s)
    """

    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 72,
        num_blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
        ff_mult: int = 2,
        max_len: int = 64,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_actions = n_actions

        self.input_proj = nn.Linear(1, d_model)
        self.scale = math.sqrt(d_model)
        self.row_emb = nn.Embedding(max_len, d_model)
        self.col_emb = nn.Embedding(n_features, d_model)

        self.blocks = nn.ModuleList([
            SAINTv2Block(d_model, heads, dropout, ff_mult)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        x : (B,T,F)
        """
        assert x.dim() == 3, f"Input x must be (B,T,F), got {x.shape}"
        B, T, F = x.shape

        tok = self.input_proj(x.unsqueeze(-1)) * self.scale

        rows = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, F)
        cols = torch.arange(F, device=x.device).view(1, 1, F).expand(B, T, F)

        tok = tok + self.row_emb(rows) + self.col_emb(cols)

        for blk in self.blocks:
            tok = blk(tok)

        h_time = tok.mean(dim=1)
        h_feat = tok.mean(dim=2)

        cls_time = h_time.mean(dim=1)
        cls_feat = h_feat.mean(dim=1)

        h = cls_time + cls_feat
        h = self.norm(h)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value


# ======================================================================
# PPO UTILITAIRES
# ======================================================================

def get_device(cfg: PPOConfig):
    if cfg.force_cpu:
        print("CPU forcé.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


def compute_gae(rewards, values, dones, gamma, lam, last_value=0.0):
    values = values + [last_value]
    gae = 0.0
    adv = []

    for t in reversed(range(len(rewards))):
        mask = 1 - int(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv.insert(0, gae)

    returns = [adv[i] + values[i] for i in range(len(adv))]
    return adv, returns


def epsilon_greedy_from_logits(logits_masked: torch.Tensor, eps: float) -> int:
    with torch.no_grad():
        if np.random.rand() > eps:
            return logits_masked.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits_masked, dim=-1)
            dist = Categorical(probs)
            return dist.sample().item()


# --------- Masques d'actions avec spécialisation (long/short/both/close) ---------

def build_mask_from_pos_scalar(pos: int, device, side: str) -> torch.Tensor:
    """
    Renvoie un bool mask (N_ACTIONS,) True = action valide.

    - side == "both"  : agent symétrique BUY/SELL
    - side == "long"  : BUY1 / BUY1.8 / HOLD
    - side == "short" : SELL1 / SELL1.8 / HOLD
    - side == "close" :
          * pos == 0  → HOLD uniquement
          * pos != 0  → CLOSE (0) + HOLD (4)
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)

    if side == "close":
        if pos == 0:
            mask[4] = True
        else:
            mask[0] = True  # CLOSE
            mask[4] = True  # HOLD
        return mask

    if pos != 0:
        mask[4] = True
        return mask

    if side == "both":
        mask[:] = True
    elif side == "long":
        mask[0] = True
        mask[2] = True
        mask[4] = True
    elif side == "short":
        mask[1] = True
        mask[3] = True
        mask[4] = True
    else:
        mask[:] = True

    return mask


def build_action_mask_from_positions(positions: torch.Tensor, side: str) -> torch.Tensor:
    """
    positions : (B,) -1, 0, +1
    Retourne un mask (B, N_ACTIONS) bool.
    """
    device = positions.device
    B = positions.shape[0]
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device=device)

    flat = (positions == 0)
    inpos = ~flat

    if side == "close":
        if flat.any():
            mask[flat, 4] = True    # HOLD
        if inpos.any():
            mask[inpos, 0] = True   # CLOSE
            mask[inpos, 4] = True   # HOLD
        return mask

    if flat.any():
        if side == "both":
            mask[flat] = True
        elif side == "long":
            mask[flat, 0] = True
            mask[flat, 2] = True
            mask[flat, 4] = True
        elif side == "short":
            mask[flat, 1] = True
            mask[flat, 3] = True
            mask[flat, 4] = True
        else:
            mask[flat] = True

    if inpos.any():
        mask[inpos, 4] = True

    return mask


# --------- Mapping unique agent_action -> env_action + risk_scale ---------

def map_agent_action_to_env_action(
    a: int,
    pos: int,
    cfg: PPOConfig,
    device: torch.device,
    state_tensor: torch.Tensor,
    policy_long: Optional[nn.Module] = None,
    policy_short: Optional[nn.Module] = None,
    epoch: int = 1,
) -> Tuple[int, float]:
    """
    Point unique de mapping entre l’action de l’agent (0..4) et
    l’action environnement (0..2) + risk_scale.

    Retourne:
        env_action ∈ {0,1,2}
        risk_scale (float)
    """
    env_action = 2
    risk_scale = 1.0

    # On force à travailler sur un batch de taille 1
    if state_tensor.dim() == 2:
        state_tensor = state_tensor.unsqueeze(0)
    elif state_tensor.dim() == 3 and state_tensor.size(0) > 1:
        state_tensor = state_tensor[:1]
    elif state_tensor.dim() != 3:
        raise ValueError(f"state_tensor doit être (B,T,F) ou (T,F), reçu {state_tensor.shape}")

    # Mode CLOSE : entrées via agents LONG/SHORT gelés
    if cfg.side == "close":
        if pos == 0:
            if policy_long is None or policy_short is None:
                raise RuntimeError("policy_long / policy_short requis pour side='close'")

            # En flat : décision d’entrée via les agents long/short pré-entraînés
            with torch.no_grad():
                logits_long, _ = policy_long(state_tensor)
                logits_long = logits_long[0]
                mask_long = build_mask_from_pos_scalar(pos, device, "long")
                logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)

                logits_short, _ = policy_short(state_tensor)
                logits_short = logits_short[0]
                mask_short = build_mask_from_pos_scalar(pos, device, "short")
                logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)

                # Scores basés sur les logits (plus stables que softmax)
                open_long_max = torch.maximum(logits_long_m[0], logits_long_m[2])
                score_long = (open_long_max - logits_long_m[4]).item()

                open_short_max = torch.maximum(logits_short_m[1], logits_short_m[3])
                score_short = (open_short_max - logits_short_m[4]).item()

                if score_long <= 0.0 and score_short <= 0.0:
                    env_action = 2
                    risk_scale = 1.0
                else:
                    if score_long >= score_short:
                        chosen_side = "long"
                        chosen_logits = logits_long_m
                    else:
                        chosen_side = "short"
                        chosen_logits = logits_short_m

                    if chosen_side == "long":
                        if chosen_logits[0] >= chosen_logits[2]:
                            a_entry = 0
                        else:
                            a_entry = 2
                    else:
                        if chosen_logits[1] >= chosen_logits[3]:
                            a_entry = 1
                        else:
                            a_entry = 3

                    if a_entry in (0, 2):  # BUY
                        env_action = 0
                        risk_scale = 1.8 if a_entry == 2 else 1.0
                    elif a_entry in (1, 3):  # SELL
                        env_action = 1
                        risk_scale = 1.8 if a_entry == 3 else 1.0
                    else:
                        env_action = 2
                        risk_scale = 1.0
        else:
            # En position : l’agent close décide CLOSE vs HOLD
            if a == 0:
                env_action = 0   # CLOSE
                risk_scale = 1.0
            else:
                env_action = 2   # HOLD
                risk_scale = 1.0

        # SHORT : on interdit le 1.8x À VIE (ou jusqu'à preuve du contraire)
        
        if epoch <= 35:
            risk_scale = 1.0

        return env_action, risk_scale

    # Modes both / long / short
    if pos == 0:
        # Flat
        if a == 4:
            env_action = 2
            risk_scale = 1.0
        elif a in (0, 2):  # BUY
            env_action = 0
            risk_scale = 1.8 if a == 2 else 1.0
        elif a in (1, 3):  # SELL
            env_action = 1
            risk_scale = 1.8 if a == 3 else 1.0
        else:
            env_action = 2
            risk_scale = 1.0
    else:
        # En position : on laisse l’env gérer la sortie (SL/TP/close agent)
        env_action = 2
        risk_scale = 1.0

    # SHORT : on interdit le 1.8x À VIE (ou jusqu'à preuve du contraire)
    
    if epoch <= 35:
        risk_scale = 1.0

    return env_action, risk_scale


# ======================================================================
# TRAINING PPO
# ======================================================================

def run_training(cfg: PPOConfig):
    df = load_mt5_data(cfg)
    train_data, val_data, test_data, stats = create_datasets(df, FEATURE_COLS)

    np.savez(NORM_STATS_PATH, mean=stats["mean"], std=stats["std"])
    print(f"Stats de normalisation sauvegardées → {NORM_STATS_PATH}")

    device = get_device(cfg)

    env = BTCTradingEnvDiscrete(train_data, cfg)
    val_env = BTCTradingEnvDiscrete(val_data, cfg)

    policy = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=cfg.d_model,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-8)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: max(0.05, 1 - e / cfg.epochs)
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    # chemins modèles
    best_path = f"best_{cfg.model_prefix}_{cfg.side}.pth"
    last_path = f"last_{cfg.model_prefix}_{cfg.side}.pth"
    best_profit_path = f"last_{cfg.model_prefix}_{cfg.side}_profit.pth"
    if os.path.exists(best_path):
        print(f"→ Chargement du modèle existant ({best_path}) pour continuation…")
        policy.load_state_dict(torch.load(best_path, map_location=device))

    
    # Modèles gelés LONG/SHORT pour le mode "close"
    policy_long = None
    policy_short = None
    if cfg.side == "close":
        policy_long = SAINTPolicySingleHead(
            n_features=OBS_N_FEATURES,
            d_model=cfg.d_model,
            num_blocks=2,
            heads=4,
            dropout=0.05,
            ff_mult=2,
            max_len=cfg.lookback,
            n_actions=N_ACTIONS
        ).to(device)
        policy_long.load_state_dict(torch.load(BEST_MODEL_LONG_PATH, map_location=device))
        policy_long.eval()

        policy_short = SAINTPolicySingleHead(
            n_features=OBS_N_FEATURES,
            d_model=cfg.d_model,
            num_blocks=2,
            heads=4,
            dropout=0.05,
            ff_mult=2,
            max_len=cfg.lookback,
            n_actions=N_ACTIONS
        ).to(device)
        policy_short.load_state_dict(torch.load(BEST_MODEL_SHORT_PATH, map_location=device))
        policy_short.eval()

        print("[CLOSE] Modèles LONG & SHORT gelés chargés pour générer les entrées pendant l'entraînement.")

    best_val_profit = -1e9
    best_metric = -1e9  # Sortino ratio
    best_state = None
    epochs_no_improve = 0
    patience = 100

    metric_history: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        batch_states = []
        batch_actions = []
        batch_oldlog = []
        batch_adv = []
        batch_returns = []
        batch_values = []
        batch_positions = []

        total_reward_epoch = 0.0
        epoch_pnl = []
        epoch_dd = []
        epoch_trades_pnl: List[float] = []

        action_counts_env = np.zeros(3, dtype=np.int64)

        policy.train()

        # --------- collecte expériences ---------
        for ep in range(cfg.episodes_per_epoch):
            state, info = env.reset()
            done = False

            ep_states = []
            ep_actions = []
            ep_logprobs = []
            ep_rewards = []
            ep_values = []
            ep_dones = []
            ep_positions = []

            while not done:
                pos = info.get("position", 0)
                ep_positions.append(pos)

                s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, value = policy(s_tensor)
                    logits = logits[0]

                    mask = build_mask_from_pos_scalar(pos, device, cfg.side)
                    logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                    dist = Categorical(logits=logits_masked)

                                        # ================================================================
                    # FORÇAGE PROGRESSIF (curriculum) – VERSION FINALE QUI GAGNE
                    # ================================================================
                    force_opening = False
                    chosen_action = None

                    if epoch <= 35 and pos == 0:
                        
                        if cfg.side == "long":
                            prob = 0.90                   # long : jusqu'à ~80%
                        elif cfg.side == "short":
                            prob = 0.90                   # short : jusqu'à ~31% max
                        else:  # both ou close
                            prob = 0.90

                        if np.random.rand() < prob:
                            force_opening = True
                            if cfg.side == "long":
                                chosen_action = 0 if np.random.rand() < 0.6 else 2
                            elif cfg.side == "short":
                                chosen_action = 1 if np.random.rand() < 0.6 else 3
                            else:
                                chosen_action = random.choice([0,1,2,3])

                    # Application
                    if force_opening and chosen_action is not None:
                        agent_action = torch.tensor(chosen_action, device=device, dtype=torch.long)
                        logprob = dist.log_prob(agent_action).squeeze()
                    else:
                        agent_action = dist.sample()
                        logprob = dist.log_prob(agent_action).squeeze()
                    a = int(agent_action.item())

                    env_action, risk_scale = map_agent_action_to_env_action(
                        a=a,
                        pos=pos,
                        cfg=cfg,
                        device=device,
                        state_tensor=s_tensor,
                        policy_long=policy_long,
                        policy_short=policy_short,
                        epoch=epoch,
                    )

                env.set_risk_scale(risk_scale)
                action_counts_env[env_action] += 1

                ns, reward, done, _, info = env.step(env_action)
                total_reward_epoch += reward

                ep_rewards.append(reward)
                ep_states.append(state)
                ep_actions.append(a)
                ep_logprobs.append(logprob.detach())
                ep_values.append(value.item())
                ep_dones.append(done)

                state = ns

            epoch_dd.append(info["drawdown"])
            epoch_pnl.append(env.capital - cfg.initial_capital)
            epoch_trades_pnl.extend(env.trades_pnl)

            if done and info.get("done_reason") == "max_drawdown":
                last_value = 0.0
            else:
                with torch.no_grad():
                    s_last = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    _, v_last = policy(s_last)
                    last_value = v_last.item()

            adv, ret = compute_gae(
                ep_rewards, ep_values, ep_dones,
                cfg.gamma, cfg.lambda_gae, last_value
            )

            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_oldlog.extend(ep_logprobs)
            batch_adv.extend(adv)
            batch_returns.extend(ret)
            batch_values.extend(ep_values)
            batch_positions.extend(ep_positions)

        # --------- tenseurs batch ---------
        states_np = np.stack(batch_states, axis=0)
        states = torch.tensor(states_np, dtype=torch.float32, device=device)

        actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
        oldlog = torch.stack(batch_oldlog).to(device).view(-1)
        advantages = torch.tensor(batch_adv, dtype=torch.float32, device=device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        values_old = torch.tensor(batch_values, dtype=torch.float32, device=device)
        positions = torch.tensor(batch_positions, dtype=torch.long, device=device)

        assert states.size(0) == oldlog.size(0) == actions.size(0) == values_old.size(0) == positions.size(0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-10, 10)
        advantages = advantages * 1.5

        # --------- PPO update ---------
        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_entropy = []
        epoch_kl = []

        n_samples = states.size(0)
        idx = np.arange(n_samples)

        for upd in range(cfg.updates_per_epoch):
            np.random.shuffle(idx)

            for start in range(0, n_samples, cfg.batch_size):
                end = start + cfg.batch_size
                ids = idx[start:end]

                sb = states[ids]
                ab = actions[ids]
                lb_old = oldlog[ids]
                adv_b = advantages[ids]
                ret_b = returns[ids]
                val_old = values_old[ids]
                pos_b = positions[ids]

                with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                    logits, value = policy(sb)
                    mask_batch = build_action_mask_from_positions(pos_b, cfg.side)
                    logits_masked = logits.masked_fill(~mask_batch, MASK_VALUE)

                    dist = Categorical(logits=logits_masked)
                    new_log = dist.log_prob(ab)
                    entropy = dist.entropy().mean()

                    ratio = (new_log - lb_old).exp()
                    surr1 = adv_b * ratio
                    surr2 = adv_b * torch.clamp(
                        ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_pred = value.squeeze(-1)
                    v_clipped = val_old + (value_pred - val_old).clamp(-0.2, 0.2)
                    unclipped_loss = (value_pred - ret_b).pow(2)
                    clipped_loss = (v_clipped - ret_b).pow(2)
                    critic_loss = torch.max(unclipped_loss, clipped_loss).mean()

                    # Entropy dynamique (patch)
                    entropy_coef_epoch = cfg.entropy_coef * math.exp(-max(0, epoch - 30) / 50.0)
                    entropy_bonus = entropy_coef_epoch * entropy

                    loss = actor_loss + cfg.value_coef * critic_loss - entropy_bonus

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    approx_kl = (lb_old - new_log).mean().item()

                epoch_actor_loss.append(actor_loss.item())
                epoch_critic_loss.append(critic_loss.item())
                epoch_entropy.append(entropy.item())
                epoch_kl.append(approx_kl)

            if np.mean(epoch_kl) > 1.5 * cfg.target_kl:
                print(f"[PPO] Early stop KL, KL={np.mean(epoch_kl):.4f}")
                break

        scheduler.step()

        # --------- stats train ---------
        profit_epoch = float(sum(epoch_pnl))
        num_trades_epoch = len(epoch_trades_pnl)
        winrate_epoch = (
            float(np.mean([p > 0 for p in epoch_trades_pnl]))
            if num_trades_epoch > 0 else 0.0
        )
        max_dd_epoch = float(max(epoch_dd) if epoch_dd else 0.0)

        total_actions_env = int(action_counts_env.sum()) if action_counts_env.sum() > 0 else 1
        buy_count, sell_count, hold_count = action_counts_env
        buy_ratio = buy_count / total_actions_env
        sell_ratio = sell_count / total_actions_env
        hold_ratio = hold_count / total_actions_env

        # --------- validation (greedy, sans epsilon) ---------
        policy.eval()
        val_pnl = []
        val_dd = []
        val_trades = []

        with torch.no_grad():
            for _ in range(2):
                s, info = val_env.reset()
                done = False
                step_count = 0
                while not done:
                    pos = info.get("position", 0)
                    st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                    logits, _ = policy(st)
                    logits = logits[0]
                    mask = build_mask_from_pos_scalar(pos, device, cfg.side)
                    logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                    # Greedy pur en validation
                    a = int(logits_masked.argmax(dim=-1).item())

                    env_action, risk_scale = map_agent_action_to_env_action(
                        a=a,
                        pos=pos,
                        cfg=cfg,
                        device=device,
                        state_tensor=st,
                        policy_long=policy_long,
                        policy_short=policy_short,
                        epoch=epoch,
                    )
                    val_env.set_risk_scale(risk_scale)

                    ns, r, done, _, info = val_env.step(env_action)
                    s = ns
                    step_count += 1

                val_pnl.append(val_env.capital - cfg.initial_capital)
                val_dd.append(info["drawdown"])
                val_trades.extend(val_env.trades_pnl)

        val_profit = float(sum(val_pnl))
        val_max_dd = float(max(val_dd) if val_dd else 0.0)
        val_num_trades = len(val_trades)
        val_winrate = (
            float(np.mean([t > 0 for t in val_trades]))
            if val_num_trades > 0 else 0.0
        )

        # --------- Métrique : Sortino ratio sur les trades ---------
        if val_num_trades > 10:
            rets = np.array(val_trades, dtype=np.float32) / cfg.initial_capital
            mean_ret = float(rets.mean())
            downside = rets[rets < 0.0]
            downside_std = float(downside.std()) if downside.size > 0 else 1e-4
            sortino = mean_ret / (downside_std + 1e-8)
        else:
            sortino = 0.0

        metric = sortino
        metric_history.append(metric)
        if len(metric_history) >= 30:
            recent_metric = float(np.mean(metric_history[-30:]))
        else:
            recent_metric = float(np.mean(metric_history))

        print(
            f"[{cfg.side.upper()}][EPOCH {epoch:03d}] "
            f"TrainPNL={profit_epoch:8.2f}  "
            f"Trades={num_trades_epoch:4d}  "
            f"Win={winrate_epoch:5.1%}  "
            f"DD={max_dd_epoch:5.1%}  "
            f"ValPNL={val_profit:8.2f}  "
            f"ValTrades={val_num_trades:4d}  "
            f"ValWin={val_winrate:5.1%}  "
            f"ValDD={val_max_dd:5.1%}  "
            f"Sortino={metric:6.3f}  "
            f"Sortino30={recent_metric:6.3f}  "
            f"ENV B:{buy_ratio:4.1%} S:{sell_ratio:4.1%} H:{hold_ratio:4.1%}  "
            f"KL={np.mean(epoch_kl):.4f}"
        )

        if val_profit > best_val_profit:
            best_val_profit = val_profit
            best_state = policy.state_dict().copy()
            torch.save(best_state, best_profit_path)
            epochs_no_improve = 0
            print(f"[{cfg.side.upper()}][EPOCH {epoch:03d}] Nouveau best model (Profit={best_val_profit:.3f}).")
        if recent_metric > best_metric:
            best_metric = recent_metric
            best_state = policy.state_dict().copy()
            torch.save(best_state, best_path)
            epochs_no_improve = 0
            print(f"[{cfg.side.upper()}][EPOCH {epoch:03d}] Nouveau best model (Sortino30={recent_metric:.3f}).")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{cfg.side.upper()}] Early stopping après {epoch} epochs (Sortino rolling ne progresse plus).")
                break

    torch.save(policy.state_dict(), last_path)

    print(f"[{cfg.side.upper()}] Entraînement terminé, passage en TEST…")

    # ================= TEST =================
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()

    test_env = BTCTradingEnvDiscrete(test_data, cfg)
    all_trades = []
    all_dd = []
    all_equity = []

    with torch.no_grad():
        for ep in range(5):
            s, info = test_env.reset()
            done = False
            step_count = 0
            while not done:
                pos = info.get("position", 0)
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = policy(st)
                logits = logits[0]
                mask = build_mask_from_pos_scalar(pos, device, cfg.side)
                logits_masked = logits.masked_fill(~mask, MASK_VALUE)

                # TEST : greedy pur, aucune exploration
                a = int(logits_masked.argmax(dim=-1).item())

                env_action, risk_scale = map_agent_action_to_env_action(
                    a=a,
                    pos=pos,
                    cfg=cfg,
                    device=device,
                    state_tensor=st,
                    policy_long=policy_long,
                    policy_short=policy_short,
                    epoch=epoch,
                )
                test_env.set_risk_scale(risk_scale)

                ns, r, done, _, info = test_env.step(env_action)
                s = ns
                step_count += 1

            dd_ep = info["drawdown"]
            all_dd.append(dd_ep)
            all_trades.extend(test_env.trades_pnl)
            all_equity.append(test_env.capital - cfg.initial_capital)

    test_profit = float(sum(all_equity))
    test_num_trades = len(all_trades)
    test_winrate = (
        float(np.mean([p > 0 for p in all_trades]))
        if test_num_trades > 0 else 0.0
    )
    test_max_dd = float(max(all_dd) if all_dd else 0.0)

    print(
        f"[{cfg.side.upper()}][TEST] Profit={test_profit:.2f} $, "
        f"trades={test_num_trades}, "
        f"winrate={test_winrate:2.0%}, "
        f"max_dd={test_max_dd:.3f}"
    )

    print(f"[{cfg.side.upper()}] Fin du script.")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    # Agent LONG-only :
    #cfg_long = PPOConfig(side="long", model_prefix="saintv2_loup_long")
    #run_training(cfg_long)

    # Agent SHORT-only :
    cfg_short = PPOConfig(side="short", model_prefix="saintv2_loup_short")
    run_training(cfg_short)

    # Agent CLOSE-only (utilise les best modèles long/short) :
    #cfg_close = PPOConfig(side="close", model_prefix="saintv2_loup_close")
    #run_training(cfg_close)
