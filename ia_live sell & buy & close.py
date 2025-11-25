import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import math
from dataclasses import dataclass
from typing import Optional, List, Dict

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from datetime import datetime

# ============================================================
# CONFIG & CONSTANTES
# ============================================================

N_ACTIONS = 5
MASK_VALUE = -1e4  # doit être cohérent avec la valeur utilisée au training
NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

# Deux modèles séparés pour l'ouverture
BEST_MODEL_LONG_PATH = "best_saintv2_loup_long_long.pth"
BEST_MODEL_SHORT_PATH = "best_saintv2_loup_short_short.pth"

# Modèle spécialisé dans la clôture anticipée
BEST_MODEL_CLOSE_PATH = "best_saintv2_loup_close_close.pth"


@dataclass
class LiveConfig:
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_M5

    lookback: int = 25

    # nombre de bougies à récupérer pour recalculer les indicateurs
    n_bars_m1: int = 2000
    n_bars_h1: int = 2000
    tp_shrink: float = 0.95
    # trading
    position_size: float = 0.06
    leverage: float = 6.0
    fee_rate: float = 0.0004  # juste info/log, pas utilisé pour MT5
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    # fréquence de décision (en secondes)
    poll_interval: int = 60  # toutes les 60s ~ M1

    # device
    force_cpu: bool = False


# ============================================================
# INDICATEURS (mêmes que dans le training)
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

    # Volatility regime (z-score vol réalisée long terme)
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
OBS_N_FEATURES = N_BASE_FEATURES


# ============================================================
# MODELE SAINTv2 (identique au training)
# ============================================================

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
    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 80,
        num_blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
        ff_mult: int = 2,
        max_len: int = 512,
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
        assert x.dim() == 3
        B, T, F = x.shape

        tok = self.input_proj(x.unsqueeze(-1)) * self.scale  # (B,T,F,D)

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


# ============================================================
# UTILS LIVE
# ============================================================

def get_device(cfg: LiveConfig):
    if cfg.force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


def build_mask_from_pos_scalar(pos: int, device, side: str) -> torch.Tensor:
    """
    Masque d'actions pour les agents LONG/SHORT :
      - Si en position : uniquement HOLD (4).
      - Si flat :
          side == "long"  -> actions 0 (BUY1), 2 (BUY1.8), 4 (HOLD)
          side == "short" -> actions 1 (SELL1), 3 (SELL1.8), 4 (HOLD)
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)

    if pos != 0:
        mask[4] = True
        return mask

    if side == "long":
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


def build_close_mask(pos: int, device) -> torch.Tensor:
    """
    Masque pour l'agent CLOSE :
      - Si flat : HOLD uniquement (4)
      - Si en position : CLOSE (0) ou HOLD (4)
    On n'utilise que 0 = CLOSE, 4 = HOLD pour cet agent.
    """
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
    if pos == 0:
        mask[4] = True
    else:
        mask[0] = True
        mask[4] = True
    return mask


def load_norm_stats(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats de normalisation introuvables : {path}")
    data = np.load(path)
    return {"mean": data["mean"], "std": data["std"]}


def normalize_features(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean, std = stats["mean"], stats["std"]
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


# ============================================================
# DATA LIVE : M1 + H1 => MERGE + FEATURES
# ============================================================

def fetch_ohlc_with_indicators(cfg: LiveConfig) -> pd.DataFrame:
    """
    Récupère les données M1 & H1 depuis MT5,
    calcule les mêmes indicateurs que dans le training,
    merge_asof(M1,H1) et renvoie le DataFrame final.
    """
    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe, 0, cfg.n_bars_m1
    )
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.htf_timeframe, 0, cfg.n_bars_h1
    )

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1 (live).")

    df_m1 = pd.DataFrame(rates_m1)
    df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
    df_m1.set_index("time", inplace=True)
    df_m1 = df_m1[["open", "high", "low", "close", "tick_volume"]]
    df_m1 = add_indicators(df_m1)

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
    return merged


def build_live_obs(df_merged: pd.DataFrame, stats: Dict[str, np.ndarray], cfg: LiveConfig) -> Optional[np.ndarray]:
    """
    Construit l'observation (lookback, OBS_N_FEATURES) à partir du DF mergé.
    On suppose que df_merged ne contient que des bougies FERMÉES ici.
    """
    if len(df_merged) < cfg.lookback + 1:
        return None

    X = df_merged[FEATURE_COLS].values.astype(np.float32)
    X_norm = normalize_features(X, stats)

    obs = X_norm[-cfg.lookback:]  # (lookback, N_BASE_FEATURES)
    return obs


# ============================================================
# POSITION LIVE (lecture MT5)
# ============================================================

def get_current_position(symbol: str) -> int:
    """
    Lis les positions MT5 sur le symbole.
    Retourne :
      0  : flat
      +1 : long
      -1 : short
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return 0

    net_volume = 0.0
    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            net_volume += p.volume
        elif p.type == mt5.POSITION_TYPE_SELL:
            net_volume -= p.volume

    if net_volume > 1e-8:
        return 1
    elif net_volume < -1e-8:
        return -1
    else:
        return 0


def compute_entry_atr(df_merged: pd.DataFrame) -> float:
    """
    ATR_14 de la dernière bougie fermée.
    """
    if "atr_14" not in df_merged.columns:
        return 0.0
    atr = float(df_merged["atr_14"].iloc[-1])  # ici df_merged ne contient que des fermées
    return max(atr, 0.0)


def compute_sl_tp(cfg: LiveConfig, entry_price: float, side: int, entry_atr: float):
    fallback = 0.0015 * entry_price
    eff_atr = max(entry_atr, fallback, 1e-8)

    sl_dist = cfg.atr_sl_mult * eff_atr
    tp_dist = cfg.atr_tp_mult * eff_atr * cfg.tp_shrink

    if side == 1:  # long
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:          # short
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    sl = max(sl, 1e-8)
    tp = max(tp, 1e-8)
    return sl, tp


def send_order(cfg: LiveConfig, side: int, risk_scale: float, df_merged_closed: pd.DataFrame):
    """
    Ouvre une position avec volume = position_size * risk_scale
    et SL/TP basés sur ATR (comme dans l'env).
    side : +1 (BUY) ou -1 (SELL)
    df_merged_closed : DF ne contenant que des bougies fermées (pour ATR)
    """
    symbol = cfg.symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Erreur : pas de tick MT5 pour", symbol)
        return

    if side == 1:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    volume = cfg.position_size * (risk_scale if risk_scale > 0 else 1.0)

    entry_atr = compute_entry_atr(df_merged_closed)
    sl, tp = compute_sl_tp(cfg, price, side, entry_atr)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 50,
        "magic": 424242,
        "comment": "SAINTv2_Live_dual_agent",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(request)
    if result is None:
        print("Erreur order_send : None")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order_send échoué, retcode={result.retcode}")
    else:
        print(f"Order exécuté : side={side}, volume={volume}, prix={price}, SL={sl}, TP={tp}")


def close_position_market(cfg: LiveConfig):
    """
    Clôture anticipée manuelle au marché (utilisée par l'agent CLOSE).
    On envoie un ordre inverse sur la position existante.
    """
    positions = mt5.positions_get(symbol=cfg.symbol)
    if positions is None or len(positions) == 0:
        print("Aucune position à fermer.")
        return

    p = positions[0]
    symbol = cfg.symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Erreur : pas de tick MT5 pour", symbol)
        return

    if p.type == mt5.POSITION_TYPE_BUY:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": p.volume,
        "type": order_type,
        "position": p.ticket,
        "price": price,
        "deviation": 50,
        "magic": 424242,
        "comment": "SAINTv2_Live_close_agent",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(request)
    if result is None:
        print("Erreur CLOSE : None")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Erreur fermeture anticipée, retcode={result.retcode}")
    else:
        print(f"Clôture manuelle exécutée au prix {price}")


# ============================================================
# LOGIQUE D'ACTION AVEC 3 AGENTS
# ============================================================

def choose_action(policy: SAINTPolicySingleHead, obs: np.ndarray, pos: int, side: str, device):
    """
    Renvoie (action 0..4, probs numpy(5,)) pour un agent donné (long ou short),
    avec masque adapté au side.
    """
    with torch.no_grad():
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,F)
        logits, _ = policy(s)
        logits = logits[0]
        mask = build_mask_from_pos_scalar(pos, device, side)
        logits_masked = logits.masked_fill(~mask, MASK_VALUE)
        probs = torch.softmax(logits_masked, dim=-1)  # (5,)
        a = int(torch.argmax(probs).item())
    return a, probs.cpu().numpy()


def choose_close_action(policy_close: SAINTPolicySingleHead, obs: np.ndarray, pos: int, device):
    """
    Agent CLOSE :
      - si flat pos == 0  -> peut seulement HOLD (4)
      - si en position    -> CLOSE (0) ou HOLD (4)
    """
    with torch.no_grad():
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,F)
        logits, _ = policy_close(s)
        logits = logits[0]
        mask = build_close_mask(pos, device)
        logits_masked = logits.masked_fill(~mask, MASK_VALUE)
        probs = torch.softmax(logits_masked, dim=-1)  # (5,)
        a = int(torch.argmax(probs).item())
    return a, probs.cpu().numpy()


def live_loop(cfg: LiveConfig):
    # --- init MT5 ---
    print("Connexion MT5 (live)…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.initialize() en live.")

    # --- device & modèles ---
    device = get_device(cfg)
    stats = load_norm_stats(NORM_STATS_PATH)

    # Agent LONG
    policy_long = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=80,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    if not os.path.exists(BEST_MODEL_LONG_PATH):
        raise FileNotFoundError(f"Modèle LONG introuvable : {BEST_MODEL_LONG_PATH}")
    policy_long.load_state_dict(torch.load(BEST_MODEL_LONG_PATH, map_location=device))
    policy_long.eval()

    # Agent SHORT
    policy_short = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=80,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    if not os.path.exists(BEST_MODEL_SHORT_PATH):
        raise FileNotFoundError(f"Modèle SHORT introuvable : {BEST_MODEL_SHORT_PATH}")
    policy_short.load_state_dict(torch.load(BEST_MODEL_SHORT_PATH, map_location=device))
    policy_short.eval()

    # Agent CLOSE
    policy_close = SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=80,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS
    ).to(device)

    if not os.path.exists(BEST_MODEL_CLOSE_PATH):
        raise FileNotFoundError(f"Modèle CLOSE introuvable : {BEST_MODEL_CLOSE_PATH}")
    policy_close.load_state_dict(torch.load(BEST_MODEL_CLOSE_PATH, map_location=device))
    policy_close.eval()

    print("Modèles LONG, SHORT & CLOSE chargés, début de la boucle live…")

    last_bar_time = None

    try:
        while True:
            try:
                # 1) Récupérer les données M1/H1 + indicateurs
                df_merged_full = fetch_ohlc_with_indicators(cfg)
            except Exception as e:
                print(f"[ERREUR MT5] {e} → pause 10s puis retry.")
                time.sleep(10)
                continue

            if len(df_merged_full) < cfg.lookback + 3:
                print("Pas assez de données pour construire l'obs, on attend…")
                time.sleep(cfg.poll_interval)
                continue

            # On utilise la DERNIÈRE LIGNE comme "bar open/encours"
            current_last_time = df_merged_full["time"].iloc[-1]

            if last_bar_time is not None and current_last_time == last_bar_time:
                # Pas de nouvelle bougie M1 dans MT5
                print("Pas de nouvelle bougie M1 → j'attends…")
                time.sleep(5)
                continue

            # Nouvelle bougie (ou reconnection) détectée
            last_bar_time = current_last_time

            # On ne travaille que sur les bougies FERMÉES :
            # on retire la dernière ligne (bougie en cours)
            df_closed = df_merged_full.iloc[:-1].reset_index(drop=True)
            closed_bar_time = df_closed["time"].iloc[-1]

            print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] Nouvelle bougie M1 FERMÉE, time={closed_bar_time}")

            # 2) Obs sur bougies fermées uniquement
            obs = build_live_obs(df_closed, stats, cfg)
            if obs is None:
                print("Impossible de construire l'obs (manque de données), on attend…")
                time.sleep(cfg.poll_interval)
                continue

            # 3) Position actuelle
            pos = get_current_position(cfg.symbol)
            print(f"Position actuelle (net) : {pos}")

            # ====================================================
            # SI DÉJÀ EN POSITION → l’agent CLOSE décide
            # ====================================================
            if pos != 0:
                print("Déjà en position → interrogation de l’agent CLOSE…")

                a_close, probs_close = choose_close_action(policy_close, obs, pos, device)
                print("Probas agent CLOSE (index 0=CLOSE, 4=HOLD) :", probs_close.round(4))
                print("Action CLOSE-agent (0=CLOSE,4=HOLD) :", a_close)

                if a_close == 0:
                    print("🔥 L’agent CLOSE demande une CLÔTURE ANTICIPÉE !")
                    close_position_market(cfg)
                else:
                    print("HOLD → on laisse SL/TP broker gérer la sortie.")

                time.sleep(cfg.poll_interval)
                continue

            # ====================================================
            # SINON : FLAT → LONG & SHORT décident d’ouvrir
            # ====================================================

            # 4) Forward des deux agents (LONG et SHORT)
            a_long, probs_long = choose_action(policy_long, obs, pos=0, side="long", device=device)
            a_short, probs_short = choose_action(policy_short, obs, pos=0, side="short", device=device)

            print("Probas LONG (0:BUY1, 2:BUY1.8, 4:HOLD)  :", probs_long.round(4))
            print("Probas SHORT(1:SELL1,3:SELL1.8,4:HOLD) :", probs_short.round(4))
            print(f"Action LONG argmax : {a_long}, Action SHORT argmax : {a_short}")

            # 5) Calcul des scores d'ouverture
            #    score = max(prob(open)) - prob(HOLD)
            prob_hold_long = probs_long[4]
            prob_open_long_max = max(probs_long[0], probs_long[2])
            score_long = prob_open_long_max - prob_hold_long

            prob_hold_short = probs_short[4]
            prob_open_short_max = max(probs_short[1], probs_short[3])
            score_short = prob_open_short_max - prob_hold_short

            print(f"Score ouverture LONG  = {score_long:.4f}")
            print(f"Score ouverture SHORT = {score_short:.4f}")

            # 6) Décision : aucun agent ne veut assez ouvrir
            if score_long <= 0 and score_short <= 0:
                print("Aucun agent n'a une probabilité d'ouverture > HOLD → HOLD global.")
                time.sleep(cfg.poll_interval)
                continue

            # 7) Choix de l'agent gagnant
            if score_long >= score_short:
                chosen_side = "long"
                chosen_probs = probs_long
                print("Agent choisi : LONG")
            else:
                chosen_side = "short"
                chosen_probs = probs_short
                print("Agent choisi : SHORT")

            # 8) Déterminer l'action concrète pour l'agent choisi
            if chosen_side == "long":
                # On regarde les actions ouvertes long : 0 (BUY1), 2 (BUY1.8)
                if chosen_probs[0] >= chosen_probs[2]:
                    a = 0
                else:
                    a = 2
            else:
                # SHORT : 1 (SELL1), 3 (SELL1.8)
                if chosen_probs[1] >= chosen_probs[3]:
                    a = 1
                else:
                    a = 3

            print(f"Action finale (0:BUY1,1:SELL1,2:BUY1.8,3:SELL1.8) : {a}")

            # 9) Mapping vers env_action + risk_scale
            # pos == 0 garanti ici
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

            print(f"Env_action (0=BUY,1=SELL,2=HOLD) : {env_action}, risk_scale={risk_scale}")

            # 10) Exécution
            if env_action == 0:
                send_order(cfg, side=1, risk_scale=risk_scale, df_merged_closed=df_closed)
            elif env_action == 1:
                send_order(cfg, side=-1, risk_scale=risk_scale, df_merged_closed=df_closed)
            else:
                print("HOLD (flat) → aucune ouverture.")

            time.sleep(cfg.poll_interval)

        # fin while True

    finally:
        mt5.shutdown()
        print("MT5 shutdown, fin du live agent.")


if __name__ == "__main__":
    cfg = LiveConfig()
    live_loop(cfg)
