import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# CONFIG & CONSTANTES
# ============================================================

# 0:BUY1, 1:SELL1, 2:BUY1.8, 3:SELL1.8, 4:HOLD
N_ACTIONS = 5
MASK_VALUE = -1e4  # EXACTEMENT comme au training

# Stats de normalisation (identiques à l'entraînement)
NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

# Modèles pré-entraînés — best Sortino (comme dans le training)
BEST_MODEL_LONG_PATH = "bestprofit_saintv2_loup_long_wf1_long_wf1.pth"
BEST_MODEL_SHORT_PATH = "bestprofit_saintv2_loup_short_wf1_short_wf1.pth"


@dataclass
class LiveConfig:
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1   # identique au training

    lookback: int = 25

    # nombre de bougies pour recalculer les indicateurs
    n_bars_m1: int = 2000
    n_bars_h1: int = 5000

    # doit matcher le training (cfg.tp_shrink = 0.7)
    tp_shrink: float = 0.7

    # trading (mêmes valeurs que PPOConfig)
    position_size: float = 0.06
    leverage: float = 6.0
    fee_rate: float = 0.0004  # juste informatif ici
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    # fréquence de décision (en secondes)
    poll_interval: int = 2

    # device
    force_cpu: bool = False

    # mode d’agent :
    #   "duel"  : duel long vs short (logique du backtest)
    #   "long"  : uniquement agent LONG (pas de short)
    #   "short" : uniquement agent SHORT (pas de long)
    side: str = "duel"

    # ======= BREAK-EVEN + TRAILING (en ATR) =======
    breakeven_atr_mult: float = 1.0
    trailing_start_atr_mult: float = 1.5
    trailing_dist_atr_mult: float = 1.0


# ============================================================
# INDICATEURS — IDENTIQUES AU TRAINING "LOUP Ω"
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    # ---------- RSI ----------
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = rsi(c, 14)

    # ---------- ATR ----------
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ---------- Vol / range ----------
    df["returns"] = c.pct_change()
    df["vol_20"] = df["returns"].rolling(20).std()
    df["range_norm"] = (h - l) / (c + 1e-8)

    # ==================================================================
    # BONUS SURPRISE : Momentum-Confirmed Entry Filter (M1 & H1)
    # ==================================================================
    df["mom_5"] = df["close"] > df["close"].shift(5)           # True si prix > close d’il y a 5 barres
    df["rsi_ok"] = (df["rsi_14"] > 28) & (df["rsi_14"] < 72)   # zone "saine" du RSI
    # Volatilité relative du jour (rolling 1440 = 24h en M1)
    df["vol_rank"] = df["vol_20"].rolling(1440).rank(pct=True)
    df["high_vol_regime"] = df["vol_rank"] > 0.65              # top 35% de vol

    return df


FEATURE_COLS_M1 = [
    "open", "high", "low", "close",
    "rsi_14",
    "returns", "vol_20", "range_norm",
    "mom_5", "rsi_ok", "high_vol_regime",
]

FEATURE_COLS_H1 = [
    "close_h1",
    "rsi_14_h1",
    "returns_h1", "vol_20_h1", "range_norm_h1",
]

FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1
N_BASE_FEATURES = len(FEATURE_COLS)

# Embedding de position identique à l'env :
#   - position (-1,0,1)
#   - entry_price_scaled
#   - current_price_scaled
#   - last_risk_scale
N_POS_FEATURES = 4
OBS_N_FEATURES = N_BASE_FEATURES + N_POS_FEATURES  # 16 + 4 = 20


# ============================================================
# MODELE SAINTv2 — COPIÉ DU TRAINING
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
    """
    Architecture identique au training :
      - actor: logits (N_ACTIONS)
      - critic: V(s)
    """
    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 80,
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
    Masque d'actions, cohérent avec le training pour side="long"/"short".
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


def build_live_obs(
    df_merged: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    cfg: LiveConfig,
    pos: int,
    entry_price: float,
    last_risk_scale: float,
) -> Optional[np.ndarray]:
    """
    Construit l'observation (lookback, OBS_N_FEATURES) à partir du DF mergé.
    Copie de BTCTradingEnvDiscrete._get_obs :
      - features M1/H1 normalisées
      - + embedding de position répété :
            [pos, entry_scaled, current_scaled, last_risk_scale]
    """
    if len(df_merged) < cfg.lookback + 1:
        return None

    X = df_merged[FEATURE_COLS].values.astype(np.float32)
    X_norm = normalize_features(X, stats)

    base = X_norm[-cfg.lookback:]  # (T, N_BASE_FEATURES)

    price_scale = 100000.0
    current_price = float(df_merged["close"].iloc[-1]) if len(df_merged) > 0 else 0.0

    if pos != 0 and entry_price > 0.0:
        entry_scaled = float(entry_price / price_scale)
    else:
        entry_scaled = 0.0

    current_scaled = float(current_price / price_scale) if current_price > 0.0 else 0.0
    pos_feature = float(pos)
    risk_feature = float(last_risk_scale)

    extra_vec = np.array(
        [pos_feature, entry_scaled, current_scaled, risk_feature],
        dtype=np.float32
    )
    extra_block = np.repeat(extra_vec[None, :], cfg.lookback, axis=0)

    obs = np.concatenate([base, extra_block], axis=-1).astype(np.float32)
    return obs


# ============================================================
# POSITION LIVE (lecture MT5)
# ============================================================

def get_current_position(symbol: str) -> Tuple[int, float]:
    """
    Lis les positions MT5 sur le symbole.
    Retourne :
      pos  : 0 / +1 / -1
      price: prix d'entrée si en position
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return 0, 0.0

    p = positions[0]
    if p.type == mt5.POSITION_TYPE_BUY:
        pos = 1
    elif p.type == mt5.POSITION_TYPE_SELL:
        pos = -1
    else:
        pos = 0

    entry_price = float(p.price_open)
    return pos, entry_price


def compute_entry_atr(df_merged: pd.DataFrame) -> float:
    """
    ATR_14 de la dernière bougie fermée.
    """
    if "atr_14" not in df_merged.columns or len(df_merged) == 0:
        return 0.0
    atr = float(df_merged["atr_14"].iloc[-1])
    return max(atr, 0.0)


def compute_sl_tp(cfg: LiveConfig, entry_price: float, side: int, entry_atr: float):
    """
    Même logique que dans l'env training : ATR SL/TP + tp_shrink.
    """
    fallback = 0.0015 * entry_price
    eff_atr = max(entry_atr, fallback, 1e-8)

    sl_dist = cfg.atr_sl_mult * eff_atr
    tp_dist = cfg.atr_tp_mult * eff_atr * cfg.tp_shrink

    if side == 1:
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    sl = max(sl, 1e-8)
    tp = max(tp, 1e-8)
    return sl, tp


def send_order(cfg: LiveConfig, side: int, risk_scale: float, df_merged_closed: pd.DataFrame):
    """
    Ouvre une position avec volume = position_size * risk_scale
    et SL/TP basés sur ATR (comme dans le backtest).
    side : +1 (BUY) ou -1 (SELL)
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
        "comment": "SAINTv2_Live_duel",
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


def modify_sl_tp(position, new_sl: float | None = None, new_tp: float | None = None):
    """
    Modifie SL/TP en respectant :
      - la distance minimale broker (trade_stops_level)
      - une marge de sécurité supplémentaire si nécessaire.
    """
    symbol = position.symbol
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"[modify_sl_tp] Impossible de récupérer symbol_info pour {symbol}")
        return

    point = info.point

    stops_level_points = getattr(info, "trade_stops_level", 0) or 0
    freeze_level_points = getattr(info, "trade_freeze_level", 0) or 0

    MIN_EXTRA_POINTS = 100
    min_points = max(stops_level_points, MIN_EXTRA_POINTS)
    min_price_dist = min_points * point

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[modify_sl_tp] Pas de tick pour {symbol}")
        return

    bid = tick.bid
    ask = tick.ask

    current_sl = float(position.sl) if position.sl > 0 else 0.0
    current_tp = float(position.tp) if position.tp > 0 else 0.0

    desired_sl = current_sl if new_sl is None else float(new_sl)
    desired_tp = current_tp if new_tp is None else float(new_tp)

    if position.type == mt5.POSITION_TYPE_BUY:
        if desired_sl > 0:
            max_sl_allowed = bid - min_price_dist
            if desired_sl > max_sl_allowed:
                print(
                    f"[WARN] new_sl ({desired_sl:.2f}) trop proche du BID ({bid:.2f}), "
                    f"clamp → {max_sl_allowed:.2f} (min_dist={min_price_dist:.5f})"
                )
                desired_sl = max_sl_allowed

        if desired_sl <= 0 or (current_sl > 0 and desired_sl <= current_sl):
            print(
                f"[INFO] SL BUY non modifié : "
                f"old_sl={current_sl:.2f}, candidate={desired_sl:.2f}"
            )
            desired_sl = current_sl

    elif position.type == mt5.POSITION_TYPE_SELL:
        if desired_sl > 0:
            min_sl_allowed = ask + min_price_dist
            if desired_sl < min_sl_allowed:
                print(
                    f"[WARN] new_sl ({desired_sl:.2f}) trop proche de l'ASK ({ask:.2f}), "
                    f"clamp → {min_sl_allowed:.2f} (min_dist={min_price_dist:.5f})"
                )
                desired_sl = min_sl_allowed

        if desired_sl <= 0 or (current_sl > 0 and desired_sl >= current_sl):
            print(
                f"[INFO] SL SELL non modifié : "
                f"old_sl={current_sl:.2f}, candidate={desired_sl:.2f}"
            )
            desired_sl = current_sl

    if (
        abs(desired_sl - current_sl) < point / 2.0
        and abs(desired_tp - current_tp) < point / 2.0
    ):
        print(
            f"[INFO] SL/TP identiques (sl={current_sl:.2f}, tp={current_tp:.2f}), "
            "aucune modification envoyée."
        )
        return

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": position.ticket,
        "sl": desired_sl,
        "tp": desired_tp,
        "magic": 424242,
        "comment": "SAINTv2_update_sl_tp",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    print(
        f"Envoi TRADE_ACTION_SLTP : "
        f"ticket={position.ticket}, old_sl={current_sl:.2f}, new_sl={desired_sl:.2f}, "
        f"bid={bid:.2f}, ask={ask:.2f}, "
        f"stops_level_pts={stops_level_points}, freeze_level_pts={freeze_level_points}"
    )

    result = mt5.order_send(request)
    if result is None:
        print("[modify_sl_tp] Erreur : result=None")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[modify_sl_tp] Erreur, retcode={result.retcode}")
    else:
        print("[modify_sl_tp] Modification SL/TP OK.")


def update_sl_be_trailing_live(cfg: LiveConfig, df_closed: pd.DataFrame, position):
    """
    Break-even + trailing, basé sur la dernière bougie FERMÉE.
    """
    if len(df_closed) == 0:
        return

    row = df_closed.iloc[-1]
    high_bar = float(row["high"])
    low_bar = float(row["low"])
    close_bar = float(row["close"])

    atr = compute_entry_atr(df_closed)
    if atr <= 0:
        return

    if position.type == mt5.POSITION_TYPE_BUY:
        side = 1
    elif position.type == mt5.POSITION_TYPE_SELL:
        side = -1
    else:
        return

    entry_price = float(position.price_open)
    current_sl = float(position.sl) if position.sl is not None else 0.0

    if side == 1:
        favorable_move = high_bar - entry_price
    else:
        favorable_move = entry_price - low_bar

    if favorable_move <= 0:
        return

    new_sl = None
    reason = ""

    be_trigger = cfg.breakeven_atr_mult * atr

    if favorable_move >= be_trigger:
        if side == 1:
            if current_sl < entry_price or current_sl == 0.0:
                new_sl = entry_price
                reason = "BREAKEVEN_LONG"
        else:
            if current_sl > entry_price or current_sl == 0.0:
                new_sl = entry_price
                reason = "BREAKEVEN_SHORT"

    trail_trigger = cfg.trailing_start_atr_mult * atr
    trail_dist = cfg.trailing_dist_atr_mult * atr

    if favorable_move >= trail_trigger:
        if side == 1:
            candidate_sl = high_bar - trail_dist
            candidate_sl = max(candidate_sl, entry_price)
            if current_sl < candidate_sl:
                new_sl = candidate_sl
                reason = "TRAIL_LONG"
        else:
            candidate_sl = low_bar + trail_dist
            candidate_sl = min(candidate_sl, entry_price)
            if current_sl == 0.0 or current_sl > candidate_sl:
                new_sl = candidate_sl
                reason = "TRAIL_SHORT"

    if new_sl is not None:
        if abs(new_sl - current_sl) > 1e-5:
            print(f"Update SL ({reason}) : old={current_sl:.2f} → new={new_sl:.2f}")
            modify_sl_tp(position, new_sl)


# ============================================================
# BOUCLE LIVE
# ============================================================

def live_loop(cfg: LiveConfig):
    print("Connexion MT5 (live)…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.initialize() en live.")

    device = get_device(cfg)
    stats = load_norm_stats(NORM_STATS_PATH)

    # Agent LONG (best Sortino) — seulement si utilisé
    policy_long = None
    if cfg.side in ("duel", "long"):
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

    # Agent SHORT (best Sortino) — seulement si utilisé
    policy_short = None
    if cfg.side in ("duel", "short"):
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

    print(f"Modèles LONG/SHORT chargés, mode side='{cfg.side}'…")

    last_bar_time = None
    last_risk_scale = 1.0  # embedding de levier identique au training

    try:
        while True:
            try:
                df_merged_full = fetch_ohlc_with_indicators(cfg)
            except Exception as e:
                print(f"[ERREUR MT5] {e} → pause 5s puis retry.")
                time.sleep(5)
                continue

            if len(df_merged_full) < cfg.lookback + 3:
                print("Pas assez de données pour construire l'obs, on attend…")
                time.sleep(cfg.poll_interval)
                continue

            current_last_time = df_merged_full["time"].iloc[-1]
            if last_bar_time is not None and current_last_time == last_bar_time:
                time.sleep(cfg.poll_interval)
                continue

            last_bar_time = current_last_time

            # Dernière bougie FERMÉE
            df_closed = df_merged_full.iloc[:-1].reset_index(drop=True)
            closed_bar_time = df_closed["time"].iloc[-1]

            print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] Nouvelle bougie M1 FERMÉE, time={closed_bar_time}")

            # Position actuelle
            pos, entry_price = get_current_position(cfg.symbol)
            print(f"Position actuelle (net) : {pos}, entry_price={entry_price}")

            if pos == 0:
                last_risk_scale = 1.0

            obs = build_live_obs(df_closed, stats, cfg, pos, entry_price, last_risk_scale)
            if obs is None:
                print("Impossible de construire l'obs (manque de données), on attend…")
                time.sleep(cfg.poll_interval)
                continue

            # ====================================================
            # EN POSITION → BREAK-EVEN + TRAILING STOP
            # ====================================================
            if pos != 0:
                print("Déjà en position → gestion SL par break-even + trailing stop.")

                positions = mt5.positions_get(symbol=cfg.symbol)
                if positions is None or len(positions) == 0:
                    print("Aucune position trouvée alors que pos != 0 (incohérence MT5).")
                    time.sleep(cfg.poll_interval)
                    continue

                position = positions[0]
                print(
                    f"Position MT5 : ticket={position.ticket}, "
                    f"type={'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL'}, "
                    f"volume={position.volume}, sl={position.sl}, tp={position.tp}"
                )

                update_sl_be_trailing_live(cfg, df_closed, position)

                time.sleep(cfg.poll_interval)
                continue

            # ====================================================
            # FLAT → DÉCISION D'ENTRÉE SELON cfg.side (logique backtest Option B)
            # ====================================================
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                if cfg.side == "duel":
                    if policy_long is None or policy_short is None:
                        a = 4
                        print("Policies LONG/SHORT non chargées → HOLD.")
                    else:
                        # ---------- LONG ----------
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        probs_long = torch.softmax(logits_long_m, dim=-1)

                        # ---------- SHORT ----------
                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        probs_short = torch.softmax(logits_short_m, dim=-1)

                        print("Logits LONG  :", logits_long_m.cpu().numpy().round(4))
                        print("Probas LONG  :", probs_long.cpu().numpy().round(4))
                        print("Logits SHORT :", logits_short_m.cpu().numpy().round(4))
                        print("Probas SHORT :", probs_short.cpu().numpy().round(4))

                        # Probabilité max parmi les 4 actions d'ouverture (0..3)
                        max_p_long = probs_long[:4].max().item()
                        max_p_short = probs_short[:4].max().item()

                        # SEUIL DE CONFIANCE MINIMUM : 0.70
                        if max_p_long >= 0.70 and max_p_long > max_p_short:
                            idx_long = int(torch.argmax(probs_long[:4]).item())
                            print(f"[DUEL] Candidat LONG idx={idx_long}, max_p_long={max_p_long:.3f}")
                            if idx_long in (0, 2):
                                a = idx_long
                            else:
                                a = 4
                        elif max_p_short >= 0.70:
                            idx_short = int(torch.argmax(probs_short[:4]).item())
                            print(f"[DUEL] Candidat SHORT idx={idx_short}, max_p_short={max_p_short:.3f}")
                            if idx_short in (1, 3):
                                a = idx_short
                            else:
                                a = 4
                        else:
                            print("[DUEL] Aucun signal avec prob >= 0.70 → HOLD.")
                            a = 4

                elif cfg.side == "long":
                    if policy_long is None:
                        print("policy_long non chargé alors que side='long' → HOLD.")
                        a = 4
                    else:
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        probs_long = torch.softmax(logits_long_m, dim=-1)

                        print("Logits LONG  :", logits_long_m.cpu().numpy().round(4))
                        print("Probas LONG  :", probs_long.cpu().numpy().round(4))

                        p_long, a_long = torch.max(probs_long, dim=-1)
                        a_long = int(a_long.item())
                        p_long = float(p_long.item())

                        print(f"BEST LONG : action={a_long}, prob={p_long:.3f}")

                        # Seuil 0.70 + rejet des HOLD
                        if a_long == 4 or p_long < 0.70:
                            print("[LONG] Signal trop faible ou HOLD → HOLD.")
                            a = 4
                        else:
                            a = a_long

                elif cfg.side == "short":
                    if policy_short is None:
                        print("policy_short non chargé alors que side='short' → HOLD.")
                        a = 4
                    else:
                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        probs_short = torch.softmax(logits_short_m, dim=-1)

                        print("Logits SHORT :", logits_short_m.cpu().numpy().round(4))
                        print("Probas SHORT :", probs_short.cpu().numpy().round(4))

                        p_short, a_short = torch.max(probs_short, dim=-1)
                        a_short = int(a_short.item())
                        p_short = float(p_short.item())

                        print(f"BEST SHORT : action={a_short}, prob={p_short:.3f}")

                        # Seuil 0.70 + rejet des HOLD
                        if a_short == 4 or p_short < 0.70:
                            print("[SHORT] Signal trop faible ou HOLD → HOLD.")
                            a = 4
                        else:
                            a = a_short
                else:
                    print(f"cfg.side invalide : {cfg.side}, on HOLD.")
                    a = 4

            print(f"Action finale (0:BUY1,1:SELL1,2:BUY1.8,3:SELL1.8,4:HOLD) : {a}")

            # Mapping vers env_action + risk_scale — cohérent avec le training (branche flat)
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

            # Sécurité : on vérifie qu'on est toujours FLAT juste avant d'envoyer l'ordre
            pos_check, _ = get_current_position(cfg.symbol)
            if pos_check != 0:
                print("Position détectée juste avant l'envoi de l'ordre → annulation de l'ouverture.")
                last_risk_scale = 1.0
                time.sleep(cfg.poll_interval)
                continue

            # Exécution
            if env_action == 0:
                send_order(cfg, side=1, risk_scale=risk_scale, df_merged_closed=df_closed)
                last_risk_scale = risk_scale
            elif env_action == 1:
                send_order(cfg, side=-1, risk_scale=risk_scale, df_merged_closed=df_closed)
                last_risk_scale = risk_scale
            else:
                print("HOLD (flat) → aucune ouverture.")
                last_risk_scale = 1.0

            time.sleep(cfg.poll_interval)

    finally:
        mt5.shutdown()
        print("MT5 shutdown, fin du live agent.")


if __name__ == "__main__":
    # side="duel"  → LONG vs SHORT
    # side="long"  → uniquement LONG
    # side="short" → uniquement SHORT
    cfg = LiveConfig(side="duel")
    live_loop(cfg)
