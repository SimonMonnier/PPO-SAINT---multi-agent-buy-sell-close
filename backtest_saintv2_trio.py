import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# CONSTANTES / CONFIG
# ============================================================

# 0:BUY1, 1:SELL1, 2:BUY1.8, 3:SELL1.8, 4:HOLD
N_ACTIONS = 5
MASK_VALUE = -1e4  # même valeur que le training / live

# Stats de normalisation (doit correspondre à ton training)
NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

# Modèles pré-entraînés (best PROFIT ici) — mêmes noms que ton training
BEST_MODEL_LONG_PATH = "bestprofit_saintv2_loup_long_wf1_long_wf1.pth"
BEST_MODEL_SHORT_PATH = "bestprofit_saintv2_loup_short_wf1_short_wf1.pth"


@dataclass
class LiveConfig:
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1  # identique au training / live

    lookback: int = 25

    # nombre de bougies à charger pour le backtest
    n_bars_m1: int = 200_000
    n_bars_h1: int = 50_000

    # même params que l'env / live
    tp_shrink: float = 0.7

    # trading
    initial_capital: float = 1000.0
    position_size: float = 0.06
    leverage: float = 6.0
    fee_rate: float = 0.0004
    atr_sl_mult: float = 1.2
    atr_tp_mult: float = 2.4

    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    # ======= BREAK-EVEN + TRAILING (en ATR) - comme en live =======
    breakeven_atr_mult: float = 1.0       # mouvement favorable pour BE
    trailing_start_atr_mult: float = 1.5  # mouvement favorable pour activer le trailing
    trailing_dist_atr_mult: float = 1.0   # distance du trailing en ATR

    # Backtest : marge mini en points si broker renvoie 0
    backtest_min_extra_points: int = 100

    # ======= Seuil de confiance minimal pour ouvrir un trade =======
    min_confidence: float = 0.70

    # device
    force_cpu: bool = False

    # mode d’agent :
    #   "duel"  : long vs short
    #   "long"  : long only
    #   "short" : short only
    side: str = "duel"

    # fréquence d'affichage de progression (en nombre de bougies M1)
    progress_interval_bars: int = 1440  # ~ 1 jour


@dataclass
class BTState:
    capital: float
    equity: float
    position: int = 0          # 0, +1, -1
    volume: float = 0.0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    entry_index: int = -1
    last_risk_scale: float = 1.0
    trades_pnl: List[float] = field(default_factory=list)
    max_equity: float = 0.0


# ============================================================
# INDICATEURS — IDENTIQUES TRAINING / LIVE ("Loup Ω")
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

# Embedding de position identique à l'env / live :
#   - position (-1,0,1)
#   - entry_price_scaled
#   - current_price_scaled
#   - last_risk_scale
N_POS_FEATURES = 4
OBS_N_FEATURES = N_BASE_FEATURES + N_POS_FEATURES  # 16 + 4 = 20


# ============================================================
# MODELE SAINTv2 — IDENTIQUE TRAINING / LIVE
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


# ============================================================
# UTILS NORMA / DEVICE
# ============================================================

def get_device(cfg: LiveConfig):
    if cfg.force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


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
# DATA M1 + H1 POUR BACKTEST (MT5)
# ============================================================

def fetch_ohlc_with_indicators(cfg: LiveConfig) -> pd.DataFrame:
    rates_m1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.timeframe, 0, cfg.n_bars_m1
    )
    rates_h1 = mt5.copy_rates_from_pos(
        cfg.symbol, cfg.htf_timeframe, 0, cfg.n_bars_h1
    )

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1.")

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
# LOGIQUE SL/TP + BREAK-EVEN / TRAILING + ACTION MASK
# ============================================================

def compute_entry_atr(df_closed: pd.DataFrame) -> float:
    if "atr_14" not in df_closed.columns or len(df_closed) == 0:
        return 0.0
    atr = float(df_closed["atr_14"].iloc[-1])
    return max(atr, 0.0)


def compute_sl_tp(cfg: LiveConfig, entry_price: float, side: int, entry_atr: float):
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


def update_sl_be_trailing_backtest(
    cfg: LiveConfig,
    df_closed: pd.DataFrame,
    state: BTState,
    min_price_dist: float
):
    """
    Version backtest de update_sl_be_trailing_live :
      - break-even + trailing
      - clamp du SL pour respecter une distance mini vis-à-vis du prix courant
        (approximation du trade_stops_level broker).
    """
    if state.position == 0:
        return
    if len(df_closed) == 0:
        return

    row = df_closed.iloc[-1]
    high_bar = float(row["high"])
    low_bar = float(row["low"])
    close_bar = float(row["close"])

    atr = compute_entry_atr(df_closed)
    if atr <= 0:
        return

    side = state.position  # +1 LONG, -1 SHORT
    entry_price = float(state.entry_price)
    current_sl = float(state.sl) if state.sl is not None else 0.0

    if side == 1:
        favorable_move = high_bar - entry_price
    else:
        favorable_move = entry_price - low_bar

    if favorable_move <= 0:
        return

    candidate_sl = None
    reason = ""

    # ----------------- BREAK-EVEN -----------------
    be_trigger = cfg.breakeven_atr_mult * atr

    if favorable_move >= be_trigger:
        be_sl = entry_price
        if side == 1:
            if current_sl < be_sl or current_sl == 0.0:
                candidate_sl = be_sl
                reason = "BREAKEVEN_LONG"
        else:
            if current_sl > be_sl or current_sl == 0.0:
                candidate_sl = be_sl
                reason = "BREAKEVEN_SHORT"

    # ----------------- TRAILING STOP -----------------
    trail_trigger = cfg.trailing_start_atr_mult * atr
    trail_dist = cfg.trailing_dist_atr_mult * atr

    if favorable_move >= trail_trigger:
        if side == 1:
            trail_sl = high_bar - trail_dist
            trail_sl = max(trail_sl, entry_price)
            if candidate_sl is None or trail_sl > candidate_sl:
                candidate_sl = trail_sl
                reason = "TRAIL_LONG"
        else:
            trail_sl = low_bar + trail_dist
            trail_sl = min(trail_sl, entry_price)
            if candidate_sl is None or trail_sl < candidate_sl:
                candidate_sl = trail_sl
                reason = "TRAIL_SHORT"

    if candidate_sl is None:
        return

    # Simu contrainte broker : distance mini
    if side == 1:
        max_sl_allowed = close_bar - min_price_dist
        if candidate_sl > max_sl_allowed:
            print(
                f"[BT WARN] SL candidate ({candidate_sl:.2f}) trop proche du prix "
                f"({close_bar:.2f}) pour LONG, clamp → {max_sl_allowed:.2f}"
            )
            candidate_sl = max_sl_allowed

        if current_sl > 0 and candidate_sl <= current_sl + 1e-8:
            print(
                f"[BT INFO] SL LONG non amélioré : old={current_sl:.2f}, "
                f"candidate={candidate_sl:.2f}"
            )
            return

    else:
        min_sl_allowed = close_bar + min_price_dist
        if candidate_sl < min_sl_allowed:
            print(
                f"[BT WARN] SL candidate ({candidate_sl:.2f}) trop proche du prix "
                f"({close_bar:.2f}) pour SHORT, clamp → {min_sl_allowed:.2f}"
            )
            candidate_sl = min_sl_allowed

        if current_sl > 0 and candidate_sl >= current_sl - 1e-8:
            print(
                f"[BT INFO] SL SHORT non amélioré : old={current_sl:.2f}, "
                f"candidate={candidate_sl:.2f}"
            )
            return

    print(f"Update SL ({reason}) : old={current_sl:.2f} → new={candidate_sl:.2f}")
    state.sl = candidate_sl


def build_mask_from_pos_scalar(pos: int, device, side: str) -> torch.Tensor:
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
    else:  # "duel"
        mask[:] = True

    return mask


# ============================================================
# BACKTEST PRINCIPAL
# ============================================================

def run_backtest(cfg: LiveConfig):
    print("Connexion MT5 pour téléchargement des données…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.initialize() pour le backtest.")

    info = mt5.symbol_info(cfg.symbol)
    if info is None:
        print(f"[WARN] symbol_info({cfg.symbol}) introuvable, on utilise des valeurs par défaut.")
        point = 0.01
        broker_stops_points = 0
    else:
        point = info.point
        broker_stops_points = getattr(info, "trade_stops_level", 0) or 0

    min_points = max(broker_stops_points, cfg.backtest_min_extra_points)
    min_price_dist = min_points * point

    print(
        f"[BACKTEST] point={point:.8f}, "
        f"trade_stops_level={broker_stops_points} pts, "
        f"min_extra={cfg.backtest_min_extra_points} pts, "
        f"min_price_dist={min_price_dist:.5f}"
    )

    try:
        df = fetch_ohlc_with_indicators(cfg)
    finally:
        mt5.shutdown()

    print(f"Données M1+H1 chargées : {len(df)} bougies M1 fusionnées.")

    if len(df) < cfg.lookback + 10:
        raise RuntimeError("Pas assez de données pour lancer le backtest.")

    device = get_device(cfg)
    stats = load_norm_stats(NORM_STATS_PATH)

    # Chargement des modèles LONG / SHORT
    policy_long = None
    policy_short = None

    if cfg.side in ("duel", "long"):
        if not os.path.exists(BEST_MODEL_LONG_PATH):
            raise FileNotFoundError(f"Modèle LONG introuvable : {BEST_MODEL_LONG_PATH}")
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
        policy_long.load_state_dict(torch.load(BEST_MODEL_LONG_PATH, map_location=device))
        policy_long.eval()

    if cfg.side in ("duel", "short"):
        if not os.path.exists(BEST_MODEL_SHORT_PATH):
            raise FileNotFoundError(f"Modèle SHORT introuvable : {BEST_MODEL_SHORT_PATH}")
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
        policy_short.load_state_dict(torch.load(BEST_MODEL_SHORT_PATH, map_location=device))
        policy_short.eval()

    print(f"Modèles LONG / SHORT chargés. Mode side='{cfg.side}'.")

    # État backtest
    state = BTState(
        capital=cfg.initial_capital,
        equity=cfg.initial_capital,
        max_equity=cfg.initial_capital
    )
    max_dd = 0.0

    n = len(df)
    start_index = cfg.lookback

    for i in range(start_index, n - 1):
        row = df.iloc[i]
        time_i = row["time"]
        time_str = time_i.strftime("%Y-%m-%d %H:%M:%S") if isinstance(time_i, pd.Timestamp) else str(time_i)

        closed_this_bar = False

        # 0) BREAK-EVEN + TRAILING (sur la barre i-1)
        if state.position != 0 and i > state.entry_index:
            df_prev_closed = df.iloc[:i].reset_index(drop=True)
            update_sl_be_trailing_backtest(cfg, df_prev_closed, state, min_price_dist)

        # 1) SL / TP sur la barre i
        if state.position != 0 and i > state.entry_index:
            high_bar = float(row["high"])
            low_bar = float(row["low"])

            exit_price = None
            exit_reason = None

            if state.position == 1:
                if low_bar <= state.sl:
                    exit_price = state.sl
                    exit_reason = "SL"
                elif high_bar >= state.tp:
                    exit_price = state.tp
                    exit_reason = "TP"
            elif state.position == -1:
                if high_bar >= state.sl:
                    exit_price = state.sl
                    exit_reason = "SL"
                elif low_bar <= state.tp:
                    exit_price = state.tp
                    exit_reason = "TP"

            if exit_price is not None:
                pnl = (
                    state.position *
                    (exit_price - state.entry_price) *
                    state.volume *
                    cfg.leverage
                )
                fee = cfg.fee_rate * exit_price * state.volume
                realized = pnl - fee

                state.capital += realized
                state.trades_pnl.append(realized)
                closed_this_bar = True

                side_txt = "LONG" if state.position == 1 else "SHORT"
                print(
                    f"[{time_str}] FERMETURE {side_txt} par {exit_reason} @ {exit_price:.2f} | "
                    f"PnL={realized:.2f} | Capital={state.capital:.2f}"
                )

                state.position = 0
                state.volume = 0.0
                state.entry_price = 0.0
                state.sl = 0.0
                state.tp = 0.0
                state.entry_index = -1
                state.last_risk_scale = 1.0
                state.max_equity = state.capital

        # 2) Equity & drawdown
        if state.position != 0:
            latent = (
                state.position *
                (float(row["close"]) - state.entry_price) *
                state.volume *
                cfg.leverage
            )
        else:
            latent = 0.0

        state.equity = state.capital + latent
        state.max_equity = max(state.max_equity, state.equity)
        dd = 0.0
        if state.max_equity > 0:
            dd = (state.max_equity - state.equity) / state.max_equity
        max_dd = max(max_dd, dd)

        # 3) Obs sur df[0..i]
        df_closed_for_obs = df.iloc[: i + 1].reset_index(drop=True)
        obs = build_live_obs(
            df_closed_for_obs, stats, cfg,
            pos=state.position,
            entry_price=state.entry_price,
            last_risk_scale=state.last_risk_scale
        )
        if obs is None:
            continue

        # 4) Décision d'ENTRÉE si FLAT – même logique que le live (duel + seuil min_confidence)
        if state.position == 0 and not closed_this_bar:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                thr = cfg.min_confidence

                if cfg.side == "duel":
                    if policy_long is None or policy_short is None:
                        a = 4
                    else:
                        # ----- LONG -----
                        logits_long, _ = policy_long(s)      # (1,5)
                        logits_long = logits_long[0]         # (5,)
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        prob_long_open = torch.softmax(logits_long_m, dim=-1)  # (5,)
                        max_p_long = prob_long_open[:4].max().item()

                        # ----- SHORT -----
                        logits_short, _ = policy_short(s)    # (1,5)
                        logits_short = logits_short[0]       # (5,)
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        prob_short_open = torch.softmax(logits_short_m, dim=-1)  # (5,)
                        max_p_short = prob_short_open[:4].max().item()

                        if max_p_long >= thr and max_p_long > max_p_short:
                            idx_long = int(torch.argmax(prob_long_open[:4]).item())
                            if idx_long in (0, 2):
                                a = idx_long
                            else:
                                a = 4
                        elif max_p_short >= thr:
                            idx_short = int(torch.argmax(prob_short_open[:4]).item())
                            if idx_short in (1, 3):
                                a = idx_short
                            else:
                                a = 4
                        else:
                            a = 4  # HOLD si aucun signal fort

                elif cfg.side == "long":
                    if policy_long is None:
                        a = 4
                    else:
                        logits_long, _ = policy_long(s)
                        logits_long = logits_long[0]
                        mask_long = build_mask_from_pos_scalar(0, device, "long")
                        logits_long_m = logits_long.masked_fill(~mask_long, MASK_VALUE)
                        probs_long = torch.softmax(logits_long_m, dim=-1)

                        p_long, a_long = torch.max(probs_long, dim=-1)
                        a_long = int(a_long.item())
                        p_long = float(p_long.item())

                        if a_long == 4 or p_long < thr:
                            a = 4
                        else:
                            a = a_long

                elif cfg.side == "short":
                    if policy_short is None:
                        a = 4
                    else:
                        logits_short, _ = policy_short(s)
                        logits_short = logits_short[0]
                        mask_short = build_mask_from_pos_scalar(0, device, "short")
                        logits_short_m = logits_short.masked_fill(~mask_short, MASK_VALUE)
                        probs_short = torch.softmax(logits_short_m, dim=-1)

                        p_short, a_short = torch.max(probs_short, dim=-1)
                        a_short = int(a_short.item())
                        p_short = float(p_short.item())

                        if a_short == 4 or p_short < thr:
                            a = 4
                        else:
                            a = a_short
                else:
                    a = 4

            # mapping vers env_action + risk_scale (comme env / live)
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

            # Sécurité explicite : on refuse toute ouverture si on n'est pas flat
            if state.position != 0:
                continue

            # ouverture de position si BUY/SELL
            if env_action in (0, 1):
                side = 1 if env_action == 0 else -1
                if env_action == 0:  # BUY
                    entry_price = float(row["close"] * (1 + 0.00006))  # +6 bps spread/slippage
                else:  # SELL
                    entry_price = float(row["close"] * (1 - 0.00006))
                volume = cfg.position_size * (risk_scale if risk_scale > 0 else 1.0)
                entry_atr = compute_entry_atr(df_closed_for_obs)
                sl, tp = compute_sl_tp(cfg, entry_price, side, entry_atr)

                state.position = side
                state.volume = volume
                state.entry_price = entry_price
                state.sl = sl
                state.tp = tp
                state.entry_index = i
                state.last_risk_scale = risk_scale

                side_txt = "LONG" if side == 1 else "SHORT"
                print(
                    f"[{time_str}] OUVERTURE {side_txt} @ {entry_price:.2f} | vol={volume:.4f} | "
                    f"SL={sl:.2f} | TP={tp:.2f} | risk_scale={risk_scale:.2f}"
                )

        # 5) Log de progression périodique
        if ((i - start_index) % cfg.progress_interval_bars == 0) or (i == n - 2):
            pnl_total = state.equity - cfg.initial_capital
            print(
                f"[{time_str}] PROGRESSION backtest | Equity={state.equity:.2f} | "
                f"PnL={pnl_total:.2f} | DDmax={max_dd*100:.1f}% | NbTrades={len(state.trades_pnl)}"
            )

    # ========================================================
    # RÉSUMÉ FINAL
    # ========================================================
    pnl_total = state.equity - cfg.initial_capital
    nb_trades = len(state.trades_pnl)
    winrate = (
        float(np.mean([p > 0 for p in state.trades_pnl])) if nb_trades > 0 else 0.0
    )
    avg_pnl = float(np.mean(state.trades_pnl)) if nb_trades > 0 else 0.0
    max_profit = max(state.trades_pnl) if nb_trades > 0 else 0.0
    max_loss = min(state.trades_pnl) if nb_trades > 0 else 0.0

    print("\n===================== RÉSULTATS BACKTEST =====================")
    print(f"Mode side           : {cfg.side}")
    print(f"Capital initial     : {cfg.initial_capital:.2f}")
    print(f"Capital final       : {state.equity:.2f}")
    print(f"PnL total           : {pnl_total:.2f}")
    print(f"Nb trades           : {nb_trades}")
    print(f"Winrate             : {winrate*100:.1f}%")
    print(f"PnL moyen / trade   : {avg_pnl:.2f}")
    print(f"Meilleur trade      : {max_profit:.2f}")
    print(f"Pire trade          : {max_loss:.2f}")
    print(f"Max drawdown (equity): {max_dd*100:.1f}%")
    print("==============================================================")


if __name__ == "__main__":
    # Choisis "duel" / "long" / "short"
    cfg = LiveConfig(
        side="duel",
        n_bars_m1=200_000,
        n_bars_h1=50_000,
    )
    run_backtest(cfg)
