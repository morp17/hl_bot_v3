"""
Módulo de Estratégias de Trading — Hyperliquid Production Bot v3.0
==================================================================
Implementa 10 estratégias de trading:

1. trend_follow      — Segue tendência com EMAs + RSI + ADX
2. mean_reversion    — Reversão à média com Bollinger Bands + RSI
3. adaptive_trend    — Trend following adaptativo com ADX
4. hybrid_regime     — 3-layer: Regime Macro + VWAP Sweep + SMC Structure
5. orderflow_delta   — Delta, CVD, divergências, absorção
6. scalping_grid     — Grid trading multi-nível (ATR-adaptativo)
7. funding_arbitrage — Arbitragem de funding rate
8. volatility_squeeze — Squeeze de volatilidade
9. funding_weighted_trend — Tendência ponderada por funding rate
10. ensemble_mode — Combina múltiplas estratégias por confluência+confiança — reduz dependência de uma única estratégia "vencedora".

Todas com:
- Type hints
- Tratamento de exceções
- Validação de inputs
- Logs estruturados
- Score de confiança (0.0–1.0) em params["confidence"]

Requisitos:
- Type hints
- Tratamento de exceções
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger as log

from ..config import BotConfig, StrategyType, Timeframe


# ──────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────

Signal = str  # "buy", "sell", "hold"
SignalWithParams = Tuple[Signal, Dict[str, Any]]


# ──────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────


def _validate_df(df: pd.DataFrame, required_cols: list = None) -> bool:
    """Valida DataFrame para uso em estratégias."""
    if df is None or df.empty:
        return False
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False
    return True


def _get_latest(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Obtém valor mais recente de uma coluna com segurança."""
    try:
        if col not in df.columns:
            return default
        val = df[col].iloc[-1]
        if pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    except Exception:
        return default


def _get_prev(df: pd.DataFrame, col: str, offset: int = 1, default: float = 0.0) -> float:
    """Obtém valor anterior de uma coluna."""
    try:
        if col not in df.columns or len(df) <= offset:
            return default
        val = df[col].iloc[-1 - offset]
        if pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    except Exception:
        return default


def _crossover(df: pd.DataFrame, fast_col: str, slow_col: str) -> bool:
    """Verifica se fast_col cruzou acima de slow_col."""
    try:
        if fast_col not in df.columns or slow_col not in df.columns:
            return False
        prev_fast = _get_prev(df, fast_col)
        prev_slow = _get_prev(df, slow_col)
        curr_fast = _get_latest(df, fast_col)
        curr_slow = _get_latest(df, slow_col)
        return prev_fast <= prev_slow and curr_fast > curr_slow
    except Exception:
        return False


def _crossunder(df: pd.DataFrame, fast_col: str, slow_col: str) -> bool:
    """Verifica se fast_col cruzou abaixo de slow_col."""
    try:
        if fast_col not in df.columns or slow_col not in df.columns:
            return False
        prev_fast = _get_prev(df, fast_col)
        prev_slow = _get_prev(df, slow_col)
        curr_fast = _get_latest(df, fast_col)
        curr_slow = _get_latest(df, slow_col)
        return prev_fast >= prev_slow and curr_fast < curr_slow
    except Exception:
        return False


def _clamp01(x: float) -> float:
    """Restringe um valor ao intervalo [0.0, 1.0], tolerando NaN/Inf."""
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
# Estratégia 1: Trend Follow
# ──────────────────────────────────────────────


def signal_trend_follow(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de seguimento de tendência.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: ("buy", "sell", "hold") + parâmetros
            (incluindo "confidence": float 0.0-1.0).
    """
    try:
        if not _validate_df(df, ["ema_fast", "ema_slow", "ema_trend", "rsi", "adx"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.TREND_FOLLOW, cfg.timeframe)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        adx_threshold = 25

        ema_fast = _get_latest(df, "ema_fast")
        ema_slow = _get_latest(df, "ema_slow")
        ema_trend = _get_latest(df, "ema_trend")
        rsi = _get_latest(df, "rsi")
        adx = _get_latest(df, "adx")
        close = _get_latest(df, "close")

        uptrend = close > ema_trend
        downtrend = close < ema_trend

        adx_strength = _clamp01((adx - adx_threshold) / 25.0)

        if (ema_fast > ema_slow and uptrend and rsi < rsi_overbought and adx > adx_threshold):
            rsi_room = _clamp01((rsi_overbought - rsi) / max(rsi_overbought - 50.0, 1.0))
            confidence = round(_clamp01(0.4 + 0.35 * adx_strength + 0.25 * rsi_room), 3)
            return "buy", {
                "reason": "trend_follow_buy",
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": rsi,
                "adx": adx,
                "strength": "strong" if adx > 30 else "medium",
                "confidence": confidence,
            }

        if (ema_fast < ema_slow and downtrend and rsi > rsi_oversold and adx > adx_threshold):
            rsi_room = _clamp01((rsi - rsi_oversold) / max(50.0 - rsi_oversold, 1.0))
            confidence = round(_clamp01(0.4 + 0.35 * adx_strength + 0.25 * rsi_room), 3)
            return "sell", {
                "reason": "trend_follow_sell",
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": rsi,
                "adx": adx,
                "strength": "strong" if adx > 30 else "medium",
                "confidence": confidence,
            }

        return "hold", {"reason": "sem_sinal_claro", "adx": adx, "confidence": 0.0}

    except Exception as e:
        log.error(f"Erro na estratégia trend_follow: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 2: Mean Reversion
# ──────────────────────────────────────────────


def signal_mean_reversion(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de reversão à média.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["bb_upper", "bb_lower", "bb_middle", "rsi"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.MEAN_REVERSION, cfg.timeframe)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        close = _get_latest(df, "close")
        bb_upper = _get_latest(df, "bb_upper")
        bb_lower = _get_latest(df, "bb_lower")
        bb_middle = _get_latest(df, "bb_middle")
        rsi = _get_latest(df, "rsi")

        if close <= bb_lower and rsi < rsi_oversold:
            rsi_conf = _clamp01((rsi_oversold - rsi) / max(rsi_oversold, 1.0))
            band_dist = ((bb_lower - close) / bb_middle) if bb_middle > 0 else 0.0
            band_conf = _clamp01(band_dist / 0.03)
            confidence = round(_clamp01(0.45 * rsi_conf + 0.55 * band_conf + 0.15), 3)
            return "buy", {
                "reason": "mean_reversion_buy",
                "bb_lower": bb_lower,
                "rsi": rsi,
                "distance_pct": ((bb_middle - close) / bb_middle) * 100 if bb_middle > 0 else 0.0,
                "confidence": confidence,
            }

        if close >= bb_upper and rsi > rsi_overbought:
            rsi_conf = _clamp01((rsi - rsi_overbought) / max(100.0 - rsi_overbought, 1.0))
            band_dist = ((close - bb_upper) / bb_middle) if bb_middle > 0 else 0.0
            band_conf = _clamp01(band_dist / 0.03)
            confidence = round(_clamp01(0.45 * rsi_conf + 0.55 * band_conf + 0.15), 3)
            return "sell", {
                "reason": "mean_reversion_sell",
                "bb_upper": bb_upper,
                "rsi": rsi,
                "distance_pct": ((close - bb_middle) / bb_middle) * 100 if bb_middle > 0 else 0.0,
                "confidence": confidence,
            }

        return "hold", {
            "reason": "dentro_das_bandas",
            "bb_pct": _get_latest(df, "bb_pct"),
            "confidence": 0.0,
        }

    except Exception as e:
        log.error(f"Erro na estratégia mean_reversion: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 3: Adaptive Trend
# ──────────────────────────────────────────────


def signal_adaptive_trend(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de tendência adaptativa.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["ema_fast", "ema_slow", "adx", "rsi"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        adx = _get_latest(df, "adx")
        ema_fast = _get_latest(df, "ema_fast")
        ema_slow = _get_latest(df, "ema_slow")
        rsi = _get_latest(df, "rsi")

        if adx > 25:
            adx_conf = _clamp01((adx - 25) / 25.0)
            if ema_fast > ema_slow and rsi < 70:
                confidence = round(_clamp01(0.5 + 0.5 * adx_conf), 3)
                return "buy", {"reason": "adaptive_trend_buy", "mode": "trend", "adx": adx, "confidence": confidence}
            if ema_fast < ema_slow and rsi > 30:
                confidence = round(_clamp01(0.5 + 0.5 * adx_conf), 3)
                return "sell", {"reason": "adaptive_trend_sell", "mode": "trend", "adx": adx, "confidence": confidence}

        if adx < 20:
            if rsi < 30:
                rsi_conf = _clamp01((30 - rsi) / 30.0)
                confidence = round(_clamp01(0.4 + 0.6 * rsi_conf), 3)
                return "buy", {"reason": "adaptive_range_buy", "mode": "range", "adx": adx, "confidence": confidence}
            if rsi > 70:
                rsi_conf = _clamp01((rsi - 70) / 30.0)
                confidence = round(_clamp01(0.4 + 0.6 * rsi_conf), 3)
                return "sell", {"reason": "adaptive_range_sell", "mode": "range", "adx": adx, "confidence": confidence}

        if _crossover(df, "ema_fast", "ema_slow"):
            return "buy", {"reason": "adaptive_hybrid_buy", "mode": "hybrid", "adx": adx, "confidence": 0.35}
        if _crossunder(df, "ema_fast", "ema_slow"):
            return "sell", {"reason": "adaptive_hybrid_sell", "mode": "hybrid", "adx": adx, "confidence": 0.35}

        return "hold", {"reason": "sem_sinal", "adx": adx, "mode": "unknown", "confidence": 0.0}

    except Exception as e:
        log.error(f"Erro na estratégia adaptive_trend: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 4: Hybrid Regime (3-layer)
# ──────────────────────────────────────────────


def signal_hybrid_regime(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de 3 camadas (Regime Macro + VWAP Sweep + SMC Structure).

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["ema_50", "ema_200", "vp_vwap", "rsi"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.HYBRID_REGIME, cfg.timeframe)
        vwap_std = params.get("vwap_std", 2.0)
        swing_lookback = params.get("smc_swing_lookback", 10)

        close = _get_latest(df, "close")
        ema_50 = _get_latest(df, "ema_50")
        ema_200 = _get_latest(df, "ema_200")
        vwap = _get_latest(df, "vp_vwap")
        rsi = _get_latest(df, "rsi")

        if close > ema_200 and ema_50 > ema_200:
            regime = "bull"
        elif close < ema_200 and ema_50 < ema_200:
            regime = "bear"
        else:
            regime = "sideways"

        vwap_distance = ((close - vwap) / vwap) * 100 if vwap > 0 else 0

        swing_high = df["high"].rolling(window=swing_lookback).max().iloc[-1]
        swing_low = df["low"].rolling(window=swing_lookback).min().iloc[-1]
        prev_swing_high = df["high"].rolling(window=swing_lookback).max().iloc[-swing_lookback - 1] if len(df) > swing_lookback + 1 else swing_high
        prev_swing_low = df["low"].rolling(window=swing_lookback).min().iloc[-swing_lookback - 1] if len(df) > swing_lookback + 1 else swing_low

        bos_bull = close > prev_swing_high and swing_high > prev_swing_high
        bos_bear = close < prev_swing_low and swing_low < prev_swing_low

        if regime == "bull" and vwap_distance < -vwap_std and rsi > 40 and bos_bull:
            vwap_conf = _clamp01((abs(vwap_distance) - vwap_std) / max(vwap_std, 0.1))
            rsi_conf = _clamp01((rsi - 40) / 30.0)
            confidence = round(_clamp01(0.35 + 0.4 * vwap_conf + 0.25 * rsi_conf), 3)
            return "buy", {
                "reason": "hybrid_bull_buy",
                "regime": regime,
                "vwap_distance": vwap_distance,
                "bos": "bullish",
                "rsi": rsi,
                "confidence": confidence,
            }

        if regime == "bear" and vwap_distance > vwap_std and rsi < 60 and bos_bear:
            vwap_conf = _clamp01((abs(vwap_distance) - vwap_std) / max(vwap_std, 0.1))
            rsi_conf = _clamp01((60 - rsi) / 30.0)
            confidence = round(_clamp01(0.35 + 0.4 * vwap_conf + 0.25 * rsi_conf), 3)
            return "sell", {
                "reason": "hybrid_bear_sell",
                "regime": regime,
                "vwap_distance": vwap_distance,
                "bos": "bearish",
                "rsi": rsi,
                "confidence": confidence,
            }

        return "hold", {
            "reason": "sem_confluencia",
            "regime": regime,
            "vwap_distance": vwap_distance,
            "bos_bull": bool(bos_bull),
            "bos_bear": bool(bos_bear),
            "confidence": 0.0,
        }

    except Exception as e:
        log.error(f"Erro na estratégia hybrid_regime: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 5: OrderFlow Delta
# ──────────────────────────────────────────────


def signal_orderflow_delta(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia baseada em OrderFlow.

    FIX (melhoria de qualidade): duas correções sobre a versão
    original:
    1. Exige PERSISTÊNCIA — delta na mesma direção por pelo menos
       `persistence_bars` barras consecutivas, não apenas a última.
       Reduz sinais disparados por ruído de 1 candle isolado.
    2. absorption_threshold (antes lido do cfg mas nunca de fato
       usado em nenhum cálculo) agora modula a CONFIANÇA do sinal de
       absorção — quanto maior o volume relativo à média em relação
       ao threshold, maior a confiança.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["delta", "cvd", "close", "volume"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.ORDERFLOW_DELTA, cfg.timeframe)
        divergence_lookback = params.get("divergence_lookback", 10)
        absorption_threshold = params.get("absorption_threshold", 1.5)
        persistence_bars = 3

        close = _get_latest(df, "close")
        delta = _get_latest(df, "delta")
        cvd = _get_latest(df, "cvd")
        cvd_ma = _get_latest(df, "cvd_ma")
        volume = _get_latest(df, "volume")
        bearish_div = _get_latest(df, "of_bearish_div")
        bullish_div = _get_latest(df, "of_bullish_div")
        absorption = _get_latest(df, "of_absorption")

        delta_strength = _clamp01(abs(delta) / volume) if volume > 0 else 0.0
        cvd_gap = _clamp01(abs(cvd - cvd_ma) / max(abs(cvd_ma), 1.0))

        # FIX: persistência — delta precisa manter o mesmo sinal pelas
        # últimas `persistence_bars` barras, não só na barra atual.
        delta_persistent_up = False
        delta_persistent_down = False
        if "delta" in df.columns and len(df) >= persistence_bars:
            recent_deltas = df["delta"].tail(persistence_bars)
            delta_persistent_up = bool((recent_deltas > 0).all())
            delta_persistent_down = bool((recent_deltas < 0).all())

        if delta > 0 and delta_persistent_up and cvd > cvd_ma and bullish_div > 0:
            confidence = round(_clamp01(0.35 + 0.35 * delta_strength + 0.30 * cvd_gap), 3)
            return "buy", {
                "reason": "orderflow_buy",
                "delta": delta,
                "cvd": cvd,
                "bullish_divergence": True,
                "absorption": bool(absorption),
                "persistence_bars": persistence_bars,
                "confidence": confidence,
            }

        if delta < 0 and delta_persistent_down and cvd < cvd_ma and bearish_div > 0:
            confidence = round(_clamp01(0.35 + 0.35 * delta_strength + 0.30 * cvd_gap), 3)
            return "sell", {
                "reason": "orderflow_sell",
                "delta": delta,
                "cvd": cvd,
                "bearish_divergence": True,
                "absorption": bool(absorption),
                "persistence_bars": persistence_bars,
                "confidence": confidence,
            }

        if absorption > 0:
            # FIX: absorption_threshold agora modula a confiança —
            # volume relativo mais forte em relação ao threshold
            # configurado aumenta a confiança do sinal de absorção.
            volume_avg = float(df["volume"].tail(20).mean()) if len(df) >= 20 else volume
            volume_ratio = (volume / volume_avg) if volume_avg > 0 else 1.0
            threshold_conf = _clamp01((volume_ratio - 1.0) / max(absorption_threshold - 1.0, 0.1))

            if delta > 0:
                confidence = round(_clamp01(0.25 + 0.35 * delta_strength + 0.25 * threshold_conf), 3)
                return "buy", {
                    "reason": "orderflow_absorption_buy",
                    "absorption": True,
                    "volume_ratio": round(volume_ratio, 3),
                    "confidence": confidence,
                }
            if delta < 0:
                confidence = round(_clamp01(0.25 + 0.35 * delta_strength + 0.25 * threshold_conf), 3)
                return "sell", {
                    "reason": "orderflow_absorption_sell",
                    "absorption": True,
                    "volume_ratio": round(volume_ratio, 3),
                    "confidence": confidence,
                }

        return "hold", {
            "reason": "sem_sinal_orderflow",
            "delta": delta,
            "cvd_vs_ma": cvd - cvd_ma,
            "confidence": 0.0,
        }

    except Exception as e:
        log.error(f"Erro na estratégia orderflow_delta: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 6: Scalping Grid (multi-nível, ATR-adaptativo)
# ──────────────────────────────────────────────


def signal_scalping_grid(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de grid trading para timeframes curtos.

    Avalia múltiplos níveis do grid (não apenas o mais próximo), usa
    espaçamento dinâmico via ATR quando disponível (fallback percentual
    fixo idêntico ao comportamento legado quando 'atr' está ausente) e
    considera confirmação de volume opcional.

    Args:
        df: DataFrame com indicadores. Requer 'close', 'ema_fast',
            'ema_slow', 'rsi'. Usa opcionalmente 'atr' e 'volume'.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros (inclui "confidence",
            "levels_touched", "grid_mode": "atr" ou "pct").
    """
    try:
        if not _validate_df(df, ["close", "ema_fast", "ema_slow", "rsi"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.SCALPING_GRID, cfg.timeframe)
        grid_levels = int(params.get("grid_levels", 5))
        grid_spread_pct = params.get("grid_spread_pct", 0.1)

        close = _get_latest(df, "close")
        rsi = _get_latest(df, "rsi")

        if close <= 0 or grid_levels <= 0:
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        atr = _get_latest(df, "atr", default=-1.0)
        if atr > 0:
            grid_step = atr * 0.5
            grid_mode = "atr"
        else:
            grid_step = close * (grid_spread_pct / 100)
            grid_mode = "pct"

        buy_levels = [close - (grid_step * (i + 1)) for i in range(grid_levels)]
        sell_levels = [close + (grid_step * (i + 1)) for i in range(grid_levels)]

        volume_confirmed = True
        volume_conf_score = 0.0
        if "volume" in df.columns and len(df) >= 20:
            try:
                vol = _get_latest(df, "volume")
                vol_avg = float(df["volume"].rolling(window=20).mean().iloc[-1])
                if vol_avg > 0:
                    volume_confirmed = vol >= vol_avg
                    volume_conf_score = _clamp01((vol - vol_avg) / vol_avg) if volume_confirmed else 0.0
            except Exception:
                pass

        touched_buy_levels = [
            i + 1 for i, lvl in enumerate(buy_levels) if close <= lvl * 1.01
        ]
        if touched_buy_levels and rsi < 45 and volume_confirmed:
            deepest_level = max(touched_buy_levels)
            depth_conf = _clamp01(deepest_level / grid_levels)
            rsi_conf = _clamp01((45 - rsi) / 45.0)
            confidence = round(
                _clamp01(0.3 + 0.35 * depth_conf + 0.2 * rsi_conf + 0.15 * volume_conf_score), 3
            )
            return "buy", {
                "reason": "scalping_grid_buy",
                "grid_level": deepest_level,
                "levels_touched": touched_buy_levels,
                "grid_mode": grid_mode,
                "buy_levels": buy_levels,
                "sell_levels": sell_levels,
                "rsi": rsi,
                "confidence": confidence,
            }

        touched_sell_levels = [
            i + 1 for i, lvl in enumerate(sell_levels) if close >= lvl * 0.99
        ]
        if touched_sell_levels and rsi > 55 and volume_confirmed:
            deepest_level = max(touched_sell_levels)
            depth_conf = _clamp01(deepest_level / grid_levels)
            rsi_conf = _clamp01((rsi - 55) / 45.0)
            confidence = round(
                _clamp01(0.3 + 0.35 * depth_conf + 0.2 * rsi_conf + 0.15 * volume_conf_score), 3
            )
            return "sell", {
                "reason": "scalping_grid_sell",
                "grid_level": deepest_level,
                "levels_touched": touched_sell_levels,
                "grid_mode": grid_mode,
                "buy_levels": buy_levels,
                "sell_levels": sell_levels,
                "rsi": rsi,
                "confidence": confidence,
            }

        return "hold", {
            "reason": "fora_do_grid",
            "grid_mode": grid_mode,
            "nearest_buy": buy_levels[0],
            "nearest_sell": sell_levels[0],
            "confidence": 0.0,
        }

    except Exception as e:
        log.error(f"Erro na estratégia scalping_grid: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 7: Funding Arbitrage
# ──────────────────────────────────────────────


def signal_funding_arbitrage(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de arbitragem de funding rate.

    Args:
        df: DataFrame com indicadores (precisa de funding_rate).
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["close"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.FUNDING_ARBITRAGE, cfg.timeframe)
        min_funding_rate = params.get("min_funding_rate", 0.0001)
        max_funding_rate = params.get("max_funding_rate", 0.01)

        funding_rate = _get_latest(df, "funding_rate", 0.0)

        if funding_rate == 0.0:
            return "hold", {"reason": "funding_rate_indisponivel", "confidence": 0.0}

        if funding_rate > max_funding_rate:
            extreme_conf = _clamp01((funding_rate - max_funding_rate) / max(max_funding_rate, 1e-6))
            confidence = round(_clamp01(0.6 + 0.4 * extreme_conf), 3)
            return "sell", {
                "reason": "funding_arb_sell",
                "funding_rate": funding_rate,
                "action": "short_funding_long",
                "confidence": confidence,
            }

        if funding_rate < -max_funding_rate:
            extreme_conf = _clamp01((abs(funding_rate) - max_funding_rate) / max(max_funding_rate, 1e-6))
            confidence = round(_clamp01(0.6 + 0.4 * extreme_conf), 3)
            return "buy", {
                "reason": "funding_arb_buy",
                "funding_rate": funding_rate,
                "action": "long_funding_short",
                "confidence": confidence,
            }

        if funding_rate > min_funding_rate:
            moderate_conf = _clamp01((funding_rate - min_funding_rate) / max(max_funding_rate - min_funding_rate, 1e-6))
            confidence = round(_clamp01(0.3 + 0.3 * moderate_conf), 3)
            return "sell", {
                "reason": "funding_arb_sell_moderate",
                "funding_rate": funding_rate,
                "strength": "moderate",
                "confidence": confidence,
            }

        if funding_rate < -min_funding_rate:
            moderate_conf = _clamp01((abs(funding_rate) - min_funding_rate) / max(max_funding_rate - min_funding_rate, 1e-6))
            confidence = round(_clamp01(0.3 + 0.3 * moderate_conf), 3)
            return "buy", {
                "reason": "funding_arb_buy_moderate",
                "funding_rate": funding_rate,
                "strength": "moderate",
                "confidence": confidence,
            }

        return "hold", {"reason": "funding_neutro", "funding_rate": funding_rate, "confidence": 0.0}

    except Exception as e:
        log.error(f"Erro na estratégia funding_arbitrage: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}

# ──────────────────────────────────────────────
# Estratégia 8: Volatility Squeeze Breakout (NOVA)
# ──────────────────────────────────────────────


def signal_volatility_squeeze(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de rompimento pós-compressão de volatilidade.

    FIX (achado em teste — falso positivo em série constante): a versão
    anterior definia "squeeze recente" comparando a cauda de bb_width
    contra o quantile(0.25) do MESMO histórico que a contém — em uma
    série sem variação (bb_width constante), quantile(0.25) é igual ao
    próprio valor constante, então a comparação "tail <= quantile" era
    SEMPRE verdadeira, mesmo sem nenhuma compressão real de volatilidade.
    Corrigido: squeeze agora é definido pelo percentil RANK da largura
    ATUAL frente a todo o histórico do lookback (width_percentile <=
    squeeze_quantile), que não sofre desse problema de auto-referência.

    Requer: bb_width, bb_upper, bb_lower, close. Usa volume
    opcionalmente para confirmação (mesmo padrão de scalping_grid).

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["bb_width", "bb_upper", "bb_lower", "close"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        lookback = 50
        if len(df) < lookback + 1:
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.VOLATILITY_SQUEEZE, cfg.timeframe)
        squeeze_quantile = params.get("squeeze_quantile", 0.25)
        volume_confirm_mult = params.get("volume_confirm_mult", 1.2)

        bb_width_series = df["bb_width"].tail(lookback)
        current_width = _get_latest(df, "bb_width")
        width_percentile = float((bb_width_series <= current_width).mean())

        close = _get_latest(df, "close")
        prev_close = _get_prev(df, "close")
        bb_upper = _get_latest(df, "bb_upper")
        bb_lower = _get_latest(df, "bb_lower")

        # FIX: squeeze definido pelo percentil da largura ATUAL frente ao
        # histórico — não mais por comparação auto-referencial da cauda
        # contra o quantile do mesmo conjunto (ver docstring acima).
        in_squeeze = width_percentile <= squeeze_quantile

        if not in_squeeze:
            return "hold", {
                "reason": "sem_squeeze_recente",
                "width_percentile": round(width_percentile, 3),
                "confidence": 0.0,
            }

        volume_confirmed = True
        volume_score = 0.0
        if "volume" in df.columns and len(df) >= 20:
            try:
                vol = _get_latest(df, "volume")
                vol_avg = float(df["volume"].rolling(window=20).mean().iloc[-1])
                if vol_avg > 0:
                    volume_confirmed = vol >= vol_avg * volume_confirm_mult
                    volume_score = _clamp01((vol - vol_avg) / vol_avg)
            except Exception:
                pass

        breakout_up = prev_close <= bb_upper and close > bb_upper
        breakout_down = prev_close >= bb_lower and close < bb_lower

        if breakout_up and volume_confirmed:
            squeeze_conf = _clamp01(1.0 - width_percentile)
            confidence = round(_clamp01(0.35 + 0.35 * squeeze_conf + 0.30 * volume_score), 3)
            return "buy", {
                "reason": "volatility_squeeze_breakout_up",
                "bb_upper": bb_upper,
                "width_percentile": round(width_percentile, 3),
                "volume_confirmed": volume_confirmed,
                "confidence": confidence,
            }

        if breakout_down and volume_confirmed:
            squeeze_conf = _clamp01(1.0 - width_percentile)
            confidence = round(_clamp01(0.35 + 0.35 * squeeze_conf + 0.30 * volume_score), 3)
            return "sell", {
                "reason": "volatility_squeeze_breakout_down",
                "bb_lower": bb_lower,
                "width_percentile": round(width_percentile, 3),
                "volume_confirmed": volume_confirmed,
                "confidence": confidence,
            }

        return "hold", {
            "reason": "squeeze_sem_rompimento_confirmado",
            "width_percentile": round(width_percentile, 3),
            "confidence": 0.0,
        }

    except Exception as e:
        log.error(f"Erro na estratégia volatility_squeeze: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Estratégia 9: Funding-Weighted Trend (NOVA)
# ──────────────────────────────────────────────


def signal_funding_weighted_trend(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de tendência ponderada por funding rate — versão
    "institucional" do funding_arbitrage isolado.

    Racional: o funding_arbitrage original (item 5 do backlog original)
    opera CONTRA o funding extremo, ignorando se há uma tendência forte
    na mesma direção do funding — isso pode significar vender contra um
    bull run saudável só porque longs estão pagando funding alto, o que
    é comportamento normal de mercado em alta sustentada, não um sinal
    de reversão por si só.

    Esta estratégia só opera quando o funding e a tendência (EMA fast/
    slow) CONCORDAM na direção de carry desfavorável ao lado dominante —
    ou seja, entra a FAVOR da tendência apenas quando o funding não está
    excessivamente contra essa direção (evita pagar carry caro), e entra
    CONTRA a tendência apenas em extremos de funding combinados com
    enfraquecimento de momentum (RSI divergente do extremo). É mais
    conservadora e seletiva que o funding_arbitrage puro.

    Requer: ema_fast, ema_slow, rsi, funding_rate.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["ema_fast", "ema_slow", "rsi"]):
            return "hold", {"reason": "dados_insuficientes", "confidence": 0.0}

        funding_rate = _get_latest(df, "funding_rate", 0.0)
        if funding_rate == 0.0:
            return "hold", {"reason": "funding_rate_indisponivel", "confidence": 0.0}

        params = cfg.get_strategy_params(StrategyType.FUNDING_ARBITRAGE, cfg.timeframe)
        max_funding_rate = params.get("max_funding_rate", 0.01)

        ema_fast = _get_latest(df, "ema_fast")
        ema_slow = _get_latest(df, "ema_slow")
        rsi = _get_latest(df, "rsi")

        uptrend = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow
        funding_extreme = abs(funding_rate) > max_funding_rate

        # Caso 1: tendência de alta + funding positivo mas NÃO extremo
        # (carry ainda saudável) -> segue a tendência.
        if uptrend and 0 < funding_rate <= max_funding_rate and rsi < 70:
            funding_headroom = _clamp01(1.0 - (funding_rate / max_funding_rate))
            confidence = round(_clamp01(0.35 + 0.4 * funding_headroom + 0.15 * _clamp01((70 - rsi) / 30.0)), 3)
            return "buy", {
                "reason": "funding_weighted_trend_follow_buy",
                "funding_rate": funding_rate,
                "mode": "carry_favoravel",
                "confidence": confidence,
            }

        if downtrend and -max_funding_rate <= funding_rate < 0 and rsi > 30:
            funding_headroom = _clamp01(1.0 - (abs(funding_rate) / max_funding_rate))
            confidence = round(_clamp01(0.35 + 0.4 * funding_headroom + 0.15 * _clamp01((rsi - 30) / 30.0)), 3)
            return "sell", {
                "reason": "funding_weighted_trend_follow_sell",
                "funding_rate": funding_rate,
                "mode": "carry_favoravel",
                "confidence": confidence,
            }

        # Caso 2: funding extremo CONTRA a tendência dominante, com RSI
        # já esticado no extremo oposto (sinal de exaustão) -> reversão
        # seletiva, não automática como no funding_arbitrage puro.
        if funding_extreme and funding_rate > max_funding_rate and uptrend and rsi > 75:
            extreme_conf = _clamp01((funding_rate - max_funding_rate) / max(max_funding_rate, 1e-6))
            rsi_conf = _clamp01((rsi - 75) / 25.0)
            confidence = round(_clamp01(0.4 + 0.35 * extreme_conf + 0.25 * rsi_conf), 3)
            return "sell", {
                "reason": "funding_weighted_exhaustion_sell",
                "funding_rate": funding_rate,
                "mode": "exaustao_reversao",
                "confidence": confidence,
            }

        if funding_extreme and funding_rate < -max_funding_rate and downtrend and rsi < 25:
            extreme_conf = _clamp01((abs(funding_rate) - max_funding_rate) / max(max_funding_rate, 1e-6))
            rsi_conf = _clamp01((25 - rsi) / 25.0)
            confidence = round(_clamp01(0.4 + 0.35 * extreme_conf + 0.25 * rsi_conf), 3)
            return "buy", {
                "reason": "funding_weighted_exhaustion_buy",
                "funding_rate": funding_rate,
                "mode": "exaustao_reversao",
                "confidence": confidence,
            }

        return "hold", {
            "reason": "sem_alinhamento_funding_tendencia",
            "funding_rate": funding_rate,
            "confidence": 0.0,
        }

    except Exception as e:
        log.error(f"Erro na estratégia funding_weighted_trend: {e}")
        return "hold", {"reason": f"erro: {e}", "confidence": 0.0}

# ──────────────────────────────────────────────
# Dispatch / Router (single-strategy)
# ──────────────────────────────────────────────

STRATEGY_MAP: Dict[str, callable] = {
    "trend_follow": signal_trend_follow,
    "mean_reversion": signal_mean_reversion,
    "adaptive_trend": signal_adaptive_trend,
    "hybrid_regime": signal_hybrid_regime,
    "orderflow_delta": signal_orderflow_delta,
    "scalping_grid": signal_scalping_grid,
    "funding_arbitrage": signal_funding_arbitrage,
    "volatility_squeeze": signal_volatility_squeeze,       # NOVA
    "funding_weighted_trend": signal_funding_weighted_trend,  # NOVA
}

# FIX (achado na análise de logs/backtest): o WARNING de "estratégia
# desabilitada" era emitido a cada barra processada — em um backtest de
# 2000 candles isso gera milhares de linhas idênticas no log, sem
# agregar informação nova após a primeira ocorrência. Este conjunto
# rastreia, por processo, quais estratégias já geraram o warning em
# nível WARNING; ocorrências repetidas caem para DEBUG (continuam
# rastreáveis, mas não poluem o log em nível operacional).
_disabled_strategy_warned: Set[str] = set()


def get_signal(
    df: pd.DataFrame,
    cfg: BotConfig,
    enforce_enabled_gate: bool = True,
) -> SignalWithParams:
    """
    Obtém sinal da estratégia configurada (modo single-strategy).

    Antes de executar, verifica se a estratégia está em
    cfg.enabled_strategies — a menos que enforce_enabled_gate=False.

    FIX (achado na análise de logs/backtest): adicionado o parâmetro
    enforce_enabled_gate para permitir que o BacktestEngine (cujo
    propósito é justamente VALIDAR uma estratégia antes de liberá-la
    em ENABLED_STRATEGIES) rode a lógica real da estratégia mesmo que
    ela ainda não esteja habilitada — sem esse bypass, backtestar uma
    estratégia não habilitada sempre resultava em total_trades=0,
    silenciosamente, contradizendo o próprio propósito do backtest.

    O comportamento em modo LIVE (main.py) é inalterado: chamadas sem
    passar o parâmetro continuam com enforce_enabled_gate=True,
    idêntico ao comportamento anterior a esta correção.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração do bot.
        enforce_enabled_gate: Se True (default), respeita
            cfg.enabled_strategies. Se False, ignora o gate e executa
            a estratégia diretamente — uso pretendido: backtest.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        strategy_name = cfg.strategy.value

        if enforce_enabled_gate and not cfg.is_strategy_enabled():
            if strategy_name not in _disabled_strategy_warned:
                _disabled_strategy_warned.add(strategy_name)
                log.warning(
                    f"[SIGNAL] Estratégia '{strategy_name}' não está em "
                    f"enabled_strategies ('{cfg.enabled_strategies}') — "
                    f"retornando hold. Habilite explicitamente no .env "
                    f"apenas após validação walk-forward. (Este aviso não "
                    f"será repetido em WARNING para esta estratégia neste "
                    f"processo — ocorrências futuras vão para DEBUG.)"
                )
            else:
                log.debug(
                    f"[SIGNAL] Estratégia '{strategy_name}' desabilitada — "
                    f"hold (aviso já emitido em WARNING anteriormente)."
                )
            return "hold", {"reason": f"estrategia_desabilitada: {strategy_name}", "confidence": 0.0}

        strategy_func = STRATEGY_MAP.get(strategy_name)

        if strategy_func is None:
            log.error(f"Estratégia desconhecida: {strategy_name}")
            return "hold", {"reason": f"estrategia_desconhecida: {strategy_name}", "confidence": 0.0}

        log.info(f"Executando estratégia: {strategy_name}")
        signal, params = strategy_func(df, cfg)
        log.info(
            f"Sinal: {signal.upper()} | Motivo: {params.get('reason', 'N/A')} | "
            f"Confiança: {params.get('confidence', 0.0):.2f}"
        )

        return signal, params

    except Exception as e:
        log.error(f"Erro ao executar estratégia {cfg.strategy}: {e}")
        return "hold", {"reason": f"erro_interno: {e}", "confidence": 0.0}


# ──────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────


def get_ensemble_signal(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Combina os sinais de TODAS as estratégias em cfg.enabled_strategies,
    ponderando por confiança, e só emite buy/sell quando há confluência
    mínima entre elas.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração do bot.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros, incluindo "votes",
            "buy_count", "sell_count" e "confidence".
    """
    try:
        enabled_names = [s.strip() for s in cfg.enabled_strategies.split(",") if s.strip()]
        if not enabled_names:
            return "hold", {"reason": "nenhuma_estrategia_habilitada", "confidence": 0.0}

        votes: List[Dict[str, Any]] = []
        for name in enabled_names:
            func = STRATEGY_MAP.get(name)
            if func is None:
                log.warning(f"[ENSEMBLE] Estratégia desconhecida em enabled_strategies: {name}")
                continue
            try:
                sig, params = func(df, cfg)
            except Exception as e:
                log.error(f"[ENSEMBLE] Erro ao executar estratégia '{name}': {e}")
                continue

            votes.append({
                "strategy": name,
                "signal": sig,
                "confidence": float(params.get("confidence", 0.0)),
                "reason": params.get("reason", ""),
            })

        buy_votes = [v for v in votes if v["signal"] == "buy"]
        sell_votes = [v for v in votes if v["signal"] == "sell"]

        min_confluence = max(1, int(getattr(cfg, "ensemble_min_confluence", 2)))
        min_avg_confidence = float(getattr(cfg, "ensemble_min_avg_confidence", 0.55))

        def _side_score(side_votes: List[Dict[str, Any]]) -> Optional[float]:
            if len(side_votes) < min_confluence:
                return None
            avg_conf = sum(v["confidence"] for v in side_votes) / len(side_votes)
            if avg_conf < min_avg_confidence:
                return None
            return avg_conf

        buy_score = _side_score(buy_votes)
        sell_score = _side_score(sell_votes)

        base_details: Dict[str, Any] = {
            "votes": votes,
            "buy_count": len(buy_votes),
            "sell_count": len(sell_votes),
            "min_confluence": min_confluence,
            "min_avg_confidence": min_avg_confidence,
        }

        log.info(
            f"[ENSEMBLE] Votos: {len(votes)} estratégia(s) avaliada(s) | "
            f"buy={len(buy_votes)} sell={len(sell_votes)} hold={len(votes) - len(buy_votes) - len(sell_votes)} | "
            f"min_confluencia={min_confluence} min_conf_media={min_avg_confidence:.2f}"
        )

        if buy_score is not None and (sell_score is None or buy_score >= sell_score):
            base_details.update({
                "reason": "ensemble_buy_confluencia",
                "confidence": round(buy_score, 3),
            })
            log.info(
                f"[ENSEMBLE] BUY confirmado por confluência: "
                f"{len(buy_votes)}/{len(votes)} estratégia(s), "
                f"confiança média={buy_score:.2f}"
            )
            return "buy", base_details

        if sell_score is not None and (buy_score is None or sell_score > buy_score):
            base_details.update({
                "reason": "ensemble_sell_confluencia",
                "confidence": round(sell_score, 3),
            })
            log.info(
                f"[ENSEMBLE] SELL confirmado por confluência: "
                f"{len(sell_votes)}/{len(votes)} estratégia(s), "
                f"confiança média={sell_score:.2f}"
            )
            return "sell", base_details

        base_details.update({"reason": "sem_confluencia_minima", "confidence": 0.0})
        return "hold", base_details

    except Exception as e:
        log.error(f"[ENSEMBLE] Erro geral no ensemble: {e}")
        return "hold", {"reason": f"erro_ensemble: {e}", "confidence": 0.0}