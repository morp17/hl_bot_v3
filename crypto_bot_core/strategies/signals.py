"""
Módulo de Estratégias de Trading — Hyperliquid Production Bot v3.0
==================================================================
Implementa 7 estratégias de trading:

1. trend_follow      — Segue tendência com EMAs + RSI + ADX
2. mean_reversion    — Reversão à média com Bollinger Bands + RSI
3. adaptive_trend    — Trend following adaptativo com ADX
4. hybrid_regime     — 3-layer: Regime Macro + VWAP Sweep + SMC Structure
5. orderflow_delta   — Delta, CVD, divergências, absorção
6. scalping_grid     — Grid trading em timeframes curtos
7. funding_arbitrage — Arbitragem de funding rate

Todas com:
- Type hints
- Tratamento de exceções
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


# ──────────────────────────────────────────────
# Estratégia 1: Trend Follow
# ──────────────────────────────────────────────


def signal_trend_follow(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de seguimento de tendência.

    Usa:
    - EMA fast/slow para direção
    - EMA trend (200) como filtro de tendência macro
    - RSI para confirmação (evitar entradas em overbought/oversold)
    - ADX para força da tendência

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: ("buy", "sell", "hold") + parâmetros.
    """
    try:
        if not _validate_df(df, ["ema_fast", "ema_slow", "ema_trend", "rsi", "adx"]):
            return "hold", {"reason": "dados_insuficientes"}

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

        # Filtro de tendência macro
        uptrend = close > ema_trend
        downtrend = close < ema_trend

        # Sinais
        if (ema_fast > ema_slow and uptrend and rsi < rsi_overbought and adx > adx_threshold):
            return "buy", {
                "reason": "trend_follow_buy",
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": rsi,
                "adx": adx,
                "strength": "strong" if adx > 30 else "medium",
            }

        if (ema_fast < ema_slow and downtrend and rsi > rsi_oversold and adx > adx_threshold):
            return "sell", {
                "reason": "trend_follow_sell",
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": rsi,
                "adx": adx,
                "strength": "strong" if adx > 30 else "medium",
            }

        return "hold", {"reason": "sem_sinal_claro", "adx": adx}

    except Exception as e:
        log.error(f"Erro na estratégia trend_follow: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Estratégia 2: Mean Reversion
# ──────────────────────────────────────────────


def signal_mean_reversion(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de reversão à média.

    Usa:
    - Bollinger Bands (preço tocando bandas externas)
    - RSI em extremos
    - Confirmação de volume

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["bb_upper", "bb_lower", "bb_middle", "rsi"]):
            return "hold", {"reason": "dados_insuficientes"}

        params = cfg.get_strategy_params(StrategyType.MEAN_REVERSION, cfg.timeframe)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        close = _get_latest(df, "close")
        bb_upper = _get_latest(df, "bb_upper")
        bb_lower = _get_latest(df, "bb_lower")
        bb_middle = _get_latest(df, "bb_middle")
        rsi = _get_latest(df, "rsi")

        # Sinal de compra (solevendido)
        if close <= bb_lower and rsi < rsi_oversold:
            return "buy", {
                "reason": "mean_reversion_buy",
                "bb_lower": bb_lower,
                "rsi": rsi,
                "distance_pct": ((bb_middle - close) / bb_middle) * 100,
            }

        # Sinal de venda (sobrecomprado)
        if close >= bb_upper and rsi > rsi_overbought:
            return "sell", {
                "reason": "mean_reversion_sell",
                "bb_upper": bb_upper,
                "rsi": rsi,
                "distance_pct": ((close - bb_middle) / bb_middle) * 100,
            }

        return "hold", {"reason": "dentro_das_bandas", "bb_pct": _get_latest(df, "bb_pct")}

    except Exception as e:
        log.error(f"Erro na estratégia mean_reversion: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Estratégia 3: Adaptive Trend
# ──────────────────────────────────────────────


def signal_adaptive_trend(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de tendência adaptativa.

    Usa ADX para determinar o regime de mercado:
    - ADX > 25: Trend mode (segue EMAs)
    - ADX < 20: Ranging mode (usa mean reversion)
    - ADX entre 20-25: Modo híbrido

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["ema_fast", "ema_slow", "adx", "rsi"]):
            return "hold", {"reason": "dados_insuficientes"}

        adx = _get_latest(df, "adx")
        ema_fast = _get_latest(df, "ema_fast")
        ema_slow = _get_latest(df, "ema_slow")
        rsi = _get_latest(df, "rsi")

        # Trend mode
        if adx > 25:
            if ema_fast > ema_slow and rsi < 70:
                return "buy", {"reason": "adaptive_trend_buy", "mode": "trend", "adx": adx}
            if ema_fast < ema_slow and rsi > 30:
                return "sell", {"reason": "adaptive_trend_sell", "mode": "trend", "adx": adx}

        # Ranging mode
        if adx < 20:
            if rsi < 30:
                return "buy", {"reason": "adaptive_range_buy", "mode": "range", "adx": adx}
            if rsi > 70:
                return "sell", {"reason": "adaptive_range_sell", "mode": "range", "adx": adx}

        # Hybrid mode
        if _crossover(df, "ema_fast", "ema_slow"):
            return "buy", {"reason": "adaptive_hybrid_buy", "mode": "hybrid", "adx": adx}
        if _crossunder(df, "ema_fast", "ema_slow"):
            return "sell", {"reason": "adaptive_hybrid_sell", "mode": "hybrid", "adx": adx}

        return "hold", {"reason": "sem_sinal", "adx": adx, "mode": "unknown"}

    except Exception as e:
        log.error(f"Erro na estratégia adaptive_trend: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Estratégia 4: Hybrid Regime (3-layer)
# ──────────────────────────────────────────────


def signal_hybrid_regime(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de 3 camadas (Regime Macro + VWAP Sweep + SMC Structure).

    Layer 1 — Regime Macro:
    - EMAs de longo prazo definem o regime (bull/bear/sideways)
    - Filtro de tendência principal

    Layer 2 — VWAP Sweep:
    - Preço varrendo VWAP com desvio padrão
    - Rejeição ou absorção nos extremos

    Layer 3 — SMC (Smart Money Concepts):
    - Estrutura de mercado (swing highs/lows)
    - Quebra de estrutura (BOS) / Change of Character (CHoCH)

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["ema_50", "ema_200", "vp_vwap", "rsi"]):
            return "hold", {"reason": "dados_insuficientes"}

        params = cfg.get_strategy_params(StrategyType.HYBRID_REGIME, cfg.timeframe)
        vwap_std = params.get("vwap_std", 2.0)
        swing_lookback = params.get("smc_swing_lookback", 10)

        close = _get_latest(df, "close")
        ema_50 = _get_latest(df, "ema_50")
        ema_200 = _get_latest(df, "ema_200")
        vwap = _get_latest(df, "vp_vwap")
        rsi = _get_latest(df, "rsi")

        # Layer 1: Regime Macro
        if close > ema_200 and ema_50 > ema_200:
            regime = "bull"
        elif close < ema_200 and ema_50 < ema_200:
            regime = "bear"
        else:
            regime = "sideways"

        # Layer 2: VWAP Sweep
        vwap_distance = ((close - vwap) / vwap) * 100 if vwap > 0 else 0

        # Layer 3: SMC Structure
        swing_high = df["high"].rolling(window=swing_lookback).max().iloc[-1]
        swing_low = df["low"].rolling(window=swing_lookback).min().iloc[-1]
        prev_swing_high = df["high"].rolling(window=swing_lookback).max().iloc[-swing_lookback - 1] if len(df) > swing_lookback + 1 else swing_high
        prev_swing_low = df["low"].rolling(window=swing_lookback).min().iloc[-swing_lookback - 1] if len(df) > swing_lookback + 1 else swing_low

        # BOS (Break of Structure) - Quebra de estrutura
        bos_bull = close > prev_swing_high and swing_high > prev_swing_high
        bos_bear = close < prev_swing_low and swing_low < prev_swing_low

        # Decisão final
        if regime == "bull" and vwap_distance < -vwap_std and rsi > 40 and bos_bull:
            return "buy", {
                "reason": "hybrid_bull_buy",
                "regime": regime,
                "vwap_distance": vwap_distance,
                "bos": "bullish",
                "rsi": rsi,
            }

        if regime == "bear" and vwap_distance > vwap_std and rsi < 60 and bos_bear:
            return "sell", {
                "reason": "hybrid_bear_sell",
                "regime": regime,
                "vwap_distance": vwap_distance,
                "bos": "bearish",
                "rsi": rsi,
            }

        return "hold", {
            "reason": "sem_confluencia",
            "regime": regime,
            "vwap_distance": vwap_distance,
            "bos_bull": bool(bos_bull),
            "bos_bear": bool(bos_bear),
        }

    except Exception as e:
        log.error(f"Erro na estratégia hybrid_regime: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Estratégia 5: OrderFlow Delta
# ──────────────────────────────────────────────


def signal_orderflow_delta(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia baseada em OrderFlow.

    Usa:
    - Delta (diferença entre compras e vendas)
    - CVD (Cumulative Volume Delta)
    - Divergências entre preço e delta
    - Absorção (volume alto com range pequeno)

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["delta", "cvd", "close", "volume"]):
            return "hold", {"reason": "dados_insuficientes"}

        params = cfg.get_strategy_params(StrategyType.ORDERFLOW_DELTA, cfg.timeframe)
        divergence_lookback = params.get("divergence_lookback", 10)
        absorption_threshold = params.get("absorption_threshold", 1.5)

        close = _get_latest(df, "close")
        delta = _get_latest(df, "delta")
        cvd = _get_latest(df, "cvd")
        cvd_ma = _get_latest(df, "cvd_ma")
        bearish_div = _get_latest(df, "of_bearish_div")
        bullish_div = _get_latest(df, "of_bullish_div")
        absorption = _get_latest(df, "of_absorption")

        # Delta positivo forte = pressão compradora
        if delta > 0 and cvd > cvd_ma and bullish_div > 0:
            return "buy", {
                "reason": "orderflow_buy",
                "delta": delta,
                "cvd": cvd,
                "bullish_divergence": True,
                "absorption": bool(absorption),
            }

        # Delta negativo forte = pressão vendedora
        if delta < 0 and cvd < cvd_ma and bearish_div > 0:
            return "sell", {
                "reason": "orderflow_sell",
                "delta": delta,
                "cvd": cvd,
                "bearish_divergence": True,
                "absorption": bool(absorption),
            }

        # Absorção = possível reversão
        if absorption > 0:
            if delta > 0:
                return "buy", {"reason": "orderflow_absorption_buy", "absorption": True}
            if delta < 0:
                return "sell", {"reason": "orderflow_absorption_sell", "absorption": True}

        return "hold", {
            "reason": "sem_sinal_orderflow",
            "delta": delta,
            "cvd_vs_ma": cvd - cvd_ma,
        }

    except Exception as e:
        log.error(f"Erro na estratégia orderflow_delta: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Estratégia 6: Scalping Grid
# ──────────────────────────────────────────────


def signal_scalping_grid(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de grid trading para timeframes curtos.

    Cria níveis de grid ao redor do preço atual.
    Compra em suportes, vende em resistências.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["close", "ema_fast", "ema_slow", "rsi"]):
            return "hold", {"reason": "dados_insuficientes"}

        params = cfg.get_strategy_params(StrategyType.SCALPING_GRID, cfg.timeframe)
        grid_levels = params.get("grid_levels", 5)
        grid_spread_pct = params.get("grid_spread_pct", 0.1)
        grid_size_pct = params.get("grid_size_pct", 20.0)

        close = _get_latest(df, "close")
        ema_fast = _get_latest(df, "ema_fast")
        rsi = _get_latest(df, "rsi")

        # Calcula níveis do grid
        grid_step = close * (grid_spread_pct / 100)
        buy_levels = [close - (grid_step * (i + 1)) for i in range(grid_levels)]
        sell_levels = [close + (grid_step * (i + 1)) for i in range(grid_levels)]

        # Sinal de compra: preço próximo ao primeiro nível de compra e RSI baixo
        if close <= buy_levels[0] * 1.01 and rsi < 45:
            return "buy", {
                "reason": "scalping_grid_buy",
                "grid_level": 1,
                "buy_levels": buy_levels,
                "sell_levels": sell_levels,
                "rsi": rsi,
            }

        # Sinal de venda: preço próximo ao primeiro nível de venda e RSI alto
        if close >= sell_levels[0] * 0.99 and rsi > 55:
            return "sell", {
                "reason": "scalping_grid_sell",
                "grid_level": 1,
                "buy_levels": buy_levels,
                "sell_levels": sell_levels,
                "rsi": rsi,
            }

        return "hold", {
            "reason": "fora_do_grid",
            "nearest_buy": buy_levels[0],
            "nearest_sell": sell_levels[0],
        }

    except Exception as e:
        log.error(f"Erro na estratégia scalping_grid: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Estratégia 7: Funding Arbitrage
# ──────────────────────────────────────────────


def signal_funding_arbitrage(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Estratégia de arbitragem de funding rate.

    Abre posição contrária ao funding dominante quando
    a taxa de funding está em extremos.

    Args:
        df: DataFrame com indicadores (precisa de funding_rate).
        cfg: Configuração.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        if not _validate_df(df, ["close"]):
            return "hold", {"reason": "dados_insuficientes"}

        params = cfg.get_strategy_params(StrategyType.FUNDING_ARBITRAGE, cfg.timeframe)
        min_funding_rate = params.get("min_funding_rate", 0.0001)
        max_funding_rate = params.get("max_funding_rate", 0.01)

        # Funding rate precisa vir de fonte externa (Info API)
        # Aqui usamos um placeholder - a taxa real será injetada
        funding_rate = _get_latest(df, "funding_rate", 0.0)

        if funding_rate == 0.0:
            return "hold", {"reason": "funding_rate_indisponivel"}

        # Funding muito positivo = mercado comprado demais = vender
        if funding_rate > max_funding_rate:
            return "sell", {
                "reason": "funding_arb_sell",
                "funding_rate": funding_rate,
                "action": "short_funding_long",
            }

        # Funding muito negativo = mercado vendido demais = comprar
        if funding_rate < -max_funding_rate:
            return "buy", {
                "reason": "funding_arb_buy",
                "funding_rate": funding_rate,
                "action": "long_funding_short",
            }

        # Funding moderado mas com direção
        if funding_rate > min_funding_rate:
            return "sell", {
                "reason": "funding_arb_sell_moderate",
                "funding_rate": funding_rate,
                "strength": "moderate",
            }

        if funding_rate < -min_funding_rate:
            return "buy", {
                "reason": "funding_arb_buy_moderate",
                "funding_rate": funding_rate,
                "strength": "moderate",
            }

        return "hold", {"reason": "funding_neutro", "funding_rate": funding_rate}

    except Exception as e:
        log.error(f"Erro na estratégia funding_arbitrage: {e}")
        return "hold", {"reason": f"erro: {e}"}


# ──────────────────────────────────────────────
# Dispatch / Router
# ──────────────────────────────────────────────

STRATEGY_MAP: Dict[str, callable] = {
    "trend_follow": signal_trend_follow,
    "mean_reversion": signal_mean_reversion,
    "adaptive_trend": signal_adaptive_trend,
    "hybrid_regime": signal_hybrid_regime,
    "orderflow_delta": signal_orderflow_delta,
    "scalping_grid": signal_scalping_grid,
    "funding_arbitrage": signal_funding_arbitrage,
}


def get_signal(df: pd.DataFrame, cfg: BotConfig) -> SignalWithParams:
    """
    Obtém sinal da estratégia configurada.

    Antes de executar, verifica se a estratégia está em
    cfg.enabled_strategies. Estratégias fora dessa lista (ex:
    mean_reversion, orderflow_delta, funding_arbitrage — sem
    validação walk-forward suficiente, ver auditoria item 9)
    retornam hold sem executar a lógica de sinal.

    Args:
        df: DataFrame com indicadores.
        cfg: Configuração do bot.

    Returns:
        Tuple[Signal, Dict]: Sinal + parâmetros.
    """
    try:
        strategy_name = cfg.strategy.value

        # NOVO: bloqueio por enabled_strategies
        if not cfg.is_strategy_enabled():
            log.warning(
                f"[SIGNAL] Estratégia '{strategy_name}' não está em "
                f"enabled_strategies ('{cfg.enabled_strategies}') — "
                f"retornando hold. Habilite explicitamente no .env "
                f"apenas após validação walk-forward."
            )
            return "hold", {"reason": f"estrategia_desabilitada: {strategy_name}"}

        strategy_func = STRATEGY_MAP.get(strategy_name)

        if strategy_func is None:
            log.error(f"Estratégia desconhecida: {strategy_name}")
            return "hold", {"reason": f"estrategia_desconhecida: {strategy_name}"}

        log.info(f"Executando estratégia: {strategy_name}")
        signal, params = strategy_func(df, cfg)
        log.info(f"Sinal: {signal.upper()} | Motivo: {params.get('reason', 'N/A')}")

        return signal, params

    except Exception as e:
        log.error(f"Erro ao executar estratégia {cfg.strategy}: {e}")
        return "hold", {"reason": f"erro_interno: {e}"}