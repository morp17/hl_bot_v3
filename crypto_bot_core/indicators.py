"""
Módulo de Indicadores Técnicos — Hyperliquid Production Bot v3.0
================================================================
Fornece todos os indicadores técnicos necessários para as estratégias:
- EMAs (fast, slow, trend)
- RSI, ATR, Bollinger Bands
- ADX, MACD, Ichimoku
- Volume Profile, CVD (Cumulative Volume Delta)
- OrderFlow: Delta, Divergence, Absorption
- Suporte multi-symbol
- Type hints, tratamento de exceções, logs estruturados
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger as log

from .config import BotConfig


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

MIN_CANDLES_REQUIRED = 50  # Mínimo de candles para indicadores confiáveis


# ──────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────


def _validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> bool:
    """
    Valida se o DataFrame tem as colunas necessárias.

    Args:
        df: DataFrame a validar.
        required_cols: Lista de colunas obrigatórias.

    Returns:
        bool: True se válido.
    """
    if df is None or df.empty:
        log.warning("DataFrame vazio ou None")
        return False

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log.warning(f"Colunas ausentes no DataFrame: {missing}")
            return False

    if len(df) < MIN_CANDLES_REQUIRED:
        log.warning(f"DataFrame muito pequeno: {len(df)} candles (mínimo {MIN_CANDLES_REQUIRED})")
        return False

    return True


def _safe_series_op(series: pd.Series, operation: str = "mean") -> float:
    """
    Executa operação segura em série pandas.

    Args:
        series: Série pandas.
        operation: "mean", "sum", "std", "min", "max"

    Returns:
        float: Resultado ou 0.0 em caso de erro.
    """
    try:
        if operation == "mean":
            return float(series.mean())
        elif operation == "sum":
            return float(series.sum())
        elif operation == "std":
            return float(series.std())
        elif operation == "min":
            return float(series.min())
        elif operation == "max":
            return float(series.max())
        return 0.0
    except Exception:
        return 0.0


def add_funding_rate(df: pd.DataFrame, funding_rate: Optional[float] = None) -> pd.DataFrame:
    """
    Adiciona a coluna funding_rate ao DataFrame.

    O funding rate não é derivável de OHLCV — deve ser buscado
    externamente (ex: HyperliquidConnector.get_mark_price() /
    Info.meta_and_asset_ctxs()) e passado como escalar aqui.

    Em backtest histórico, funding_rate por barra normalmente não
    está disponível sem uma chamada separada a funding_history();
    por padrão, se None, a coluna fica ausente e
    signal_funding_arbitrage() retorna hold (comportamento seguro,
    não gera falso sinal).

    Args:
        df: DataFrame OHLCV.
        funding_rate: Taxa de funding atual (escalar). Se None,
            a coluna não é adicionada.

    Returns:
        DataFrame com a coluna funding_rate (broadcast do escalar)
        se funding_rate foi fornecido; inalterado caso contrário.
    """
    try:
        if funding_rate is None:
            return df
        if not _validate_dataframe(df):
            return df
        df["funding_rate"] = float(funding_rate)
        return df
    except Exception as e:
        log.error(f"Erro ao adicionar funding_rate: {e}")
        return df


# ──────────────────────────────────────────────
# Indicadores Base
# ──────────────────────────────────────────────


def add_ema(df: pd.DataFrame, period: int, col_name: str = None) -> pd.DataFrame:
    """
    Adiciona EMA ao DataFrame.

    Args:
        df: DataFrame com coluna 'close'.
        period: Período da EMA.
        col_name: Nome da coluna (auto se None).

    Returns:
        DataFrame com EMA adicionada.
    """
    try:
        if not _validate_dataframe(df, ["close"]):
            return df

        name = col_name or f"ema_{period}"
        df[name] = df["close"].ewm(span=period, adjust=False).mean()
        return df
    except Exception as e:
        log.error(f"Erro ao calcular EMA {period}: {e}")
        return df


def add_rsi(df: pd.DataFrame, period: int = 14, col_name: str = "rsi") -> pd.DataFrame:
    try:
        if not _validate_dataframe(df, ["close"]):
            return df

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[col_name] = 100.0 - (100.0 / (1.0 + rs))
        # ALTERADO: quando avg_loss==0 (só ganhos), RSI correto é 100, não 50
        df[col_name] = df[col_name].where(avg_loss != 0, 100.0)
        df[col_name] = df[col_name].fillna(50.0)  # fallback para NaN genuíno (ex: warm-up)

        return df
    except Exception as e:
        log.error(f"Erro ao calcular RSI: {e}")
        if col_name not in df.columns:
            df[col_name] = 50.0
        return df


def add_atr(df: pd.DataFrame, period: int = 14, col_name: str = "atr") -> pd.DataFrame:
    """
    Adiciona ATR (Average True Range) ao DataFrame.

    Args:
        df: DataFrame com colunas 'high', 'low', 'close'.
        period: Período do ATR.
        col_name: Nome da coluna.

    Returns:
        DataFrame com ATR adicionada.
    """
    try:
        if not _validate_dataframe(df, ["high", "low", "close"]):
            return df

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[col_name] = tr.ewm(span=period, adjust=False).mean()

        return df
    except Exception as e:
        log.error(f"Erro ao calcular ATR: {e}")
        return df


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    col_prefix: str = "bb",
) -> pd.DataFrame:
    """
    Adiciona Bollinger Bands ao DataFrame.

    Args:
        df: DataFrame com coluna 'close'.
        period: Período da média móvel.
        std_dev: Número de desvios padrão.
        col_prefix: Prefixo das colunas.

    Returns:
        DataFrame com BB adicionadas.
    """
    try:
        if not _validate_dataframe(df, ["close"]):
            return df

        df[f"{col_prefix}_middle"] = df["close"].rolling(window=period).mean()
        bb_std = df["close"].rolling(window=period).std()

        df[f"{col_prefix}_upper"] = df[f"{col_prefix}_middle"] + (bb_std * std_dev)
        df[f"{col_prefix}_lower"] = df[f"{col_prefix}_middle"] - (bb_std * std_dev)
        df[f"{col_prefix}_width"] = (df[f"{col_prefix}_upper"] - df[f"{col_prefix}_lower"]) / df[f"{col_prefix}_middle"]
        df[f"{col_prefix}_pct"] = (df["close"] - df[f"{col_prefix}_lower"]) / (df[f"{col_prefix}_upper"] - df[f"{col_prefix}_lower"]).replace(0, np.nan)

        return df
    except Exception as e:
        log.error(f"Erro ao calcular Bollinger Bands: {e}")
        return df


def add_adx(df: pd.DataFrame, period: int = 14, col_name: str = "adx") -> pd.DataFrame:
    """
    Adiciona ADX (Average Directional Index) ao DataFrame.

    Args:
        df: DataFrame com colunas 'high', 'low', 'close'.
        period: Período do ADX.
        col_name: Nome da coluna.

    Returns:
        DataFrame com ADX adicionada.
    """
    try:
        if not _validate_dataframe(df, ["high", "low", "close"]):
            return df

        # True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Directional Movement
        up_move = df["high"].diff()
        down_move = -df["low"].diff()

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df[col_name] = dx.ewm(span=period, adjust=False).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df
    except Exception as e:
        log.error(f"Erro ao calcular ADX: {e}")
        return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col_prefix: str = "macd",
) -> pd.DataFrame:
    """
    Adiciona MACD ao DataFrame.

    Args:
        df: DataFrame com coluna 'close'.
        fast: Período rápido.
        slow: Período lento.
        signal: Período do sinal.
        col_prefix: Prefixo das colunas.

    Returns:
        DataFrame com MACD adicionada.
    """
    try:
        if not _validate_dataframe(df, ["close"]):
            return df

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        df[f"{col_prefix}_line"] = ema_fast - ema_slow
        df[f"{col_prefix}_signal"] = df[f"{col_prefix}_line"].ewm(span=signal, adjust=False).mean()
        df[f"{col_prefix}_histogram"] = df[f"{col_prefix}_line"] - df[f"{col_prefix}_signal"]

        return df
    except Exception as e:
        log.error(f"Erro ao calcular MACD: {e}")
        return df


def add_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Adiciona Ichimoku Cloud ao DataFrame.

    Args:
        df: DataFrame com colunas 'high', 'low', 'close'.
        tenkan_period: Período Tenkan-sen.
        kijun_period: Período Kijun-sen.
        senkou_b_period: Período Senkou Span B.
        displacement: Deslocamento.

    Returns:
        DataFrame com Ichimoku adicionada.
    """
    try:
        if not _validate_dataframe(df, ["high", "low"]):
            return df

        # Tenkan-sen (Conversion Line)
        tenkan_high = df["high"].rolling(window=tenkan_period).max()
        tenkan_low = df["low"].rolling(window=tenkan_period).min()
        df["tenkan_sen"] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = df["high"].rolling(window=kijun_period).max()
        kijun_low = df["low"].rolling(window=kijun_period).min()
        df["kijun_sen"] = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(displacement)

        # Senkou Span B (Leading Span B)
        senkou_b_high = df["high"].rolling(window=senkou_b_period).max()
        senkou_b_low = df["low"].rolling(window=senkou_b_period).min()
        df["senkou_span_b"] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)

        # Chikou Span (Lagging Span)
        df["chikou_span"] = df["close"].shift(-displacement)

        return df
    except Exception as e:
        log.error(f"Erro ao calcular Ichimoku: {e}")
        return df


# ──────────────────────────────────────────────
# Indicadores de Volume / OrderFlow
# ──────────────────────────────────────────────


def add_volume_profile(
    df: pd.DataFrame,
    period: int = 20,
    col_prefix: str = "vp",
) -> pd.DataFrame:
    """
    Adiciona Volume Profile simplificado.

    Args:
        df: DataFrame com colunas 'volume', 'close'.
        period: Período de análise.
        col_prefix: Prefixo das colunas.

    Returns:
        DataFrame com Volume Profile.
    """
    try:
        if not _validate_dataframe(df, ["volume", "close"]):
            return df

        # Volume médio
        df[f"{col_prefix}_avg"] = df["volume"].rolling(window=period).mean()

        # Volume relativo (ratio vs média)
        df[f"{col_prefix}_ratio"] = df["volume"] / df[f"{col_prefix}_avg"].replace(0, np.nan)

        # Volume acumulado
        df[f"{col_prefix}_cumsum"] = df["volume"].cumsum()

        # Preço médio ponderado por volume (VWAP)
        df[f"{col_prefix}_vwap"] = (df["close"] * df["volume"]).rolling(window=period).sum() / df["volume"].rolling(window=period).sum().replace(0, np.nan)

        return df
    except Exception as e:
        log.error(f"Erro ao calcular Volume Profile: {e}")
        return df


def add_cvd(df: pd.DataFrame, col_name: str = "cvd") -> pd.DataFrame:
    """
    Adiciona CVD (Cumulative Volume Delta) ao DataFrame.

    FIX (melhoria de qualidade — análise de orderflow_delta): a proxy
    anterior usava apenas sign(close - close.shift()) — um candle de
    alta com pavio superior grande (pressão vendedora real no
    fechamento) era contado como 100% comprador, mesmo que o
    fechamento tenha ficado perto da mínima da barra. Substituído por
    Close Location Value (CLV), técnica padrão (usada em Chaikin Money
    Flow/Accumulation-Distribution) que pondera pela POSIÇÃO do
    fechamento dentro do range [low, high] da barra, não apenas a
    direção close-a-close:

        clv = ((close - low) - (high - close)) / (high - low)
        delta = volume * clv

    clv varia de -1 (fechou na mínima, pressão totalmente vendedora)
    a +1 (fechou na máxima, pressão totalmente compradora) — captura
    a intenção real da barra melhor que um sinal binário.

    Requer 'high'/'low' além de 'volume'/'close' (diferente da versão
    anterior) — barras com high==low (sem range, ex: dados
    corrompidos/gaps) recebem delta=0 por segurança.

    Args:
        df: DataFrame com colunas 'high', 'low', 'volume', 'close'.
        col_name: Nome da coluna.

    Returns:
        DataFrame com CVD.
    """
    try:
        if not _validate_dataframe(df, ["high", "low", "volume", "close"]):
            return df

        high_low_range = (df["high"] - df["low"]).replace(0, np.nan)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low_range
        clv = clv.fillna(0.0).clip(-1.0, 1.0)

        df["delta"] = (df["volume"] * clv).fillna(0.0)

        df[col_name] = df["delta"].cumsum()
        df[f"{col_name}_ma"] = df[col_name].rolling(window=20).mean()

        return df
    except Exception as e:
        log.error(f"Erro ao calcular CVD: {e}")
        return df


def add_orderflow(
    df: pd.DataFrame,
    lookback: int = 10,
    col_prefix: str = "of",
    absorption_volume_percentile: float = 0.8,
    absorption_range_percentile: float = 0.2,
) -> pd.DataFrame:
    """
    Adiciona indicadores de OrderFlow.

    FIX (melhoria de qualidade — análise de orderflow_delta): os
    percentis de volume/range para detecção de absorção agora são
    PARÂMETROS explícitos (antes hardcoded como 0.8/0.2 direto no
    corpo da função), permitindo que get_strategy_params() de fato
    influencie o cálculo — antes, absorption_threshold em
    STRATEGY_DEFAULTS['orderflow_delta'] era lido pela estratégia mas
    nunca chegava até aqui, era um parâmetro morto.

    NOTA: absorption_threshold em STRATEGY_DEFAULTS continua sendo
    usado como fator de CONFIRMAÇÃO na estratégia (ver
    signal_orderflow_delta), não como percentil de detecção — os
    parâmetros aqui (absorption_volume_percentile/
    absorption_range_percentile) controlam a SENSIBILIDADE da
    detecção de absorção nos indicadores; absorption_threshold
    controla o quão forte o sinal precisa ser para a estratégia agir
    sobre uma absorção já detectada. São complementares, não
    duplicados.

    Args:
        df: DataFrame com colunas 'high', 'low', 'close', 'volume'.
        lookback: Período de lookback para divergências.
        col_prefix: Prefixo das colunas.
        absorption_volume_percentile: Percentil mínimo de volume
            (rank) para considerar absorção candidata.
        absorption_range_percentile: Percentil máximo de range da
            barra (quantile) para considerar absorção candidata.

    Returns:
        DataFrame com OrderFlow.
    """
    try:
        if not _validate_dataframe(df, ["high", "low", "close", "volume"]):
            return df

        if "cvd" not in df.columns:
            df = add_cvd(df)

        price_high = df["close"].rolling(window=lookback).max()
        price_low = df["close"].rolling(window=lookback).min()
        cvd_high = df["cvd"].rolling(window=lookback).max()
        cvd_low = df["cvd"].rolling(window=lookback).min()

        df[f"{col_prefix}_bearish_div"] = ((df["close"] >= price_high) & (df["cvd"] <= cvd_high)).astype(int)
        df[f"{col_prefix}_bullish_div"] = ((df["close"] <= price_low) & (df["cvd"] >= cvd_low)).astype(int)

        df[f"{col_prefix}_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df[f"{col_prefix}_volume_rank"] = df["volume"].rank(pct=True)
        df[f"{col_prefix}_absorption"] = (
            (df[f"{col_prefix}_volume_rank"] > absorption_volume_percentile)
            & (df[f"{col_prefix}_range"] < df[f"{col_prefix}_range"].quantile(absorption_range_percentile))
        ).astype(int)

        return df
    except Exception as e:
        log.error(f"Erro ao calcular OrderFlow: {e}")
        return df


# ──────────────────────────────────────────────
# Indicadores Compostos
# ──────────────────────────────────────────────


def add_all_indicators(
    df: pd.DataFrame,
    cfg: BotConfig,
    funding_rate: Optional[float] = None,   # NOVO
) -> pd.DataFrame:
    """
    Adiciona todos os indicadores necessários ao DataFrame.

    Args:
        df: DataFrame com colunas OHLCV.
        cfg: Configuração do bot (para parâmetros de estratégia).
        funding_rate: Taxa de funding atual, buscada externamente
            (ex: connector.get_mark_price / meta_and_asset_ctxs).
            Necessário apenas se a estratégia funding_arbitrage
            estiver habilitada; None preserva o comportamento
            anterior (estratégia retorna hold).

    Returns:
        DataFrame com todos os indicadores.
    """
    try:
        if not _validate_dataframe(df, ["open", "high", "low", "close", "volume"]):
            log.warning("DataFrame inválido para add_all_indicators")
            return df

        log.info(f"Calculando indicadores para {len(df)} candles...")

        # EMAs - nomes alinhados com as estratégias
        df = add_ema(df, 9, "ema_fast")
        df = add_ema(df, 21, "ema_slow")
        df = add_ema(df, 50, "ema_50")      # usado por hybrid_regime
        df = add_ema(df, 200, "ema_200")     # usado por hybrid_regime
        df["ema_trend"] = df["ema_200"]      # compatibilidade com trend_follow

        # RSI
        df = add_rsi(df, 14)

        # ATR
        df = add_atr(df, 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Bollinger Bands
        df = add_bollinger_bands(df, 20, 2.0)

        # ADX
        df = add_adx(df, 14)

        # MACD
        df = add_macd(df, 12, 26, 9)

        # Ichimoku
        df = add_ichimoku(df)

        # Volume Profile
        df = add_volume_profile(df, 20)

        # CVD e OrderFlow
        df = add_cvd(df)
        df = add_orderflow(df, 10)

        # Funding Rate (NOVO — só popula se fornecido externamente)
        df = add_funding_rate(df, funding_rate)

        log.info(f"Indicadores calculados: {len(df.columns)} colunas")
        return df

    except Exception as e:
        log.error(f"Erro ao calcular todos os indicadores: {e}")
        return df


def get_latest_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Retorna os valores mais recentes de todos os indicadores.

    Args:
        df: DataFrame com indicadores.

    Returns:
        Dict com valores atuais.
    """
    try:
        if df is None or df.empty:
            return {}

        latest = df.iloc[-1].to_dict()
        result: Dict[str, Any] = {}

        for key, value in latest.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    result[key] = 0.0
                else:
                    result[key] = float(value)
            elif isinstance(value, (pd.Timestamp,)):
                result[key] = str(value)
            else:
                try:
                    result[key] = float(value)
                except (ValueError, TypeError):
                    result[key] = 0.0

        return result
    except Exception as e:
        log.error(f"Erro ao obter valores recentes: {e}")
        return {}
