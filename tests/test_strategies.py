"""
Testes do Módulo de Estratégias — Hyperliquid Production Bot v3.0
=================================================================
Testa todas as 7 estratégias, funções utilitárias e o router.

Requisitos:
- Type hints
- Tratamento de exceções
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.config import BotConfig, StrategyType, Timeframe
from crypto_bot_core.strategies.signals import (
    STRATEGY_MAP,
    _crossunder,
    _crossover,
    _get_latest,
    _get_prev,
    _validate_df,
    get_signal,
    signal_adaptive_trend,
    signal_funding_arbitrage,
    signal_hybrid_regime,
    signal_mean_reversion,
    signal_orderflow_delta,
    signal_scalping_grid,
    signal_trend_follow,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame de exemplo com todos os indicadores."""
    np.random.seed(42)
    n = 100
    close = np.random.randn(n).cumsum() + 100
    return pd.DataFrame(
        {
            "close": close,
            "high": close + np.random.rand(n) * 2,
            "low": close - np.random.rand(n) * 2,
            "volume": np.random.rand(n) * 1000,
            "ema_fast": close + np.random.randn(n) * 0.5,
            "ema_slow": close + np.random.randn(n) * 1.0,
            "ema_trend": close + np.random.randn(n) * 2.0,
            "ema_50": close + np.random.randn(n) * 1.5,
            "ema_200": close + np.random.randn(n) * 3.0,
            "rsi": np.random.rand(n) * 100,
            "adx": np.random.rand(n) * 50,
            "bb_upper": close + 2 * np.std(close),
            "bb_lower": close - 2 * np.std(close),
            "bb_middle": close,
            "bb_pct": np.random.rand(n),
            "vp_vwap": close + np.random.randn(n) * 0.3,
            "delta": np.random.randn(n) * 10,
            "cvd": np.random.randn(n).cumsum(),
            "cvd_ma": np.random.randn(n).cumsum() * 0.9,
            "of_bearish_div": np.zeros(n),
            "of_bullish_div": np.zeros(n),
            "of_absorption": np.zeros(n),
            "funding_rate": np.random.randn(n) * 0.005,
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """DataFrame vazio."""
    return pd.DataFrame()


@pytest.fixture
def mock_cfg() -> MagicMock:
    """Mock da configuração do bot."""
    cfg = MagicMock(spec=BotConfig)
    cfg.strategy = StrategyType.TREND_FOLLOW
    cfg.timeframe = Timeframe.H1
    cfg.get_strategy_params.return_value = {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "vwap_std": 2.0,
        "smc_swing_lookback": 10,
        "divergence_lookback": 10,
        "absorption_threshold": 1.5,
        "grid_levels": 5,
        "grid_spread_pct": 0.1,
        "grid_size_pct": 20.0,
        "min_funding_rate": 0.0001,
        "max_funding_rate": 0.01,
    }
    return cfg


# ──────────────────────────────────────────────
# Testes: _validate_df
# ──────────────────────────────────────────────


class TestValidateDF:
    """Testes para _validate_df."""

    def test_valid_df(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar True para DataFrame válido."""
        assert _validate_df(sample_df) is True

    def test_valid_df_with_required_cols(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar True quando colunas requeridas existem."""
        assert _validate_df(sample_df, ["close", "volume"]) is True

    def test_empty_df(self, empty_df: pd.DataFrame) -> None:
        """Deve retornar False para DataFrame vazio."""
        assert _validate_df(empty_df) is False

    def test_none_df(self) -> None:
        """Deve retornar False para None."""
        assert _validate_df(None) is False  # type: ignore

    def test_missing_required_cols(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar False quando colunas requeridas faltam."""
        assert _validate_df(sample_df, ["coluna_inexistente"]) is False


# ──────────────────────────────────────────────
# Testes: _get_latest
# ──────────────────────────────────────────────


class TestGetLatest:
    """Testes para _get_latest."""

    def test_get_latest_valid(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar o último valor de uma coluna."""
        expected = float(sample_df["close"].iloc[-1])
        assert _get_latest(sample_df, "close") == expected

    def test_get_latest_missing_col(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar default para coluna inexistente."""
        assert _get_latest(sample_df, "nao_existe", 42.0) == 42.0

    def test_get_latest_nan(self) -> None:
        """Deve retornar default para valor NaN."""
        df = pd.DataFrame({"x": [1.0, np.nan]})
        assert _get_latest(df, "x") == 0.0

    def test_get_latest_inf(self) -> None:
        """Deve retornar default para valor Inf."""
        df = pd.DataFrame({"x": [1.0, np.inf]})
        assert _get_latest(df, "x") == 0.0


# ──────────────────────────────────────────────
# Testes: _get_prev
# ──────────────────────────────────────────────


class TestGetPrev:
    """Testes para _get_prev."""

    def test_get_prev_valid(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar o valor anterior de uma coluna."""
        expected = float(sample_df["close"].iloc[-2])
        assert _get_prev(sample_df, "close") == expected

    def test_get_prev_offset_maior_que_df(self) -> None:
        """Deve retornar default quando offset > tamanho do df."""
        df = pd.DataFrame({"x": [1.0]})
        assert _get_prev(df, "x", offset=5) == 0.0

    def test_get_prev_missing_col(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar default para coluna inexistente."""
        assert _get_prev(sample_df, "nao_existe", default=-1.0) == -1.0


# ──────────────────────────────────────────────
# Testes: _crossover
# ──────────────────────────────────────────────


class TestCrossover:
    """Testes para _crossover."""

    def test_crossover_true(self) -> None:
        """Deve retornar True quando fast cruza acima de slow."""
        df = pd.DataFrame({"fast": [1, 2, 3], "slow": [2, 2.5, 2.8]})
        assert _crossover(df, "fast", "slow") is True

    def test_crossover_false(self) -> None:
        """Deve retornar False quando não há cruzamento."""
        df = pd.DataFrame({"fast": [1, 2, 3], "slow": [0.5, 1.5, 2.5]})
        assert _crossover(df, "fast", "slow") is False

    def test_crossover_missing_col(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar False para coluna inexistente."""
        assert _crossover(sample_df, "nao_existe", "close") is False


# ──────────────────────────────────────────────
# Testes: _crossunder
# ──────────────────────────────────────────────


class TestCrossunder:
    """Testes para _crossunder."""

    def test_crossunder_true(self) -> None:
        """Deve retornar True quando fast cruza abaixo de slow."""
        # fast: [3, 2.6, 1] -> prev_fast=2.6, curr_fast=1
        # slow: [2, 2.5, 2.8] -> prev_slow=2.5, curr_slow=2.8
        # prev_fast(2.6) >= prev_slow(2.5) AND curr_fast(1) < curr_slow(2.8) = True
        df = pd.DataFrame({"fast": [3, 2.6, 1], "slow": [2, 2.5, 2.8]})
        assert _crossunder(df, "fast", "slow") is True

    def test_crossunder_false(self) -> None:
        """Deve retornar False quando não há cruzamento."""
        df = pd.DataFrame({"fast": [3, 2, 1], "slow": [3.5, 2.5, 1.5]})
        assert _crossunder(df, "fast", "slow") is False

    def test_crossunder_missing_col(self, sample_df: pd.DataFrame) -> None:
        """Deve retornar False para coluna inexistente."""
        assert _crossunder(sample_df, "nao_existe", "close") is False


# ──────────────────────────────────────────────
# Testes: signal_trend_follow
# ──────────────────────────────────────────────


class TestSignalTrendFollow:
    """Testes para signal_trend_follow."""

    def test_buy_signal(self, mock_cfg: MagicMock) -> None:
        """Deve gerar sinal de compra quando condições são atendidas."""
        df = pd.DataFrame(
            {
                "ema_fast": [100, 101, 102],
                "ema_slow": [99, 100, 100],
                "ema_trend": [98, 99, 99],
                "rsi": [50, 55, 60],
                "adx": [20, 25, 30],
                "close": [100, 101, 103],
            }
        )
        signal, params = signal_trend_follow(df, mock_cfg)
        assert signal == "buy"
        assert "trend_follow_buy" in params.get("reason", "")

    def test_sell_signal(self, mock_cfg: MagicMock) -> None:
        """Deve gerar sinal de venda quando condições são atendidas."""
        df = pd.DataFrame(
            {
                "ema_fast": [102, 101, 100],
                "ema_slow": [100, 100, 101],
                "ema_trend": [99, 100, 102],
                "rsi": [60, 55, 50],
                "adx": [20, 25, 30],
                "close": [102, 101, 99],
            }
        )
        signal, params = signal_trend_follow(df, mock_cfg)
        assert signal == "sell"
        assert "trend_follow_sell" in params.get("reason", "")

    def test_hold_insufficient_data(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando faltam colunas."""
        df = pd.DataFrame({"close": [100]})
        signal, params = signal_trend_follow(df, mock_cfg)
        assert signal == "hold"
        assert "dados_insuficientes" in params.get("reason", "")

    def test_hold_no_signal(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando não há sinal claro."""
        df = pd.DataFrame(
            {
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "ema_trend": [100, 100, 100],
                "rsi": [50, 50, 50],
                "adx": [10, 10, 10],
                "close": [100, 100, 100],
            }
        )
        signal, params = signal_trend_follow(df, mock_cfg)
        assert signal == "hold"

    def test_exception_handling(self, mock_cfg: MagicMock) -> None:
        """Deve tratar exceções internas."""
        signal, params = signal_trend_follow(None, mock_cfg)  # type: ignore
        assert signal == "hold"


# ──────────────────────────────────────────────
# Testes: signal_mean_reversion
# ──────────────────────────────────────────────


class TestSignalMeanReversion:
    """Testes para signal_mean_reversion."""

    def test_buy_signal(self, mock_cfg: MagicMock) -> None:
        """Deve gerar sinal de compra em oversold."""
        df = pd.DataFrame(
            {
                "close": [90, 88, 85],
                "bb_upper": [110, 110, 110],
                "bb_lower": [90, 88, 86],
                "bb_middle": [100, 100, 100],
                "rsi": [35, 32, 28],
            }
        )
        signal, params = signal_mean_reversion(df, mock_cfg)
        assert signal == "buy"
        assert "mean_reversion_buy" in params.get("reason", "")

    def test_sell_signal(self, mock_cfg: MagicMock) -> None:
        """Deve gerar sinal de venda em overbought."""
        df = pd.DataFrame(
            {
                "close": [110, 112, 115],
                "bb_upper": [110, 112, 114],
                "bb_lower": [90, 90, 90],
                "bb_middle": [100, 100, 100],
                "rsi": [65, 68, 72],
            }
        )
        signal, params = signal_mean_reversion(df, mock_cfg)
        assert signal == "sell"
        assert "mean_reversion_sell" in params.get("reason", "")

    def test_hold_within_bands(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando preço está dentro das bandas."""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "bb_upper": [110, 110, 110],
                "bb_lower": [90, 90, 90],
                "bb_middle": [100, 100, 100],
                "rsi": [50, 52, 55],
                "bb_pct": [0.5, 0.55, 0.6],
            }
        )
        signal, params = signal_mean_reversion(df, mock_cfg)
        assert signal == "hold"
        assert "dentro_das_bandas" in params.get("reason", "")

    def test_hold_insufficient_data(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando faltam colunas."""
        df = pd.DataFrame({"close": [100]})
        signal, params = signal_mean_reversion(df, mock_cfg)
        assert signal == "hold"
        assert "dados_insuficientes" in params.get("reason", "")


# ──────────────────────────────────────────────
# Testes: signal_adaptive_trend
# ──────────────────────────────────────────────


class TestSignalAdaptiveTrend:
    """Testes para signal_adaptive_trend."""

    def test_trend_buy(self, mock_cfg: MagicMock) -> None:
        """Deve comprar em modo trend com ADX alto."""
        df = pd.DataFrame(
            {
                "ema_fast": [100, 101, 102],
                "ema_slow": [99, 100, 100],
                "adx": [30, 32, 35],
                "rsi": [50, 55, 60],
            }
        )
        signal, params = signal_adaptive_trend(df, mock_cfg)
        assert signal == "buy"
        assert params.get("mode") == "trend"

    def test_trend_sell(self, mock_cfg: MagicMock) -> None:
        """Deve vender em modo trend com ADX alto."""
        df = pd.DataFrame(
            {
                "ema_fast": [102, 101, 100],
                "ema_slow": [100, 100, 101],
                "adx": [30, 32, 35],
                "rsi": [60, 55, 50],
            }
        )
        signal, params = signal_adaptive_trend(df, mock_cfg)
        assert signal == "sell"
        assert params.get("mode") == "trend"

    def test_range_buy(self, mock_cfg: MagicMock) -> None:
        """Deve comprar em modo range com RSI baixo."""
        df = pd.DataFrame(
            {
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "adx": [15, 14, 13],
                "rsi": [35, 32, 28],
            }
        )
        signal, params = signal_adaptive_trend(df, mock_cfg)
        assert signal == "buy"
        assert params.get("mode") == "range"

    def test_range_sell(self, mock_cfg: MagicMock) -> None:
        """Deve vender em modo range com RSI alto."""
        df = pd.DataFrame(
            {
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "adx": [15, 14, 13],
                "rsi": [65, 68, 72],
            }
        )
        signal, params = signal_adaptive_trend(df, mock_cfg)
        assert signal == "sell"
        assert params.get("mode") == "range"

    def test_hold_no_signal(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando não há sinal."""
        df = pd.DataFrame(
            {
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "adx": [22, 22, 22],
                "rsi": [50, 50, 50],
            }
        )
        signal, params = signal_adaptive_trend(df, mock_cfg)
        assert signal == "hold"


# ──────────────────────────────────────────────
# Testes: signal_hybrid_regime
# ──────────────────────────────────────────────


class TestSignalHybridRegime:
    """Testes para signal_hybrid_regime."""

    def test_bull_buy(self, mock_cfg: MagicMock) -> None:
        """Deve comprar em regime bull com confluência."""
        # Precisa de 20+ linhas para swing_lookback=10 funcionar
        n = 25
        close = [100 + i * 0.5 for i in range(n)]
        high = [v + 1 for v in close]
        low = [v - 1 for v in close]
        df = pd.DataFrame(
            {
                "close": close,
                "high": high,
                "low": low,
                "ema_50": [100 + i * 0.3 for i in range(n)],
                "ema_200": [100 + i * 0.1 for i in range(n)],
                "vp_vwap": [v + 3 for v in close],  # vwap acima do preço
                "rsi": [50 + i * 0.5 for i in range(n)],
            }
        )
        signal, params = signal_hybrid_regime(df, mock_cfg)
        assert signal == "buy"
        assert params.get("regime") == "bull"

    def test_bear_sell(self, mock_cfg: MagicMock) -> None:
        """Deve vender em regime bear com confluência."""
        n = 25
        close = [100 - i * 0.5 for i in range(n)]
        high = [v + 1 for v in close]
        low = [v - 1 for v in close]
        df = pd.DataFrame(
            {
                "close": close,
                "high": high,
                "low": low,
                "ema_50": [100 - i * 0.3 for i in range(n)],
                "ema_200": [100 - i * 0.1 for i in range(n)],
                "vp_vwap": [v - 3 for v in close],  # vwap abaixo do preço
                "rsi": [50 - i * 0.5 for i in range(n)],
            }
        )
        signal, params = signal_hybrid_regime(df, mock_cfg)
        assert signal == "sell"
        assert params.get("regime") == "bear"

    def test_hold_sideways(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold em regime sideways."""
        df = pd.DataFrame(
            {
                "close": [100, 100, 100],
                "high": [101, 101, 101],
                "low": [99, 99, 99],
                "ema_50": [100, 100, 100],
                "ema_200": [100, 100, 100],
                "vp_vwap": [100, 100, 100],
                "rsi": [50, 50, 50],
            }
        )
        signal, params = signal_hybrid_regime(df, mock_cfg)
        assert signal == "hold"

    def test_hold_insufficient_data(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando faltam colunas."""
        df = pd.DataFrame({"close": [100]})
        signal, params = signal_hybrid_regime(df, mock_cfg)
        assert signal == "hold"
        assert "dados_insuficientes" in params.get("reason", "")


# ──────────────────────────────────────────────
# Testes: signal_orderflow_delta
# ──────────────────────────────────────────────


class TestSignalOrderFlow:
    """Testes para signal_orderflow_delta."""

    def test_buy_delta_positive(self, mock_cfg: MagicMock) -> None:
        """Deve comprar com delta positivo e divergência bullish."""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                "delta": [1, 2, 3],
                "cvd": [10, 12, 15],
                "cvd_ma": [9, 10, 11],
                "of_bearish_div": [0, 0, 0],
                "of_bullish_div": [0, 0, 1],
                "of_absorption": [0, 0, 0],
            }
        )
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "buy"
        assert "orderflow_buy" in params.get("reason", "")

    def test_sell_delta_negative(self, mock_cfg: MagicMock) -> None:
        """Deve vender com delta negativo e divergência bearish."""
        df = pd.DataFrame(
            {
                "close": [102, 101, 100],
                "volume": [1000, 1100, 1200],
                "delta": [-1, -2, -3],
                "cvd": [15, 12, 10],
                "cvd_ma": [16, 14, 13],
                "of_bearish_div": [0, 0, 1],
                "of_bullish_div": [0, 0, 0],
                "of_absorption": [0, 0, 0],
            }
        )
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "sell"
        assert "orderflow_sell" in params.get("reason", "")

    def test_absorption_buy(self, mock_cfg: MagicMock) -> None:
        """Deve comprar com absorção e delta positivo."""
        df = pd.DataFrame(
            {
                "close": [100, 100, 100],
                "volume": [1000, 2000, 3000],
                "delta": [0.5, 0.5, 0.5],
                "cvd": [10, 10, 10],
                "cvd_ma": [10, 10, 10],
                "of_bearish_div": [0, 0, 0],
                "of_bullish_div": [0, 0, 0],
                "of_absorption": [0, 0, 1],
            }
        )
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "buy"
        assert "absorption" in params.get("reason", "")

    def test_hold_no_signal(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold sem sinais claros."""
        df = pd.DataFrame(
            {
                "close": [100, 100, 100],
                "volume": [1000, 1000, 1000],
                "delta": [0, 0, 0],
                "cvd": [10, 10, 10],
                "cvd_ma": [10, 10, 10],
                "of_bearish_div": [0, 0, 0],
                "of_bullish_div": [0, 0, 0],
                "of_absorption": [0, 0, 0],
            }
        )
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "hold"


# ──────────────────────────────────────────────
# Testes: signal_scalping_grid
# ──────────────────────────────────────────────


class TestSignalScalpingGrid:
    """Testes para signal_scalping_grid."""

    def test_buy(self, mock_cfg: MagicMock) -> None:
        """Deve comprar quando preço está no nível de compra do grid."""
        # Corrigido: código agora usa margem de 1% (1.01)
        # close=99.5, grid_spread_pct=0.1, grid_step=99.5*0.001=0.0995
        # buy_levels[0] = 99.5 - 0.0995 = 99.4005
        # buy_levels[0]*1.01 = 99.4005*1.01 = 100.3945
        # close(99.5) <= 100.3945 -> True!
        # rsi=44 < 45 -> True
        df = pd.DataFrame(
            {
                "close": [100, 99.9, 99.5],
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "rsi": [50, 48, 44],
            }
        )
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "buy", f"Esperado buy, obtido {signal}. Params: {params}"
        assert "scalping_grid_buy" in params.get("reason", "")

    def test_sell(self, mock_cfg: MagicMock) -> None:
        """Deve vender quando preço está no nível de venda do grid."""
        df = pd.DataFrame(
            {
                "close": [100, 100.1, 100.2],
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "rsi": [50, 52, 56],
            }
        )
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "sell"
        assert "scalping_grid_sell" in params.get("reason", "")

    def test_hold_outside_grid(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando preço está fora do grid."""
        df = pd.DataFrame(
            {
                "close": [100, 100, 100],
                "ema_fast": [100, 100, 100],
                "ema_slow": [100, 100, 100],
                "rsi": [50, 50, 50],
            }
        )
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "hold"
        assert "fora_do_grid" in params.get("reason", "")


# ──────────────────────────────────────────────
# Testes: signal_funding_arbitrage
# ──────────────────────────────────────────────


class TestSignalFundingArbitrage:
    """Testes para signal_funding_arbitrage."""

    def test_sell_high_funding(self, mock_cfg: MagicMock) -> None:
        """Deve vender quando funding está muito positivo."""
        df = pd.DataFrame({"close": [100, 101, 102], "funding_rate": [0.005, 0.008, 0.015]})
        signal, params = signal_funding_arbitrage(df, mock_cfg)
        assert signal == "sell"
        assert "funding_arb_sell" in params.get("reason", "")

    def test_buy_low_funding(self, mock_cfg: MagicMock) -> None:
        """Deve comprar quando funding está muito negativo."""
        df = pd.DataFrame({"close": [100, 101, 102], "funding_rate": [-0.005, -0.008, -0.015]})
        signal, params = signal_funding_arbitrage(df, mock_cfg)
        assert signal == "buy"
        assert "funding_arb_buy" in params.get("reason", "")

    def test_sell_moderate_positive(self, mock_cfg: MagicMock) -> None:
        """Deve vender com funding moderadamente positivo."""
        df = pd.DataFrame({"close": [100, 101, 102], "funding_rate": [0.0005, 0.0008, 0.001]})
        signal, params = signal_funding_arbitrage(df, mock_cfg)
        assert signal == "sell"
        assert "moderate" in params.get("strength", "")

    def test_buy_moderate_negative(self, mock_cfg: MagicMock) -> None:
        """Deve comprar com funding moderadamente negativo."""
        df = pd.DataFrame({"close": [100, 101, 102], "funding_rate": [-0.0005, -0.0008, -0.001]})
        signal, params = signal_funding_arbitrage(df, mock_cfg)
        assert signal == "buy"
        assert "moderate" in params.get("strength", "")

    def test_hold_neutral(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold com funding neutro (entre -min_funding_rate e +min_funding_rate)."""
        # funding_rate=0.0 é tratado como "indisponível" pela função
        # Usar um valor entre -min_funding_rate e +min_funding_rate (excluindo 0)
        df = pd.DataFrame({"close": [100, 101, 102], "funding_rate": [0.00005, 0.00005, 0.00005]})
        signal, params = signal_funding_arbitrage(df, mock_cfg)
        assert signal == "hold"
        assert "funding_neutro" in params.get("reason", "")

    def test_hold_no_funding_col(self, mock_cfg: MagicMock) -> None:
        """Deve retornar hold quando não há coluna funding_rate."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        signal, params = signal_funding_arbitrage(df, mock_cfg)
        assert signal == "hold"
        assert "funding_rate_indisponivel" in params.get("reason", "")


# ──────────────────────────────────────────────
# Testes: Router (STRATEGY_MAP + get_signal)
# ──────────────────────────────────────────────


class TestStrategyMap:
    """Testes para STRATEGY_MAP."""

    def test_all_strategies_mapped(self) -> None:
        """Todas as 7 estratégias devem estar mapeadas."""
        expected = [
            "trend_follow",
            "mean_reversion",
            "adaptive_trend",
            "hybrid_regime",
            "orderflow_delta",
            "scalping_grid",
            "funding_arbitrage",
        ]
        for name in expected:
            assert name in STRATEGY_MAP, f"Estratégia {name} não encontrada em STRATEGY_MAP"
            assert callable(STRATEGY_MAP[name]), f"Estratégia {name} não é callable"

    def test_strategy_map_functions(self) -> None:
        """Cada função mapeada deve ser a correta."""
        assert STRATEGY_MAP["trend_follow"] is signal_trend_follow
        assert STRATEGY_MAP["mean_reversion"] is signal_mean_reversion
        assert STRATEGY_MAP["adaptive_trend"] is signal_adaptive_trend
        assert STRATEGY_MAP["hybrid_regime"] is signal_hybrid_regime
        assert STRATEGY_MAP["orderflow_delta"] is signal_orderflow_delta
        assert STRATEGY_MAP["scalping_grid"] is signal_scalping_grid
        assert STRATEGY_MAP["funding_arbitrage"] is signal_funding_arbitrage


class TestGetSignal:
    """Testes para get_signal."""

    def test_get_signal_trend_follow(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para trend_follow corretamente."""
        mock_cfg.strategy = StrategyType.TREND_FOLLOW
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_mean_reversion(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para mean_reversion corretamente."""
        mock_cfg.strategy = StrategyType.MEAN_REVERSION
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_adaptive_trend(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para adaptive_trend corretamente."""
        mock_cfg.strategy = StrategyType.ADAPTIVE_TREND
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_hybrid_regime(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para hybrid_regime corretamente."""
        mock_cfg.strategy = StrategyType.HYBRID_REGIME
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_orderflow(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para orderflow_delta corretamente."""
        mock_cfg.strategy = StrategyType.ORDERFLOW_DELTA
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_scalping_grid(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para scalping_grid corretamente."""
        mock_cfg.strategy = StrategyType.SCALPING_GRID
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_funding_arb(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve rotear para funding_arbitrage corretamente."""
        mock_cfg.strategy = StrategyType.FUNDING_ARBITRAGE
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal in ("buy", "sell", "hold")
        assert isinstance(params, dict)

    def test_get_signal_unknown_strategy(self, sample_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        """Deve retornar hold para estratégia desconhecida."""
        # Mock strategy como objeto sem value para simular estratégia inválida
        mock_cfg.strategy = MagicMock()
        mock_cfg.strategy.value = "estrategia_inexistente"
        signal, params = get_signal(sample_df, mock_cfg)
        assert signal == "hold"
        assert "estrategia_desconhecida" in params.get("reason", "")

    def test_get_signal_exception(self, mock_cfg: MagicMock) -> None:
        """Deve tratar exceção no get_signal."""
        # Força exceção fazendo cfg.strategy.value lançar erro
        mock_cfg.strategy = None
        signal, params = get_signal(pd.DataFrame({"close": [100]}), mock_cfg)  # type: ignore
        assert signal == "hold"
        assert "erro_interno" in params.get("reason", "")
