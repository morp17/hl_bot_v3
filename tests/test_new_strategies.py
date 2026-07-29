"""
Testes para as novas estratégias — volatility_squeeze e
funding_weighted_trend (item 4 da segunda rodada de auditoria).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.config import BotConfig, StrategyType, Timeframe
from crypto_bot_core.strategies.signals import (
    STRATEGY_MAP,
    signal_funding_weighted_trend,
    signal_volatility_squeeze,
)


@pytest.fixture
def mock_cfg_squeeze() -> MagicMock:
    cfg = MagicMock(spec=BotConfig)
    cfg.strategy = StrategyType.VOLATILITY_SQUEEZE
    cfg.timeframe = Timeframe.H1
    cfg.get_strategy_params.return_value = {
        "squeeze_lookback": 50, "squeeze_quantile": 0.25, "volume_confirm_mult": 1.2,
    }
    return cfg


@pytest.fixture
def mock_cfg_funding() -> MagicMock:
    cfg = MagicMock(spec=BotConfig)
    cfg.strategy = StrategyType.FUNDING_WEIGHTED_TREND
    cfg.timeframe = Timeframe.H1
    cfg.get_strategy_params.return_value = {
        "min_funding_rate": 0.0001, "max_funding_rate": 0.01,
        "rsi_exhaustion_high": 75, "rsi_exhaustion_low": 25,
    }
    return cfg


# ──────────────────────────────────────────────
# volatility_squeeze
# ──────────────────────────────────────────────


class TestVolatilitySqueezeStrategyMapRegistration:
    def test_registered_in_strategy_map(self) -> None:
        assert "volatility_squeeze" in STRATEGY_MAP
        assert STRATEGY_MAP["volatility_squeeze"] is signal_volatility_squeeze


class TestVolatilitySqueezeInsufficientData:
    def test_missing_columns_returns_hold(self, mock_cfg_squeeze: MagicMock) -> None:
        df = pd.DataFrame({"close": [100, 101, 102]})
        signal, params = signal_volatility_squeeze(df, mock_cfg_squeeze)
        assert signal == "hold"
        assert params["reason"] == "dados_insuficientes"
        assert params["confidence"] == 0.0

    def test_too_few_rows_returns_hold(self, mock_cfg_squeeze: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100] * 10,
            "bb_width": [0.02] * 10,
            "bb_upper": [102] * 10,
            "bb_lower": [98] * 10,
        })
        signal, params = signal_volatility_squeeze(df, mock_cfg_squeeze)
        assert signal == "hold"
        assert params["reason"] == "dados_insuficientes"


class TestVolatilitySqueezeBreakout:
    def _build_squeeze_then_breakout_df(self, direction: str = "up") -> pd.DataFrame:
        """
        Constrói um DataFrame sintético: 50 barras de bb_width alto
        (sem squeeze), seguido de 5 barras de squeeze (bb_width baixo),
        seguido de 1 barra de rompimento com volume alto.
        """
        n_wide = 50
        n_squeeze = 5
        wide_width = np.full(n_wide, 5.0)
        squeeze_width = np.full(n_squeeze, 0.5)  # bem mais estreito -> percentil baixo
        bb_width = np.concatenate([wide_width, squeeze_width])

        close = np.full(len(bb_width), 100.0)
        bb_upper = np.full(len(bb_width), 102.0)
        bb_lower = np.full(len(bb_width), 98.0)
        volume = np.full(len(bb_width), 1000.0)

        if direction == "up":
            close = np.append(close, 103.0)  # rompe acima de bb_upper
        else:
            close = np.append(close, 97.0)  # rompe abaixo de bb_lower
        bb_upper = np.append(bb_upper, 102.0)
        bb_lower = np.append(bb_lower, 98.0)
        bb_width = np.append(bb_width, 0.5)
        volume = np.append(volume, 5000.0)  # volume bem acima da média (confirmação)

        return pd.DataFrame({
            "close": close, "bb_upper": bb_upper, "bb_lower": bb_lower,
            "bb_width": bb_width, "volume": volume,
        })

    def test_breakout_up_with_squeeze_and_volume(self, mock_cfg_squeeze: MagicMock) -> None:
        df = self._build_squeeze_then_breakout_df("up")
        signal, params = signal_volatility_squeeze(df, mock_cfg_squeeze)
        assert signal == "buy"
        assert params["reason"] == "volatility_squeeze_breakout_up"
        assert 0.0 < params["confidence"] <= 1.0
        assert params["volume_confirmed"] is True

    def test_breakout_down_with_squeeze_and_volume(self, mock_cfg_squeeze: MagicMock) -> None:
        df = self._build_squeeze_then_breakout_df("down")
        signal, params = signal_volatility_squeeze(df, mock_cfg_squeeze)
        assert signal == "sell"
        assert params["reason"] == "volatility_squeeze_breakout_down"

    def test_no_squeeze_no_breakout_signal(self, mock_cfg_squeeze: MagicMock) -> None:
        """Sem squeeze recente (bb_width sempre alto), mesmo com rompimento
        de preço, não deve gerar sinal."""
        n = 55
        df = pd.DataFrame({
            "close": [100.0] * (n - 1) + [103.0],
            "bb_upper": [102.0] * n,
            "bb_lower": [98.0] * n,
            "bb_width": [5.0] * n,  # nunca estreita -> sem squeeze
            "volume": [1000.0] * (n - 1) + [5000.0],
        })
        signal, params = signal_volatility_squeeze(df, mock_cfg_squeeze)
        assert signal == "hold"
        assert params["reason"] == "sem_squeeze_recente"

    def test_squeeze_without_volume_confirmation_holds(self, mock_cfg_squeeze: MagicMock) -> None:
        df = self._build_squeeze_then_breakout_df("up")
        # Substitui o volume do rompimento por um valor baixo (sem confirmação)
        df.loc[df.index[-1], "volume"] = 100.0
        signal, params = signal_volatility_squeeze(df, mock_cfg_squeeze)
        assert signal == "hold"
        assert params["reason"] == "squeeze_sem_rompimento_confirmado"

    def test_exception_handling(self, mock_cfg_squeeze: MagicMock) -> None:
        signal, params = signal_volatility_squeeze(None, mock_cfg_squeeze)  # type: ignore
        assert signal == "hold"
        assert params["confidence"] == 0.0


# ──────────────────────────────────────────────
# funding_weighted_trend
# ──────────────────────────────────────────────


class TestFundingWeightedTrendStrategyMapRegistration:
    def test_registered_in_strategy_map(self) -> None:
        assert "funding_weighted_trend" in STRATEGY_MAP
        assert STRATEGY_MAP["funding_weighted_trend"] is signal_funding_weighted_trend


class TestFundingWeightedTrendInsufficientData:
    def test_missing_columns_returns_hold(self, mock_cfg_funding: MagicMock) -> None:
        df = pd.DataFrame({"close": [100, 101, 102]})
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "hold"
        assert params["reason"] == "dados_insuficientes"

    def test_no_funding_rate_returns_hold(self, mock_cfg_funding: MagicMock) -> None:
        df = pd.DataFrame({
            "ema_fast": [100, 101, 102], "ema_slow": [99, 100, 100], "rsi": [50, 55, 60],
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "hold"
        assert params["reason"] == "funding_rate_indisponivel"


class TestFundingWeightedTrendCarryFavorable:
    """Casos 'carry favorável' — segue tendência quando funding não é extremo."""

    def test_uptrend_with_healthy_positive_funding_buys(self, mock_cfg_funding: MagicMock) -> None:
        df = pd.DataFrame({
            "ema_fast": [100, 101, 102], "ema_slow": [99, 100, 100],
            "rsi": [50, 55, 60], "funding_rate": [0.0005, 0.0005, 0.0005],  # bem abaixo de max=0.01
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "buy"
        assert params["reason"] == "funding_weighted_trend_follow_buy"
        assert params["mode"] == "carry_favoravel"

    def test_downtrend_with_healthy_negative_funding_sells(self, mock_cfg_funding: MagicMock) -> None:
        df = pd.DataFrame({
            "ema_fast": [102, 101, 100], "ema_slow": [100, 100, 101],
            "rsi": [60, 55, 50], "funding_rate": [-0.0005, -0.0005, -0.0005],
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "sell"
        assert params["mode"] == "carry_favoravel"

    def test_uptrend_but_funding_already_extreme_does_not_follow(self, mock_cfg_funding: MagicMock) -> None:
        """Funding já no limite máximo (carry caro) não deve gerar buy de
        'carry favorável' — só entraria pelo caminho de exaustão (RSI
        extremo), que não é o caso aqui (RSI=60)."""
        df = pd.DataFrame({
            "ema_fast": [100, 101, 102], "ema_slow": [99, 100, 100],
            "rsi": [50, 55, 60], "funding_rate": [0.02, 0.02, 0.02],  # acima de max=0.01
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "hold"


class TestFundingWeightedTrendExhaustionReversal:
    """Casos de exaustão — reversão seletiva apenas com funding extremo + RSI esticado."""

    def test_extreme_positive_funding_with_overbought_rsi_sells(self, mock_cfg_funding: MagicMock) -> None:
        df = pd.DataFrame({
            "ema_fast": [100, 101, 102], "ema_slow": [99, 100, 100],
            "rsi": [70, 78, 82], "funding_rate": [0.015, 0.02, 0.025],  # extremo, acima de max=0.01
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "sell"
        assert params["reason"] == "funding_weighted_exhaustion_sell"
        assert params["mode"] == "exaustao_reversao"

    def test_extreme_negative_funding_with_oversold_rsi_buys(self, mock_cfg_funding: MagicMock) -> None:
        df = pd.DataFrame({
            "ema_fast": [102, 101, 100], "ema_slow": [100, 100, 101],
            "rsi": [30, 22, 18], "funding_rate": [-0.015, -0.02, -0.025],
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "buy"
        assert params["reason"] == "funding_weighted_exhaustion_buy"

    def test_extreme_funding_without_rsi_exhaustion_holds(self, mock_cfg_funding: MagicMock) -> None:
        """Funding extremo sozinho, sem RSI no extremo oposto, não deve
        gerar reversão automática (diferença chave vs. funding_arbitrage puro)."""
        df = pd.DataFrame({
            "ema_fast": [100, 101, 102], "ema_slow": [99, 100, 100],
            "rsi": [50, 55, 60], "funding_rate": [0.02, 0.02, 0.02],
        })
        signal, params = signal_funding_weighted_trend(df, mock_cfg_funding)
        assert signal == "hold"
        assert params["reason"] == "sem_alinhamento_funding_tendencia"

    def test_exception_handling(self, mock_cfg_funding: MagicMock) -> None:
        signal, params = signal_funding_weighted_trend(None, mock_cfg_funding)  # type: ignore
        assert signal == "hold"
        assert params["confidence"] == 0.0