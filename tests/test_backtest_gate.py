"""
Testes para o bypass de enabled_strategies no BacktestEngine
(crypto_bot_core/backtest.py) — confirma que o backtest, por padrão,
permite validar estratégias ainda não habilitadas em produção, e que a
flag --respect-enabled-strategies (via bypass_enabled_strategies=False)
restaura o comportamento fiel ao live.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.backtest import BacktestEngine
from crypto_bot_core.config import BotConfig, StrategyType
from crypto_bot_core.indicators import add_all_indicators


def _make_synthetic_df(bars: int = 200) -> pd.DataFrame:
    """DataFrame OHLCV sintético simples para o backtest."""
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(bars) * 0.5)
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(bars) * 0.3)
    low = close - np.abs(np.random.randn(bars) * 0.3)
    open_p = close + np.random.randn(bars) * 0.1
    volume = np.abs(np.random.randn(bars) * 100) + 50

    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close, "volume": volume,
    })
    df.index = pd.date_range(start="2024-01-01", periods=bars, freq="1h", tz="UTC")
    return df


@pytest.fixture
def df_with_indicators() -> pd.DataFrame:
    df = _make_synthetic_df(200)
    cfg = BotConfig(_env_file=None)
    return add_all_indicators(df, cfg)


class TestBypassEnabledStrategiesWiring:
    """Verifica que BacktestEngine propaga enforce_enabled_gate corretamente para get_signal."""

    def test_default_bypasses_gate(self, df_with_indicators: pd.DataFrame) -> None:
        """Por padrão (sem passar bypass_enabled_strategies), o backtest
        deve chamar get_signal com enforce_enabled_gate=False."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.ORDERFLOW_DELTA)
        assert cfg.is_strategy_enabled() is False  # orderflow_delta não habilitada por padrão

        engine = BacktestEngine(cfg, initial_capital=1000.0)
        assert engine.bypass_enabled_strategies is True

        with patch(
            "crypto_bot_core.backtest.get_signal", return_value=("hold", {"confidence": 0.0})
        ) as mock_signal:
            engine.run(df_with_indicators)
            assert mock_signal.called
            _, kwargs = mock_signal.call_args
            assert kwargs.get("enforce_enabled_gate") is False

    def test_respect_flag_enforces_gate(self, df_with_indicators: pd.DataFrame) -> None:
        """Com bypass_enabled_strategies=False, deve chamar get_signal
        com enforce_enabled_gate=True (comportamento fiel ao live)."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.ORDERFLOW_DELTA)
        engine = BacktestEngine(cfg, initial_capital=1000.0, bypass_enabled_strategies=False)

        with patch(
            "crypto_bot_core.backtest.get_signal", return_value=("hold", {"confidence": 0.0})
        ) as mock_signal:
            engine.run(df_with_indicators)
            assert mock_signal.called
            _, kwargs = mock_signal.call_args
            assert kwargs.get("enforce_enabled_gate") is True

    def test_bypass_allows_trades_for_disabled_strategy(self, df_with_indicators: pd.DataFrame) -> None:
        """Com bypass=True, um sinal real de buy/sell (forçado via mock)
        deve gerar pelo menos um trade mesmo para uma estratégia fora de
        ENABLED_STRATEGIES."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.ORDERFLOW_DELTA)
        engine = BacktestEngine(cfg, initial_capital=1000.0, bypass_enabled_strategies=True)

        call_count = {"n": 0}

        def fake_signal(df, cfg, enforce_enabled_gate=True):
            call_count["n"] += 1
            if call_count["n"] == 5:
                return "buy", {"confidence": 0.9, "reason": "forced_buy"}
            return "hold", {"confidence": 0.0}

        with patch("crypto_bot_core.backtest.get_signal", side_effect=fake_signal):
            result = engine.run(df_with_indicators)

        assert result.total_trades >= 1

    def test_respect_flag_blocks_trades_for_disabled_strategy(self, df_with_indicators: pd.DataFrame) -> None:
        """Com bypass=False, usando get_signal REAL (sem mock): como a
        estratégia não está habilitada e o gate é respeitado, o resultado
        deve ser sempre hold -> 0 trades. Este é o teste que reproduz o
        comportamento original relatado no log (total_trades=0)."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.ORDERFLOW_DELTA)
        engine = BacktestEngine(cfg, initial_capital=1000.0, bypass_enabled_strategies=False)

        result = engine.run(df_with_indicators)
        assert result.total_trades == 0


class TestBacktestEngineInitDoesNotRaise:
    """Confirma que a inicialização não lança erro em nenhuma combinação
    de bypass_enabled_strategies x estratégia habilitada/desabilitada."""

    def test_init_bypass_true_strategy_disabled(self) -> None:
        cfg = BotConfig(_env_file=None, strategy=StrategyType.SCALPING_GRID)
        engine = BacktestEngine(cfg, bypass_enabled_strategies=True)
        assert engine is not None

    def test_init_bypass_false_strategy_disabled(self) -> None:
        cfg = BotConfig(_env_file=None, strategy=StrategyType.SCALPING_GRID)
        engine = BacktestEngine(cfg, bypass_enabled_strategies=False)
        assert engine is not None

    def test_init_strategy_enabled_bypass_irrelevant(self) -> None:
        """Quando a estratégia JÁ está habilitada, bypass_enabled_strategies
        não deve fazer diferença de comportamento (apenas de log)."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.TREND_FOLLOW)
        assert cfg.is_strategy_enabled() is True
        engine = BacktestEngine(cfg, bypass_enabled_strategies=True)
        assert engine is not None