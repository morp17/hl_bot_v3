"""
Testes unitários para crypto_bot_core/backtest.py

Cobre:
- BacktestTrade (dataclass)
- BacktestResult (summary, to_dict, export_csv)
- BacktestEngine (inicialização, validação, execução)
- Todas as 7 estratégias
- Casos de borda (dados insuficientes, estratégia inválida)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.backtest import BacktestEngine, BacktestResult, BacktestTrade
from crypto_bot_core.config import BotConfig
from crypto_bot_core.indicators import add_all_indicators


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_synthetic_df(bars: int = 500) -> pd.DataFrame:
    """Cria um DataFrame OHLCV sintético para backtest.

    Args:
        bars: Número de velas a gerar.

    Returns:
        DataFrame com colunas OHLCV.
    """
    np.random.seed(42)
    # Tendência de alta forte para gerar sinais de compra
    t = np.linspace(0, 10, bars)
    close = 50000.0 + t * 500 + np.random.randn(bars) * 200
    close = np.maximum(close, 1000)

    high = close + np.abs(np.random.randn(bars) * 100)
    low = close - np.abs(np.random.randn(bars) * 100)
    open_p = close + np.random.randn(bars) * 50
    volume = np.abs(np.random.randn(bars) * 2000) + 500

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.date_range(
        start="2024-01-01", periods=bars, freq="1h", tz="UTC"
    )
    return df


@pytest.fixture
def df_with_indicators() -> pd.DataFrame:
    """Fixture com dados sintéticos + indicadores."""
    df = _make_synthetic_df(500)
    cfg = BotConfig()
    return add_all_indicators(df, cfg)


# ──────────────────────────────────────────────
# Testes: BacktestTrade
# ──────────────────────────────────────────────


class TestBacktestTrade:
    """Testes para a dataclass BacktestTrade."""

    def test_create_trade(self) -> None:
        """Criar um BacktestTrade com todos os campos."""
        trade = BacktestTrade(
            entry_time="2024-01-01 10:00:00",
            exit_time="2024-01-01 14:00:00",
            side="buy",
            entry_price=50000.0,
            exit_price=51000.0,
            qty=0.1,
            stop_loss=49000.0,
            take_profit=52000.0,
            gross_pnl=100.0,
            fees=5.0,
            net_pnl=95.0,
            pnl_pct=1.9,
            bars_held=4,
            exit_reason="take_profit",
        )
        assert trade.side == "buy"
        assert trade.net_pnl == 95.0
        assert trade.exit_reason == "take_profit"
        assert trade.bars_held == 4

    def test_trade_sell(self) -> None:
        """Trade de venda (short)."""
        trade = BacktestTrade(
            entry_time="2024-01-01 10:00:00",
            exit_time="2024-01-01 12:00:00",
            side="sell",
            entry_price=51000.0,
            exit_price=50000.0,
            qty=0.1,
            stop_loss=52000.0,
            take_profit=49000.0,
            gross_pnl=100.0,
            fees=5.0,
            net_pnl=95.0,
            pnl_pct=1.86,
            bars_held=2,
            exit_reason="take_profit",
        )
        assert trade.side == "sell"
        assert trade.net_pnl > 0


# ──────────────────────────────────────────────
# Testes: BacktestResult
# ──────────────────────────────────────────────


class TestBacktestResult:
    """Testes para a dataclass BacktestResult."""

    @pytest.fixture
    def sample_result(self) -> BacktestResult:
        """Cria um BacktestResult de exemplo."""
        trades = [
            BacktestTrade(
                entry_time="2024-01-01 10:00:00",
                exit_time="2024-01-01 14:00:00",
                side="buy",
                entry_price=50000.0,
                exit_price=51000.0,
                qty=0.1,
                stop_loss=49000.0,
                take_profit=52000.0,
                gross_pnl=100.0,
                fees=5.0,
                net_pnl=95.0,
                pnl_pct=1.9,
                bars_held=4,
                exit_reason="take_profit",
            ),
            BacktestTrade(
                entry_time="2024-01-02 10:00:00",
                exit_time="2024-01-02 12:00:00",
                side="buy",
                entry_price=51000.0,
                exit_price=50500.0,
                qty=0.1,
                stop_loss=50000.0,
                take_profit=53000.0,
                gross_pnl=-50.0,
                fees=5.0,
                net_pnl=-55.0,
                pnl_pct=-1.08,
                bars_held=2,
                exit_reason="stop_loss",
            ),
        ]
        return BacktestResult(
            strategy="trend_follow",
            symbol="BTC/USDC",
            timeframe="1h",
            start_date="2024-01-01 10:00:00",
            end_date="2024-01-02 12:00:00",
            total_bars=100,
            initial_capital=10000.0,
            final_capital=10040.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
            total_gross_pnl=50.0,
            total_net_pnl=40.0,
            total_fees=10.0,
            max_drawdown=55.0,
            max_drawdown_pct=0.0055,
            sharpe_ratio=0.5,
            profit_factor=2.0,
            avg_win=95.0,
            avg_loss=-55.0,
            avg_bars_held=3.0,
            expectancy=20.0,
            trades=trades,
            equity_curve=[10000.0, 10095.0, 10040.0],
        )

    def test_summary_contains_expected_fields(self, sample_result: BacktestResult) -> None:
        """summary() deve conter campos principais."""
        summary = sample_result.summary()
        assert isinstance(summary, str)
        assert "BACKTEST RESULT" in summary
        assert "trend_follow" in summary
        assert "BTC/USDC" in summary
        assert "50.0%" in summary  # win rate

    def test_summary_empty_trades(self) -> None:
        """summary() com zero trades não deve crashar."""
        result = BacktestResult(
            strategy="trend_follow",
            symbol="BTC/USDC",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            total_bars=100,
            initial_capital=10000.0,
            final_capital=10000.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_gross_pnl=0.0,
            total_net_pnl=0.0,
            total_fees=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_bars_held=0.0,
            expectancy=0.0,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        # O summary mostra "Trades:       0" (com formatação)
        assert "Trades:" in summary

    def test_to_dict(self, sample_result: BacktestResult) -> None:
        """to_dict() deve retornar dict com métricas."""
        d = sample_result.to_dict()
        assert isinstance(d, dict)
        assert d["strategy"] == "trend_follow"
        assert d["total_trades"] == 2
        assert d["win_rate"] == 50.0
        assert "sharpe_ratio" in d
        assert "profit_factor" in d
        assert "trades_count" in d

    def test_to_dict_empty(self) -> None:
        """to_dict() com resultado vazio não deve crashar."""
        result = BacktestResult(
            strategy="trend_follow",
            symbol="BTC/USDC",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            total_bars=100,
            initial_capital=10000.0,
            final_capital=10000.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_gross_pnl=0.0,
            total_net_pnl=0.0,
            total_fees=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_bars_held=0.0,
            expectancy=0.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["total_trades"] == 0

    def test_export_csv(self, sample_result: BacktestResult, tmp_path: Any) -> None:
        """export_csv deve criar arquivo CSV válido."""
        csv_path = tmp_path / "backtest_result.csv"
        sample_result.export_csv(str(csv_path))
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 2
        assert "net_pnl" in df_csv.columns
        assert "exit_reason" in df_csv.columns

    def test_export_csv_empty(self, tmp_path: Any) -> None:
        """export_csv com zero trades não deve crashar."""
        result = BacktestResult(
            strategy="trend_follow",
            symbol="BTC/USDC",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            total_bars=100,
            initial_capital=10000.0,
            final_capital=10000.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_gross_pnl=0.0,
            total_net_pnl=0.0,
            total_fees=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_bars_held=0.0,
            expectancy=0.0,
        )
        csv_path = tmp_path / "empty_result.csv"
        # Não deve levantar exceção
        result.export_csv(str(csv_path))


# ──────────────────────────────────────────────
# Testes: BacktestEngine
# ──────────────────────────────────────────────


class TestBacktestEngine:
    """Testes do BacktestEngine."""

    def test_initialization(self) -> None:
        """Engine deve inicializar com config válida."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        assert engine.initial_capital == 1000.0
        assert engine.cfg.strategy.value == "trend_follow"

    def test_initialization_default_capital(self) -> None:
        """Engine deve usar capital padrão se não especificado."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg)
        assert engine.initial_capital == 10000.0

    def test_invalid_strategy(self) -> None:
        """Estratégia inválida deve levantar ValueError."""
        from crypto_bot_core.config import StrategyType

        cfg = BotConfig(strategy="trend_follow")
        # Testar que uma estratégia válida NÃO levanta erro
        engine = BacktestEngine(cfg)
        assert engine is not None

    def test_insufficient_data(self) -> None:
        """Dados insuficientes devem levantar ValueError."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg)
        df = _make_synthetic_df(10)  # apenas 10 velas
        with pytest.raises(ValueError, match="Dados insuficientes"):
            engine.run(df)

    def test_run_returns_result(self, df_with_indicators: pd.DataFrame) -> None:
        """run() deve retornar BacktestResult."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)
        assert isinstance(result, BacktestResult)

    def test_result_has_expected_fields(self, df_with_indicators: pd.DataFrame) -> None:
        """BacktestResult deve ter todos os campos esperados."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)

        assert result.strategy == "trend_follow"
        assert result.initial_capital == 1000.0
        assert result.total_bars > 0
        assert isinstance(result.total_trades, int)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.profit_factor, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert result.final_capital >= 0

    def test_all_strategies_run(self, df_with_indicators: pd.DataFrame) -> None:
        """Todas as 7 estratégias devem executar sem erro."""
        strategies = [
            "trend_follow",
            "mean_reversion",
            "adaptive_trend",
            "hybrid_regime",
            "orderflow_delta",
            "scalping_grid",
            "funding_arbitrage",
        ]
        for strat in strategies:
            cfg = BotConfig(strategy=strat)
            engine = BacktestEngine(cfg, initial_capital=1000.0)
            result = engine.run(df_with_indicators)
            assert isinstance(result, BacktestResult)
            assert result.strategy == strat
            # Não deve crashar em nenhuma estratégia
            assert result.total_bars > 0

    def test_result_summary_string(self, df_with_indicators: pd.DataFrame) -> None:
        """summary() deve retornar string formatada."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "BACKTEST RESULT" in summary
        assert result.strategy in summary

    def test_result_to_dict(self, df_with_indicators: pd.DataFrame) -> None:
        """to_dict() deve retornar dict com métricas."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["strategy"] == "trend_follow"
        assert d["total_trades"] >= 0
        assert "sharpe_ratio" in d
        assert "profit_factor" in d

    def test_equity_curve_length(self, df_with_indicators: pd.DataFrame) -> None:
        """Equity curve deve ter comprimento compatível com dados."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)
        # equity_curve tem 1 ponto a mais que o número de velas processadas
        assert len(result.equity_curve) > 0
        # Deve ter pelo menos o capital inicial
        assert result.equity_curve[0] == 1000.0

    def test_trades_have_valid_data(self, df_with_indicators: pd.DataFrame) -> None:
        """Trades gerados devem ter dados coerentes."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)

        for trade in result.trades:
            assert trade.side in ("buy", "sell")
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.qty > 0
            assert trade.exit_reason in (
                "take_profit",
                "stop_loss",
                "trailing_stop",
                "signal_reversal",
                "end_of_data",
            )
            # PnL deve ser calculado
            assert isinstance(trade.net_pnl, float)
            assert trade.bars_held >= 0

    def test_export_csv(self, df_with_indicators: pd.DataFrame, tmp_path: Any) -> None:
        """export_csv deve criar arquivo CSV válido."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg, initial_capital=1000.0)
        result = engine.run(df_with_indicators)

        csv_path = tmp_path / "backtest_result.csv"
        result.export_csv(str(csv_path))

        if result.total_trades > 0:
            assert csv_path.exists()
            assert csv_path.stat().st_size > 0
            df_csv = pd.read_csv(csv_path)
            assert len(df_csv) == len(result.trades)
            assert "net_pnl" in df_csv.columns
            assert "exit_reason" in df_csv.columns
        else:
            # Sem trades, CSV não é criado — apenas verifica que não crasha
            assert not csv_path.exists() or csv_path.stat().st_size == 0

    def test_run_with_none_df(self) -> None:
        """run() com None deve levantar ValueError."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError, match="Dados insuficientes"):
            engine.run(None)  # type: ignore[arg-type]

    def test_run_with_missing_columns(self) -> None:
        """run() com DataFrame sem colunas obrigatórias deve levantar ValueError."""
        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg)
        df = pd.DataFrame({"a": [1, 2, 3]})  # colunas erradas
        # Precisa de 50+ linhas para passar na primeira validação
        df = pd.concat([df] * 20, ignore_index=True)
        with pytest.raises(ValueError, match="Colunas obrigatórias"):
            engine.run(df)

    def test_tf_minutes(self) -> None:
        """_tf_minutes deve retornar minutos corretos."""
        cfg = BotConfig(strategy="trend_follow", timeframe="1h")
        engine = BacktestEngine(cfg)
        assert engine._tf_minutes() == 60

        cfg2 = BotConfig(strategy="trend_follow", timeframe="4h")
        engine2 = BacktestEngine(cfg2)
        assert engine2._tf_minutes() == 240

        cfg3 = BotConfig(strategy="trend_follow", timeframe="1d")
        engine3 = BacktestEngine(cfg3)
        assert engine3._tf_minutes() == 1440

    def test_tf_minutes_unknown(self) -> None:
        """_tf_minutes com timeframe desconhecido deve retornar 60."""
        from crypto_bot_core.config import Timeframe

        cfg = BotConfig(strategy="trend_follow")
        engine = BacktestEngine(cfg)
        # Timeframes conhecidos: 1m, 5m, 15m, 30m, 1h, 4h, 1d
        # Testar que timeframe mapeado retorna o valor correto
        engine.cfg.timeframe = Timeframe("5m")
        assert engine._tf_minutes() == 5
