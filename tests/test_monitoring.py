"""
Testes do Módulo de Monitoramento — Hyperliquid Production Bot v3.0
====================================================================
"""

from __future__ import annotations

import time

import pytest

from crypto_bot_core.monitoring import Monitor


class TestMonitor:
    def test_init(self) -> None:
        m = Monitor()
        assert m.metrics.total_trades == 0
        assert m.metrics.uptime_seconds == 0.0

    def test_record_step(self) -> None:
        m = Monitor()
        m.record_step(success=True)
        assert m.metrics.steps_completed == 1
        assert m.metrics.steps_failed == 0
        assert m.metrics.last_step_ts > 0

    def test_record_step_failed(self) -> None:
        m = Monitor()
        m.record_step(success=False)
        assert m.metrics.steps_completed == 1
        assert m.metrics.steps_failed == 1

    def test_record_trade_profit(self) -> None:
        m = Monitor()
        m.record_trade(100.0)
        assert m.metrics.total_trades == 1
        assert m.metrics.winning_trades == 1
        assert m.metrics.total_pnl == 100.0

    def test_record_trade_loss(self) -> None:
        m = Monitor()
        m.record_trade(-50.0)
        assert m.metrics.total_trades == 1
        assert m.metrics.losing_trades == 1
        assert m.metrics.total_pnl == -50.0

    def test_record_error(self) -> None:
        m = Monitor()
        m.record_error("test error")
        assert m.metrics.last_error == "test error"

    def test_update_balance(self) -> None:
        m = Monitor()
        m.update_balance(10000)
        assert m.metrics.balance == 10000
        assert m.metrics.peak_balance == 10000

    def test_update_balance_drawdown(self) -> None:
        m = Monitor()
        m.update_balance(10000)
        m.update_balance(8000)
        assert m.metrics.drawdown_pct == 0.20

    def test_health_check_healthy(self) -> None:
        m = Monitor()
        m.record_step(success=True)
        m.metrics.last_step_ts = time.time()
        health = m.health_check()
        assert health["status"] == "healthy"

    def test_health_check_warning(self) -> None:
        m = Monitor()
        m.record_step(success=True)
        m.metrics.last_step_ts = time.time() - 400  # >5 min
        health = m.health_check()
        assert health["status"] == "warning"

    def test_get_metrics(self) -> None:
        m = Monitor()
        m.record_trade(100)
        m.record_trade(-50)
        m.record_step(success=True)
        m.update_balance(10000)
        metrics = m.get_metrics()
        assert metrics["trading"]["total_trades"] == 2
        assert metrics["trading"]["win_rate_pct"] == 50.0
        assert metrics["capital"]["balance"] == 10000.0

    def test_to_dict(self) -> None:
        m = Monitor()
        d = m.to_dict()
        assert "metrics" in d
        assert "health" in d
