"""
Testes do Módulo de Proteção de Capital — Hyperliquid Production Bot v3.0
==========================================================================
Testa os 4 níveis de proteção: filtros, drawdown, exposição, circuit breaker.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from crypto_bot_core.capital_protection import CapitalProtection


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def cp() -> CapitalProtection:
    """CapitalProtection com configuração padrão."""
    return CapitalProtection(
        initial_balance=10000.0,
        max_daily_loss_pct=0.10,
        max_drawdown_pct=0.20,
        max_consecutive_losses=5,
        cooldown_after_loss_sec=300,
        max_exposure_pct=0.50,
        max_position_pct=0.20,
        circuit_breaker_loss_pct=0.15,
        circuit_breaker_cooldown_sec=3600,
    )


# ──────────────────────────────────────────────
# Testes: Nível 1 — Filtros de Mercado
# ──────────────────────────────────────────────


class TestLevel1MarketFilters:
    """Testes para filtros de mercado (Nível 1)."""

    def test_trade_hours_no_filter(self, cp: CapitalProtection) -> None:
        """Sem filtro de horário."""
        cp.trade_hour_start_utc = 0
        cp.trade_hour_end_utc = 0
        ok, _ = cp.check_trade_hours()
        assert ok is True

    def test_trade_hours_within(self, cp: CapitalProtection) -> None:
        """Dentro do horário."""
        cp.trade_hour_start_utc = 8
        cp.trade_hour_end_utc = 22
        with patch("crypto_bot_core.capital_protection.datetime") as mock_dt:
            mock_dt.now.return_value.hour = 14
            mock_dt.now.return_value.strftime.return_value = "2024-01-01"
            ok, _ = cp.check_trade_hours()
            assert ok is True

    def test_trade_hours_outside(self, cp: CapitalProtection) -> None:
        """Fora do horário."""
        cp.trade_hour_start_utc = 8
        cp.trade_hour_end_utc = 22
        with patch("crypto_bot_core.capital_protection.datetime") as mock_dt:
            mock_dt.now.return_value.hour = 6
            mock_dt.now.return_value.strftime.return_value = "2024-01-01"
            ok, _ = cp.check_trade_hours()
            assert ok is False

    def test_spread_ok(self, cp: CapitalProtection) -> None:
        """Spread dentro do limite."""
        ok, _ = cp.check_spread(0.001)  # 0.1%
        assert ok is True

    def test_spread_high(self, cp: CapitalProtection) -> None:
        """Spread acima do limite."""
        ok, reason = cp.check_spread(0.01)  # 1%
        assert ok is False
        assert "spread" in reason

    def test_funding_ok(self, cp: CapitalProtection) -> None:
        """Funding dentro do limite."""
        ok, _ = cp.check_funding_rate(0.0005)
        assert ok is True

    def test_funding_high(self, cp: CapitalProtection) -> None:
        """Funding acima do limite."""
        ok, reason = cp.check_funding_rate(0.005)
        assert ok is False
        assert "funding" in reason


# ──────────────────────────────────────────────
# Testes: Nível 2 — Drawdown e Perda Diária
# ──────────────────────────────────────────────


class TestLevel2Drawdown:
    """Testes para drawdown e perda diária (Nível 2)."""

    def test_drawdown_ok(self, cp: CapitalProtection) -> None:
        """Drawdown dentro do limite."""
        cp.state.current_balance = 9000  # 10% drawdown
        cp.state.peak_balance = 10000
        ok, _ = cp.check_drawdown()
        assert ok is True

    def test_drawdown_exceeded(self, cp: CapitalProtection) -> None:
        """Drawdown acima do limite."""
        cp.state.current_balance = 7000  # 30% drawdown
        cp.state.peak_balance = 10000
        ok, reason = cp.check_drawdown()
        assert ok is False
        assert "drawdown" in reason

    def test_daily_loss_ok(self, cp: CapitalProtection) -> None:
        """Perda diária dentro do limite."""
        cp.state.daily_loss = -500  # 5% de perda
        cp.state.peak_balance = 10000
        ok, _ = cp.check_daily_loss()
        assert ok is True

    def test_daily_loss_exceeded(self, cp: CapitalProtection) -> None:
        """Perda diária acima do limite."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cp.state.daily_loss = -2000  # 20% de perda
        cp.state.peak_balance = 10000
        cp.state.daily_reset_day = today  # mesmo dia de hoje para evitar reset
        ok, reason = cp.check_daily_loss()
        assert ok is False
        assert "daily_loss" in reason

    def test_consecutive_losses_ok(self, cp: CapitalProtection) -> None:
        """Perdas consecutivas dentro do limite."""
        cp.state.consecutive_losses = 3
        ok, _ = cp.check_consecutive_losses()
        assert ok is True

    def test_consecutive_losses_exceeded(self, cp: CapitalProtection) -> None:
        """Perdas consecutivas acima do limite."""
        cp.state.consecutive_losses = 5
        ok, reason = cp.check_consecutive_losses()
        assert ok is False
        assert "consecutive_losses" in reason

    def test_cooldown_active(self, cp: CapitalProtection) -> None:
        """Cooldown ativo após perda."""
        cp.state.last_loss_ts = time.monotonic() - 10  # 10s atrás
        ok, reason = cp.check_cooldown()
        assert ok is False
        assert "cooldown" in reason

    def test_cooldown_expired(self, cp: CapitalProtection) -> None:
        """Cooldown expirado."""
        cp.state.last_loss_ts = time.monotonic() - 600  # 600s atrás
        ok, _ = cp.check_cooldown()
        assert ok is True


# ──────────────────────────────────────────────
# Testes: Nível 3 — Exposição
# ──────────────────────────────────────────────


class TestLevel3Exposure:
    """Testes para exposição (Nível 3)."""

    def test_exposure_ok(self, cp: CapitalProtection) -> None:
        """Exposição dentro do limite."""
        ok, _ = cp.check_exposure(
            current_positions_value=3000,  # 30%
            new_position_value=1000,  # +10% = 40%
        )
        assert ok is True

    def test_exposure_exceeded(self, cp: CapitalProtection) -> None:
        """Exposição acima do limite."""
        ok, reason = cp.check_exposure(
            current_positions_value=5000,  # 50%
            new_position_value=1000,  # +10% = 60%
        )
        assert ok is False
        assert "exposicao" in reason

    def test_position_size_ok(self, cp: CapitalProtection) -> None:
        """Tamanho da posição dentro do limite."""
        ok, _ = cp.check_exposure(
            current_positions_value=0,
            new_position_value=1500,  # 15%
        )
        assert ok is True

    def test_position_size_exceeded(self, cp: CapitalProtection) -> None:
        """Tamanho da posição acima do limite."""
        ok, reason = cp.check_exposure(
            current_positions_value=0,
            new_position_value=3000,  # 30% > 20%
        )
        assert ok is False
        assert "posicao" in reason


# ──────────────────────────────────────────────
# Testes: Nível 4 — Circuit Breaker
# ──────────────────────────────────────────────


class TestLevel4CircuitBreaker:
    """Testes para circuit breaker (Nível 4)."""

    def test_circuit_breaker_inactive(self, cp: CapitalProtection) -> None:
        """Circuit breaker inativo."""
        ok, _ = cp.check_circuit_breaker()
        assert ok is True

    def test_circuit_breaker_active(self, cp: CapitalProtection) -> None:
        """Circuit breaker ativo."""
        cp.state.is_paused = True
        cp.state.pause_until = time.time() + 3600
        cp.state.pause_reason = "test"
        ok, reason = cp.check_circuit_breaker()
        assert ok is False
        assert "circuit_breaker" in reason

    def test_circuit_breaker_expired(self, cp: CapitalProtection) -> None:
        """Circuit breaker expirado."""
        cp.state.is_paused = True
        cp.state.pause_until = time.time() - 1  # 1s atrás
        cp.state.pause_reason = "test"
        ok, _ = cp.check_circuit_breaker()
        assert ok is True
        assert cp.state.is_paused is False

    def test_trigger_circuit_breaker(self, cp: CapitalProtection) -> None:
        """Acionar circuit breaker."""
        cp.trigger_circuit_breaker("perda_20%")
        assert cp.state.is_paused is True
        assert "perda_20%" in cp.state.pause_reason
        assert cp.state.pause_until > time.time()


# ──────────────────────────────────────────────
# Testes: Record Trade
# ──────────────────────────────────────────────


class TestRecordTrade:
    """Testes para record_trade_result.

    NOTA: record_trade_result NÃO modifica current_balance ou peak_balance.
    O saldo é sincronizado exclusivamente via _sync_from_exchange() e step()
    no main.py. record_trade_result apenas atualiza estatísticas (daily_loss,
    consecutive_losses) e verifica circuit breaker.
    """

    def test_profit(self, cp: CapitalProtection) -> None:
        """Registrar trade lucrativo — não modifica saldo."""
        cp.record_trade_result(500)
        # Saldo NÃO é modificado por record_trade_result
        assert cp.state.current_balance == 10000
        assert cp.state.peak_balance == 10000
        assert cp.state.consecutive_losses == 0

    def test_loss(self, cp: CapitalProtection) -> None:
        """Registrar trade com perda — não modifica saldo."""
        cp.record_trade_result(-500)
        # Saldo NÃO é modificado por record_trade_result
        assert cp.state.current_balance == 10000
        assert cp.state.peak_balance == 10000
        # Mas estatísticas de perda são atualizadas
        assert cp.state.daily_loss == -500
        assert cp.state.consecutive_losses == 1
        assert cp.state.last_loss_ts > 0

    def test_circuit_breaker_on_loss(self, cp: CapitalProtection) -> None:
        """Perda grande deve acionar circuit breaker.

        O circuit breaker verifica (peak - current) / peak.
        Como record_trade_result não modifica current_balance,
        precisamos simular o saldo real (que seria atualizado via step()).
        """
        # Simular saldo real após perda (como faria step() da exchange)
        cp.state.current_balance = 8000  # 20% de perda do pico de 10000
        cp.record_trade_result(-2000)
        # Circuit breaker deve ter sido acionado
        assert cp.state.is_paused is True
        assert "perda" in cp.state.pause_reason


# ──────────────────────────────────────────────
# Testes: Check All
# ──────────────────────────────────────────────


class TestCheckAll:
    """Testes para check_all."""

    def test_all_ok(self, cp: CapitalProtection) -> None:
        """Todas as verificações ok."""
        ok, reasons = cp.check_all()
        assert ok is True
        assert reasons == []

    def test_some_fail(self, cp: CapitalProtection) -> None:
        """Algumas verificações falham."""
        cp.state.current_balance = 5000  # 50% drawdown
        cp.state.peak_balance = 10000
        ok, reasons = cp.check_all()
        assert ok is False
        assert len(reasons) > 0


# ──────────────────────────────────────────────
# Testes: To Dict
# ──────────────────────────────────────────────


class TestToDict:
    """Testes para to_dict."""

    def test_to_dict(self, cp: CapitalProtection) -> None:
        """Deve converter para dicionário."""
        d = cp.to_dict()
        assert "state" in d
        assert "limits" in d
        assert d["state"]["current_balance"] == 10000.0
