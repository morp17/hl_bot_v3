"""
Testes do Módulo de Execução — Hyperliquid Production Bot v3.0
===============================================================
Testa OrderExecutor com mocks do connector Hyperliquid.

Requisitos:
- Type hints
- Tratamento de exceções
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from crypto_bot_core.execution import (
    OrderExecutor,
    _build_limit_order,
    _build_trigger_order,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def mock_cfg() -> MagicMock:
    """Mock da configuração."""
    cfg = MagicMock()
    cfg.hyperliquid_private_key = "0x" + "ab" * 32
    cfg.hyperliquid_account_address = "0x" + "cd" * 20
    cfg.sandbox = True
    return cfg


@pytest.fixture
def executor(mock_cfg: MagicMock) -> OrderExecutor:
    """OrderExecutor com connector mockado."""
    with patch(
        "crypto_bot_core.execution.get_connector"
    ) as mock_get_connector:
        mock_connector = MagicMock()
        # Mock _get_sz_decimals para retornar valor inteiro
        mock_connector._get_sz_decimals.return_value = 5
        mock_get_connector.return_value = mock_connector
        exe = OrderExecutor(mock_cfg)
        exe._connector = mock_connector
        yield exe


# ──────────────────────────────────────────────
# Testes: _build_limit_order
# ──────────────────────────────────────────────


class TestBuildLimitOrder:
    """Testes para _build_limit_order."""

    def test_basic_limit_order(self) -> None:
        """Deve construir payload de ordem limit básica."""
        order = _build_limit_order("BTC", True, 0.1, 50000.0)
        assert order["coin"] == "BTC"
        assert order["is_buy"] is True
        assert order["sz"] == 0.1
        assert order["limit_px"] == 50000.0
        assert order["order_type"]["limit"]["tif"] == "Gtc"
        assert order["reduce_only"] is False

    def test_reduce_only(self) -> None:
        """Deve construir ordem reduce_only."""
        order = _build_limit_order("ETH", False, 1.0, 3000.0, reduce_only=True)
        assert order["reduce_only"] is True

    def test_with_cloid(self) -> None:
        """Deve incluir cloid quando fornecido."""
        order = _build_limit_order("BTC", True, 0.1, 50000.0, cloid="test_cloid")
        assert order["cloid"] == "test_cloid"


# ──────────────────────────────────────────────
# Testes: _build_trigger_order
# ──────────────────────────────────────────────


class TestBuildTriggerOrder:
    """Testes para _build_trigger_order."""

    def test_tp_order(self) -> None:
        """Deve construir payload de take profit."""
        order = _build_trigger_order("BTC", False, 0.1, 52000.0, 51000.0, "tp")
        assert order["coin"] == "BTC"
        assert order["is_buy"] is False
        assert order["order_type"]["trigger"]["tpsl"] == "tp"
        assert order["order_type"]["trigger"]["triggerPx"] == 51000.0
        assert order["order_type"]["trigger"]["isMarket"] is True
        assert order["reduce_only"] is True

    def test_sl_order(self) -> None:
        """Deve construir payload de stop loss."""
        order = _build_trigger_order("ETH", True, 1.0, 2900.0, 2950.0, "sl")
        assert order["order_type"]["trigger"]["tpsl"] == "sl"
        assert order["order_type"]["trigger"]["triggerPx"] == 2950.0

    def test_market_false(self) -> None:
        """Deve usar limit quando is_market=False."""
        order = _build_trigger_order("BTC", False, 0.1, 52000.0, 51000.0, "tp", is_market=False)
        assert order["order_type"]["trigger"]["isMarket"] is False


# ──────────────────────────────────────────────
# Testes: OrderExecutor
# ──────────────────────────────────────────────


class TestOrderExecutorInit:
    """Testes de inicialização do OrderExecutor."""

    def test_init(self, mock_cfg: MagicMock) -> None:
        """Deve inicializar com configuração."""
        exe = OrderExecutor(mock_cfg)
        assert exe.cfg is mock_cfg
        assert exe._connector is None


class TestOrderExecutorPlaceOrder:
    """Testes para OrderExecutor.place_order."""

    def test_place_buy(self, executor: OrderExecutor) -> None:
        """Deve enviar ordem de compra."""
        executor.connector.place_order.return_value = {"oid": "12345", "status": "resting"}
        result = executor.place_order("BTC/USDC", "buy", 0.1, 50000.0)
        assert result is not None
        assert result["oid"] == "12345"

    def test_place_sell(self, executor: OrderExecutor) -> None:
        """Deve enviar ordem de venda."""
        executor.connector.place_order.return_value = {"oid": "67890", "status": "resting"}
        result = executor.place_order("ETH/USDC", "sell", 1.0, 3000.0)
        assert result is not None
        assert result["oid"] == "67890"

    def test_invalid_side(self, executor: OrderExecutor) -> None:
        """Deve rejeitar side inválido."""
        result = executor.place_order("BTC/USDC", "hold", 0.1, 50000.0)
        assert result is None

    def test_zero_qty(self, executor: OrderExecutor) -> None:
        """Deve rejeitar qty zero."""
        result = executor.place_order("BTC/USDC", "buy", 0, 50000.0)
        assert result is None

    def test_zero_price(self, executor: OrderExecutor) -> None:
        """Deve rejeitar price zero."""
        result = executor.place_order("BTC/USDC", "buy", 0.1, 0)
        assert result is None

    def test_connector_failure(self, executor: OrderExecutor) -> None:
        """Deve tratar falha do connector."""
        executor.connector.place_order.return_value = None
        result = executor.place_order("BTC/USDC", "buy", 0.1, 50000.0)
        assert result is None


class TestOrderExecutorPlaceTpsl:
    """Testes para OrderExecutor.place_tpsl_order."""

    def test_place_tp(self, executor: OrderExecutor) -> None:
        """Deve enviar ordem de take profit."""
        executor.connector.place_tpsl_order.return_value = {"oid": "tp_123"}
        result = executor.place_tpsl_order(
            "BTC/USDC", "sell", 0.1, 52000.0, 51000.0, "tp"
        )
        assert result is not None
        assert result["oid"] == "tp_123"

    def test_place_sl(self, executor: OrderExecutor) -> None:
        """Deve enviar ordem de stop loss."""
        executor.connector.place_tpsl_order.return_value = {"oid": "sl_456"}
        result = executor.place_tpsl_order(
            "BTC/USDC", "buy", 0.1, 48000.0, 49000.0, "sl"
        )
        assert result is not None
        assert result["oid"] == "sl_456"

    def test_invalid_tpsl(self, executor: OrderExecutor) -> None:
        """Deve rejeitar tpsl inválido."""
        result = executor.place_tpsl_order(
            "BTC/USDC", "sell", 0.1, 52000.0, 51000.0, "invalid"
        )
        assert result is None

    def test_zero_trigger(self, executor: OrderExecutor) -> None:
        """Deve rejeitar trigger_px zero."""
        result = executor.place_tpsl_order(
            "BTC/USDC", "sell", 0.1, 52000.0, 0, "tp"
        )
        assert result is None


class TestOrderExecutorBulkTpsl:
    """Testes para OrderExecutor.place_bulk_tpsl."""

    def test_bulk_success(self, executor: OrderExecutor) -> None:
        """Deve enviar bulk TP/SL com sucesso."""
        executor.connector.place_bulk_tpsl.return_value = [
            {"oid": "entry_1"},
            {"oid": "tp_1"},
            {"oid": "sl_1"},
        ]
        results = executor.place_bulk_tpsl(
            "BTC/USDC", "buy", 0.1, 50000.0, 52000.0, 48000.0
        )
        assert len(results) == 3
        assert results[0]["oid"] == "entry_1"

    def test_bulk_invalid_side(self, executor: OrderExecutor) -> None:
        """Deve rejeitar side inválido no bulk."""
        results = executor.place_bulk_tpsl(
            "BTC/USDC", "hold", 0.1, 50000.0, 52000.0, 48000.0
        )
        assert results is None

    def test_bulk_zero_qty(self, executor: OrderExecutor) -> None:
        """Deve rejeitar qty zero no bulk."""
        results = executor.place_bulk_tpsl(
            "BTC/USDC", "buy", 0, 50000.0, 52000.0, 48000.0
        )
        assert results is None


class TestOrderExecutorCancel:
    """Testes para OrderExecutor.cancel_order."""

    def test_cancel_success(self, executor: OrderExecutor) -> None:
        """Deve cancelar ordem com sucesso."""
        executor.connector.cancel_order.return_value = True
        assert executor.cancel_order("BTC/USDC", "12345") is True

    def test_cancel_failure(self, executor: OrderExecutor) -> None:
        """Deve tratar falha no cancelamento."""
        executor.connector.cancel_order.return_value = False
        assert executor.cancel_order("BTC/USDC", "12345") is False

    def test_cancel_empty_oid(self, executor: OrderExecutor) -> None:
        """Deve rejeitar oid vazio."""
        assert executor.cancel_order("BTC/USDC", "") is False


class TestOrderExecutorCancelAll:
    """Testes para OrderExecutor.cancel_all_orders."""

    def test_cancel_all(self, executor: OrderExecutor) -> None:
        """Deve cancelar todas as ordens."""
        executor.connector.cancel_all_orders.return_value = 3
        assert executor.cancel_all_orders("BTC/USDC") == 3

    def test_cancel_all_zero(self, executor: OrderExecutor) -> None:
        """Deve retornar 0 quando não há ordens."""
        executor.connector.cancel_all_orders.return_value = 0
        assert executor.cancel_all_orders("BTC/USDC") == 0


class TestOrderExecutorLeverage:
    """Testes para OrderExecutor.update_leverage."""

    def test_update_leverage(self, executor: OrderExecutor) -> None:
        """Deve atualizar alavancagem."""
        executor.connector.update_leverage.return_value = True
        assert executor.update_leverage("BTC/USDC", 5) is True

    def test_leverage_too_low(self, executor: OrderExecutor) -> None:
        """Deve rejeitar alavancagem < 1."""
        assert executor.update_leverage("BTC/USDC", 0) is False

    def test_leverage_too_high(self, executor: OrderExecutor) -> None:
        """Deve rejeitar alavancagem > 50."""
        assert executor.update_leverage("BTC/USDC", 100) is False


class TestOrderExecutorClosePosition:
    """Testes para OrderExecutor.close_position."""

    def test_close_long(self, executor: OrderExecutor) -> None:
        """Deve fechar posição long."""
        executor.connector.close_position.return_value = {"oid": "close_1"}
        result = executor.close_position("BTC/USDC", 0.1, "buy")
        assert result is not None
        assert result["status"] == "closed"

    def test_close_zero_qty(self, executor: OrderExecutor) -> None:
        """Deve rejeitar qty zero."""
        assert executor.close_position("BTC/USDC", 0, "buy") is None

    def test_close_invalid_side(self, executor: OrderExecutor) -> None:
        """Deve rejeitar side inválido."""
        assert executor.close_position("BTC/USDC", 0.1, "hold") is None
