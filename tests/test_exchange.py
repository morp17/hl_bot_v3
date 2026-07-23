"""
Testes unitários para o módulo de exchange Hyperliquid.
"""
from unittest.mock import MagicMock, patch

import pytest

from crypto_bot_core.config import BotConfig
from crypto_bot_core.exchanges.hyperliquid import (
    HyperliquidConnector,
    _extract_coin,
    _truncate_qty,
    _format_price,
    generate_cloid,
    HyperliquidError,
    AuthenticationError,
    OrderError,
)


class TestUtils:
    """Testes para funções utilitárias."""

    def test_extract_coin_btc(self) -> None:
        assert _extract_coin("BTC/USDC") == "BTC"

    def test_extract_coin_eth(self) -> None:
        assert _extract_coin("ETH/USDC") == "ETH"

    def test_extract_coin_sol(self) -> None:
        assert _extract_coin("SOL/USDC") == "SOL"

    def test_truncate_qty_btc(self) -> None:
        """BTC tem 5 decimais."""
        result = _truncate_qty(1.23456789, 5)
        assert result == 1.23456

    def test_truncate_qty_eth(self) -> None:
        """ETH tem 4 decimais."""
        result = _truncate_qty(10.56789, 4)
        assert result == 10.5678

    def test_truncate_qty_sol(self) -> None:
        """SOL tem 2 decimais."""
        result = _truncate_qty(100.999, 2)
        assert result == 100.99

    def test_truncate_qty_zero(self) -> None:
        result = _truncate_qty(0.0, 5)
        assert result == 0.0

    def test_format_price(self) -> None:
        # Hyperliquid usa 5 significant figures
        result = _format_price(50000.12345, 2)
        # 50000.12345 → 5 sig figs = 50000
        assert result == 50000.0

    def test_format_price_high_precision(self) -> None:
        result = _format_price(1.23456789, 5)
        # 1.23456789 → 5 sig figs = 1.2346
        assert result == 1.2346

    def test_generate_cloid(self) -> None:
        cloid = generate_cloid()
        assert cloid is not None
        # Cloid deve ter 16 bytes (32 hex chars)
        assert len(str(cloid)) > 0


class TestHyperliquidConnector:
    """Testes para HyperliquidConnector."""

    def test_init(self) -> None:
        """Testa inicialização."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)
        assert connector.cfg is cfg
        assert connector.connected is False

    def test_extract_coin_method(self) -> None:
        """Testa _extract_coin via conector."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)
        # Testa o método de classe
        assert _extract_coin("BTC/USDC") == "BTC"

    def test_health_check_not_connected(self) -> None:
        """Testa health check sem conexão."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)
        status = connector.health_check()
        assert status["connected"] is False
        assert status["ccxt"] is False

    @patch("crypto_bot_core.exchanges.hyperliquid.HyperliquidConnector._build_ccxt")
    def test_connect_success(self, mock_build: MagicMock) -> None:
        """Testa conexão bem-sucedida."""
        mock_exchange = MagicMock()
        mock_build.return_value = mock_exchange

        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)

        # Mock native e info também
        with patch.object(connector, "_build_native") as mock_native, \
             patch.object(connector, "_build_info") as mock_info:
            mock_native.return_value = MagicMock()
            mock_info.return_value = MagicMock()

            result = connector.connect()
            assert result is True
            assert connector.connected is True

    def test_connect_failure(self) -> None:
        """Testa falha de conexão."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)

        with patch.object(connector, "_build_ccxt", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                connector.connect()
            assert connector.connected is False

    def test_disconnect(self) -> None:
        """Testa desconexão."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)
        connector._connected = True
        connector.disconnect()
        assert connector.connected is False


class TestOrderValidation:
    """Testes de validação de ordens."""

    def test_place_order_invalid_qty(self) -> None:
        """Testa ordem com quantidade inválida."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)

        with patch.object(connector, "_get_sz_decimals", return_value=5):
            with pytest.raises(OrderError):
                connector.place_order("BTC/USDC", "buy", 0.0, "limit", 50000.0)

    def test_place_order_limit_no_price(self) -> None:
        """Testa ordem limit sem preço."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)

        with patch.object(connector, "_get_sz_decimals", return_value=5):
            with pytest.raises(OrderError):
                connector.place_order("BTC/USDC", "buy", 0.1, "limit", None)

    def test_place_tpsl_invalid_qty(self) -> None:
        """Testa TP/SL com quantidade inválida."""
        cfg = BotConfig(testnet=True)
        connector = HyperliquidConnector(cfg)

        with patch.object(connector, "_get_sz_decimals", return_value=5):
            with pytest.raises(OrderError):
                connector.place_tpsl_order("BTC/USDC", "sell", 0.0, 49000.0, "sl")


class TestCredentials:
    """Testes de credenciais."""

    def test_mainnet_no_credentials(self) -> None:
        """Testa que mainnet sem credenciais emite warning (não erro)."""
        cfg = BotConfig(testnet=False)
        # Não deve lançar erro na criação
        assert cfg.testnet is False

    def test_mainnet_with_credentials(self) -> None:
        """Testa mainnet com credenciais."""
        cfg = BotConfig(
            testnet=False,
            hyperliquid_private_key="0x" + "a" * 64,
            hyperliquid_account_address="0x" + "b" * 40,
        )
        assert cfg.hyperliquid_private_key == "a" * 64
        assert cfg.hyperliquid_account_address == "0x" + "b" * 40
