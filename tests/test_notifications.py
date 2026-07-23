"""
Testes do Módulo de Notificações — Hyperliquid Production Bot v3.0
===================================================================
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crypto_bot_core.notifications import Notificator


@pytest.fixture
def notificator() -> Notificator:
    return Notificator(
        telegram_token="test_token",
        telegram_chat_id="12345",
        discord_webhook="https://discord.com/api/webhooks/test",
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_user="user",
        smtp_pass="pass",
        email_from="bot@test.com",
        email_to="admin@test.com",
    )


class TestNotificator:
    def test_send_telegram(self, notificator: Notificator) -> None:
        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            assert notificator.send_telegram("test") is True

    def test_send_telegram_not_configured(self) -> None:
        n = Notificator()
        assert n.send_telegram("test") is False

    def test_send_discord(self, notificator: Notificator) -> None:
        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            assert notificator.send_discord("test") is True

    def test_send_discord_not_configured(self) -> None:
        n = Notificator()
        assert n.send_discord("test") is False

    def test_send_email(self, notificator: Notificator) -> None:
        with patch("smtplib.SMTP") as mock_smtp:
            mock_instance = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_instance
            assert notificator.send_email("subject", "body") is True

    def test_send_email_not_configured(self) -> None:
        n = Notificator()
        assert n.send_email("subject", "body") is False

    def test_send_trade_alert(self, notificator: Notificator) -> None:
        with patch.object(notificator, "send_telegram") as mock_tg:
            with patch.object(notificator, "send_discord") as mock_dc:
                notificator.send_trade_alert("open", "BTC/USDC", "buy", 0.1, 50000)
                assert mock_tg.called
                assert mock_dc.called

    def test_send_error(self, notificator: Notificator) -> None:
        with patch.object(notificator, "send_telegram") as mock_tg:
            with patch.object(notificator, "send_discord") as mock_dc:
                with patch.object(notificator, "send_email") as mock_em:
                    notificator.send_error("test error", "test_context")
                    assert mock_tg.called
                    assert mock_dc.called
                    assert mock_em.called
