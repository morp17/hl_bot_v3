"""
Testes do Módulo de Staking — Hyperliquid Production Bot v3.0
==============================================================
Testa StakingManager com mocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crypto_bot_core.staking import StakingManager


@pytest.fixture
def mock_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.hyperliquid_private_key = "0x" + "ab" * 32
    cfg.hyperliquid_account_address = "0x" + "cd" * 20
    cfg.sandbox = True
    return cfg


@pytest.fixture
def manager(mock_cfg: MagicMock) -> StakingManager:
    with patch("crypto_bot_core.staking.get_connector") as mock_get:
        mock_connector = MagicMock()
        mock_get.return_value = mock_connector
        mgr = StakingManager(mock_cfg)
        mgr._connector = mock_connector
        yield mgr


class TestStakingManager:
    def test_delegate_stake(self, manager: StakingManager) -> None:
        manager.connector.delegate_stake.return_value = {"status": "ok"}
        result = manager.delegate_stake("0xvalidator123", 100.0)
        assert result is not None
        assert result["status"] == "delegated"

    def test_delegate_invalid_validator(self, manager: StakingManager) -> None:
        result = manager.delegate_stake("", 100.0)
        assert result is None

    def test_delegate_zero_amount(self, manager: StakingManager) -> None:
        result = manager.delegate_stake("0xvalidator", 0)
        assert result is None

    def test_undelegate_stake(self, manager: StakingManager) -> None:
        manager.connector.undelegate_stake.return_value = {"status": "ok"}
        result = manager.undelegate_stake("0xvalidator123", 50.0)
        assert result is not None
        assert result["status"] == "undelegated"

    def test_get_staking_summary(self, manager: StakingManager) -> None:
        manager.connector.get_user_staking_summary.return_value = {
            "total_staked": 1000.0,
            "rewards": 50.0,
        }
        result = manager.get_staking_summary()
        assert result is not None
        assert result["total_staked"] == 1000.0

    def test_get_staking_rewards(self, manager: StakingManager) -> None:
        manager.connector.get_user_staking_rewards.return_value = [
            {"amount": 10.0, "timestamp": 1234567890}
        ]
        result = manager.get_staking_rewards()
        assert result is not None
        assert len(result) == 1

    def test_vault_transfer_deposit(self, manager: StakingManager) -> None:
        manager.connector.vault_transfer.return_value = {"status": "ok"}
        result = manager.vault_transfer("0xvault123", 500.0, is_deposit=True)
        assert result is not None
        assert result["action"] == "deposit"

    def test_vault_transfer_withdraw(self, manager: StakingManager) -> None:
        manager.connector.vault_transfer.return_value = {"status": "ok"}
        result = manager.vault_transfer("0xvault123", 200.0, is_deposit=False)
        assert result is not None
        assert result["action"] == "withdraw"

    def test_vault_transfer_invalid_address(self, manager: StakingManager) -> None:
        result = manager.vault_transfer("", 100.0)
        assert result is None
