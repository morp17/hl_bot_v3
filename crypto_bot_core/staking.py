"""
Módulo de Staking — Hyperliquid Production Bot v3.0
====================================================
Gerencia staking de HYPE e operações com vaults.

Baseado nos exemplos do SDK:
- basic_staking.py: delegação/undelegation de stake
- basic_vault.py: transferências para vaults

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger as log

from .config import BotConfig
from .exchanges.hyperliquid import get_connector


# ──────────────────────────────────────────────
# StakingManager
# ──────────────────────────────────────────────


class StakingManager:
    """
    Gerenciador de staking HYPE e vaults.

    Fornece interface para:
    - Delegar/undelegar stake para validadores
    - Consultar resumo de staking e recompensas
    - Transferir fundos para vaults
    """

    def __init__(self, cfg: BotConfig) -> None:
        """
        Inicializa o gerenciador de staking.

        Args:
            cfg: Configuração do bot.
        """
        self.cfg = cfg
        self._connector = None

    @property
    def connector(self):
        """Obtém o connector (lazy initialization)."""
        if self._connector is None:
            self._connector = get_connector(self.cfg)
        return self._connector

    def delegate_stake(
        self, validator: str, amount_usdc: float
    ) -> Optional[Dict[str, Any]]:
        """
        Delega HYPE para um validador.

        Args:
            validator: Endereço do validador.
            amount_usdc: Quantidade em USDC a delegar.

        Returns:
            Dict com resultado ou None em caso de erro.
        """
        try:
            if not validator:
                raise ValueError("validator não pode ser vazio")
            if amount_usdc <= 0:
                raise ValueError(f"amount_usdc deve ser > 0: {amount_usdc}")

            # Converter USDC para wei (1 USDC = 10^6)
            wei_amount = int(amount_usdc * 1_000_000)

            result = self.connector.delegate_stake(validator, wei_amount)

            if result:
                log.info(
                    f"[STAKING] Delegado {amount_usdc} USDC para validador "
                    f"{validator[:10]}..."
                )
                return {"status": "delegated", "validator": validator, "amount": amount_usdc}

            log.error(f"[STAKING] Falha ao delegar para {validator}")
            return None

        except ValueError as e:
            log.error(f"[STAKING] Erro de validação: {e}")
            return None
        except Exception as e:
            log.error(f"[STAKING] Erro ao delegar stake: {e}")
            return None

    def undelegate_stake(
        self, validator: str, amount_usdc: float
    ) -> Optional[Dict[str, Any]]:
        """
        Remove delegação de HYPE de um validador.

        Args:
            validator: Endereço do validador.
            amount_usdc: Quantidade em USDC a remover.

        Returns:
            Dict com resultado ou None.
        """
        try:
            if not validator:
                raise ValueError("validator não pode ser vazio")
            if amount_usdc <= 0:
                raise ValueError(f"amount_usdc deve ser > 0: {amount_usdc}")

            wei_amount = int(amount_usdc * 1_000_000)

            result = self.connector.undelegate_stake(validator, wei_amount)

            if result:
                log.info(
                    f"[STAKING] Removido {amount_usdc} USDC do validador "
                    f"{validator[:10]}..."
                )
                return {"status": "undelegated", "validator": validator, "amount": amount_usdc}

            log.error(f"[STAKING] Falha ao remover delegação de {validator}")
            return None

        except ValueError as e:
            log.error(f"[STAKING] Erro de validação: {e}")
            return None
        except Exception as e:
            log.error(f"[STAKING] Erro ao remover stake: {e}")
            return None

    def get_staking_summary(self) -> Optional[Dict[str, Any]]:
        """
        Obtém resumo de staking do usuário.

        Returns:
            Dict com resumo ou None.
        """
        try:
            summary = self.connector.get_user_staking_summary()

            if summary:
                log.info("[STAKING] Resumo obtido com sucesso")
                return summary

            log.warning("[STAKING] Resumo indisponível")
            return None

        except Exception as e:
            log.error(f"[STAKING] Erro ao obter resumo: {e}")
            return None

    def get_staking_rewards(self) -> Optional[List[Dict[str, Any]]]:
        """
        Obtém histórico de recompensas de staking.

        Returns:
            Lista de recompensas ou None.
        """
        try:
            rewards = self.connector.get_user_staking_rewards()

            if rewards is not None:
                log.info(f"[STAKING] {len(rewards)} recompensa(s) encontrada(s)")
                return rewards

            log.warning("[STAKING] Recompensas indisponíveis")
            return None

        except Exception as e:
            log.error(f"[STAKING] Erro ao obter recompensas: {e}")
            return None

    def vault_transfer(
        self, vault_address: str, amount_usdc: float, is_deposit: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Transfere fundos para/de um vault.

        Args:
            vault_address: Endereço do vault.
            amount_usdc: Quantidade em USDC.
            is_deposit: True para depósito, False para saque.

        Returns:
            Dict com resultado ou None.
        """
        try:
            if not vault_address:
                raise ValueError("vault_address não pode ser vazio")
            if amount_usdc <= 0:
                raise ValueError(f"amount_usdc deve ser > 0: {amount_usdc}")

            # Connector espera: vault_transfer(vault_address, is_deposit, usd)
            result = self.connector.vault_transfer(
                vault_address=vault_address,
                is_deposit=is_deposit,
                usd=int(amount_usdc),
            )

            action = "depositado" if is_deposit else "retirado"
            if result:
                log.info(
                    f"[VAULT] {action} {amount_usdc} USDC "
                    f"do vault {vault_address[:10]}..."
                )
                return {
                    "status": "transferred",
                    "vault": vault_address,
                    "amount": amount_usdc,
                    "action": "deposit" if is_deposit else "withdraw",
                }

            log.error(f"[VAULT] Falha ao transferir para {vault_address}")
            return None

        except ValueError as e:
            log.error(f"[VAULT] Erro de validação: {e}")
            return None
        except Exception as e:
            log.error(f"[VAULT] Erro ao transferir: {e}")
            return None
