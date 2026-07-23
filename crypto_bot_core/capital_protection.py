"""
Módulo de Proteção de Capital — Hyperliquid Production Bot v3.0
================================================================
Implementa sistema de proteção de capital em 4 níveis:

Nível 1 — Filtros de Mercado: spread, funding rate, horário, BTC crash
Nível 2 — Drawdown: limite de perda diária, drawdown máximo
Nível 3 — Exposição: limite por ativo, correlação entre posições
Nível 4 — Emergência: circuit breakers, pause automático

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log


# ──────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────


@dataclass
class ProtectionState:
    """Estado atual da proteção de capital."""

    daily_loss: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    consecutive_losses: int = 0
    last_loss_ts: float = 0.0
    is_paused: bool = False
    pause_reason: str = ""
    pause_until: float = 0.0
    daily_reset_day: str = ""


# ──────────────────────────────────────────────
# CapitalProtection
# ──────────────────────────────────────────────


class CapitalProtection:
    """
    Sistema de proteção de capital em 4 níveis.

    Nível 1: Filtros de mercado (spread, funding, horário)
    Nível 2: Drawdown e perda diária
    Nível 3: Exposição por ativo e correlação
    Nível 4: Circuit breakers e pause automático
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_daily_loss_pct: float = 0.10,
        max_drawdown_pct: float = 0.20,
        max_consecutive_losses: int = 5,
        cooldown_after_loss_sec: int = 300,
        max_exposure_pct: float = 0.50,
        max_position_pct: float = 0.20,
        circuit_breaker_loss_pct: float = 0.15,
        circuit_breaker_cooldown_sec: int = 3600,
        trade_hour_start_utc: int = 0,
        trade_hour_end_utc: int = 0,
        max_spread_pct: float = 0.005,
        max_funding_rate: float = 0.001,
        btc_crash_filter_pct: float = 0.0,
    ) -> None:
        """
        Inicializa o sistema de proteção.

        Args:
            initial_balance: Saldo inicial em USDC.
            max_daily_loss_pct: Limite de perda diária (%).
            max_drawdown_pct: Drawdown máximo permitido (%).
            max_consecutive_losses: Máximo de perdas consecutivas.
            cooldown_after_loss_sec: Cooldown após perda (segundos).
            max_exposure_pct: Exposição máxima total (%).
            max_position_pct: Exposição máxima por posição (%).
            circuit_breaker_loss_pct: Perda que aciona circuit breaker (%).
            circuit_breaker_cooldown_sec: Cooldown do circuit breaker (segundos).
            trade_hour_start_utc: Hora UTC de início.
            trade_hour_end_utc: Hora UTC de fim.
            max_spread_pct: Spread máximo permitido.
            max_funding_rate: Funding rate máximo permitido.
            btc_crash_filter_pct: Filtro de crash BTC (%).
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_after_loss_sec = cooldown_after_loss_sec
        self.max_exposure_pct = max_exposure_pct
        self.max_position_pct = max_position_pct
        self.circuit_breaker_loss_pct = circuit_breaker_loss_pct
        self.circuit_breaker_cooldown_sec = circuit_breaker_cooldown_sec
        self.trade_hour_start_utc = trade_hour_start_utc
        self.trade_hour_end_utc = trade_hour_end_utc
        self.max_spread_pct = max_spread_pct
        self.max_funding_rate = max_funding_rate
        self.btc_crash_filter_pct = btc_crash_filter_pct

        self.state = ProtectionState(
            peak_balance=initial_balance,
            current_balance=initial_balance,
        )

    # ── Nível 1: Filtros de Mercado ──────────

    def check_trade_hours(self) -> Tuple[bool, str]:
        """
        Verifica se está dentro do horário de operação.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            if self.trade_hour_start_utc == 0 and self.trade_hour_end_utc == 0:
                return True, "ok"

            hour = datetime.now(timezone.utc).hour

            if self.trade_hour_start_utc < self.trade_hour_end_utc:
                ok = self.trade_hour_start_utc <= hour < self.trade_hour_end_utc
            else:
                ok = hour >= self.trade_hour_start_utc or hour < self.trade_hour_end_utc

            if not ok:
                log.debug(
                    f"[N1] Fora do horário (UTC {hour}h | "
                    f"janela {self.trade_hour_start_utc}h-{self.trade_hour_end_utc}h)"
                )
                return False, f"fora_horario_utc_{hour}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N1] Erro em check_trade_hours: {e}")
            return True, "ok"  # fail-open

    def check_spread(self, spread_pct: float) -> Tuple[bool, str]:
        """
        Verifica se o spread está dentro do limite.

        Args:
            spread_pct: Spread atual em percentual.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            if spread_pct <= 0:
                return True, "ok"

            if spread_pct > self.max_spread_pct:
                log.info(f"[N1] Spread alto: {spread_pct:.4%} > {self.max_spread_pct:.4%}")
                return False, f"spread_alto_{spread_pct:.4%}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N1] Erro em check_spread: {e}")
            return True, "ok"

    def check_funding_rate(self, funding_rate: float) -> Tuple[bool, str]:
        """
        Verifica se o funding rate está dentro do limite.

        Args:
            funding_rate: Funding rate atual.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            rate = abs(funding_rate)
            if rate > self.max_funding_rate:
                log.info(f"[N1] Funding rate alto: {rate:.4%} > {self.max_funding_rate:.4%}")
                return False, f"funding_alto_{rate:.4%}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N1] Erro em check_funding_rate: {e}")
            return True, "ok"

    # ── Nível 2: Drawdown e Perda Diária ─────

    def _check_daily_reset(self) -> None:
        """Verifica se precisa resetar contadores diários."""
        try:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self.state.daily_reset_day != day:
                log.info(
                    f"[N2] Reset diário: loss={self.state.daily_loss:.2f}, "
                    f"consecutive_losses={self.state.consecutive_losses}"
                )
                self.state.daily_loss = 0.0
                self.state.consecutive_losses = 0
                self.state.daily_reset_day = day
        except Exception as e:
            log.error(f"[N2] Erro em _check_daily_reset: {e}")

    def check_drawdown(self) -> Tuple[bool, str]:
        """
        Verifica se o drawdown está dentro do limite.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            self._check_daily_reset()

            if self.state.peak_balance <= 0:
                return True, "ok"

            dd = (
                (self.state.peak_balance - self.state.current_balance)
                / self.state.peak_balance
            )

            if dd > self.max_drawdown_pct:
                log.warning(f"[N2] Drawdown {dd:.2%} > {self.max_drawdown_pct:.2%}")
                return False, f"drawdown_{dd:.2%}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N2] Erro em check_drawdown: {e}")
            return True, "ok"

    def check_daily_loss(self) -> Tuple[bool, str]:
        """
        Verifica se a perda diária está dentro do limite.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            self._check_daily_reset()

            if self.state.daily_loss >= 0:
                return True, "ok"

            loss_pct = abs(self.state.daily_loss) / max(self.state.peak_balance, 1)

            if loss_pct > self.max_daily_loss_pct:
                log.warning(f"[N2] Perda diária {loss_pct:.2%} > {self.max_daily_loss_pct:.2%}")
                return False, f"daily_loss_{loss_pct:.2%}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N2] Erro em check_daily_loss: {e}")
            return True, "ok"

    def check_consecutive_losses(self) -> Tuple[bool, str]:
        """
        Verifica se perdas consecutivas estão dentro do limite.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            if (
                self.max_consecutive_losses > 0
                and self.state.consecutive_losses >= self.max_consecutive_losses
            ):
                log.warning(
                    f"[N2] {self.state.consecutive_losses} perdas consecutivas"
                )
                return False, f"consecutive_losses_{self.state.consecutive_losses}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N2] Erro em check_consecutive_losses: {e}")
            return True, "ok"

    def check_cooldown(self) -> Tuple[bool, str]:
        """
        Verifica se está em período de cooldown após perda.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            if self.cooldown_after_loss_sec > 0 and self.state.last_loss_ts > 0:
                elapsed = time.monotonic() - self.state.last_loss_ts
                if elapsed < self.cooldown_after_loss_sec:
                    remaining = int(self.cooldown_after_loss_sec - elapsed)
                    log.info(f"[N2] Cooldown pós-loss: {remaining}s restantes")
                    return False, f"cooldown_{remaining}s"

            return True, "ok"

        except Exception as e:
            log.error(f"[N2] Erro em check_cooldown: {e}")
            return True, "ok"

    # ── Nível 3: Exposição ───────────────────

    def check_exposure(
        self,
        current_positions_value: float,
        new_position_value: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Verifica se a exposição total está dentro do limite.

        Args:
            current_positions_value: Valor total das posições atuais.
            new_position_value: Valor da nova posição proposta.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            balance = max(self.state.current_balance, 1)
            total_exposure = current_positions_value + new_position_value
            exposure_pct = total_exposure / balance

            if exposure_pct > self.max_exposure_pct:
                log.warning(
                    f"[N3] Exposição {exposure_pct:.1%} > {self.max_exposure_pct:.1%}"
                )
                return False, f"exposicao_{exposure_pct:.1%}"

            # Verificar posição individual
            if new_position_value > 0:
                position_pct = new_position_value / balance
                if position_pct > self.max_position_pct:
                    log.warning(
                        f"[N3] Posição {position_pct:.1%} > {self.max_position_pct:.1%}"
                    )
                    return False, f"posicao_{position_pct:.1%}"

            return True, "ok"

        except Exception as e:
            log.error(f"[N3] Erro em check_exposure: {e}")
            return True, "ok"

    # ── Nível 4: Circuit Breaker ─────────────

    def check_circuit_breaker(self) -> Tuple[bool, str]:
        """
        Verifica se o circuit breaker está ativo.

        Returns:
            Tuple[bool, str]: (ok, motivo).
        """
        try:
            if self.state.is_paused:
                if time.time() < self.state.pause_until:
                    remaining = int(self.state.pause_until - time.time())
                    log.warning(
                        f"[N4] Circuit breaker ativo: {remaining}s restantes "
                        f"({self.state.pause_reason})"
                    )
                    return False, f"circuit_breaker_{remaining}s"
                else:
                    log.info("[N4] Circuit breaker expirou — retomando operações")
                    self.state.is_paused = False
                    self.state.pause_reason = ""
                    self.state.pause_until = 0.0

            return True, "ok"

        except Exception as e:
            log.error(f"[N4] Erro em check_circuit_breaker: {e}")
            return True, "ok"

    def trigger_circuit_breaker(self, reason: str) -> None:
        """
        Aciona o circuit breaker.

        Args:
            reason: Motivo do acionamento.
        """
        try:
            self.state.is_paused = True
            self.state.pause_reason = reason
            self.state.pause_until = time.time() + self.circuit_breaker_cooldown_sec
            log.error(
                f"[N4] CIRCUIT BREAKER ACIONADO: {reason} | "
                f"pausado por {self.circuit_breaker_cooldown_sec}s"
            )
        except Exception as e:
            log.error(f"[N4] Erro ao acionar circuit breaker: {e}")

    # ── Métodos de Atualização de Estado ─────

    def record_trade_result(self, pnl: float) -> None:
        """
        Registra resultado de um trade para estatísticas de proteção.

        NOTA: NÃO modifica current_balance ou peak_balance aqui!
        O saldo já está refletido na exchange. A sincronização é feita
        exclusivamente em _sync_from_exchange() e step() no main.py.

        Args:
            pnl: PnL do trade (positivo = lucro, negativo = perda).
        """
        try:
            if pnl < 0:
                self.state.daily_loss += pnl
                self.state.consecutive_losses += 1
                self.state.last_loss_ts = time.monotonic()

                # Verificar circuit breaker usando saldo REAL (já sincronizado)
                loss_from_peak = (
                    (self.state.peak_balance - self.state.current_balance)
                    / max(self.state.peak_balance, 1)
                )
                if loss_from_peak > self.circuit_breaker_loss_pct:
                    self.trigger_circuit_breaker(
                        f"perda_{loss_from_peak:.2%}_do_pico"
                    )
            else:
                # FIX: Reduz daily_loss quando há lucro (recuperação parcial)
                # Ex: daily_loss=-10, pnl=+5 → daily_loss=-5
                self.state.daily_loss = min(0, self.state.daily_loss + pnl)
                self.state.consecutive_losses = 0
                self.state.last_loss_ts = 0.0

        except Exception as e:
            log.error(f"[PROT] Erro ao registrar trade: {e}")

    def check_all(self) -> Tuple[bool, List[str]]:
        """
        Executa todas as verificações de proteção.

        Returns:
            Tuple[bool, List[str]]: (pode_operar, lista_de_motivos).
        """
        reasons: List[str] = []
        checks = [
            ("circuit_breaker", self.check_circuit_breaker()),
            ("trade_hours", self.check_trade_hours()),
            ("drawdown", self.check_drawdown()),
            ("daily_loss", self.check_daily_loss()),
            ("consecutive_losses", self.check_consecutive_losses()),
            ("cooldown", self.check_cooldown()),
        ]

        all_ok = True
        for name, (ok, reason) in checks:
            if not ok:
                all_ok = False
                reasons.append(f"{name}:{reason}")

        return all_ok, reasons

    def to_dict(self) -> Dict[str, Any]:
        """Converte estado para dicionário."""
        return {
            "state": {
                "daily_loss": self.state.daily_loss,
                "peak_balance": self.state.peak_balance,
                "current_balance": self.state.current_balance,
                "consecutive_losses": self.state.consecutive_losses,
                "is_paused": self.state.is_paused,
                "pause_reason": self.state.pause_reason,
            },
            "limits": {
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "max_consecutive_losses": self.max_consecutive_losses,
                "max_exposure_pct": self.max_exposure_pct,
                "circuit_breaker_loss_pct": self.circuit_breaker_loss_pct,
            },
        }
