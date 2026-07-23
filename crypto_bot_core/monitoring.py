"""
Módulo de Monitoramento — Hyperliquid Production Bot v3.0
==========================================================
Métricas Prometheus, health checks e heartbeat.

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger as log


# ──────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────


@dataclass
class BotMetrics:
    """Métricas do bot para Prometheus / health check."""

    # Trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    current_positions: int = 0

    # Performance
    uptime_seconds: float = 0.0
    last_step_ts: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0

    # Conexão
    last_signal: str = "hold"
    last_error: str = ""
    is_connected: bool = False

    # Capital
    balance: float = 0.0
    peak_balance: float = 0.0
    drawdown_pct: float = 0.0


# ──────────────────────────────────────────────
# Monitor
# ──────────────────────────────────────────────


class Monitor:
    """
    Sistema de monitoramento do bot.

    Coleta métricas, health checks e heartbeat.
    """

    def __init__(self) -> None:
        """Inicializa o monitor."""
        self.metrics = BotMetrics()
        self._start_time = time.time()
        self._health_history: List[Dict[str, Any]] = []
        self._max_history = 100

    def record_step(self, success: bool = True) -> None:
        """
        Registra um ciclo do bot.

        Args:
            success: True se o ciclo foi bem-sucedido.
        """
        try:
            self.metrics.steps_completed += 1
            if not success:
                self.metrics.steps_failed += 1
            self.metrics.last_step_ts = time.time()
            self.metrics.uptime_seconds = time.time() - self._start_time
        except Exception as e:
            log.error(f"[MONITOR] Erro ao registrar step: {e}")

    def record_trade(self, pnl: float) -> None:
        """
        Registra um trade completado.

        Args:
            pnl: PnL do trade.
        """
        try:
            self.metrics.total_trades += 1
            self.metrics.total_pnl += pnl
            if pnl >= 0:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1
        except Exception as e:
            log.error(f"[MONITOR] Erro ao registrar trade: {e}")

    def record_error(self, error: str) -> None:
        """
        Registra um erro.

        Args:
            error: Mensagem de erro.
        """
        try:
            self.metrics.last_error = error
            log.error(f"[MONITOR] Erro registrado: {error}")
        except Exception as e:
            log.error(f"[MONITOR] Erro ao registrar erro: {e}")

    def update_balance(self, balance: float) -> None:
        """
        Atualiza saldo e drawdown.

        Args:
            balance: Saldo atual.
        """
        try:
            self.metrics.balance = balance
            self.metrics.peak_balance = max(self.metrics.peak_balance, balance)
            if self.metrics.peak_balance > 0:
                self.metrics.drawdown_pct = (
                    (self.metrics.peak_balance - balance)
                    / self.metrics.peak_balance
                )
        except Exception as e:
            log.error(f"[MONITOR] Erro ao atualizar saldo: {e}")

    def update_signal(self, signal: str) -> None:
        """Atualiza último sinal."""
        self.metrics.last_signal = signal

    def update_connection(self, connected: bool) -> None:
        """Atualiza status de conexão."""
        self.metrics.is_connected = connected

    def health_check(self) -> Dict[str, Any]:
        """
        Retorna health check completo.

        Returns:
            Dict com status de saúde do bot.
        """
        try:
            now = time.time()
            time_since_last_step = now - self.metrics.last_step_ts if self.metrics.last_step_ts > 0 else 9999

            status = "healthy"
            if time_since_last_step > 300:  # 5 min sem step
                status = "warning"
            if time_since_last_step > 600:  # 10 min sem step
                status = "critical"

            error_rate = (
                self.metrics.steps_failed / max(self.metrics.steps_completed, 1)
            )
            if error_rate > 0.5:  # >50% de erros
                status = "degraded"

            return {
                "status": status,
                "uptime_seconds": self.metrics.uptime_seconds,
                "last_step_ts": self.metrics.last_step_ts,
                "time_since_last_step": time_since_last_step,
                "steps_completed": self.metrics.steps_completed,
                "steps_failed": self.metrics.steps_failed,
                "error_rate": error_rate,
                "is_connected": self.metrics.is_connected,
                "last_error": self.metrics.last_error,
                "timestamp": now,
            }

        except Exception as e:
            log.error(f"[MONITOR] Erro no health check: {e}")
            return {"status": "error", "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas completas.

        Returns:
            Dict com todas as métricas.
        """
        try:
            win_rate = (
                (self.metrics.winning_trades / max(self.metrics.total_trades, 1)) * 100
            )

            return {
                "trading": {
                    "total_trades": self.metrics.total_trades,
                    "winning_trades": self.metrics.winning_trades,
                    "losing_trades": self.metrics.losing_trades,
                    "win_rate_pct": round(win_rate, 2),
                    "total_pnl": round(self.metrics.total_pnl, 2),
                    "current_positions": self.metrics.current_positions,
                },
                "performance": {
                    "uptime_hours": round(self.metrics.uptime_seconds / 3600, 2),
                    "steps_completed": self.metrics.steps_completed,
                    "steps_failed": self.metrics.steps_failed,
                    "last_signal": self.metrics.last_signal,
                },
                "capital": {
                    "balance": round(self.metrics.balance, 2),
                    "peak_balance": round(self.metrics.peak_balance, 2),
                    "drawdown_pct": round(self.metrics.drawdown_pct * 100, 2),
                },
                "connection": {
                    "is_connected": self.metrics.is_connected,
                    "last_error": self.metrics.last_error,
                },
            }

        except Exception as e:
            log.error(f"[MONITOR] Erro ao obter métricas: {e}")
            return {"error": str(e)}

    def to_dict(self) -> Dict[str, Any]:
        """Converte estado para dicionário."""
        return {
            "metrics": self.get_metrics(),
            "health": self.health_check(),
        }
