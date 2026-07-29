"""
Módulo de Trilha de Auditoria de Trades — Hyperliquid Production Bot v3.0
===========================================================================
Registro append-only e persistente de todos os trades fechados em modo
LIVE (interno ou fechado externamente na exchange).

FIX (auditoria — item 12): antes desta correção, o único registro
estruturado e persistente de trades era o CSV gerado por
BacktestResult.export_csv() — exclusivo do modo backtest. Em modo
live, cada trade fechado só existia:
- Nos logs rotacionados (data/bot_{date}.log, retenção de 7 dias);
- No estado em memória do dashboard (perdido a cada restart).

Para um bot operando capital real, é esperado um registro append-only,
com timestamp, símbolo, lado, preços, PnL líquido e motivo de saída,
que sobreviva a restarts e rotação de logs — usado para reconciliação
contábil, auditoria fiscal e análise de performance histórica.

Formato: JSONL (uma linha JSON por trade), por ser:
- Append-only nativo (não requer reescrever o arquivo inteiro a cada
  gravação, como um CSV/JSON array faria);
- Resiliente a interrupção no meio de uma escrita (uma linha corrompida
  não invalida as demais);
- Trivialmente importável para pandas/análise posterior
  (`pd.read_json(path, lines=True)`).

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Logs estruturados
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from loguru import logger as log


DEFAULT_LEDGER_PATH = os.path.join("data", "trades_live.jsonl")


class TradeLedger:
    """
    Registro append-only de trades fechados em modo live.

    Thread-safe via lock interno — o dashboard (thread separada) e o
    loop principal do bot podem, em tese, gravar/ler concorrentemente.
    """

    def __init__(self, path: str = DEFAULT_LEDGER_PATH) -> None:
        """
        Inicializa o ledger, garantindo que o diretório de destino exista.

        Args:
            path: Caminho do arquivo JSONL de destino.
        """
        self.path = path
        self._lock = threading.Lock()

        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        except OSError as e:
            log.error(f"[LEDGER] Erro ao criar diretório para {self.path}: {e}")

    def record(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        gross_pnl: float,
        fee: float,
        funding_cost: float,
        net_pnl: float,
        exit_reason: str,
        strategy: Optional[str] = None,
        open_time: Optional[float] = None,
    ) -> None:
        """
        Registra um trade fechado como uma linha JSON no arquivo.

        Args:
            symbol: Símbolo (ex: BTC/USDC).
            side: "buy" ou "sell".
            entry_price: Preço de entrada.
            exit_price: Preço de saída.
            qty: Quantidade.
            gross_pnl: PnL bruto (sem taxas/funding).
            fee: Taxa paga (taker).
            funding_cost: Custo/receita líquida de funding acumulada
                durante o holding (positivo = custo, negativo = receita).
            net_pnl: PnL líquido final (gross - fee - funding_cost).
            exit_reason: Motivo da saída ("take_profit", "stop_loss",
                "trailing_stop", "close_external", "manual", etc.).
            strategy: Nome da estratégia que gerou a entrada (opcional).
            open_time: Timestamp epoch de abertura da posição (opcional,
                usado para calcular bars_held/tempo em posição).
        """
        try:
            now = time.time()
            entry = {
                "closed_at": now,
                "closed_at_iso": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)
                ),
                "symbol": symbol,
                "side": side,
                "entry_price": round(entry_price, 8),
                "exit_price": round(exit_price, 8),
                "qty": round(qty, 8),
                "gross_pnl": round(gross_pnl, 8),
                "fee": round(fee, 8),
                "funding_cost": round(funding_cost, 8),
                "net_pnl": round(net_pnl, 8),
                "exit_reason": exit_reason,
                "strategy": strategy,
                "open_time": open_time,
                "hold_seconds": (now - open_time) if open_time else None,
            }

            line = json.dumps(entry, ensure_ascii=False)

            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

            log.debug(f"[LEDGER] Trade registrado: {symbol} net_pnl={net_pnl:+.4f}")

        except Exception as e:
            # FALHA NO LEDGER NUNCA DEVE INTERROMPER O FLUXO DE TRADING —
            # é um registro auxiliar de auditoria, não parte crítica do
            # caminho de execução de ordens.
            log.error(f"[LEDGER] Erro ao registrar trade (não bloqueante): {e}")

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Lê todos os trades registrados (uso ocasional/administrativo —
        não é chamado no hot path do bot).

        Returns:
            List[Dict[str, Any]]: Lista de trades, na ordem em que
                foram gravados. Retorna lista vazia se o arquivo não
                existir ou em caso de erro.
        """
        try:
            if not os.path.exists(self.path):
                return []

            trades: List[Dict[str, Any]] = []
            with self._lock:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            log.warning(
                                f"[LEDGER] Linha {line_num} corrompida em "
                                f"{self.path}, ignorando: {e}"
                            )
            return trades

        except Exception as e:
            log.error(f"[LEDGER] Erro ao ler ledger: {e}")
            return []

    def summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo agregado simples dos trades registrados.

        Returns:
            Dict[str, Any]: total_trades, total_net_pnl, win_rate_pct.
        """
        try:
            trades = self.read_all()
            if not trades:
                return {"total_trades": 0, "total_net_pnl": 0.0, "win_rate_pct": 0.0}

            total_net = sum(t.get("net_pnl", 0.0) for t in trades)
            wins = sum(1 for t in trades if t.get("net_pnl", 0.0) >= 0)
            win_rate = (wins / len(trades)) * 100.0

            return {
                "total_trades": len(trades),
                "total_net_pnl": round(total_net, 4),
                "win_rate_pct": round(win_rate, 2),
            }

        except Exception as e:
            log.error(f"[LEDGER] Erro ao gerar resumo: {e}")
            return {"total_trades": 0, "total_net_pnl": 0.0, "win_rate_pct": 0.0}


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_ledger_instance: Optional[TradeLedger] = None


def get_ledger(path: str = DEFAULT_LEDGER_PATH) -> TradeLedger:
    """
    Retorna instância singleton do TradeLedger.

    Args:
        path: Caminho do arquivo JSONL (só é aplicado na primeira
            chamada; chamadas subsequentes retornam a mesma instância).

    Returns:
        TradeLedger: Instância singleton.
    """
    global _ledger_instance
    if _ledger_instance is None:
        _ledger_instance = TradeLedger(path)
    return _ledger_instance