"""
Módulo de Gerenciamento de Risco — Hyperliquid Production Bot v3.0
==================================================================
Gerencia:
- Dimensionamento de posição (Kelly, ATR, % do capital)
- Cálculo de stops (SL/TP estruturais e percentuais)
- Filtros de entrada (horário, BTC crash, spread, funding)
- PositionManager (PnL, drawdown, perdas consecutivas, cooldown)
- Verificação de segurança contra liquidação

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

import numpy as np
import pandas as pd
from loguru import logger as log

from .config import BotConfig


# ──────────────────────────────────────────────
# Constantes de Segurança
# ──────────────────────────────────────────────

# Folga de segurança entre SL e preço de liquidação da exchange
# NOTA: hardcoded intencionalmente — não confundir com o campo
# RiskConfig.liquidation_safety_buffer_pct, que foi removido por
# nunca ter sido de fato conectado a este valor (débito técnico
# identificado na auditoria; se quiser tornar configurável no
# futuro, seria necessário reintroduzir o campo E ler daqui).
LIQUIDATION_SAFETY_BUFFER_PCT = 0.20

# Número de consultas "posição ausente" antes de concluir fechamento externo
CHECK_EXITS_MISS_THRESHOLD = 2


# ──────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────


@dataclass
class Position:
    """Representa uma posição aberta."""

    symbol: str
    side: str  # "buy" ou "sell"
    entry_price: float
    qty: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_price: float = 0.0
    open_time: float = 0.0
    order_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Valida campos após inicialização."""
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side inválido: {self.side}")
        if self.qty <= 0:
            raise ValueError(f"qty deve ser > 0: {self.qty}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price deve ser > 0: {self.entry_price}")


# ──────────────────────────────────────────────
# Funções de Risco
# ──────────────────────────────────────────────


def calc_position_size(
    balance: float,
    price: float,
    atr: float,
    risk_per_trade: float = 0.02,
    max_capital_pct: float = 0.20,
    stop_loss_pct: float = 0.02,   # NOVO — precisa bater com o valor usado em calc_stops
) -> float:
    """
    Calcula o tamanho da posição usando risco percentual.

    A distância de stop usada aqui é IDÊNTICA à usada em calc_stops():
    max(price * stop_loss_pct, atr * 2). Isso garante que o risco em
    dólares efetivamente exposto (qty * distância_real_do_stop) bata
    com risk_per_trade nominal, e não apenas com uma estimativa
    desacoplada do stop real que será colocado na ordem.

    NOTA: se calc_stops() usar SL estrutural (swing low/high), a
    distância real pode divergir desta estimativa em até 3x — isso é
    uma limitação conhecida, não citar como "risco exato garantido".

    Args:
        balance: Saldo disponível em USDC.
        price: Preço atual do ativo.
        atr: ATR (Average True Range).
        risk_per_trade: Fração do capital a arriscar (ex: 0.02 = 2%).
        max_capital_pct: Fração máxima do capital por posição.
        stop_loss_pct: Percentual de stop — DEVE ser o mesmo valor
            passado depois para calc_stops() no mesmo ciclo de entrada.

    Returns:
        float: Quantidade calculada (arredondada para 6 casas).
    """
    try:
        if price <= 0 or balance <= 0:
            log.warning(f"[SIZING] Preço ({price}) ou saldo ({balance}) inválido")
            return 0.0

        risk_amt = balance * risk_per_trade
        stop_dist = max(price * stop_loss_pct, atr * 2)  # ALTERADO: era max(atr*2, price*0.001)
        qty_risk = risk_amt / stop_dist
        max_qty = (balance * max_capital_pct) / price
        qty = min(qty_risk, max_qty)

        if qty < qty_risk:
            effective_risk = qty * stop_dist / balance
            log.debug(
                f"[SIZING] Cap {max_capital_pct:.0%} ativo: "
                f"qty_risco={qty_risk:.6f} → {qty:.6f} | "
                f"risco efetivo={effective_risk:.2%} vs configurado={risk_per_trade:.2%}"
            )

        return round(qty, 6)

    except Exception as e:
        log.error(f"[SIZING] Erro ao calcular tamanho da posição: {e}")
        return 0.0


def calc_stops(
    price: float,
    side: str,
    atr: float,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[float, float]:
    """
    Calcula Stop Loss e Take Profit.

    Se df for fornecido, tenta usar swing low/high estrutural como SL.
    Fallback: max(pct, ATR×2).

    Args:
        price: Preço de entrada.
        side: "buy" ou "sell".
        atr: ATR atual.
        stop_loss_pct: Percentual do stop loss.
        take_profit_pct: Percentual do take profit.
        df: DataFrame com high/low para SL estrutural.

    Returns:
        Tuple[float, float]: (stop_loss, take_profit).
    """
    try:
        if price <= 0:
            log.warning(f"[STOPS] Preço inválido: {price}")
            return 0.0, 0.0

        if side not in ("buy", "sell"):
            log.warning(f"[STOPS] Side inválido: {side}")
            return 0.0, 0.0

        sl_pct = price * stop_loss_pct
        sl_atr = atr * 2
        tp_d = max(price * take_profit_pct, atr * 4)

        sl_structural: Optional[float] = None
        if df is not None and len(df) >= 11:
            try:
                length = 5
                highs = df["high"]
                lows = df["low"]
                swing_high = highs == highs.rolling(length * 2 + 1, center=True).max()
                swing_low = lows == lows.rolling(length * 2 + 1, center=True).min()

                if side == "buy":
                    sl_candidates = df.loc[swing_low, "low"].values
                    sl_candidates = sl_candidates[sl_candidates < price]
                    if len(sl_candidates) > 0:
                        nearest_swing_low = sl_candidates[-1]
                        sl_structural = nearest_swing_low - atr * 0.3
                else:
                    sh_candidates = df.loc[swing_high, "high"].values
                    sh_candidates = sh_candidates[sh_candidates > price]
                    if len(sh_candidates) > 0:
                        nearest_swing_high = sh_candidates[-1]
                        sl_structural = nearest_swing_high + atr * 0.3
            except Exception as e:
                log.debug(f"[STOPS] Erro no cálculo estrutural: {e}")

        if sl_structural is not None:
            sl_dist = abs(price - sl_structural)
            sl_min = max(sl_pct, sl_atr)
            if sl_min <= sl_dist <= sl_pct * 3:
                sl_d = sl_dist
                log.debug(f"[STOPS] SL estrutural: {sl_structural:.4f} (dist {sl_dist / price:.2%})")
            else:
                sl_d = sl_min
        else:
            sl_d = max(sl_pct, sl_atr)

        if side == "buy":
            return round(price - sl_d, 4), round(price + tp_d, 4)
        return round(price + sl_d, 4), round(price - tp_d, 4)

    except Exception as e:
        log.error(f"[STOPS] Erro ao calcular stops: {e}")
        return 0.0, 0.0


def trade_hours_ok(
    trade_hour_start_utc: int = 0,
    trade_hour_end_utc: int = 0,
) -> bool:
    """
    Verifica se está dentro do horário de operação.

    Args:
        trade_hour_start_utc: Hora UTC de início (0 = desligado).
        trade_hour_end_utc: Hora UTC de fim (0 = desligado).

    Returns:
        bool: True se pode operar.
    """
    try:
        if trade_hour_start_utc == 0 and trade_hour_end_utc == 0:
            return True

        hour = datetime.now(timezone.utc).hour

        if trade_hour_start_utc < trade_hour_end_utc:
            ok = trade_hour_start_utc <= hour < trade_hour_end_utc
        else:
            ok = hour >= trade_hour_start_utc or hour < trade_hour_end_utc

        if not ok:
            log.debug(
                f"[FILTRO] Fora do horário (UTC {hour}h | "
                f"janela {trade_hour_start_utc}h-{trade_hour_end_utc}h)"
            )
        return ok

    except Exception as e:
        log.error(f"[FILTRO] Erro em trade_hours_ok: {e}")
        return True  # fail-open


# ──────────────────────────────────────────────
# PositionManager
# ──────────────────────────────────────────────


class PositionManager:
    """
    Gerenciador de posições abertas, PnL, drawdown e risco.

    Attributes:
        cfg: Configuração do bot.
        positions: Lista de posições abertas.
        pnl_today: PnL acumulado no dia.
        peak_balance: Maior saldo já atingido.
        current_balance: Saldo atual.
        consecutive_losses: Perdas consecutivas.
    """

    def __init__(
        self,
        capital_usd: float,
        max_open_trades: int = 3,
        max_drawdown_pct: float = 0.20,
        daily_loss_limit_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        cooldown_after_loss_sec: int = 300,
        trailing_stop: bool = False,
        trailing_activation_pct: float = 0.02,
        trailing_stop_pct: float = 0.01,
        taker_fee: float = 0.0005,
        max_position_pct: float = 0.20,   # NOVO
    ) -> None:
        """
        Inicializa o gerenciador de posições.

        Args:
            capital_usd: Capital inicial em USD.
            max_open_trades: Máximo de trades simultâneos.
            max_drawdown_pct: Drawdown máximo permitido (fração, ex: 0.20 = 20%).
            daily_loss_limit_pct: Limite de perda diária (fração).
            max_consecutive_losses: Máximo de perdas consecutivas antes de pausar.
            cooldown_after_loss_sec: Cooldown após perda (segundos).
            trailing_stop: Se True, habilita trailing stop.
            trailing_activation_pct: Lucro percentual mínimo para ativar o trailing.
            trailing_stop_pct: Distância do trailing stop em relação ao preço.
            taker_fee: Taxa taker da exchange (fração).
            max_position_pct: Fração máxima do capital por posição
                (deve refletir cfg.max_position_pct/100.0 — usado para
                calcular exposição máxima total em can_open()).
        """
        self.max_open_trades = max_open_trades
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_after_loss_sec = cooldown_after_loss_sec
        self.trailing_stop = trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.taker_fee = taker_fee
        self.max_position_pct = max_position_pct   # NOVO

        self.positions: List[Position] = []
        self.pnl_today = 0.0
        self.peak_balance = float(capital_usd)
        self.current_balance = float(capital_usd)
        self.consecutive_losses = 0
        self._last_loss_ts: float = 0.0
        self._pnl_day_utc: Optional[str] = None
        self._exit_sync_misses: Dict[str, int] = {}

    def maybe_reset_daily_pnl(self) -> None:
        """Reseta PnL diário e perdas consecutivas na virada do dia."""
        try:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._pnl_day_utc is None:
                self._pnl_day_utc = day
                return
            if self._pnl_day_utc != day:
                log.info(
                    f"[RISCO] Virada do dia — pnl_today reset "
                    f"(era {self.pnl_today:+.4f}) | "
                    f"losses consecutivos reset (era {self.consecutive_losses})"
                )
                self.pnl_today = 0.0
                self.consecutive_losses = 0
                self._pnl_day_utc = day
        except Exception as e:
            log.error(f"[RISCO] Erro em maybe_reset_daily_pnl: {e}")

    def can_open(self) -> Tuple[bool, str]:
        """
        Verifica se pode abrir nova posição.

        Returns:
            Tuple[bool, str]: (pode_abrir, motivo).
        """
        try:
            self.maybe_reset_daily_pnl()

            if len(self.positions) >= self.max_open_trades:
                return False, "max_open_trades_atingido"

            dd = (
                (self.peak_balance - self.current_balance)
                / max(self.peak_balance, 1)
                if self.peak_balance > 0
                else 0
            )
            if dd > self.max_drawdown_pct:
                log.warning(f"[RISCO] Drawdown máximo ({dd:.2%}) — pausado")
                return False, f"drawdown_{dd:.2%}"

            dld = (
                abs(self.pnl_today) / max(self.peak_balance, 1)
                if self.pnl_today < 0
                else 0
            )
            if dld > self.daily_loss_limit_pct:
                log.warning(f"[RISCO] Perda diária ({dld:.2%}) — pausado")
                return False, f"daily_loss_{dld:.2%}"

            if (
                self.max_consecutive_losses > 0
                and self.consecutive_losses >= self.max_consecutive_losses
            ):
                log.warning(
                    f"[RISCO] {self.consecutive_losses} perdas consecutivas — "
                    f"pausado até próximo reset diário"
                )
                return False, f"consecutive_losses_{self.consecutive_losses}"

            # Cooldown após loss
            if self.cooldown_after_loss_sec > 0 and self._last_loss_ts > 0:
                elapsed = time.monotonic() - self._last_loss_ts
                if elapsed < self.cooldown_after_loss_sec:
                    remaining = int(self.cooldown_after_loss_sec - elapsed)
                    log.info(f"[RISCO] Cooldown pós-loss: aguardando {remaining}s")
                    return False, f"cooldown_{remaining}s"

            # Verificar exposição total
            if self.positions:
                total_exposure = sum(
                    p.qty * p.entry_price for p in self.positions
                )
                exposure_pct = total_exposure / max(self.current_balance, 1)
                max_exposure = self.max_open_trades * self.max_position_pct   # ALTERADO: era * 0.20
                if exposure_pct >= max_exposure * 0.95:
                    log.info(
                        f"[RISCO] Exposição total {exposure_pct:.1%} "
                        f"próxima do limite ({max_exposure:.0%})"
                    )
                    return False, f"exposure_{exposure_pct:.1%}"

            return True, "ok"

        except Exception as e:
            log.error(f"[RISCO] Erro em can_open: {e}")
            return False, f"erro: {e}"

    def update_trailing(self, price: float) -> None:
        """
        Atualiza trailing stop das posições.

        Args:
            price: Preço atual.
        """
        try:
            if not self.trailing_stop:
                return

            updated = False
            for i, pos in enumerate(self.positions, 1):
                if pos.side == "buy":
                    if price <= pos.entry_price:
                        continue
                    if self.trailing_activation_pct > 0:
                        profit_pct = (price - pos.entry_price) / pos.entry_price
                        if profit_pct < self.trailing_activation_pct:
                            continue
                    nt = price * (1 - self.trailing_stop_pct)
                    if nt > pos.trailing_stop_price:
                        pos.trailing_stop_price = nt
                        log.info(f"[TRAIL] Posição {i} LONG {pos.symbol}: {nt:.4f}")
                        updated = True
                else:
                    if price >= pos.entry_price:
                        continue
                    if self.trailing_activation_pct > 0:
                        profit_pct = (pos.entry_price - price) / pos.entry_price
                        if profit_pct < self.trailing_activation_pct:
                            continue
                    nt = price * (1 + self.trailing_stop_pct)
                    if nt < pos.trailing_stop_price:
                        pos.trailing_stop_price = nt
                        log.info(f"[TRAIL] Posição {i} SHORT {pos.symbol}: {nt:.4f}")
                        updated = True

        except Exception as e:
            log.error(f"[TRAIL] Erro ao atualizar trailing: {e}")

    def check_exits(self, price: float) -> List[Position]:
        """
        Verifica se alguma posição atingiu SL, TP ou trailing.

        FIX CRÍTICO #1 (auditoria, achado em teste ao vivo): a checagem
        de trailing stop rodava incondicionalmente, sem verificar se
        self.trailing_stop estava de fato habilitado, e sem verificar
        se pos.trailing_stop_price já havia sido inicializado com um
        valor válido (>0). Corrigido abaixo.

        FIX CRÍTICO #2 (auditoria, segundo achado em teste ao vivo):
        posições RESTAURADAS de uma sessão anterior via
        main.py::_sync_from_exchange() são criadas sem stop_loss/
        take_profit (ambos ficam no default 0.0 da dataclass Position,
        pois esse dado não é buscado de volta da exchange). Para uma
        posição SELL, 'price >= stop_loss' com stop_loss=0.0 é SEMPRE
        verdadeiro — fechando a posição no ciclo seguinte à restauração,
        rotulado incorretamente como "SL atingido" mesmo sem o preço
        ter se movido. Guarda abaixo: só avalia hit_sl/hit_tp quando o
        respectivo valor foi de fato inicializado (>0).

        Args:
            price: Preço atual.

        Returns:
            List[Position]: Lista de posições a fechar.
        """
        try:
            to_close: List[Position] = []
            for pos in self.positions:
                # FIX: só avalia TP/SL se tiverem sido inicializados
                # (>0) — evita fechamento espúrio de posições com
                # stop_loss/take_profit ainda no default 0.0 (ex:
                # restauradas via sync sem essa informação).
                hit_tp = False
                if pos.take_profit > 0:
                    hit_tp = (
                        (pos.side == "buy" and price >= pos.take_profit)
                        or (pos.side == "sell" and price <= pos.take_profit)
                    )
                hit_sl = False
                if pos.stop_loss > 0:
                    hit_sl = (
                        (pos.side == "buy" and price <= pos.stop_loss)
                        or (pos.side == "sell" and price >= pos.stop_loss)
                    )

                # FIX: só avalia trailing se estiver habilitado E já
                # tiver um trailing_stop_price válido (>0).
                hit_tr = False
                if self.trailing_stop and pos.trailing_stop_price > 0:
                    hit_tr = (
                        (pos.side == "buy" and price <= pos.trailing_stop_price)
                        or (pos.side == "sell" and price >= pos.trailing_stop_price)
                    )

                if hit_tp:
                    log.info(f"[TP] {pos.symbol} @ {price}")
                    to_close.append(pos)
                elif hit_sl:
                    log.info(f"[SL] {pos.symbol} @ {price}")
                    to_close.append(pos)
                elif hit_tr:
                    log.info(f"[TRAIL] {pos.symbol} @ {price}")
                    to_close.append(pos)

                if pos.stop_loss <= 0 and pos.take_profit <= 0:
                    log.warning(
                        f"[EXITS] {pos.symbol} sem SL/TP definidos "
                        f"(provavelmente restaurada via sync) — posição "
                        f"NÃO tem proteção de stop até ser corrigido "
                        f"manualmente ou até _sync_from_exchange popular "
                        f"esses valores a partir das ordens reais na DEX."
                    )

            return to_close

        except Exception as e:
            log.error(f"[EXITS] Erro em check_exits: {e}")
            return []

    def add(self, pos: Position) -> None:
        """
        Adiciona posição à lista.

        Args:
            pos: Posição a adicionar.
        """
        try:
            if not isinstance(pos, Position):
                raise TypeError(f"Esperado Position, recebido {type(pos).__name__}")
            self.positions.append(pos)
            log.info(f"[POS] Adicionada: {pos.side.upper()} {pos.symbol} qty={pos.qty} @ {pos.entry_price}")
        except Exception as e:
            log.error(f"[POS] Erro ao adicionar posição: {e}")

    def record_close(self, pos: Position, exit_price: float) -> Dict[str, Any]:
        """
        Registra fechamento de posição e atualiza PnL.

        NOTA: NÃO modifica current_balance ou peak_balance aqui!
        O saldo já está refletido na exchange. A sincronização é feita
        exclusivamente em _sync_from_exchange() e step() no main.py.

        Args:
            pos: Posição fechada.
            exit_price: Preço de saída.

        Returns:
            Dict[str, Any]: Resultado do fechamento.
        """
        try:
            fee = (pos.entry_price + exit_price) * pos.qty * self.taker_fee
            gross = (
                (exit_price - pos.entry_price)
                * pos.qty
                * (1 if pos.side == "buy" else -1)
            )
            net = gross - fee

            self.pnl_today += net

            result: Dict[str, Any] = {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "qty": pos.qty,
                "gross_pnl": gross,
                "fee": fee,
                "net_pnl": net,
            }

            if net < 0:
                self.consecutive_losses += 1
                self._last_loss_ts = time.monotonic()
                log.info(
                    f"[TRADE] PnL: {net:+.4f} USDC | "
                    f"Losses consecutivos: {self.consecutive_losses}"
                )
            else:
                self.consecutive_losses = 0
                self._last_loss_ts = 0.0
                log.info(f"[TRADE] PnL: {net:+.4f} USDC")

            if (
                self.max_consecutive_losses > 0
                and self.consecutive_losses >= self.max_consecutive_losses
            ):
                log.warning(
                    f"[RISCO] Limite de {self.max_consecutive_losses} "
                    f"perdas consecutivas atingido"
                )

            self.positions.remove(pos)
            return result

        except ValueError:
            log.warning(f"[POS] Posição {pos.symbol} não encontrada na lista")
            return {"error": "posicao_nao_encontrada"}
        except Exception as e:
            log.error(f"[POS] Erro ao registrar fechamento: {e}")
            return {"error": str(e)}

    def to_dict(self) -> Dict[str, Any]:
        """Converte estado para dicionário."""
        return {
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "qty": p.qty,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop_price": p.trailing_stop_price,
                    "open_time": p.open_time,
                }
                for p in self.positions
            ],
            "pnl_today": self.pnl_today,
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "consecutive_losses": self.consecutive_losses,
        }