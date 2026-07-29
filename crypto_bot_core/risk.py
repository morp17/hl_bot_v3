"""
Módulo de Gerenciamento de Risco — Hyperliquid Production Bot v3.0
==================================================================
Gerencia:
- Dimensionamento de posição (Kelly, ATR, % do capital)
- Cálculo de stops (SL/TP estruturais e percentuais)
- Filtros de entrada (horário, BTC crash, spread, funding)
- PositionManager (PnL, drawdown, perdas consecutivas, cooldown)
- Verificação de segurança contra liquidação
- Acúmulo de custo/receita de funding durante o holding (perpétuos)

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

LIQUIDATION_SAFETY_BUFFER_PCT = 0.20
DEFAULT_MAINTENANCE_MARGIN_PCT = 0.03
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

    # FIX (auditoria — item 6): campos novos para acúmulo de funding.
    # accrued_funding é um CUSTO acumulado (positivo = reduz o PnL
    # líquido no fechamento; negativo = aumenta, quando a posição
    # RECEBE funding em vez de pagar). Ver PositionManager.accrue_funding()
    # para a lógica de sinal por lado (long/short).
    accrued_funding: float = 0.0
    last_funding_ts: float = 0.0

    def __post_init__(self) -> None:
        """Valida campos após inicialização."""
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side inválido: {self.side}")
        if self.qty <= 0:
            raise ValueError(f"qty deve ser > 0: {self.qty}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price deve ser > 0: {self.entry_price}")
        if self.last_funding_ts <= 0:
            # Se não informado, começa a contar a partir da abertura
            # (ou de agora, se open_time também não foi informado).
            self.last_funding_ts = self.open_time or time.time()


# ──────────────────────────────────────────────
# Funções de Risco
# ──────────────────────────────────────────────


def calc_position_size(
    balance: float,
    price: float,
    atr: float,
    risk_per_trade: float = 0.02,
    max_capital_pct: float = 0.20,
    stop_loss_pct: float = 0.02,
) -> float:
    """
    Calcula o tamanho da posição usando risco percentual.

    A distância de stop usada aqui é IDÊNTICA à usada em calc_stops():
    max(price * stop_loss_pct, atr * 2). Isso garante que o risco em
    dólares efetivamente exposto (qty * distância_real_do_stop) bata
    com risk_per_trade nominal, e não apenas com uma estimativa
    desacoplada do stop real que será colocado na ordem.

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
        stop_dist = max(price * stop_loss_pct, atr * 2)
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


def calc_liquidation_price_estimate(
    entry_price: float,
    side: str,
    leverage: int,
    is_cross: bool = False,
    maintenance_margin_pct: float = DEFAULT_MAINTENANCE_MARGIN_PCT,
) -> float:
    """
    Estima o preço de liquidação de uma posição ANTES dela existir.

    IMPORTANTE — isto é uma ESTIMATIVA, não o preço de liquidação
    exato da Hyperliquid (ver detalhes na versão anterior deste
    docstring, mantidos por completo abaixo).

    A exchange usa margem de manutenção variável por ativo/tier de
    risco/notional, não um percentual fixo único. O valor real só
    existe de fato DEPOIS que a posição é aberta (via
    user_state()["assetPositions"][i]["position"]["liquidationPx"]).
    Esta função serve para um "gate" de segurança PRÉ-entrada.

    Fórmula (aproximação padrão para perpétuos com margem isolada):
        LONG:  liq_price ≈ entry * (1 - 1/leverage + maintenance_margin_pct)
        SHORT: liq_price ≈ entry * (1 + 1/leverage - maintenance_margin_pct)

    Args:
        entry_price: Preço de entrada planejado.
        side: "buy" ou "sell".
        leverage: Alavancagem (1-50).
        is_cross: Se True, margem cruzada (fórmula tende a ser
            conservadora — superestima a distância até a liquidação).
        maintenance_margin_pct: Margem de manutenção assumida (fração).

    Returns:
        float: Preço de liquidação estimado, ou 0.0 se inputs inválidos.
    """
    try:
        if entry_price <= 0 or leverage <= 0:
            log.warning(
                f"[LIQ] Inputs inválidos para estimativa de liquidação: "
                f"entry={entry_price}, leverage={leverage}"
            )
            return 0.0
        if side not in ("buy", "sell"):
            log.warning(f"[LIQ] Side inválido: {side}")
            return 0.0

        initial_margin_frac = 1.0 / leverage

        if side == "buy":
            liq = entry_price * (1 - initial_margin_frac + maintenance_margin_pct)
            return max(liq, 0.0)
        else:
            liq = entry_price * (1 + initial_margin_frac - maintenance_margin_pct)
            return max(liq, 0.0)

    except Exception as e:
        log.error(f"[LIQ] Erro ao estimar preço de liquidação: {e}")
        return 0.0


def validate_stop_loss_safety(
    stop_loss: float,
    liquidation_price: float,
    side: str,
    buffer_pct: float = LIQUIDATION_SAFETY_BUFFER_PCT,
) -> Tuple[bool, str]:
    """
    Valida se o Stop Loss planejado tem folga suficiente em relação
    ao preço de liquidação estimado (LIQUIDATION_SAFETY_BUFFER_PCT).

    Args:
        stop_loss: Preço do stop loss planejado.
        liquidation_price: Preço de liquidação estimado.
        side: "buy" ou "sell".
        buffer_pct: Folga mínima exigida entre SL e liquidação.

    Returns:
        Tuple[bool, str]: (seguro, motivo).
    """
    try:
        if liquidation_price <= 0 or stop_loss <= 0:
            return True, "liquidacao_nao_calculavel"

        if side == "buy":
            if stop_loss <= liquidation_price:
                return False, (
                    f"sl_{stop_loss:.4f}_abaixo_ou_igual_liquidacao_"
                    f"{liquidation_price:.4f}"
                )
            dist = (stop_loss - liquidation_price) / liquidation_price
        else:
            if stop_loss >= liquidation_price:
                return False, (
                    f"sl_{stop_loss:.4f}_acima_ou_igual_liquidacao_"
                    f"{liquidation_price:.4f}"
                )
            dist = (liquidation_price - stop_loss) / liquidation_price

        if dist < buffer_pct:
            return False, (
                f"sl_proximo_liquidacao_dist={dist:.2%}_"
                f"min_exigido={buffer_pct:.2%}"
            )

        return True, "ok"

    except Exception as e:
        log.error(f"[LIQ] Erro ao validar segurança do SL: {e}")
        return True, "erro_validacao"


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
        return True


# ──────────────────────────────────────────────
# PositionManager
# ──────────────────────────────────────────────


class PositionManager:
    """
    Gerenciador de posições abertas, PnL, drawdown e risco.
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
        max_position_pct: float = 0.20,
    ) -> None:
        """
        Inicializa o gerenciador de posições.

        Args:
            capital_usd: Capital inicial em USD.
            max_open_trades: Máximo de trades simultâneos.
            max_drawdown_pct: Drawdown máximo permitido (fração).
            daily_loss_limit_pct: Limite de perda diária (fração).
            max_consecutive_losses: Máximo de perdas consecutivas.
            cooldown_after_loss_sec: Cooldown após perda (segundos).
            trailing_stop: Se True, habilita trailing stop.
            trailing_activation_pct: Lucro mínimo para ativar o trailing.
            trailing_stop_pct: Distância do trailing stop.
            taker_fee: Taxa taker da exchange (fração).
            max_position_pct: Fração máxima do capital por posição.
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
        self.max_position_pct = max_position_pct

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

            if self.cooldown_after_loss_sec > 0 and self._last_loss_ts > 0:
                elapsed = time.monotonic() - self._last_loss_ts
                if elapsed < self.cooldown_after_loss_sec:
                    remaining = int(self.cooldown_after_loss_sec - elapsed)
                    log.info(f"[RISCO] Cooldown pós-loss: aguardando {remaining}s")
                    return False, f"cooldown_{remaining}s"

            if self.positions:
                total_exposure = sum(
                    p.qty * p.entry_price for p in self.positions
                )
                exposure_pct = total_exposure / max(self.current_balance, 1)
                max_exposure = self.max_open_trades * self.max_position_pct
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

    def accrue_funding(
        self,
        symbol: str,
        funding_rate: float,
        mark_price: float,
    ) -> None:
        """
        Acumula custo/receita de funding para posições abertas de um
        símbolo, proporcional ao tempo decorrido desde a última chamada.

        FIX (auditoria — item 6): até esta correção, o funding rate era
        usado apenas como FILTRO de entrada (capital_protection.
        check_funding_rate) e como sinal de estratégia
        (funding_arbitrage), mas NUNCA entrava no PnL de posições já
        abertas. Para perpétuos, o funding pago/recebido durante o
        holding é uma componente real de custo/retorno — ignorá-lo
        deixa o PnL live sistematicamente otimista (ou pessimista, se
        a posição está do lado que recebe funding).

        Convenção de sinal (padrão de mercado para perpétuos):
        - funding_rate > 0 → LONGS pagam, SHORTS recebem.
        - funding_rate < 0 → SHORTS pagam, LONGS recebem.

        `Position.accrued_funding` é armazenado como CUSTO (positivo
        reduz o PnL líquido no fechamento). Por isso:
            direction = +1 para "buy" (long paga quando funding>0)
            direction = -1 para "sell" (short recebe quando funding>0,
                          ou seja, custo NEGATIVO = receita)

        IMPORTANTE — aproximação assumida: trata o funding_rate
        retornado por get_current_funding_rate() como uma taxa por
        HORA (Hyperliquid liquida funding a cada hora), acumulando
        proporcionalmente ao tempo decorrido em vez de aplicar de uma
        vez por hora cheia. Isso suaviza o acúmulo entre ciclos do bot
        (tipicamente 60s) sem exigir alinhamento exato com o horário
        de liquidação da exchange — para fins de PnL informativo/
        interno, a diferença é desprezível frente ao ganho de
        simplicidade. Não deve ser usado como fonte de verdade fiscal
        exata (para isso, usar o funding real reportado por
        get_user_fills()/histórico da conta).

        Args:
            symbol: Símbolo a atualizar.
            funding_rate: Taxa de funding atual (fração, ex: 0.0001).
            mark_price: Mark price atual (para calcular notional).
        """
        try:
            if not funding_rate or mark_price <= 0:
                return

            now = time.time()
            for pos in self.positions:
                if pos.symbol != symbol:
                    continue

                last_ts = pos.last_funding_ts or pos.open_time or now
                elapsed_hours = (now - last_ts) / 3600.0
                if elapsed_hours <= 0:
                    continue

                notional = pos.qty * mark_price
                direction = 1.0 if pos.side == "buy" else -1.0
                cost = funding_rate * notional * elapsed_hours * direction

                pos.accrued_funding += cost
                pos.last_funding_ts = now

                log.debug(
                    f"[FUNDING] {symbol} {pos.side.upper()}: +{cost:+.6f} "
                    f"USDC acumulado (total={pos.accrued_funding:+.6f}, "
                    f"rate={funding_rate:.6f}, elapsed={elapsed_hours:.4f}h)"
                )

        except Exception as e:
            log.error(f"[FUNDING] Erro ao acumular funding para {symbol}: {e}")

    def check_exits(self, price: float) -> List[Position]:
        """
        Verifica se alguma posição atingiu SL, TP ou trailing.

        FIX CRÍTICO #1: a checagem de trailing stop agora só roda se
        self.trailing_stop estiver habilitado E pos.trailing_stop_price
        já tiver sido inicializado com um valor válido (>0).

        FIX CRÍTICO #2: posições restauradas via sync sem SL/TP
        (0.0 no default) não disparam mais fechamento espúrio — só
        avalia hit_sl/hit_tp quando o respectivo valor foi de fato
        inicializado (>0).

        Args:
            price: Preço atual.

        Returns:
            List[Position]: Lista de posições a fechar.
        """
        try:
            to_close: List[Position] = []
            for pos in self.positions:
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

        FIX (auditoria — item 6): o PnL líquido agora deduz
        pos.accrued_funding (custo/receita de funding acumulado
        durante o holding via accrue_funding()), além da taxa taker.
        O dict de retorno inclui "funding_cost" separadamente, para
        que o TradeLedger e as notificações possam exibir a
        decomposição completa do resultado.

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
            funding_cost = pos.accrued_funding
            net = gross - fee - funding_cost

            self.pnl_today += net

            result: Dict[str, Any] = {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "qty": pos.qty,
                "gross_pnl": gross,
                "fee": fee,
                "funding_cost": funding_cost,
                "net_pnl": net,
                "open_time": pos.open_time,
            }

            if net < 0:
                self.consecutive_losses += 1
                self._last_loss_ts = time.monotonic()
                log.info(
                    f"[TRADE] PnL: {net:+.4f} USDC (bruto={gross:+.4f}, "
                    f"fee={fee:.4f}, funding={funding_cost:+.4f}) | "
                    f"Losses consecutivos: {self.consecutive_losses}"
                )
            else:
                self.consecutive_losses = 0
                self._last_loss_ts = 0.0
                log.info(
                    f"[TRADE] PnL: {net:+.4f} USDC (bruto={gross:+.4f}, "
                    f"fee={fee:.4f}, funding={funding_cost:+.4f})"
                )

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
                    "accrued_funding": p.accrued_funding,
                }
                for p in self.positions
            ],
            "pnl_today": self.pnl_today,
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "consecutive_losses": self.consecutive_losses,
        }