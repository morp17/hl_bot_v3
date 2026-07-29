"""
Módulo de Execução de Ordens — Hyperliquid Production Bot v3.0
===============================================================
Gerencia:
- Ordens limit/market via SDK nativo
- TP/SL nativos via trigger orders (grouping="normalTpsl")
- OCO (One Cancels Other)
- Bulk orders (entrada + TP + SL em uma chamada)
- Cancelamento de ordens
- Ajuste de alavancagem
- Lock por símbolo contra reentrância entre threads (NOVO)

FIX (auditoria — item 7): place_bulk_tpsl() CANCELA todas as ordens
abertas de um símbolo antes de criar as novas (entrada + TP + SL). Isso
é seguro em um único fluxo de execução sequencial, mas o processo tem
múltiplas threads que podem, em princípio, chamar métodos deste
executor para o MESMO símbolo concorrentemente — hoje a thread do
dashboard (FastAPI, ver crypto_bot_core/dashboard/main.py) e o loop
principal do bot (thread principal via asyncio) compartilham a mesma
instância de OrderExecutor/HyperliquidConnector. Uma futura ação manual
via dashboard (ex: "fechar posição agora") ou uma evolução para
processamento paralelo de símbolos poderiam colidir com um ciclo do
bot em andamento para o mesmo símbolo — ex: o dashboard cancela ordens
no meio de um bulk_orders() do bot, ou duas chamadas de place_bulk_tpsl
para o mesmo símbolo se intercalam, deixando o símbolo com TP/SL
inconsistentes ou duplicados.

Correção: cada símbolo tem um threading.Lock próprio (criado sob
demanda, protegido por um lock mestre para evitar corrida na criação
da entrada do dicionário). Todo método que MUTA ordens de um símbolo
(place_order, place_tpsl_order, place_bulk_tpsl, cancel_order,
cancel_all_orders, close_position) adquire o lock do símbolo antes de
agir e o libera ao final — serializando operações concorrentes para o
MESMO símbolo, sem bloquear símbolos diferentes entre si (cada um tem
seu próprio lock, então BTC e ETH continuam podendo ser processados
sem se esperar).

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log

from .config import BotConfig
from .exchanges.hyperliquid import (
    _extract_coin,
    _format_price,
    _truncate_qty,
    generate_cloid,
    get_connector,
)


# ──────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────

OrderResult = Optional[Dict[str, Any]]
BulkOrderResult = Optional[Dict[str, Any]]


# ──────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────


def _build_limit_order(
    coin: str,
    is_buy: bool,
    sz: float,
    limit_px: float,
    reduce_only: bool = False,
    cloid: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Constrói payload de ordem limit para Hyperliquid.

    Args:
        coin: Nome do ativo (ex: "BTC").
        is_buy: True para compra, False para venda.
        sz: Quantidade.
        limit_px: Preço limite.
        reduce_only: Se True, apenas reduz posição.
        cloid: Client order ID opcional.

    Returns:
        Dict com payload da ordem.
    """
    order: Dict[str, Any] = {
        "coin": coin,
        "is_buy": is_buy,
        "sz": sz,
        "limit_px": limit_px,
        "order_type": {"limit": {"tif": "Gtc"}},
        "reduce_only": reduce_only,
    }
    if cloid is not None:
        order["cloid"] = cloid
    return order


def _build_trigger_order(
    coin: str,
    is_buy: bool,
    sz: float,
    limit_px: float,
    trigger_px: float,
    tpsl: str,  # "tp" ou "sl"
    is_market: bool = True,
    reduce_only: bool = True,
    cloid: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Constrói payload de ordem trigger (TP/SL) para Hyperliquid.

    Args:
        coin: Nome do ativo.
        is_buy: Direção da ordem trigger.
        sz: Quantidade.
        limit_px: Preço limite (usado mesmo para market como referência).
        trigger_px: Preço de trigger.
        tpsl: "tp" para take profit, "sl" para stop loss.
        is_market: Se True, executa como market ao ser triggerado.
        reduce_only: Se True, apenas reduz posição.
        cloid: Client order ID opcional.

    Returns:
        Dict com payload da ordem trigger.
    """
    order: Dict[str, Any] = {
        "coin": coin,
        "is_buy": is_buy,
        "sz": sz,
        "limit_px": limit_px,
        "order_type": {
            "trigger": {
                "triggerPx": trigger_px,
                "isMarket": is_market,
                "tpsl": tpsl,
            }
        },
        "reduce_only": reduce_only,
    }
    if cloid is not None:
        order["cloid"] = cloid
    return order


# ──────────────────────────────────────────────
# Executor de Ordens
# ──────────────────────────────────────────────


class OrderExecutor:
    """
    Executor de ordens com suporte a TP/SL nativos da Hyperliquid.

    Encapsula a lógica de:
    - Place entry order
    - Place TP/SL trigger orders
    - Bulk place (entry + TP + SL em uma chamada)
    - Cancel orders
    - Update leverage

    FIX (auditoria — item 7): thread-safe por símbolo. Ver docstring
    do módulo para detalhes do risco de reentrância corrigido.
    """

    def __init__(self, cfg: BotConfig) -> None:
        """
        Inicializa o executor.

        Args:
            cfg: Configuração do bot.
        """
        self.cfg = cfg
        self._connector: Optional[HyperliquidConnector] = None

        # FIX (auditoria — item 7): lock por símbolo + lock mestre para
        # proteger a criação de novas entradas no dicionário de locks
        # (a criação em si é uma operação rápida e não bloqueante, mas
        # precisa ser atômica para não criar dois locks distintos para
        # o mesmo símbolo em caso de corrida entre threads).
        self._symbol_locks: Dict[str, threading.Lock] = {}
        self._symbol_locks_master: threading.Lock = threading.Lock()

    @property
    def connector(self) -> HyperliquidConnector:
        """Obtém o connector (lazy initialization)."""
        if self._connector is None:
            self._connector = get_connector(self.cfg)
        return self._connector

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        """
        Retorna o lock exclusivo de um símbolo, criando-o sob demanda
        de forma thread-safe.

        FIX (auditoria — item 7): usa um lock mestre apenas durante a
        criação da entrada do dicionário — uma vez que o lock do
        símbolo existe, o mestre não é mais necessário para adquiri-lo
        (double-checked locking), evitando que TODAS as operações de
        TODOS os símbolos se serializem por trás de um único lock
        global (o que anularia o ganho de granularidade por símbolo).

        Args:
            symbol: Símbolo (ex: "BTC/USDC").

        Returns:
            threading.Lock: Lock exclusivo deste símbolo.
        """
        lock = self._symbol_locks.get(symbol)
        if lock is not None:
            return lock

        with self._symbol_locks_master:
            # Re-checa dentro do lock mestre (outra thread pode ter
            # criado entre o get() acima e a aquisição do master lock).
            lock = self._symbol_locks.get(symbol)
            if lock is None:
                lock = threading.Lock()
                self._symbol_locks[symbol] = lock
            return lock

    def _prepare_order_params(
        self, symbol: str, qty: float, price: float
    ) -> Tuple[str, float, float, int]:
        """
        Prepara parâmetros de ordem (coin, qty ajustada, price ajustado, sz_decimals).

        Args:
            symbol: Símbolo no formato BASE/QUOTE.
            qty: Quantidade.
            price: Preço.

        Returns:
            Tuple[str, float, float, int]: (coin, qty_adj, price_adj, sz_decimals).
        """
        try:
            coin = _extract_coin(symbol)
            # Usa o método do connector que tem cache e usa Info do SDK
            sz_decimals = self.connector._get_sz_decimals(coin)
            qty_adj = _truncate_qty(float(qty), sz_decimals)
            price_adj = _format_price(float(price), sz_decimals)

            if qty_adj <= 0:
                raise ValueError(
                    f"Quantidade ajustada = 0 (original={qty}, szDecimals={sz_decimals})"
                )

            log.debug(
                f"[PRECISAO] {coin}: qty {qty}→{qty_adj} | "
                f"price {price}→{price_adj} (szDecimals={sz_decimals})"
            )

            return coin, qty_adj, price_adj, sz_decimals

        except Exception as e:
            log.error(f"[ORDEM] Erro ao preparar parâmetros: {e}")
            raise

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """
        Envia uma ordem limit (GTC).

        FIX (auditoria — item 7): protegido pelo lock do símbolo.

        Args:
            symbol: Símbolo (ex: "BTC/USDC").
            side: "buy" ou "sell".
            qty: Quantidade.
            price: Preço limite.
            reduce_only: Se True, apenas reduz posição.

        Returns:
            Dict com resultado ou None em caso de erro.
        """
        with self._get_symbol_lock(symbol):
            try:
                if side not in ("buy", "sell"):
                    raise ValueError(f"side inválido: {side}")
                if qty <= 0:
                    raise ValueError(f"qty deve ser > 0: {qty}")
                if price <= 0:
                    raise ValueError(f"price deve ser > 0: {price}")

                coin, qty_adj, price_adj, _ = self._prepare_order_params(symbol, qty, price)
                is_buy = side.lower() == "buy"
                cloid_obj = generate_cloid()

                result = self.connector.place_order(
                    symbol=symbol,
                    side=side,
                    size=qty_adj,
                    price=price_adj,
                    order_type="limit",
                    cloid=cloid_obj,
                )

                if result is None:
                    log.error(f"[ORDEM] Falha ao enviar ordem {side.upper()} {qty} {coin} @ {price}")
                    return None

                log.info(
                    f"[ORDEM] {side.upper()} {qty_adj} {coin} @ {price_adj} | "
                    f"oid={result.get('oid')} | status=enviada"
                )

                return result

            except ValueError as e:
                log.error(f"[ORDEM] Erro de validação: {e}")
                return None
            except Exception as e:
                log.error(f"[ORDEM] Erro ao enviar ordem: {e}")
                return None

    def place_tpsl_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        trigger_px: float,
        tpsl: str,
        is_market: bool = True,
    ) -> OrderResult:
        """
        Envia uma ordem trigger (TP ou SL).

        FIX (auditoria — item 7): protegido pelo lock do símbolo.

        Args:
            symbol: Símbolo.
            side: "buy" ou "sell" (direção da ordem trigger).
            qty: Quantidade.
            price: Preço limite de referência.
            trigger_px: Preço de trigger.
            tpsl: "tp" ou "sl".
            is_market: Se True, executa como market.

        Returns:
            Dict com resultado ou None.
        """
        with self._get_symbol_lock(symbol):
            try:
                if tpsl not in ("tp", "sl"):
                    raise ValueError(f"tpsl deve ser 'tp' ou 'sl': {tpsl}")
                if trigger_px <= 0:
                    raise ValueError(f"trigger_px deve ser > 0: {trigger_px}")

                coin, qty_adj, price_adj, _ = self._prepare_order_params(symbol, qty, price)
                is_buy = side.lower() == "buy"

                result = self.connector.place_tpsl_order(
                    symbol=symbol,
                    side=side,
                    size=qty_adj,
                    trigger_px=trigger_px,
                    tpsl_type=tpsl,
                    is_market=is_market,
                )

                if result is None:
                    log.error(
                        f"[TPSL] Falha ao enviar {tpsl.upper()} "
                        f"{side.upper()} {qty} {coin} trigger@{trigger_px}"
                    )
                    return None

                log.info(
                    f"[TPSL] {tpsl.upper()} {side.upper()} {qty_adj} {coin} "
                    f"trigger@{trigger_px} | oid={result.get('oid')}"
                )

                return result

            except ValueError as e:
                log.error(f"[TPSL] Erro de validação: {e}")
                return None
            except Exception as e:
                log.error(f"[TPSL] Erro ao enviar ordem trigger: {e}")
                return None

    def place_bulk_tpsl(
        self,
        symbol: str,
        side: str,
        entry_qty: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
    ) -> BulkOrderResult:
        """
        Envia ordem de entrada + TP + SL em uma única chamada bulk_orders
        via connector (que usa o SDK nativo com grouping="normalTpsl").

        FIX CRÍTICO (auditoria — item 7): este é o método mais sensível
        a reentrância — o connector.place_bulk_tpsl() CANCELA todas as
        ordens abertas do símbolo ANTES de criar as novas (ver
        exchanges/hyperliquid.py::place_bulk_tpsl, PASSO 1). Se duas
        chamadas para o MESMO símbolo se intercalassem (ex: bot + ação
        manual futura via dashboard, ou uma futura evolução para
        paralelismo entre símbolos), uma poderia cancelar ordens que a
        outra acabou de criar, deixando a posição sem TP/SL efetivos —
        um cenário de risco direto de capital. Agora todo o método
        roda sob o lock do símbolo, serializando qualquer tentativa
        concorrente.

        O SDK native.bulk_orders() retorna um Dict com:
            {"status": "ok", "response": {"data": {"statuses": [...]}}}

        Args:
            symbol: Símbolo.
            side: "buy" ou "sell".
            entry_qty: Quantidade da entrada.
            entry_price: Preço de entrada.
            tp_price: Preço do Take Profit.
            sl_price: Preço do Stop Loss.

        Returns:
            Dict com resposta do SDK ou None em caso de erro.
        """
        with self._get_symbol_lock(symbol):
            try:
                if side not in ("buy", "sell"):
                    raise ValueError(f"side inválido: {side}")
                if entry_qty <= 0:
                    raise ValueError(f"entry_qty deve ser > 0: {entry_qty}")

                coin, qty_adj, price_adj, _ = self._prepare_order_params(
                    symbol, entry_qty, entry_price
                )

                sz_dec = self.connector._get_sz_decimals(coin)
                tp_px_adj = _format_price(float(tp_price), sz_dec)
                sl_px_adj = _format_price(float(sl_price), sz_dec)

                result = self.connector.place_bulk_tpsl(
                    symbol=symbol,
                    entry_side=side,
                    entry_size=qty_adj,
                    entry_price=price_adj,
                    stop_loss_px=sl_px_adj,
                    take_profit_px=tp_px_adj,
                )

                if result is None:
                    log.error(f"[BULK] Falha ao enviar bulk TP/SL para {coin}")
                    return None

                log.info(
                    f"[BULK] {side.upper()} {qty_adj} {coin} @ {price_adj} "
                    f"| TP={tp_px_adj} | SL={sl_px_adj}"
                )

                return result

            except ValueError as e:
                log.error(f"[BULK] Erro de validação: {e}")
                return None
            except Exception as e:
                log.error(f"[BULK] Erro ao enviar bulk TP/SL: {e}")
                return None

    def cancel_order(self, symbol: str, oid: str) -> bool:
        """
        Cancela uma ordem.

        FIX (auditoria — item 7): protegido pelo lock do símbolo.

        Args:
            symbol: Símbolo.
            oid: Order ID.

        Returns:
            True se cancelada com sucesso.
        """
        with self._get_symbol_lock(symbol):
            try:
                if not oid:
                    raise ValueError("oid não fornecido")

                result = self.connector.cancel_order(symbol, int(oid))

                if result:
                    log.info(f"[CANCEL] {symbol} oid={oid} cancelada")
                    return True

                log.warning(f"[CANCEL] Falha ao cancelar {symbol} oid={oid}")
                return False

            except ValueError as e:
                log.error(f"[CANCEL] Erro de validação: {e}")
                return False
            except Exception as e:
                log.error(f"[CANCEL] Erro ao cancelar ordem: {e}")
                return False

    def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancela todas as ordens de um símbolo.

        FIX (auditoria — item 7): protegido pelo lock do símbolo.

        Args:
            symbol: Símbolo.

        Returns:
            Número de ordens canceladas.
        """
        with self._get_symbol_lock(symbol):
            try:
                count = self.connector.cancel_all_orders(symbol)
                if count > 0:
                    log.info(f"[CANCEL] {count} ordem(ns) cancelada(s) em {symbol}")
                return count

            except Exception as e:
                log.error(f"[CANCEL] Erro ao cancelar todas: {e}")
                return 0

    def update_leverage(self, symbol: str, leverage: int, is_cross: bool = True) -> bool:
        """
        Atualiza alavancagem para um símbolo.

        NOTA: não protegido pelo lock de ordens do símbolo — atualização
        de alavancagem não interage com o ciclo cancel-then-create de
        place_bulk_tpsl() e é tipicamente chamada apenas uma vez no
        startup (ver main.py::__init__), então o risco de reentrância
        aqui é desprezível; manter sem lock evita serialização
        desnecessária dessa chamada com operações de ordem em curso.

        Args:
            symbol: Símbolo.
            leverage: Alavancagem (1-50).
            is_cross: True para cross margin, False para isolated.

        Returns:
            True se atualizada com sucesso.
        """
        try:
            if leverage < 1 or leverage > 50:
                raise ValueError(f"leverage deve estar entre 1 e 50: {leverage}")

            coin = _extract_coin(symbol)
            result = self.connector.update_leverage(coin, leverage, is_cross)

            if result:
                log.info(f"[LEVERAGE] {coin}: {leverage}x {'cross' if is_cross else 'isolated'}")
                return True

            log.warning(f"[LEVERAGE] Falha ao atualizar {coin} para {leverage}x")
            return False

        except ValueError as e:
            log.error(f"[LEVERAGE] Erro de validação: {e}")
            return False
        except Exception as e:
            log.error(f"[LEVERAGE] Erro ao atualizar alavancagem: {e}")
            return False

    def close_position(self, symbol: str, qty: float, side: str) -> OrderResult:
        """
        Fecha uma posição.

        FIX (auditoria — item 7): protegido pelo lock do símbolo — evita
        que um fechamento (ex: SL/TP atingido no ciclo do bot) colida
        com uma tentativa concorrente de reenviar bulk TP/SL para o
        mesmo símbolo.

        Args:
            symbol: Símbolo.
            qty: Quantidade a fechar.
            side: Lado da posição ("buy" para long, "sell" para short).

        Returns:
            Dict com resultado ou None.
        """
        with self._get_symbol_lock(symbol):
            try:
                if qty <= 0:
                    raise ValueError(f"qty deve ser > 0: {qty}")
                if side not in ("buy", "sell"):
                    raise ValueError(f"side inválido: {side}")

                result = self.connector.close_position(symbol, qty, side)

                if result:
                    log.info(f"[CLOSE] {side.upper()} {qty} {symbol} fechada")
                    return {"status": "closed", "symbol": symbol, "qty": qty}

                log.error(f"[CLOSE] Falha ao fechar {side.upper()} {qty} {symbol}")
                return None

            except ValueError as e:
                log.error(f"[CLOSE] Erro de validação: {e}")
                return None
            except Exception as e:
                log.error(f"[CLOSE] Erro ao fechar posição: {e}")
                return None