"""
Módulo de Conexão Hyperliquid — v3.0
=====================================
Gerencia toda a comunicação com a Hyperliquid DEX:
- SDK nativo (hyperliquid-python-sdk) para ordens e autenticação
- ccxt para consultas de mercado (OHLCV, ticker, order book)
- WebSocket para dados em tempo real
- Suporte multi-symbol (BTC, ETH, SOL)
- Type hints, tratamento de exceções, logs estruturados
"""

from __future__ import annotations

import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Dict, List, Optional, Tuple

import ccxt
from ccxt import Exchange as CcxtExchange
from hyperliquid.exchange import Exchange as HLExchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import Cloid
from loguru import logger as log

from ..config import BotConfig, SymbolConfig


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

DEFAULT_SLIPPAGE = 0.05
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2
WS_RECONNECT_DELAY_SEC = 5


# ──────────────────────────────────────────────
# Exceções Customizadas
# ──────────────────────────────────────────────


class HyperliquidError(Exception):
    """Erro base para operações Hyperliquid."""


class ConnectionError(HyperliquidError):
    """Erro de conexão com a API."""


class OrderError(HyperliquidError):
    """Erro ao executar ordem."""


class AuthenticationError(HyperliquidError):
    """Erro de autenticação."""


class RateLimitError(HyperliquidError):
    """Rate limit atingido."""


# ──────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────


def _get_credentials(cfg: BotConfig) -> Tuple[str, str]:
    """
    Extrai credenciais da configuração.

    Args:
        cfg: Configuração do bot.

    Returns:
        Tuple[str, str]: (private_key, account_address)

    Raises:
        AuthenticationError: Se credenciais estiverem ausentes no mainnet.
    """
    pk = cfg.hyperliquid_private_key
    addr = cfg.hyperliquid_account_address

    if not cfg.testnet:
        if not pk or not addr:
            raise AuthenticationError(
                "Credenciais não configuradas para mainnet. "
                "Defina HYPERLIQUID_PRIVATE_KEY e HYPERLIQUID_ACCOUNT_ADDRESS no .env"
            )

    return pk, addr


def _extract_coin(symbol: str) -> str:
    """
    Extrai o nome da moeda de um símbolo.

    Args:
        symbol: Símbolo no formato BTC/USDC.

    Returns:
        str: Nome da moeda (ex: BTC).
    """
    return symbol.replace("/USDC", "").replace("/USD", "")


def _get_sz_decimals(info: Info, coin: str) -> int:
    """
    Obtém o número de decimais para tamanho de ordem de um ativo.

    Args:
        info: Instância Info do SDK.
        coin: Nome da moeda (ex: BTC).

    Returns:
        int: Número de decimais (ex: 5 para BTC).
    """
    try:
        meta = info.meta()
        assets = meta.get("universe", [])
        for asset in assets:
            if asset.get("name", "").upper() == coin.upper():
                sz_decimals = asset.get("szDecimals", 5)
                log.debug(f"szDecimals para {coin}: {sz_decimals}")
                return sz_decimals
        log.warning(f"szDecimals não encontrado para {coin}, usando default 5")
        return 5
    except Exception as e:
        log.error(f"Erro ao obter szDecimals para {coin}: {e}")
        return 5


def _truncate_qty(qty: float, sz_decimals: int) -> float:
    """
    Trunca quantidade para o número de decimais permitido.

    Args:
        qty: Quantidade a truncar.
        sz_decimals: Número de decimais.

    Returns:
        float: Quantidade truncada.
    """
    if sz_decimals <= 0:
        return round(qty)
    factor = 10 ** sz_decimals
    return float(Decimal(str(qty)).quantize(Decimal(str(1 / factor)), rounding=ROUND_DOWN))


def _format_price(price: float, sz_decimals: int) -> float:
    """
    Formata preço para o formato aceito pela Hyperliquid.

    A Hyperliquid usa 5 SIGNIFICANT FIGURES para preços (não casas decimais fixas).
    Ex: 1929.05 → 1929.0 (5 sig figs)
        0.0012345 → 0.00123 (5 sig figs)
        123456 → 123450 (5 sig figs)

    Args:
        price: Preço a formatar.
        sz_decimals: Número de decimais (usado como fallback).

    Returns:
        float: Preço formatado com 5 significant figures.
    """
    if price <= 0:
        return 0.0

    import math
    # Calcular 5 significant figures
    if price >= 1:
        # Para preços >= 1: arredondar para 5 sig figs
        # Ex: 1929.05 → log10(1929.05) ≈ 3.28 → floor = 3 → 5-3-1 = 1 casa decimal
        sig_figs = 5
        decimals = sig_figs - int(math.floor(math.log10(abs(price)))) - 1
        decimals = max(0, decimals)
        return round(price, decimals)
    else:
        # Para preços < 1: arredondar para 5 sig figs
        # Ex: 0.0012345 → log10(0.0012345) ≈ -2.9 → floor = -3 → 5-(-3)-1 = 7 casas
        sig_figs = 5
        decimals = sig_figs - int(math.floor(math.log10(abs(price)))) - 1
        decimals = max(0, decimals)
        return round(price, decimals)


def generate_cloid() -> Cloid:
    """
    Gera Client Order ID único para idempotência de ordens.

    Returns:
        Cloid: Identificador único baseado em timestamp + random.
    """
    # Usa timestamp + random para garantir unicidade
    timestamp = int(time.time() * 1000)
    random_part = int.from_bytes(os.urandom(4), "big")
    cloid_int = (timestamp << 32) | (random_part & 0xFFFFFFFF)
    return Cloid.from_int(cloid_int)


# ──────────────────────────────────────────────
# Classe Principal — HyperliquidConnector
# ──────────────────────────────────────────────


class HyperliquidConnector:
    """
    Conector principal para a Hyperliquid DEX.

    Gerencia:
    - Conexão ccxt para consultas de mercado
    - Conexão SDK nativa para ordens
    - WebSocket para dados em tempo real
    - Cache de metadados (szDecimals, ativos)
    """

    def __init__(self, cfg: BotConfig) -> None:
        """
        Inicializa o conector.

        Args:
            cfg: Configuração do bot.
        """
        self.cfg = cfg
        self._ccxt: Optional[CcxtExchange] = None
        self._native: Optional[HLExchange] = None
        self._info: Optional[Info] = None
        self._ws_connections: Dict[str, Any] = {}
        self._sz_decimals_cache: Dict[str, int] = {}
        self._asset_meta_cache: Optional[Dict[str, Any]] = None
        self._connected = False
        self._cache: Dict[str, Dict[str, Any]] = {}  # Cache TTL para métodos Info

        log.info("HyperliquidConnector inicializado")

    # ── Propriedades ──

    @property
    def ccxt(self) -> CcxtExchange:
        """Retorna instância ccxt (conexão de consulta)."""
        if self._ccxt is None:
            self._ccxt = self._build_ccxt()
        return self._ccxt

    @property
    def native(self) -> HLExchange:
        """Retorna instância do SDK nativo (conexão de ordens)."""
        if self._native is None:
            self._native = self._build_native()
        return self._native

    @property
    def info(self) -> Info:
        """Retorna instância Info do SDK."""
        if self._info is None:
            self._info = self._build_info()
        return self._info

    @property
    def connected(self) -> bool:
        """Indica se o conector está ativo."""
        return self._connected

    # ── Métodos de Conexão ──

    def connect(self) -> bool:
        """
        Estabelece todas as conexões.

        Returns:
            bool: True se conectou com sucesso.
        """
        try:
            log.info("Conectando à Hyperliquid...")
            # Força inicialização
            _ = self.ccxt
            _ = self.native
            _ = self.info
            self._connected = True
            log.info("Conectado à Hyperliquid com sucesso")
            return True
        except Exception as e:
            log.error(f"Falha ao conectar à Hyperliquid: {e}")
            self._connected = False
            raise ConnectionError(f"Não foi possível conectar: {e}") from e

    def disconnect(self) -> None:
        """Fecha todas as conexões."""
        try:
            # Fecha WebSockets
            for ws_id, ws in self._ws_connections.items():
                try:
                    ws.close()
                except Exception:
                    pass
            self._ws_connections.clear()

            # Fecha ccxt
            if self._ccxt:
                try:
                    self._ccxt.close()
                except Exception:
                    pass

            self._connected = False
            log.info("Desconectado da Hyperliquid")
        except Exception as e:
            log.error(f"Erro ao desconectar: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Verifica saúde das conexões.

        Returns:
            Dict com status de cada conexão.
        """
        status = {
            "connected": self._connected,
            "ccxt": self._ccxt is not None,
            "native": self._native is not None,
            "info": self._info is not None,
            "ws_connections": len(self._ws_connections),
        }

        # Testa conectividade real
        try:
            mids = self.info.all_mids()
            status["api_reachable"] = True
            status["assets_count"] = len(mids) if mids else 0
        except Exception as e:
            status["api_reachable"] = False
            status["api_error"] = str(e)

        return status

    # ── Métodos de Mercado ──

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 350,
    ) -> Optional[List[List[float]]]:
        """
        Busca dados OHLCV via ccxt com validação de sanity check.

        Tenta múltiplos formatos de símbolo:
        - BTC/USDC (padrão)
        - BTC/USDC:USDC (swap)

        Aplica sanity check nos preços retornados para detectar dados
        corrompidos (ex: ccxt retornar preço de outro ativo).

        Args:
            symbol: Símbolo (ex: BTC/USDC).
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            limit: Número de candles.

        Returns:
            Lista de candles [timestamp, open, high, low, close, volume] ou None.
        """
        # Faixas de preço esperadas para sanity check
        PRICE_RANGES: Dict[str, Tuple[float, float]] = {
            "BTC": (20000, 200000),
            "ETH": (1000, 10000),
            "SOL": (20, 500),
            "HYPE": (5, 200),
            "ARB": (0.1, 10),
            "OP": (0.5, 20),
            "PURR": (0.01, 5),
        }

        # Mapeamento explícito: força perpetual swap para evitar confusão do ccxt
        SYMBOL_MAP: Dict[str, str] = {
            "BTC/USDC": "BTC/USDC:USDC",
            "ETH/USDC": "ETH/USDC:USDC",
            "SOL/USDC": "SOL/USDC:USDC",
        }

        # Lista de formatos de símbolo para tentar (mapeado primeiro, depois original)
        symbols_to_try = [SYMBOL_MAP.get(symbol, symbol)]
        if SYMBOL_MAP.get(symbol, symbol) != symbol:
            symbols_to_try.append(symbol)
        elif "/" in symbol:
            base = symbol.split("/")[0]
            alt = f"{base}/USDC:USDC"
            if alt != symbol:
                symbols_to_try.append(alt)

        for sym in symbols_to_try:
            for attempt in range(MAX_RETRIES):
                try:
                    ohlcv = self.ccxt.fetch_ohlcv(sym, timeframe, limit=limit)
                    if ohlcv and len(ohlcv) > 0:
                        # Sanity check: validar preço do último candle
                        last_close = float(ohlcv[-1][4])
                        base = symbol.split("/")[0]
                        expected_range = PRICE_RANGES.get(base)

                        if expected_range:
                            min_px, max_px = expected_range
                            if not (min_px <= last_close <= max_px):
                                log.error(
                                    f"[SANITY] OHLCV {sym}: preço suspeito "
                                    f"close={last_close:.2f} (esperado "
                                    f"{min_px:.0f}-{max_px:.0f} para {base}). "
                                    f"Tentando próximo formato..."
                                )
                                continue  # Tenta próximo símbolo

                        log.debug(f"OHLCV {sym} {timeframe}: {len(ohlcv)} candles "
                                  f"(último close={last_close:.2f})")
                        return ohlcv
                    log.debug(f"OHLCV vazio para {sym} {timeframe}, tentando próximo formato...")
                    break  # Sai do loop de retry, tenta próximo símbolo
                except ccxt.RateLimitExceeded as e:
                    wait = RETRY_DELAY_SEC * (attempt + 1) * 2
                    log.warning(f"Rate limit ao buscar OHLCV {sym}, aguardando {wait}s...")
                    time.sleep(wait)
                except ccxt.NetworkError as e:
                    log.error(f"Erro de rede ao buscar OHLCV {sym}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY_SEC)
                except Exception as e:
                    log.debug(f"Erro ao buscar OHLCV {sym}: {e}")
                    break  # Tenta próximo formato

        log.warning(f"Falha ao buscar OHLCV {symbol} em todos os formatos")
        return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Busca ticker atual via ccxt.

        Args:
            symbol: Símbolo (ex: BTC/USDC).

        Returns:
            Dict com dados do ticker ou None.
        """
        try:
            ticker = self.ccxt.fetch_ticker(symbol)
            log.debug(f"Ticker {symbol}: bid={ticker.get('bid')}, ask={ticker.get('ask')}")
            return ticker
        except Exception as e:
            log.error(f"Erro ao buscar ticker {symbol}: {e}")
            return None

    def fetch_order_book(self, symbol: str, limit: int = 25) -> Optional[Dict[str, Any]]:
        """
        Busca order book via ccxt.

        Args:
            symbol: Símbolo (ex: BTC/USDC).
            limit: Profundidade.

        Returns:
            Dict com bids/asks ou None.
        """
        try:
            ob = self.ccxt.fetch_order_book(symbol, limit)
            return ob
        except Exception as e:
            log.error(f"Erro ao buscar order book {symbol}: {e}")
            return None

    def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        Busca saldo da conta via ccxt.

        O ccxt hyperliquid requer o parâmetro 'user' com o endereço da wallet.

        Returns:
            Dict com saldos ou None.
        """
        try:
            # Hyperliquid ccxt precisa do endereço da wallet como parâmetro 'user'
            balance = self.ccxt.fetch_balance(params={"user": self.cfg.hyperliquid_account_address})
            return balance
        except Exception as e:
            log.error(f"Erro ao buscar balance: {e}")
            return None

    def fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Busca posições abertas via ccxt.

        O ccxt hyperliquid requer o parâmetro 'user' com o endereço da wallet,
        similar ao fetch_balance.

        Returns:
            Lista de posições.
        """
        try:
            positions = self.ccxt.fetch_positions(params={"user": self.cfg.hyperliquid_account_address})
            return positions
        except Exception as e:
            log.error(f"Erro ao buscar posições: {e}")
            return []

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca ordens abertas.

        Args:
            symbol: Símbolo opcional para filtrar.

        Returns:
            Lista de ordens abertas.
        """
        try:
            orders = self.ccxt.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            log.error(f"Erro ao buscar ordens abertas: {e}")
            return []

    # ── Métodos de Ordem (SDK Nativo) ──

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "limit",
        price: Optional[float] = None,
        reduce_only: bool = False,
        cloid: Optional[Cloid] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Coloca uma ordem na Hyperliquid usando SDK nativo.

        Args:
            symbol: Símbolo (ex: BTC/USDC).
            side: "buy" ou "sell".
            size: Quantidade em moeda base.
            order_type: "limit" ou "market".
            price: Preço (obrigatório para limit).
            reduce_only: Se True, ordem de redução.
            cloid: Client Order ID para idempotência.

        Returns:
            Dict com resposta da ordem ou None.

        Raises:
            OrderError: Se a ordem falhar.
        """
        coin = _extract_coin(symbol)
        sz_decimals = self._get_sz_decimals(coin)
        qty = _truncate_qty(abs(size), sz_decimals)

        if qty <= 0:
            raise OrderError(f"Quantidade inválida: {size} -> {qty}")

        try:
            if order_type == "market":
                # Ordem a mercado
                result = self.native.market_open(
                    name=coin,
                    is_buy=(side.lower() == "buy"),
                    sz=qty,
                    slippage=DEFAULT_SLIPPAGE,
                )
            else:
                # Ordem limitada
                if price is None:
                    raise OrderError("Preço é obrigatório para ordem limit")

                formatted_price = _format_price(price, sz_decimals)
                cloid = cloid or generate_cloid()

                order_result = self.native.order(
                    name=coin,
                    is_buy=(side.lower() == "buy"),
                    sz=qty,
                    limit_px=formatted_price,
                    order_type={"limit": {"tif": "Gtc"}},
                    reduce_only=reduce_only,
                    cloid=cloid,
                )
                result = order_result

            log.info(f"Ordem executada: {side.upper()} {qty} {coin} @ {price or 'MARKET'}")
            return result

        except Exception as e:
            log.error(f"Erro ao colocar ordem {side} {qty} {coin}: {e}")
            raise OrderError(f"Falha na ordem: {e}") from e

    def place_tpsl_order(
        self,
        symbol: str,
        side: str,
        size: float,
        trigger_px: float,
        tpsl_type: str,
        is_market: bool = True,
        cloid: Optional[Cloid] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Coloca ordem TP/SL nativa (trigger order).

        Args:
            symbol: Símbolo (ex: BTC/USDC).
            side: "buy" ou "sell".
            size: Quantidade.
            trigger_px: Preço de trigger.
            tpsl_type: "tp" (take profit) ou "sl" (stop loss).
            is_market: Se True, executa a mercado ao trigger.
            cloid: Client Order ID.

        Returns:
            Dict com resposta ou None.
        """
        coin = _extract_coin(symbol)
        sz_decimals = self._get_sz_decimals(coin)
        qty = _truncate_qty(abs(size), sz_decimals)

        if qty <= 0:
            raise OrderError(f"Quantidade inválida para TP/SL: {size}")

        try:
            cloid = cloid or generate_cloid()

            order_result = self.native.order(
                name=coin,
                is_buy=(side.lower() == "buy"),
                sz=qty,
                limit_px=trigger_px,
                order_type={
                    "trigger": {
                        "triggerPx": trigger_px,
                        "isMarket": is_market,
                        "tpsl": tpsl_type,
                    }
                },
                reduce_only=True,
                cloid=cloid,
            )

            log.info(f"Ordem {tpsl_type.upper()} colocada: {side.upper()} {qty} {coin} @ {trigger_px}")
            return order_result

        except Exception as e:
            log.error(f"Erro ao colocar ordem {tpsl_type} {coin}: {e}")
            raise OrderError(f"Falha na ordem {tpsl_type}: {e}") from e

    def place_bulk_tpsl(
        self,
        symbol: str,
        entry_side: str,
        entry_size: float,
        entry_price: float,
        stop_loss_px: float,
        take_profit_px: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Coloca ordem de entrada + TP + SL em uma única chamada bulk_orders.

        ANTES de criar novas ordens, CANCELA todas as ordens abertas do mesmo
        símbolo para evitar acúmulo de múltiplos pares TP/SL.

        Usa o formato HIGH-LEVEL do SDK (coin, is_buy, sz, limit_px, order_type),
        que o SDK converte internamente para wire-level via order_request_to_order_wire().

        O SDK native.bulk_orders() retorna um Dict com:
            {"status": "ok", "response": {"data": {"statuses": [...]}}}

        Args:
            symbol: Símbolo.
            entry_side: "buy" ou "sell".
            entry_size: Tamanho da entrada.
            entry_price: Preço de entrada.
            stop_loss_px: Preço do Stop Loss.
            take_profit_px: Preço do Take Profit.

        Returns:
            Dict com resposta do SDK ou None em caso de erro.
        """
        coin = _extract_coin(symbol)
        sz_decimals = self._get_sz_decimals(coin)
        qty = _truncate_qty(abs(entry_size), sz_decimals)
        is_buy = entry_side.lower() == "buy"

        try:
            # PASSO 1: Cancelar TP/SL antigos deste símbolo para evitar duplicatas
            try:
                old_orders = self.info.open_orders(self.cfg.hyperliquid_account_address)
                if old_orders:
                    cancel_list = []
                    for o in old_orders:
                        if isinstance(o, dict) and o.get("coin") == coin:
                            oid = o.get("oid")
                            if oid is not None:
                                cancel_list.append({"coin": coin, "oid": int(oid)})
                    if cancel_list:
                        self.native.bulk_cancel(cancel_requests=cancel_list)
                        log.info(
                            f"[BULK] Canceladas {len(cancel_list)} ordem(ns) antiga(s) "
                            f"de {coin} antes de criar novos TP/SL"
                        )
            except Exception as e:
                log.debug(f"[BULK] Erro ao cancelar ordens antigas de {coin}: {e}")

            # PASSO 2: Criar novas ordens
            entry_cloid = generate_cloid()
            sl_cloid = generate_cloid()
            tp_cloid = generate_cloid()

            # Formato HIGH-LEVEL (o que o SDK espera em bulk_orders)
            orders = [
                {
                    "coin": coin,
                    "is_buy": is_buy,
                    "sz": qty,
                    "limit_px": _format_price(entry_price, sz_decimals),
                    "order_type": {"limit": {"tif": "Gtc"}},
                    "reduce_only": False,
                    "cloid": entry_cloid,
                },
                {
                    "coin": coin,
                    "is_buy": not is_buy,
                    "sz": qty,
                    "limit_px": _format_price(stop_loss_px, sz_decimals),
                    "order_type": {
                        "trigger": {
                            "triggerPx": stop_loss_px,
                            "isMarket": True,
                            "tpsl": "sl",
                        }
                    },
                    "reduce_only": True,
                    "cloid": sl_cloid,
                },
                {
                    "coin": coin,
                    "is_buy": not is_buy,
                    "sz": qty,
                    "limit_px": _format_price(take_profit_px, sz_decimals),
                    "order_type": {
                        "trigger": {
                            "triggerPx": take_profit_px,
                            "isMarket": True,
                            "tpsl": "tp",
                        }
                    },
                    "reduce_only": True,
                    "cloid": tp_cloid,
                },
            ]

            # Usa bulk_orders com grouping="normalTpsl"
            result = self.native.bulk_orders(
                order_requests=orders,
                grouping="normalTpsl",
            )

            log.info(f"Bulk TP/SL: entrada {entry_side.upper()} {qty} {coin} @ {entry_price}")
            return result

        except Exception as e:
            log.error(f"Erro no bulk TP/SL {coin}: {e}")
            raise OrderError(f"Falha no bulk TP/SL: {e}") from e

    def cancel_order(self, symbol: str, oid: int) -> Optional[Dict[str, Any]]:
        """
        Cancela uma ordem específica.

        Args:
            symbol: Símbolo.
            oid: ID da ordem.

        Returns:
            Dict com resposta ou None.
        """
        coin = _extract_coin(symbol)
        try:
            result = self.native.cancel(name=coin, oid=oid)
            log.info(f"Ordem cancelada: {oid} {coin}")
            return result
        except Exception as e:
            log.error(f"Erro ao cancelar ordem {oid}: {e}")
            return None

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancela todas as ordens abertas.

        Usa o formato high-level CancelRequest (coin, oid) que o SDK espera.

        Args:
            symbol: Símbolo opcional para filtrar.

        Returns:
            int: Número de ordens canceladas.
        """
        try:
            if symbol:
                coin = _extract_coin(symbol)
                # Busca ordens abertas para este símbolo para obter os OIDs
                open_orders = self.info.open_orders(self.cfg.hyperliquid_account_address)
                cancel_requests = []
                for order in open_orders:
                    if order.get("coin", "").upper() == coin.upper():
                        cancel_requests.append({
                            "coin": order["coin"],
                            "oid": order["oid"],
                        })
                if cancel_requests:
                    result = self.native.bulk_cancel(cancel_requests)
                    log.info(f"Ordens canceladas: {len(cancel_requests)} em {symbol}")
                    return len(cancel_requests)
                log.debug(f"[CANCEL] Nenhuma ordem aberta para {symbol}")
                return 0
            else:
                # Cancela todas as ordens
                open_orders = self.info.open_orders(self.cfg.hyperliquid_account_address)
                if not open_orders:
                    log.debug("[CANCEL] Nenhuma ordem aberta")
                    return 0

                cancel_requests = [
                    {"coin": o["coin"], "oid": o["oid"]}
                    for o in open_orders
                ]
                result = self.native.bulk_cancel(cancel_requests)
                log.info(f"Ordens canceladas: {len(cancel_requests)} (todas)")
                return len(cancel_requests)

        except Exception as e:
            log.error(f"Erro ao cancelar ordens: {e}")
            return 0

    def close_position(self, symbol: str, qty: float, side: str) -> Optional[str]:
        """
        Fecha posição usando ordem a mercado.

        Usa o SDK nativo market_close() com parâmetro 'coin=' (correto),
        não 'name=' (que não existe na API).

        Args:
            symbol: Símbolo.
            qty: Quantidade a fechar.
            side: Lado da posição ("buy" ou "sell").

        Returns:
            str: ID da ordem ou None.
        """
        coin = _extract_coin(symbol)
        sz_decimals = self._get_sz_decimals(coin)
        qty = _truncate_qty(abs(qty), sz_decimals)

        if qty <= 0:
            log.warning(f"Quantidade inválida para fechar posição {symbol}: {qty}")
            return None

        try:
            # SDK nativo: market_close(coin=, sz=, slippage=) — NÃO use name=
            result = self.native.market_close(
                coin=coin,
                sz=qty,
                slippage=DEFAULT_SLIPPAGE,
            )
            log.info(f"Posição fechada: {qty} {coin}")
            return str(result)
        except Exception as e:
            log.error(f"Erro ao fechar posição {coin}: {e}")
            return None

    def update_leverage(self, symbol: str, leverage: int, is_cross: bool = False) -> bool:
        """
        Atualiza alavancagem para um símbolo.

        Args:
            symbol: Símbolo.
            leverage: Alavancagem (1-50).
            is_cross: Se True, margem cruzada.

        Returns:
            bool: True se sucesso.
        """
        coin = _extract_coin(symbol)
        try:
            self.native.update_leverage(leverage=leverage, name=coin, is_cross=is_cross)
            log.info(f"Alavancagem {coin}: {leverage}x {'cross' if is_cross else 'isolated'}")
            return True
        except Exception as e:
            log.error(f"Erro ao atualizar alavancagem {coin}: {e}")
            return False

    # ── Métodos Info ──

    def get_user_state(self) -> Optional[Dict[str, Any]]:
        """
        Retorna estado completo da conta com cache de curta duração.

        O cache (TTL: 2s) evita rate limiting quando múltiplas partes
        do bot chamam este método repetidamente no mesmo ciclo.
        """
        cache_key = "user_state"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and (now - cached["ts"] < 2.0):
            return cached["data"]

        try:
            data = self.info.user_state(self.cfg.hyperliquid_account_address)
            self._cache[cache_key] = {"data": data, "ts": now}
            return data
        except Exception as e:
            log.error(f"Erro ao buscar user state: {e}")
            return None

    def get_all_mids(self) -> Optional[Dict[str, str]]:
        """Retorna preços médios de todos os ativos."""
        try:
            return self.info.all_mids()
        except Exception as e:
            log.error(f"Erro ao buscar all_mids: {e}")
            return None

    def get_mark_price(self, coin: str) -> Optional[float]:
        """
        Obtém o mark price real de um ativo via metaAndAssetCtxs.

        O mark price (markPx) é o preço usado pela Hyperliquid para:
        - Cálculo de P&L não realizado
        - Margem e liquidações
        - Gatilhos de stop-loss e take-profit

        Diferente do mid price (média bid/ask), o mark price é um blend
        suavizado do oracle, trades recentes e preços de exchanges externas.

        Args:
            coin: Nome do ativo (ex: "ETH").

        Returns:
            float: Mark price real ou None se indisponível.
        """
        try:
            # Normaliza: remove /USDC, :USDC, etc.
            coin = coin.split("/")[0].split(":")[0]

            # 1. Tentar markPx real via metaAndAssetCtxs (fonte oficial)
            meta_and_ctxs = self.info.meta_and_asset_ctxs()
            if meta_and_ctxs and len(meta_and_ctxs) == 2:
                meta, asset_ctxs = meta_and_ctxs
                universe = meta.get("universe", [])
                for i, asset in enumerate(universe):
                    if asset.get("name") == coin:
                        mark_px_raw = asset_ctxs[i].get("markPx")
                        if mark_px_raw is not None:
                            mark_px = float(mark_px_raw)
                            log.debug(f"Mark price {coin}: {mark_px} (via metaAndAssetCtxs)")
                            return mark_px
                        break

            # 2. Fallback para all_mids (mid price — não é mark price real)
            log.warning(
                f"markPx não encontrado em metaAndAssetCtxs para {coin}, "
                f"usando all_mids como fallback (mid price, não mark price)"
            )
            all_mids = self.get_all_mids()
            if all_mids and coin in all_mids:
                mid_px = float(all_mids[coin])
                log.debug(f"Mid price (fallback) {coin}: {mid_px}")
                return mid_px

            log.warning(f"Mark price não encontrado para {coin} em nenhuma fonte")
            return None
        except Exception as e:
            log.error(f"Erro ao buscar mark price para {coin}: {e}")
            return None

    def get_current_funding_rate(self, coin: str) -> Optional[float]:
        """
        Obtém o funding rate atual (não histórico) de um ativo via
        metaAndAssetCtxs.

        Diferente de get_funding_history(), que retorna uma série
        histórica, este método retorna a taxa vigente no momento,
        útil para:
        - Alimentar indicators.add_funding_rate() no ciclo ao vivo
        - Alimentar capital_protection.check_funding_rate()

        Args:
            coin: Nome do ativo (ex: "ETH").

        Returns:
            float: Funding rate atual (ex: 0.0001 = 0.01%) ou None
                se indisponível.
        """
        try:
            coin = coin.split("/")[0].split(":")[0]

            meta_and_ctxs = self.info.meta_and_asset_ctxs()
            if meta_and_ctxs and len(meta_and_ctxs) == 2:
                meta, asset_ctxs = meta_and_ctxs
                universe = meta.get("universe", [])
                for i, asset in enumerate(universe):
                    if asset.get("name") == coin:
                        funding_raw = asset_ctxs[i].get("funding")
                        if funding_raw is not None:
                            funding = float(funding_raw)
                            log.debug(f"Funding rate atual {coin}: {funding}")
                            return funding
                        break

            log.warning(f"Funding rate não encontrado para {coin} em metaAndAssetCtxs")
            return None
        except Exception as e:
            log.error(f"Erro ao buscar funding rate atual para {coin}: {e}")
            return None    

    def get_funding_history(self, coin: str, start_time: int, end_time: Optional[int] = None) -> List[Any]:
        """Retorna histórico de funding."""
        try:
            return self.info.funding_history(coin, start_time, end_time)
        except Exception as e:
            log.error(f"Erro ao buscar funding history {coin}: {e}")
            return []

    def get_l2_snapshot(self, coin: str) -> Optional[Dict[str, Any]]:
        """Retorna snapshot do L2 book."""
        try:
            return self.info.l2_snapshot(coin)
        except Exception as e:
            log.error(f"Erro ao buscar L2 snapshot {coin}: {e}")
            return None

    def get_candles_snapshot(self, coin: str, interval: str, start_time: int, end_time: int) -> List[Any]:
        """Retorna snapshot de candles."""
        try:
            return self.info.candles_snapshot(coin, interval, start_time, end_time)
        except Exception as e:
            log.error(f"Erro ao buscar candles snapshot {coin}: {e}")
            return []

    def get_user_fills(self) -> List[Any]:
        """Retorna preenchimentos do usuário."""
        try:
            return self.info.user_fills(self.cfg.hyperliquid_account_address)
        except Exception as e:
            log.error(f"Erro ao buscar fills: {e}")
            return []

    def get_user_staking_summary(self) -> Optional[Dict[str, Any]]:
        """Retorna resumo de staking."""
        try:
            return self.info.user_staking_summary(self.cfg.hyperliquid_account_address)
        except Exception as e:
            log.error(f"Erro ao buscar staking summary: {e}")
            return None

    def get_user_staking_rewards(self) -> List[Any]:
        """Retorna recompensas de staking."""
        try:
            return self.info.user_staking_rewards(self.cfg.hyperliquid_account_address)
        except Exception as e:
            log.error(f"Erro ao buscar staking rewards: {e}")
            return []

    def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Retorna portfólio do usuário."""
        try:
            return self.info.portfolio(self.cfg.hyperliquid_account_address)
        except Exception as e:
            log.error(f"Erro ao buscar portfolio: {e}")
            return None

    # ── WebSocket ──

    def subscribe_ws(
        self,
        subscription: Dict[str, Any],
        callback: Callable[[Any], None],
    ) -> Optional[int]:
        """
        Inscreve em um canal WebSocket.

        Args:
            subscription: Dict com tipo e coin (ex: {"type": "allMids"}).
            callback: Função callback para dados recebidos.

        Returns:
            int: ID da subscription ou None.
        """
        try:
            sub_id = self.info.subscribe(subscription, callback)
            log.info(f"Inscrito no WebSocket: {subscription}")
            return sub_id
        except Exception as e:
            log.error(f"Erro ao inscrever no WebSocket: {e}")
            return None

    def unsubscribe_ws(self, subscription: Dict[str, Any], sub_id: int) -> bool:
        """Remove inscrição WebSocket."""
        try:
            self.info.unsubscribe(subscription, sub_id)
            return True
        except Exception as e:
            log.error(f"Erro ao remover inscrição WebSocket: {e}")
            return False

    # ── Staking ──

    def delegate_stake(self, validator: str, amount_wei: int) -> Optional[Dict[str, Any]]:
        """
        Delega HYPE para staking.

        Args:
            validator: Endereço do validador.
            amount_wei: Quantidade em wei.

        Returns:
            Dict com resultado ou None.
        """
        try:
            result = self.native.token_delegate(
                validator=validator,
                wei=amount_wei,
                is_undelegate=False,
            )
            log.info(f"Staking delegado: {amount_wei} wei para {validator}")
            return result
        except Exception as e:
            log.error(f"Erro ao delegar staking: {e}")
            return None

    def undelegate_stake(self, validator: str, amount_wei: int) -> Optional[Dict[str, Any]]:
        """Remove delegação de staking."""
        try:
            result = self.native.token_delegate(
                validator=validator,
                wei=amount_wei,
                is_undelegate=True,
            )
            log.info(f"Staking removido: {amount_wei} wei de {validator}")
            return result
        except Exception as e:
            log.error(f"Erro ao remover staking: {e}")
            return None

    def vault_transfer(self, vault_address: str, is_deposit: bool, usd: int) -> Optional[Dict[str, Any]]:
        """
        Transfere fundos para/de um vault.

        Args:
            vault_address: Endereço do vault.
            is_deposit: True para depositar, False para sacar.
            usd: Quantidade em centavos de USD.

        Returns:
            Dict com resultado ou None.
        """
        try:
            result = self.native.vault_usd_transfer(
                vault_address=vault_address,
                is_deposit=is_deposit,
                usd=usd,
            )
            action = "depositado" if is_deposit else "sacado"
            log.info(f"Vault {action}: {usd} cents para {vault_address}")
            return result
        except Exception as e:
            log.error(f"Erro ao transferir para vault: {e}")
            return None

    # ── Métodos Internos ──

    def _build_ccxt(self) -> CcxtExchange:
        """Constrói e configura cliente ccxt."""
        try:
            exchange_id = "hyperliquid"
            exchange_class = getattr(ccxt, exchange_id)
            ex = exchange_class({
                "apiKey": self.cfg.hyperliquid_account_address,
                "secret": self.cfg.hyperliquid_private_key,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                },
            })

            if self.cfg.testnet:
                ex.urls["api"] = {
                    "public": "https://api.hyperliquid-testnet.xyz",
                    "private": "https://api.hyperliquid-testnet.xyz",
                }

            log.info("Cliente ccxt configurado")
            return ex
        except Exception as e:
            log.error(f"Erro ao configurar ccxt: {e}")
            raise ConnectionError(f"Falha ao configurar ccxt: {e}") from e

    def _build_native(self) -> HLExchange:
        """Constrói e configura cliente SDK nativo.

        O SDK oficial (Exchange) espera um objeto LocalAccount (eth_account),
        não um dict. Usamos Account.from_key() para criar a wallet corretamente.
        """
        try:
            pk, addr = _get_credentials(self.cfg)

            # SDK oficial requer LocalAccount (eth_account), não dict
            from eth_account import Account
            wallet = Account.from_key(pk)

            if self.cfg.testnet:
                exchange = HLExchange(
                    wallet=wallet,
                    base_url=constants.TESTNET_API_URL,
                    account_address=addr,
                )
            else:
                exchange = HLExchange(
                    wallet=wallet,
                    base_url=constants.MAINNET_API_URL,
                    account_address=addr,
                )

            log.info("Cliente SDK nativo configurado")
            return exchange
        except Exception as e:
            log.error(f"Erro ao configurar SDK nativo: {e}")
            raise ConnectionError(f"Falha ao configurar SDK nativo: {e}") from e

    def _build_info(self) -> Info:
        """Constrói cliente Info do SDK."""
        try:
            if self.cfg.testnet:
                info = Info(
                    base_url=constants.TESTNET_API_URL,
                    skip_ws=True,
                )
            else:
                info = Info(
                    base_url=constants.MAINNET_API_URL,
                    skip_ws=True,
                )
            return info
        except Exception as e:
            log.error(f"Erro ao configurar Info: {e}")
            raise ConnectionError(f"Falha ao configurar Info: {e}") from e

    def _get_sz_decimals(self, coin: str) -> int:
        """Obtém szDecimals com cache."""
        if coin in self._sz_decimals_cache:
            return self._sz_decimals_cache[coin]

        sz_decimals = _get_sz_decimals(self.info, coin)
        self._sz_decimals_cache[coin] = sz_decimals
        return sz_decimals


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

_connector_instance: Optional[HyperliquidConnector] = None


def get_connector(cfg: Optional[BotConfig] = None, reload: bool = False) -> HyperliquidConnector:
    """
    Retorna instância singleton do conector.

    Args:
        cfg: Configuração (usa global se None).
        reload: Se True, recria a instância.

    Returns:
        HyperliquidConnector.
    """
    global _connector_instance
    if _connector_instance is None or reload:
        from ..config import get_config
        cfg = cfg or get_config()
        _connector_instance = HyperliquidConnector(cfg)
        _connector_instance.connect()
    return _connector_instance
