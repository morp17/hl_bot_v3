"""
Hyperliquid Production Bot v3.0 — Ponto de Entrada Principal
=============================================================
Bot de trading automatizado para Hyperliquid DEX.

Modos de operação:
- live: Trading real (requer credenciais)
- backtest: Backtesting com dados históricos
- dashboard: Servidor web local para monitoramento (modo standalone
  de debug — no modo live, o dashboard já sobe automaticamente como
  thread em background dentro do mesmo processo)

Uso:
    python main.py --mode live
    python main.py --mode backtest --symbol BTC/USDC
    python main.py --mode dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger as log

from crypto_bot_core.capital_protection import CapitalProtection
from crypto_bot_core.config import BotConfig, StrategyType, SymbolConfig, get_config
from crypto_bot_core.execution import OrderExecutor
from crypto_bot_core.indicators import add_all_indicators, get_latest_values
from crypto_bot_core.monitoring import Monitor
from crypto_bot_core.notifications import Notificator
from crypto_bot_core.risk import (
    LIQUIDATION_SAFETY_BUFFER_PCT,
    CHECK_EXITS_MISS_THRESHOLD,
    Position,
    PositionManager,
    calc_position_size,
    calc_stops,
    trade_hours_ok,
)
from crypto_bot_core.health_server import HealthServer, update_health
from crypto_bot_core.lock import LockManager
from crypto_bot_core.staking import StakingManager
from crypto_bot_core.strategies.signals import get_signal


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

# Valor mínimo de ordem exigido pela Hyperliquid. Abaixo disso a API
# rejeita com "Order must have minimum value of $10" — checar ANTES
# de enviar evita uma chamada de API fadada a falhar e o log de erro
# ruidoso correspondente a cada ciclo.
MIN_ORDER_NOTIONAL_USD = 10.0


# ──────────────────────────────────────────────
# HyperliquidBot
# ──────────────────────────────────────────────


class HyperliquidBot:
    """
    Bot principal de trading para Hyperliquid.

    Integra todos os módulos:
    - Config → Estratégia → Sinais → Risco → Execução → Monitoramento
    """

    def __init__(self, cfg: BotConfig) -> None:
        """
        Inicializa o bot.

        Args:
            cfg: Configuração carregada.

        Raises:
            RuntimeError: Se outra instância do bot já estiver rodando
                          (lock file detectado).
        """
        self.cfg = cfg
        self.running = False
        self.paused = False

        # Lock file para prevenir múltiplas instâncias
        self._lock = LockManager()
        if not self._lock.acquire():
            raise RuntimeError(
                "Outra instância do bot já está rodando. "
                "Se tiver certeza que não, remova o lock file manualmente "
                "ou use --force para sobrescrever."
            )

        # Módulos
        r = cfg.risk  # atalho para RiskConfig
        n = cfg.notifications  # atalho para NotificationConfig
        s = cfg.staking  # atalho para StakingConfig

        self.executor = OrderExecutor(cfg)
        self.position_manager = PositionManager(
            capital_usd=cfg.capital_usd,
            max_open_trades=r.max_open_trades,
            max_drawdown_pct=r.max_drawdown_pct / 100.0,
            daily_loss_limit_pct=r.daily_loss_limit_pct / 100.0,
            max_consecutive_losses=r.max_consecutive_losses,
            cooldown_after_loss_sec=r.cooldown_after_loss_sec,
            trailing_stop=r.trailing_stop,
            trailing_activation_pct=r.trailing_stop_activation_pct / 100.0,
            trailing_stop_pct=r.trailing_stop_distance_pct / 100.0,
            taker_fee=r.taker_fee,
            max_position_pct=cfg.max_position_pct / 100.0,   # FIX item 1/4
        )
        self.capital_protection = CapitalProtection(
            initial_balance=cfg.capital_usd,
            max_daily_loss_pct=r.daily_loss_limit_pct / 100.0,
            max_drawdown_pct=r.max_drawdown_pct / 100.0,
            max_consecutive_losses=r.max_consecutive_losses,
            cooldown_after_loss_sec=r.cooldown_after_loss_sec,
            max_exposure_pct=r.max_exposure_pct / 100.0,
            max_position_pct=cfg.max_position_pct / 100.0,
            circuit_breaker_loss_pct=r.circuit_breaker_loss_pct / 100.0,
            circuit_breaker_cooldown_sec=r.circuit_breaker_cooldown_sec,
            trade_hour_start_utc=r.trade_hour_start_utc,
            trade_hour_end_utc=r.trade_hour_end_utc,
            max_spread_pct=r.max_spread_pct / 100.0,
            max_funding_rate=r.max_funding_rate / 100.0,
            # FIX item 4 auditoria: btc_crash_filter_pct removido de
            # RiskConfig (campo morto sem verificação implementada) —
            # também removido daqui.
        )
        self.notificator = Notificator(
            telegram_token=n.telegram_bot_token or None,
            telegram_chat_id=n.telegram_chat_id or None,
            discord_webhook=n.discord_webhook_url or None,
            smtp_host=n.smtp_host or None,
            smtp_port=n.smtp_port,
            smtp_user=n.smtp_user or None,
            smtp_pass=n.smtp_password or None,
            email_from=n.email_from or None,
            email_to=n.email_to or None,
        )
        self.staking = StakingManager(cfg)
        self.monitor = Monitor()

        # Healthcheck server (thread separada, porta 8081)
        self._health_server = HealthServer(
            host=cfg.dashboard.host,
            port=cfg.monitoring.health_check_port,
        )
        self._health_server.start()

        # Dashboard server (FIX arquitetural: roda como thread DENTRO
        # deste processo, não mais como --mode dashboard separado —
        # ver crypto_bot_core/dashboard/main.py::start_dashboard_thread)
        from crypto_bot_core.dashboard.main import start_dashboard_thread
        self._dashboard_thread = start_dashboard_thread(cfg)

        # Estado
        self._symbols: List[SymbolConfig] = []   # ALTERADO: era List[str]
        self._last_signals: Dict[str, str] = {}
        self._synced_from_exchange: bool = False

        # ── Aplicar leverage/margem por símbolo (FIX item 3: antes nunca era chamado) ──
        try:
            symbol_configs = self.cfg.parse_symbols()
            for sc in symbol_configs:
                if not sc.enabled:
                    continue
                ok = self.executor.update_leverage(
                    symbol=sc.symbol,
                    leverage=sc.leverage,
                    is_cross=not sc.isolated_margin,
                )
                if not ok:
                    log.warning(
                        f"[INIT] Falha ao aplicar leverage {sc.leverage}x "
                        f"em {sc.symbol} — verifique conexão/credenciais"
                    )
        except Exception as e:
            log.error(f"[INIT] Erro ao aplicar leverage por símbolo: {e}")

        log.info("=" * 50)
        log.info("Hyperliquid Production Bot v3.0")
        log.info(f"Estratégia: {cfg.strategy.value}")
        log.info(f"Símbolos: {cfg.symbols}")
        log.info(f"Capital: ${cfg.capital_usd:,.2f}")
        log.info("=" * 50)

    def _load_symbols(self) -> List[SymbolConfig]:
        """Carrega configurações completas por símbolo (não só o nome)."""
        try:
            return self.cfg.parse_symbols()
        except Exception as e:
            log.error(f"Erro ao carregar símbolos: {e}")
            # fallback mínimo
            from crypto_bot_core.config import SymbolConfig as _SC
            return [
                _SC(symbol=s, coin=s.split("/")[0])
                for s in ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
            ]

    def _normalize_symbol(self, raw_symbol: str) -> str:
        """
        Normaliza símbolo do formato ccxt (ex: 'ETH/USDC:USDC') para
        o formato padrão do bot (ex: 'ETH/USDC').

        O ccxt Hyperliquid retorna símbolos no formato 'BASE/QUOTE:QUOTE'
        para perpétuos (ex: 'ETH/USDC:USDC'), enquanto o bot usa
        o formato simples 'BASE/QUOTE' (ex: 'ETH/USDC').
        """
        # Remove sufixo :USDC, :USD, etc.
        if ":" in raw_symbol:
            raw_symbol = raw_symbol.split(":")[0]
        return raw_symbol

    def _sync_from_exchange(self) -> None:
        """
        Sincroniza posições e ordens abertas da DEX no startup.

        REGRAS:
        1. CANCELA APENAS ordens órfãs (TP/SL sem posição correspondente).
           NÃO cancela ordens limit que ainda não preencheram (FIX bugC8).
        2. Restaura posições reais da Hyperliquid para o PositionManager
           local, normalizando símbolos do formato ccxt (:USDC).
        3. Ajusta peak_balance para o saldo REAL da exchange, não o .env.
        """
        try:
            connector = self.executor.connector
            log.info("[SYNC] Sincronizando posições da DEX...")

            # ── PASSO 0: Buscar posições abertas primeiro ──
            dex_positions = connector.fetch_positions()
            active_symbols: set = set()
            if dex_positions:
                for dp in dex_positions:
                    if not isinstance(dp, dict):
                        continue
                    raw_symbol = dp.get("symbol", "")
                    symbol = self._normalize_symbol(raw_symbol)
                    contracts = float(dp.get("contracts", 0) or 0)
                    if symbol and contracts > 0:
                        active_symbols.add(symbol)

            # ── PASSO 1: Cancelar APENAS ordens órfãs ──
            # FIX CRÍTICO (achado em teste ao vivo): além de decidir
            # cancelar ou manter, agora também coletamos os preços das
            # ordens MANTIDAS (TP/SL reais já na exchange) por símbolo,
            # para popular corretamente Position.stop_loss/take_profit
            # no PASSO 2 — antes esses campos ficavam sempre em 0.0 para
            # posições restauradas, e check_exits() fechava a posição
            # no ciclo seguinte por engano (price >= 0.0 é sempre
            # verdadeiro para SELL). Ver correção complementar em
            # risk.py::check_exits (guarda contra SL/TP não inicializado).
            kept_order_prices_by_symbol: Dict[str, List[float]] = {}
            try:
                open_orders = connector.info.open_orders(connector.cfg.hyperliquid_account_address)
                if open_orders and len(open_orders) > 0:
                    cancel_list = []
                    for o in open_orders:
                        if not isinstance(o, dict):
                            continue
                        coin = o.get("coin", "")
                        oid = o.get("oid")
                        if not coin or oid is None:
                            continue
                        symbol = f"{coin}/USDC"
                        if symbol not in active_symbols:
                            cancel_list.append({"coin": coin, "oid": int(oid)})
                        else:
                            log.debug(f"[SYNC] Mantendo ordem {coin} oid={oid} (símbolo com posição ativa)")
                            try:
                                px = float(o.get("limitPx", 0) or 0)
                                if px > 0:
                                    kept_order_prices_by_symbol.setdefault(symbol, []).append(px)
                            except (ValueError, TypeError):
                                pass
                    if cancel_list:
                        connector.native.bulk_cancel(cancel_requests=cancel_list)
                        log.info(f"[SYNC] Canceladas {len(cancel_list)} ordem(ns) órfã(s) da DEX")
                    else:
                        log.info("[SYNC] Nenhuma ordem órfã para cancelar")
            except Exception as e:
                log.debug(f"[SYNC] Erro ao cancelar ordens órfãs: {e}")

            # ── PASSO 2: Restaurar posições da DEX ──
            if dex_positions:
                for dp in dex_positions:
                    if not isinstance(dp, dict):
                        continue
                    raw_symbol = dp.get("symbol", "")
                    symbol = self._normalize_symbol(raw_symbol)
                    contracts = float(dp.get("contracts", 0) or 0)
                    if not symbol or contracts <= 0:
                        continue

                    side = "buy" if str(dp.get("side", "")).lower() == "long" else "sell"
                    entry_price = float(dp.get("entryPrice", 0) or 0)
                    if entry_price <= 0:
                        continue

                    existing = [p for p in self.position_manager.positions if p.symbol == symbol]
                    if existing:
                        log.debug(f"[SYNC] Posição {symbol} já existe localmente")
                        continue

                    # FIX CRÍTICO: inferir stop_loss/take_profit reais a
                    # partir dos preços das ordens trigger mantidas na
                    # exchange (coletadas no PASSO 1), em vez de deixar
                    # em 0.0. Para SELL: SL fica ACIMA da entrada, TP
                    # fica ABAIXO. Para BUY: o inverso.
                    order_prices = kept_order_prices_by_symbol.get(symbol, [])
                    restored_sl, restored_tp = 0.0, 0.0
                    if len(order_prices) >= 2:
                        if side == "sell":
                            restored_sl = max(order_prices)
                            restored_tp = min(order_prices)
                        else:
                            restored_sl = min(order_prices)
                            restored_tp = max(order_prices)
                        log.info(
                            f"[SYNC] {symbol}: SL/TP restaurados das ordens "
                            f"reais da exchange: SL={restored_sl} TP={restored_tp}"
                        )
                    elif len(order_prices) == 1:
                        log.warning(
                            f"[SYNC] {symbol}: apenas 1 ordem trigger encontrada "
                            f"({order_prices[0]}) — não é possível distinguir "
                            f"SL de TP com segurança. Posição restaurada SEM "
                            f"stop automático local (proteção real ainda existe "
                            f"na exchange via a ordem trigger restante, mas "
                            f"check_exits() do bot não vai fechá-la)."
                        )
                    else:
                        log.warning(
                            f"[SYNC] {symbol}: nenhuma ordem trigger encontrada "
                            f"na exchange — posição restaurada SEM stop_loss/"
                            f"take_profit locais. Considere fechar manualmente "
                            f"ou recriar TP/SL via dashboard/exchange."
                        )

                    pos = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        qty=contracts,
                        stop_loss=restored_sl,
                        take_profit=restored_tp,
                        trailing_stop_price=restored_sl,  # consistente com _process_symbol
                        open_time=time.time(),
                    )
                    self.position_manager.add(pos)
                    log.info(f"[SYNC] Posição restaurada da DEX: {side.upper()} {contracts} {symbol} @ {entry_price}")

            # ── PASSO 3: Log ordens abertas restantes (se houver) ──
            try:
                remaining_orders = connector.info.open_orders(connector.cfg.hyperliquid_account_address)
                if remaining_orders:
                    log.info(f"[SYNC] {len(remaining_orders)} ordem(ns) restante(s) na DEX (após cancelamento)")
                    for o in remaining_orders:
                        if isinstance(o, dict):
                            coin = o.get("coin", "")
                            oid = o.get("oid", "")
                            sz = o.get("sz", "0")
                            limit_px = o.get("limitPx", "0")
                            log.debug(f"[SYNC] Ordem: {coin} oid={oid} sz={sz} @ {limit_px}")
            except Exception as e:
                log.debug(f"[SYNC] Erro ao buscar ordens restantes: {e}")

            # ── PASSO 4: Atualizar saldo real da DEX ──
            try:
                balance_info = connector.fetch_balance()
                if balance_info and isinstance(balance_info, dict):
                    usdc_data = balance_info.get("USDC", {})
                    if isinstance(usdc_data, dict):
                        free_balance = float(usdc_data.get("free", 0) or 0)
                    else:
                        free_balance = float(balance_info.get("total", 0) or 0)

                    if free_balance > 0:
                        self.capital_protection.state.current_balance = free_balance

                        old_peak = self.capital_protection.state.peak_balance
                        if free_balance < old_peak:
                            self.capital_protection.state.peak_balance = free_balance
                            log.info(
                                f"[SYNC] Peak reajustado: ${old_peak:.2f} → "
                                f"${free_balance:.2f} (saldo real < capital config)"
                            )
                        else:
                            self.capital_protection.state.peak_balance = max(
                                old_peak, free_balance
                            )

                        self.position_manager.current_balance = self.capital_protection.state.current_balance
                        self.position_manager.peak_balance = self.capital_protection.state.peak_balance

                        log.info(f"[SYNC] Saldo atualizado: ${free_balance:.2f}")
            except Exception as e:
                log.debug(f"[SYNC] Erro ao buscar saldo: {e}")

            self._synced_from_exchange = True
            log.info(f"[SYNC] Sincronização concluída. {len(self.position_manager.positions)} posição(ões) ativa(s)")

        except Exception as e:
            log.error(f"[SYNC] Erro na sincronização: {e}")

    def _verify_positions_on_exchange(self) -> None:
        """
        Verifica periodicamente se as posições locais ainda existem na DEX.

        NOTA 2 (FIX bugC8): Posições com menos de VERIFY_GRACE_PERIOD segundos
        NÃO são removidas automaticamente, para não tratar como "fechada
        externamente" uma ordem limit que ainda está resting.
        """
        VERIFY_GRACE_PERIOD = 120  # segundos — período de carência
        now = time.time()

        try:
            if not self.position_manager.positions:
                return

            connector = self.executor.connector
            dex_positions = connector.fetch_positions()

            active_symbols: set = set()
            if dex_positions:
                for dp in dex_positions:
                    if not isinstance(dp, dict):
                        continue
                    raw_symbol = dp.get("symbol", "")
                    symbol = self._normalize_symbol(raw_symbol)
                    contracts = float(dp.get("contracts", 0) or 0)
                    if symbol and contracts > 0:
                        active_symbols.add(symbol)

            for pos in list(self.position_manager.positions):
                age = now - pos.open_time
                if pos.symbol not in active_symbols and age < VERIFY_GRACE_PERIOD:
                    log.debug(
                        f"[VERIFY] {pos.symbol} ausente na DEX mas tem "
                        f"apenas {age:.0f}s (grace={VERIFY_GRACE_PERIOD}s) — "
                        f"ordem limit ainda não preenchida, mantendo"
                    )
                    continue

                if pos.symbol not in active_symbols:
                    log.warning(
                        f"[VERIFY] Posição {pos.symbol} não encontrada na DEX "
                        f"(fechada externamente) — removendo do gerenciamento"
                    )
                    coin = pos.symbol.replace("/USDC", "").replace("/USD", "")
                    mark_px = connector.get_mark_price(coin)
                    exit_price = mark_px if mark_px and mark_px > 0 else pos.entry_price

                    fee = (pos.entry_price + exit_price) * pos.qty * 0.0005
                    gross = (
                        (exit_price - pos.entry_price)
                        * pos.qty
                        * (1 if pos.side == "buy" else -1)
                    )
                    net_pnl = gross - fee

                    self.position_manager.positions.remove(pos)

                    self.notificator.send_trade_alert(
                        action="close_external",
                        symbol=pos.symbol,
                        side=pos.side,
                        qty=pos.qty,
                        price=exit_price,
                        pnl=net_pnl,
                    )
                    log.info(
                        f"[VERIFY] {pos.symbol} fechada externamente: "
                        f"PnL estimado={net_pnl:+.4f}"
                    )

        except Exception as e:
            log.debug(f"[VERIFY] Erro ao verificar posições na DEX: {e}")

    async def _process_symbol(self, sym_cfg: SymbolConfig) -> None:
        """
        Processa um símbolo: busca dados, gera sinal, executa.

        Args:
            sym_cfg: Configuração específica do símbolo (estratégia,
                timeframe, risco e leverage podem sobrescrever os
                globais — ver SymbolConfig).
        """
        symbol = sym_cfg.symbol
        try:
            log.debug(f"Processando {symbol}...")

            connector = self.executor.connector

            # Estratégia/timeframe efetivos: override do símbolo > global (FIX item 4)
            effective_strategy = sym_cfg.strategy or self.cfg.strategy
            effective_timeframe = sym_cfg.timeframe or self.cfg.timeframe

            # 1. Buscar dados OHLCV
            ohlcv_data = connector.fetch_ohlcv(symbol, effective_timeframe.value, limit=350)

            if ohlcv_data is None or len(ohlcv_data) == 0:
                log.warning(f"[{symbol}] Dados OHLCV indisponíveis")
                return

            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # 1b. Buscar funding rate atual (FIX item 5)
            coin = symbol.replace("/USDC", "").replace("/USD", "")
            funding_rate = connector.get_current_funding_rate(coin)

            # 2. Adicionar indicadores (com funding rate real)
            df = add_all_indicators(df, self.cfg, funding_rate=funding_rate)

            # 3. Gerar sinal — troca temporária de cfg.strategy/timeframe
            #    se houver override no símbolo (limitação documentada:
            #    get_strategy_params ainda lê cfg global, não por símbolo)
            original_strategy, original_timeframe = self.cfg.strategy, self.cfg.timeframe
            if sym_cfg.strategy or sym_cfg.timeframe:
                self.cfg.strategy = effective_strategy
                self.cfg.timeframe = effective_timeframe
            try:
                signal, params = get_signal(df, self.cfg)
            finally:
                self.cfg.strategy, self.cfg.timeframe = original_strategy, original_timeframe

            self._last_signals[symbol] = signal
            self.monitor.update_signal(signal)

            # 4. Buscar spread para checagem de proteção (FIX item 2 capital_protection)
            spread_pct = None
            try:
                ticker = connector.fetch_ticker(symbol)
                if ticker and ticker.get("bid") and ticker.get("ask"):
                    bid, ask = float(ticker["bid"]), float(ticker["ask"])
                    mid = (bid + ask) / 2
                    if mid > 0:
                        spread_pct = (ask - bid) / mid
            except Exception as e:
                log.debug(f"[{symbol}] Erro ao buscar spread: {e}")

            # 5. Verificar proteções (agora com spread e funding)
            protection_ok, reasons = self.capital_protection.check_all(
                spread_pct=spread_pct,
                funding_rate=funding_rate,
            )
            if not protection_ok:
                log.info(f"[{symbol}] Proteção bloqueou: {reasons}")
                return

            # 6. Extrair valores mais recentes
            latest = get_latest_values(df)
            close_price = latest.get("close", 0)

            mark_px = connector.get_mark_price(coin)
            if mark_px and close_price > 0:
                distance_pct = abs(close_price - mark_px) / mark_px * 100
                if distance_pct > 50:
                    log.warning(
                        f"[{symbol}] PREÇO SUSPEITO: close={close_price:.2f} "
                        f"vs mark={mark_px:.2f} (dif={distance_pct:.1f}%) - "
                        f"USANDO MARK PRICE"
                    )
                    close_price = mark_px

            # 7. Verificar posição existente
            existing_positions = [
                p for p in self.position_manager.positions
                if p.symbol == symbol
            ]
            if existing_positions:
                log.debug(f"[{symbol}] Posição já existe ({len(existing_positions)}), pulando abertura")
                await self._check_exits_for_symbol(symbol, close_price, mark_px)
                return

            # 8. Verificar limite de trades
            if signal in ("buy", "sell"):
                can_open, reason = self.position_manager.can_open()
                if not can_open:
                    log.info(f"[{symbol}] PositionManager bloqueou: {reason}")
                    return

                # 9. Sizing — usa risco/max_position por símbolo (FIX item 1+4)
                #    e stop_loss_pct compartilhado com calc_stops (FIX item 2)
                atr = latest.get("atr", 0)
                balance = self.position_manager.current_balance

                effective_risk_pct = sym_cfg.risk_per_trade_pct / 100.0
                effective_max_pos_pct = sym_cfg.max_position_pct / 100.0
                effective_sl_pct = self.cfg.risk.stop_loss_pct / 100.0

                qty = calc_position_size(
                    balance=balance,
                    price=close_price,
                    atr=atr,
                    risk_per_trade=effective_risk_pct,
                    max_capital_pct=effective_max_pos_pct,
                    stop_loss_pct=effective_sl_pct,
                )

                if qty <= 0:
                    log.warning(f"[{symbol}] Quantidade calculada = 0")
                    return

                # 9b. Guarda de valor mínimo de ordem (achado em teste ao
                # vivo — Hyperliquid rejeita notional < $10; melhor
                # detectar aqui do que gastar uma chamada de API fadada
                # a falhar e poluir os logs a cada ciclo).
                notional = qty * close_price
                if notional < MIN_ORDER_NOTIONAL_USD:
                    log.warning(
                        f"[{symbol}] Ordem abaixo do mínimo da exchange "
                        f"(${notional:.2f} < ${MIN_ORDER_NOTIONAL_USD:.2f}) — "
                        f"pulando. Aumente capital_usd ou risk_per_trade_pct "
                        f"para operar este símbolo com o capital atual."
                    )
                    return

                # 10. Stops — mesma stop_loss_pct usada no sizing acima
                sl, tp = calc_stops(
                    price=close_price,
                    side=signal,
                    atr=atr,
                    stop_loss_pct=effective_sl_pct,
                    take_profit_pct=self.cfg.risk.take_profit_pct / 100.0,
                    df=df,
                )

                # 11. Executar ordem
                result = self.executor.place_bulk_tpsl(
                    symbol=symbol,
                    side=signal,
                    entry_qty=qty,
                    entry_price=close_price,
                    tp_price=tp,
                    sl_price=sl,
                )

                order_accepted = False
                order_filled = False
                if result is not None:
                    if isinstance(result, dict):
                        api_status = result.get("status", "")
                        if api_status == "ok":
                            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                            if statuses:
                                first_status = statuses[0]
                                if "filled" in first_status:
                                    order_accepted = True
                                    order_filled = True
                                    oid = first_status.get("filled", {}).get("oid")
                                    log.info(f"[{symbol}] Ordem PREENCHIDA: oid={oid}")
                                elif "resting" in first_status:
                                    order_accepted = True
                                    order_filled = False
                                    oid = first_status.get("resting", {}).get("oid")
                                    log.info(f"[{symbol}] Ordem ACEITA (resting): oid={oid}")
                                elif "error" in first_status:
                                    log.error(f"[{symbol}] Hyperliquid rejeitou: {first_status['error']}")
                                else:
                                    log.warning(f"[{symbol}] Status inesperado: {first_status}")
                            else:
                                log.warning(f"[{symbol}] Resposta sem statuses: {result}")
                        elif api_status == "err":
                            log.error(f"[{symbol}] Hyperliquid erro: {result}")
                        else:
                            log.warning(f"[{symbol}] Status desconhecido: {api_status}")
                    else:
                        log.warning(f"[{symbol}] Resposta inesperada (tipo {type(result).__name__}): {result}")
                else:
                    log.error(f"[{symbol}] place_bulk_tpsl retornou None")

                if order_accepted:
                    if order_filled:
                        pos = Position(
                            symbol=symbol,
                            side=signal,
                            entry_price=close_price,
                            qty=qty,
                            stop_loss=sl,
                            take_profit=tp,
                            # FIX CRÍTICO: inicializar trailing_stop_price
                            # com o SL — evita que o default 0.0 dispare
                            # check_exits() para SELL instantaneamente
                            # (ver correção em risk.py::check_exits).
                            trailing_stop_price=sl,
                            open_time=time.time(),
                        )
                        self.position_manager.add(pos)
                        try:
                            self._lock.heartbeat()
                        except Exception:
                            pass

                        self.monitor.record_step(success=True)

                        self.notificator.send_trade_alert(
                            action="open",
                            symbol=symbol,
                            side=signal,
                            qty=qty,
                            price=close_price,
                        )

                        log.info(
                            f"[{symbol}] {signal.upper()} {qty} @ {close_price} "
                            f"| TP={tp} | SL={sl}"
                        )
                    else:
                        log.info(
                            f"[{symbol}] Ordem {signal.upper()} {qty} @ {close_price} "
                            f"em resting (limit) — aguardando preenchimento"
                        )
                else:
                    log.error(f"[{symbol}] Falha ao executar ordem")
                    self.monitor.record_step(success=False)

            if existing_positions or signal in ("buy", "sell"):
                await self._check_exits_for_symbol(symbol, close_price, mark_px)

        except Exception as e:
            log.error(f"[{symbol}] Erro no processamento: {e}")
            self.monitor.record_step(success=False)
            self.monitor.record_error(str(e))

    async def _check_exits_for_symbol(
        self,
        symbol: str,
        close_price: float,
        mark_px: Optional[float] = None,
    ) -> None:
        """
        Verifica se posições do símbolo atual devem ser fechadas.

        Usa o mark price real da DEX em vez do close_price do OHLCV
        para evitar falsos positivos de SL/TP.

        Args:
            symbol: Símbolo a verificar.
            close_price: Preço de fechamento do OHLCV.
            mark_px: Mark price real da DEX (opcional).
        """
        try:
            check_price = close_price
            if mark_px and mark_px > 0:
                check_price = mark_px

            positions_for_symbol = [
                p for p in self.position_manager.positions
                if p.symbol == symbol
            ]
            if not positions_for_symbol:
                return

            to_close = self.position_manager.check_exits(check_price)

            for pos in to_close:
                try:
                    connector = self.executor.connector
                    result_close_dex = connector.close_position(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side=pos.side,
                    )
                    if result_close_dex:
                        log.info(f"[{symbol}] Posição fechada na DEX: {result_close_dex}")
                except Exception as e:
                    log.warning(f"[{symbol}] Erro ao fechar na DEX (pode já ter sido fechada): {e}")

                result_close = self.position_manager.record_close(pos, check_price)
                if "error" not in result_close:
                    self.capital_protection.record_trade_result(
                        result_close.get("net_pnl", 0)
                    )
                    self.monitor.record_trade(result_close.get("net_pnl", 0))
                    self.notificator.send_trade_alert(
                        action="close",
                        symbol=pos.symbol,
                        side=pos.side,
                        qty=pos.qty,
                        price=check_price,
                        pnl=result_close.get("net_pnl"),
                    )

            if positions_for_symbol:
                self.position_manager.update_trailing(check_price)

        except Exception as e:
            log.error(f"[{symbol}] Erro em check_exits: {e}")

    async def step(self) -> None:
        """Executa um ciclo completo do bot."""
        try:
            if self.paused:
                return

            if not self._symbols:
                self._symbols = self._load_symbols()

            if not self._synced_from_exchange:
                self._sync_from_exchange()

            for sym_cfg in self._symbols:
                if not sym_cfg.enabled:
                    continue
                await self._process_symbol(sym_cfg)   # ALTERADO: passa SymbolConfig, não string

            self._verify_positions_on_exchange()

            # Atualizar saldo
            try:
                connector = self.executor.connector
                balance_info = connector.fetch_balance()
                if balance_info and isinstance(balance_info, dict):
                    usdc_data = balance_info.get("USDC", {})
                    if isinstance(usdc_data, dict):
                        free_balance = float(usdc_data.get("free", 0) or 0)
                    else:
                        free_balance = float(balance_info.get("total", 0) or 0)

                    if free_balance > 0:
                        self.capital_protection.state.current_balance = free_balance
                        if free_balance > self.capital_protection.state.peak_balance:
                            self.capital_protection.state.peak_balance = free_balance

                        self.position_manager.current_balance = self.capital_protection.state.current_balance
                        self.position_manager.peak_balance = self.capital_protection.state.peak_balance
                        self.monitor.update_balance(free_balance)
                        self.monitor.metrics.peak_balance = self.capital_protection.state.peak_balance
            except Exception as e:
                log.debug(f"Erro ao atualizar saldo: {e}")

            # Staking automático (se configurado)
            if self.cfg.staking.enabled and self.cfg.staking.validator_address:
                try:
                    summary = self.staking.get_staking_summary()
                    if summary:
                        log.debug(f"[STAKING] Resumo: {summary}")
                except Exception as e:
                    log.debug(f"[STAKING] Erro ao consultar: {e}")

            self.monitor.record_step(success=True)

            # ── Atualizar Dashboard State ──
            try:
                from crypto_bot_core.dashboard.main import update_state

                positions_data = []
                for pos in self.position_manager.positions:
                    positions_data.append({
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "qty": pos.qty,
                        "entry_price": pos.entry_price,
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                    })

                metrics = self.monitor.get_metrics()
                metrics.update({
                    "total_pnl": self.position_manager.pnl_today,
                    # FIX: BotMetrics não tem atributo win_rate — o valor
                    # já calculado está em metrics["trading"]["win_rate_pct"].
                    "win_rate": metrics["trading"]["win_rate_pct"],
                    "current_balance": self.capital_protection.state.current_balance,
                    "peak_balance": self.capital_protection.state.peak_balance,
                    "drawdown_pct": (
                        (self.capital_protection.state.peak_balance - self.capital_protection.state.current_balance)
                        / max(self.capital_protection.state.peak_balance, 1) * 100
                    ),
                })

                update_state({
                    "status": "online",
                    "positions": positions_data,
                    "metrics": metrics,
                    "last_update": time.time(),
                })
            except Exception as e:
                log.debug(f"Erro ao atualizar dashboard: {e}")

            # ── Atualizar Healthcheck State ──
            try:
                update_health({
                    "status": "online",
                    "last_step": time.time(),
                    "positions": len(self.position_manager.positions),
                    # FIX: BotMetrics não tem atributo 'errors' — o
                    # campo real de contagem de falhas é steps_failed.
                    "errors": self.monitor.metrics.steps_failed,
                })
            except Exception as e:
                log.debug(f"Erro ao atualizar healthcheck: {e}")

        except Exception as e:
            log.error(f"Erro no ciclo principal: {e}")
            self.monitor.record_step(success=False)
            try:
                update_health({"status": "error", "last_step": time.time()})
            except Exception:
                pass

    async def run(self, interval: int = 60) -> None:
        """
        Loop principal do bot.

        Args:
            interval: Intervalo entre ciclos em segundos.
        """
        self.running = True
        log.info(f"Bot iniciado. Intervalo: {interval}s")

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                # Windows não suporta add_signal_handler
                pass

        try:
            while self.running:
                cycle_start = time.time()
                await self.step()

                elapsed = time.time() - cycle_start
                sleep_time = max(0, interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            log.info("Bot cancelado")
        except Exception as e:
            log.error(f"Erro fatal no loop principal: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Para o bot gracefulmente."""
        self.running = False
        log.info("Bot parando...")

        metrics = self.monitor.get_metrics()
        log.info(f"Trades: {metrics['trading']['total_trades']}")
        log.info(f"PnL Total: ${metrics['trading']['total_pnl']:+,.2f}")
        log.info(f"Win Rate: {metrics['trading']['win_rate_pct']}%")

        try:
            self._health_server.stop()
        except Exception:
            pass

        try:
            self._lock.release()
        except Exception:
            pass

        log.info("Bot parado com sucesso.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperliquid Production Bot v3.0")
    parser.add_argument("--mode", choices=["live", "backtest", "dashboard"], default="live")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--config", default=".env")
    parser.add_argument(
        "--load-json-overrides",
        action="store_true",
        help="Carrega bot_config.json por cima do .env no startup "
             "(útil para retomar ajustes feitos via dashboard em uma "
             "sessão anterior). Sem esta flag, apenas .env é usado, e "
             "o dashboard começa com os defaults do .env.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configura logging."""
    log.remove()
    log.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
        level="INFO",
    )
    log.add(
        "data/bot_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )


def _run_backtest(args: argparse.Namespace, cfg: BotConfig) -> None:
    """Executa o modo backtest.

    Busca dados históricos via exchange, calcula indicadores,
    executa o BacktestEngine e exibe/exports o resultado.

    Args:
        args: Argumentos da linha de comando.
        cfg: Configuração do bot.
    """
    try:
        from crypto_bot_core.backtest import BacktestEngine
        from crypto_bot_core.indicators import add_all_indicators
        from crypto_bot_core.exchanges.hyperliquid import get_connector

        symbol = args.symbol or cfg.symbols.split(",")[0].strip()
        log.info(f"[BACKTEST] Iniciando backtest para {symbol}...")

        connector = get_connector(cfg)
        ohlcv = connector.fetch_ohlcv(
            symbol,
            cfg.timeframe.value,
            limit=2000,
        )

        if not ohlcv or len(ohlcv) < 50:
            log.error(
                f"[BACKTEST] Dados históricos insuficientes para {symbol}: "
                f"{len(ohlcv) if ohlcv else 0} candles"
            )
            return

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        log.info(
            f"[BACKTEST] {len(df)} candles obtidos para {symbol} "
            f"({df.index[0].strftime('%Y-%m-%d')} → "
            f"{df.index[-1].strftime('%Y-%m-%d')})"
        )

        df = add_all_indicators(df, cfg)

        engine = BacktestEngine(
            cfg,
            initial_capital=cfg.backtest.initial_capital,
        )
        result = engine.run(df)

        print()
        print(result.summary())
        print()

        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        csv_path = (
            f"backtest_{safe_symbol}_{cfg.strategy.value}_"
            f"{cfg.timeframe.value}.csv"
        )
        result.export_csv(csv_path)

        log.info(f"[BACKTEST] Resultado exportado para {csv_path}")

    except Exception as e:
        log.error(f"[BACKTEST] Erro fatal: {e}")
        sys.exit(1)


async def main_async(args: argparse.Namespace, cfg: BotConfig) -> None:
    """Modo assíncrono: live trading."""
    bot = HyperliquidBot(cfg)
    await bot.run(interval=args.interval)


def main_sync(args: argparse.Namespace, cfg: BotConfig) -> None:
    """Modo síncrono: dashboard ou backtest."""
    if args.mode == "dashboard":
        log.info("Iniciando dashboard local (modo standalone de debug)...")
        from crypto_bot_core.dashboard.main import start_dashboard
        start_dashboard(cfg)
    elif args.mode == "backtest":
        _run_backtest(args, cfg)


def main() -> None:
    """Função principal.

    FIX CRÍTICO (item de dispatch, achado em teste ao vivo): a versão
    anterior carregava a configuração e retornava sem nunca despachar
    para main_async()/main_sync() — o processo terminava silenciosamente
    após 2 linhas de log, sem erro algum, sem nunca de fato rodar o bot.
    """
    args = parse_args()
    setup_logging()

    try:
        cfg = get_config()
        if args.load_json_overrides:
            cfg = BotConfig.load_json("bot_config.json")

        if args.mode == "live":
            asyncio.run(main_async(args, cfg))
        else:
            main_sync(args, cfg)

    except KeyboardInterrupt:
        log.info("Interrompido pelo usuário")
    except Exception as e:
        log.error(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()