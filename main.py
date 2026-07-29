"""
Hyperliquid Production Bot v3.0 — Ponto de Entrada Principal
=============================================================
Bot de trading automatizado para Hyperliquid DEX.

Modos de operação:
- live: Trading real (requer credenciais)
- backtest: Backtesting com dados históricos
- dashboard: Servidor web local para monitoramento

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
    calc_liquidation_price_estimate,
    calc_position_size,
    calc_stops,
    trade_hours_ok,
    validate_stop_loss_safety,
)
from crypto_bot_core.health_server import HealthServer, update_health
from crypto_bot_core.lock import LockManager
from crypto_bot_core.staking import StakingManager
from crypto_bot_core.strategies.signals import get_signal, get_ensemble_signal
from crypto_bot_core.trade_ledger import get_ledger


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

MIN_ORDER_NOTIONAL_USD = 10.0
RECONCILE_FULL_SYNC_INTERVAL_CYCLES = 30
STATE_RECONCILE_EPSILON_USD = 0.5


# ──────────────────────────────────────────────
# HyperliquidBot
# ──────────────────────────────────────────────


class HyperliquidBot:
    """
    Bot principal de trading para Hyperliquid.

    Integra todos os módulos:
    - Config → Estratégia(s) → Sinais → Risco → Execução → Monitoramento
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

        self._lock = LockManager()
        if not self._lock.acquire():
            raise RuntimeError(
                "Outra instância do bot já está rodando. "
                "Se tiver certeza que não, remova o lock file manualmente "
                "ou use --force para sobrescrever."
            )

        r = cfg.risk
        n = cfg.notifications
        s = cfg.staking

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
            max_position_pct=cfg.max_position_pct / 100.0,
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
            max_funding_rate=r.max_funding_rate_fraction,   # FIX item 2: usa property centralizada
            max_correlated_exposure_pct=r.max_correlated_exposure_pct / 100.0,
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
        self.ledger = get_ledger()

        self._health_server = HealthServer(
            host=cfg.dashboard.host,
            port=cfg.monitoring.health_check_port,
        )
        self._health_server.start()

        from crypto_bot_core.dashboard.main import start_dashboard_thread
        self._dashboard_thread = start_dashboard_thread(cfg)

        self._symbols: List[SymbolConfig] = []
        self._last_signals: Dict[str, str] = {}
        self._synced_from_exchange: bool = False
        self._cycle_count: int = 0

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
        if cfg.ensemble_mode:
            log.info(
                f"Modo de sinal: ENSEMBLE (estratégias: {cfg.enabled_strategies} | "
                f"min_confluencia={cfg.ensemble_min_confluence} | "
                f"min_confianca_media={cfg.ensemble_min_avg_confidence:.2f})"
            )
        else:
            log.info(f"Modo de sinal: SINGLE-STRATEGY ({cfg.strategy.value})")
        log.info(f"Símbolos: {cfg.symbols}")
        log.info(f"Capital: ${cfg.capital_usd:,.2f}")
        log.info("=" * 50)

    def _load_symbols(self) -> List[SymbolConfig]:
        """Carrega configurações completas por símbolo (não só o nome)."""
        try:
            return self.cfg.parse_symbols()
        except Exception as e:
            log.error(f"Erro ao carregar símbolos: {e}")
            from crypto_bot_core.config import SymbolConfig as _SC
            return [
                _SC(symbol=s, coin=s.split("/")[0])
                for s in ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
            ]

    def _normalize_symbol(self, raw_symbol: str) -> str:
        """
        Normaliza símbolo do formato ccxt (ex: 'ETH/USDC:USDC') para
        o formato padrão do bot (ex: 'ETH/USDC').
        """
        if ":" in raw_symbol:
            raw_symbol = raw_symbol.split(":")[0]
        return raw_symbol

    def _sync_from_exchange(self) -> None:
        """Sincroniza posições e ordens abertas da DEX (idempotente, chamada periodicamente)."""
        try:
            connector = self.executor.connector
            log.info("[SYNC] Sincronizando posições da DEX...")

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
                            f"({order_prices[0]}) — posição restaurada SEM stop "
                            f"automático local."
                        )
                    else:
                        log.warning(
                            f"[SYNC] {symbol}: nenhuma ordem trigger encontrada "
                            f"na exchange — posição restaurada SEM stop_loss/"
                            f"take_profit locais."
                        )

                    pos = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        qty=contracts,
                        stop_loss=restored_sl,
                        take_profit=restored_tp,
                        trailing_stop_price=restored_sl,
                        open_time=time.time(),
                    )
                    self.position_manager.add(pos)
                    log.info(f"[SYNC] Posição restaurada da DEX: {side.upper()} {contracts} {symbol} @ {entry_price}")

            try:
                remaining_orders = connector.info.open_orders(connector.cfg.hyperliquid_account_address)
                if remaining_orders:
                    log.info(f"[SYNC] {len(remaining_orders)} ordem(ns) restante(s) na DEX (após cancelamento)")
            except Exception as e:
                log.debug(f"[SYNC] Erro ao buscar ordens restantes: {e}")

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
                        else:
                            self.capital_protection.state.peak_balance = max(old_peak, free_balance)

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
        """Verifica periodicamente se as posições locais ainda existem na DEX."""
        VERIFY_GRACE_PERIOD = 120
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
                    funding_cost = pos.accrued_funding
                    net_pnl = gross - fee - funding_cost

                    self.position_manager.positions.remove(pos)

                    self.capital_protection.record_trade_result(net_pnl)
                    self.monitor.record_trade(net_pnl)
                    self.ledger.record(
                        symbol=pos.symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        qty=pos.qty,
                        gross_pnl=gross,
                        fee=fee,
                        funding_cost=funding_cost,
                        net_pnl=net_pnl,
                        exit_reason="close_external",
                        open_time=pos.open_time,
                    )

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
                        f"PnL estimado={net_pnl:+.4f} (funding={funding_cost:+.4f})"
                    )

        except Exception as e:
            log.debug(f"[VERIFY] Erro ao verificar posições na DEX: {e}")

    def _reconcile_state(self) -> None:
        """Checagem cruzada entre as fontes de estado do bot."""
        try:
            cp_balance = self.capital_protection.state.current_balance
            pm_balance = self.position_manager.current_balance

            if abs(cp_balance - pm_balance) > STATE_RECONCILE_EPSILON_USD:
                log.error(
                    f"[RECONCILE] DIVERGÊNCIA DE SALDO: CapitalProtection="
                    f"${cp_balance:.2f} vs PositionManager=${pm_balance:.2f}. "
                    f"Forçando resincronização completa com a DEX."
                )
                self._sync_from_exchange()
                return

            cp_peak = self.capital_protection.state.peak_balance
            pm_peak = self.position_manager.peak_balance
            if abs(cp_peak - pm_peak) > STATE_RECONCILE_EPSILON_USD:
                log.warning(
                    f"[RECONCILE] Divergência de peak_balance: "
                    f"CapitalProtection=${cp_peak:.2f} vs "
                    f"PositionManager=${pm_peak:.2f} — sincronizando."
                )
                self.position_manager.peak_balance = cp_peak

            local_positions = len(self.position_manager.positions)
            try:
                dex_positions = self.executor.connector.fetch_positions()
                dex_active = sum(
                    1 for dp in (dex_positions or [])
                    if isinstance(dp, dict) and float(dp.get("contracts", 0) or 0) > 0
                )
                if local_positions != dex_active:
                    log.warning(
                        f"[RECONCILE] Contagem de posições diverge: "
                        f"local={local_positions} vs DEX={dex_active}."
                    )
            except Exception as e:
                log.debug(f"[RECONCILE] Erro ao verificar posições na DEX: {e}")

        except Exception as e:
            log.error(f"[RECONCILE] Erro na reconciliação de estado: {e}")

    async def _process_symbol(self, sym_cfg: SymbolConfig) -> None:
        """
        Processa um símbolo: busca dados, gera sinal, executa.

        Args:
            sym_cfg: Configuração específica do símbolo.
        """
        symbol = sym_cfg.symbol
        try:
            log.debug(f"Processando {symbol}...")

            connector = self.executor.connector

            effective_strategy = sym_cfg.strategy or self.cfg.strategy
            effective_timeframe = sym_cfg.timeframe or self.cfg.timeframe

            ohlcv_data = connector.fetch_ohlcv(symbol, effective_timeframe.value, limit=350)

            if ohlcv_data is None or len(ohlcv_data) == 0:
                log.warning(f"[{symbol}] Dados OHLCV indisponíveis")
                return

            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            coin = symbol.replace("/USDC", "").replace("/USD", "")
            funding_rate = connector.get_current_funding_rate(coin)

            df = add_all_indicators(df, self.cfg, funding_rate=funding_rate)

            # FIX (auditoria — item 9): geração de sinal agora suporta
            # dois modos — ensemble (combina todas as estratégias
            # habilitadas, ponderadas por confiança) ou single-strategy
            # (comportamento original, uma única estratégia por vez).
            # O modo é escolhido via cfg.ensemble_mode (default False —
            # não altera o comportamento de instalações existentes).
            if self.cfg.ensemble_mode:
                signal, params = get_ensemble_signal(df, self.cfg)
            else:
                original_strategy, original_timeframe = self.cfg.strategy, self.cfg.timeframe
                if sym_cfg.strategy or sym_cfg.timeframe:
                    self.cfg.strategy = effective_strategy
                    self.cfg.timeframe = effective_timeframe
                try:
                    signal, params = get_signal(df, self.cfg)
                finally:
                    self.cfg.strategy, self.cfg.timeframe = original_strategy, original_timeframe

            signal_confidence = float(params.get("confidence", 0.0))
            self._last_signals[symbol] = signal
            self.monitor.update_signal(signal)

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

            protection_ok, reasons = self.capital_protection.check_all(
                spread_pct=spread_pct,
                funding_rate=funding_rate,
            )
            if not protection_ok:
                log.info(f"[{symbol}] Proteção bloqueou: {reasons}")
                return

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

            if funding_rate:
                funding_ref_price = mark_px if (mark_px and mark_px > 0) else close_price
                self.position_manager.accrue_funding(
                    symbol=symbol,
                    funding_rate=funding_rate,
                    mark_price=funding_ref_price,
                )

            existing_positions = [
                p for p in self.position_manager.positions
                if p.symbol == symbol
            ]
            if existing_positions:
                log.debug(f"[{symbol}] Posição já existe ({len(existing_positions)}), pulando abertura")
                await self._check_exits_for_symbol(symbol, close_price, mark_px)
                return

            if signal in ("buy", "sell"):
                can_open, reason = self.position_manager.can_open()
                if not can_open:
                    log.info(f"[{symbol}] PositionManager bloqueou: {reason}")
                    return

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

                notional = qty * close_price
                if notional < MIN_ORDER_NOTIONAL_USD:
                    log.warning(
                        f"[{symbol}] Ordem abaixo do mínimo da exchange "
                        f"(${notional:.2f} < ${MIN_ORDER_NOTIONAL_USD:.2f}) — pulando."
                    )
                    return

                total_notional_all = sum(
                    p.qty * p.entry_price for p in self.position_manager.positions
                )

                exposure_ok, exposure_reason = self.capital_protection.check_exposure(
                    current_positions_value=total_notional_all,
                    new_position_value=notional,
                )
                if not exposure_ok:
                    log.info(f"[{symbol}] Bloqueado por exposição (Nível 3): {exposure_reason}")
                    return

                corr_ok, corr_reason = self.capital_protection.check_correlated_exposure(
                    total_notional_all_positions=total_notional_all,
                    new_position_notional=notional,
                )
                if not corr_ok:
                    log.info(
                        f"[{symbol}] Bloqueado por exposição correlacionada "
                        f"(Nível 3b): {corr_reason}"
                    )
                    return

                sl, tp = calc_stops(
                    price=close_price,
                    side=signal,
                    atr=atr,
                    stop_loss_pct=effective_sl_pct,
                    take_profit_pct=self.cfg.risk.take_profit_pct / 100.0,
                    df=df,
                )

                liq_price_est = calc_liquidation_price_estimate(
                    entry_price=close_price,
                    side=signal,
                    leverage=sym_cfg.leverage,
                    is_cross=not sym_cfg.isolated_margin,
                )
                sl_safe, sl_reason = validate_stop_loss_safety(
                    stop_loss=sl,
                    liquidation_price=liq_price_est,
                    side=signal,
                )
                if not sl_safe:
                    log.error(
                        f"[{symbol}] ORDEM ABORTADA por segurança de liquidação: "
                        f"SL={sl:.4f} próximo da liquidação estimada "
                        f"(${liq_price_est:.4f}) com leverage {sym_cfg.leverage}x — "
                        f"{sl_reason}."
                    )
                    self.monitor.record_error(f"SL inseguro em {symbol}: {sl_reason}")
                    return

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
                            f"| TP={tp} | SL={sl} | conf={signal_confidence:.2f} "
                            f"| Liq.est≈{liq_price_est:.2f}"
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
        """Verifica se posições do símbolo atual devem ser fechadas."""
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
                exit_reason = "manual"
                if pos.take_profit > 0 and (
                    (pos.side == "buy" and check_price >= pos.take_profit)
                    or (pos.side == "sell" and check_price <= pos.take_profit)
                ):
                    exit_reason = "take_profit"
                elif pos.stop_loss > 0 and (
                    (pos.side == "buy" and check_price <= pos.stop_loss)
                    or (pos.side == "sell" and check_price >= pos.stop_loss)
                ):
                    exit_reason = "stop_loss"
                elif pos.trailing_stop_price > 0:
                    exit_reason = "trailing_stop"

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
                    self.capital_protection.record_trade_result(result_close.get("net_pnl", 0))
                    self.monitor.record_trade(result_close.get("net_pnl", 0))

                    self.ledger.record(
                        symbol=result_close["symbol"],
                        side=result_close["side"],
                        entry_price=result_close["entry_price"],
                        exit_price=result_close["exit_price"],
                        qty=result_close["qty"],
                        gross_pnl=result_close["gross_pnl"],
                        fee=result_close["fee"],
                        funding_cost=result_close.get("funding_cost", 0.0),
                        net_pnl=result_close["net_pnl"],
                        exit_reason=exit_reason,
                        strategy=(
                            "ensemble" if self.cfg.ensemble_mode else self.cfg.strategy.value
                        ),
                        open_time=result_close.get("open_time"),
                    )

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

            self._cycle_count += 1

            if not self._symbols:
                self._symbols = self._load_symbols()

            if not self._synced_from_exchange:
                self._sync_from_exchange()
            elif self._cycle_count % RECONCILE_FULL_SYNC_INTERVAL_CYCLES == 0:
                log.info(
                    f"[SYNC] Resincronização periódica completa "
                    f"(ciclo {self._cycle_count})"
                )
                self._sync_from_exchange()

            for sym_cfg in self._symbols:
                if not sym_cfg.enabled:
                    continue
                await self._process_symbol(sym_cfg)

            self._verify_positions_on_exchange()
            self._reconcile_state()

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

            if self.cfg.staking.enabled and self.cfg.staking.validator_address:
                try:
                    summary = self.staking.get_staking_summary()
                    if summary:
                        log.debug(f"[STAKING] Resumo: {summary}")
                except Exception as e:
                    log.debug(f"[STAKING] Erro ao consultar: {e}")

            self.monitor.record_step(success=True)

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
                        "accrued_funding": pos.accrued_funding,
                    })

                metrics = self.monitor.get_metrics()
                metrics.update({
                    "total_pnl": self.position_manager.pnl_today,
                    "win_rate": metrics["trading"]["win_rate_pct"],
                    "current_balance": self.capital_protection.state.current_balance,
                    "peak_balance": self.capital_protection.state.peak_balance,
                    "drawdown_pct": (
                        (self.capital_protection.state.peak_balance - self.capital_protection.state.current_balance)
                        / max(self.capital_protection.state.peak_balance, 1) * 100
                    ),
                    "ledger_summary": self.ledger.summary(),
                    "signal_mode": "ensemble" if self.cfg.ensemble_mode else "single_strategy",
                })

                update_state({
                    "status": "online",
                    "positions": positions_data,
                    "metrics": metrics,
                    "last_update": time.time(),
                })
            except Exception as e:
                log.debug(f"Erro ao atualizar dashboard: {e}")

            try:
                update_health({
                    "status": "online",
                    "last_step": time.time(),
                    "positions": len(self.position_manager.positions),
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

        ledger_summary = self.ledger.summary()
        log.info(
            f"[LEDGER] Trilha de auditoria: {ledger_summary['total_trades']} "
            f"trade(s) registrado(s), PnL líquido acumulado "
            f"${ledger_summary['total_net_pnl']:+,.2f}"
        )

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
        help="Carrega bot_config.json por cima do .env no startup.",
    )
    parser.add_argument(
        "--respect-enabled-strategies",
        action="store_true",
        help="No modo backtest, respeita cfg.enabled_strategies (mesmo "
             "comportamento do live) em vez do default do backtest, que "
             "ignora esse gate para permitir validar estratégias ainda "
             "não habilitadas.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="No modo backtest, usa get_ensemble_signal() (combina "
             "todas as estratégias de ENABLED_STRATEGIES ponderadas "
             "por confiança) em vez da estratégia única (STRATEGY). "
             "Equivalente a rodar o backtest com ENSEMBLE_MODE=true, "
             "sem precisar alterar o .env — útil para validar o modo "
             "ensemble antes de habilitá-lo em produção. Se omitida, "
             "usa cfg.ensemble_mode (valor do .env) como default.",
    )
    return parser.parse_args()


def _run_backtest(args: argparse.Namespace, cfg: BotConfig) -> None:
    """Executa o modo backtest.

    Busca dados históricos via exchange, calcula indicadores,
    executa o BacktestEngine e exibe/exports o resultado.

    FIX (suporte a ensemble_mode no backtest): a flag --ensemble (ou
    ENSEMBLE_MODE=true no .env) faz o backtest usar
    get_ensemble_signal() em vez de get_signal() — necessário para
    validar o modo ensemble via dados históricos antes de habilitá-lo
    em produção (ver BacktestEngine.ensemble_mode).

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

        bypass = not args.respect_enabled_strategies
        if bypass and not cfg.is_strategy_enabled():
            log.info(
                f"[BACKTEST] Estratégia '{cfg.strategy.value}' será testada "
                f"mesmo NÃO estando em ENABLED_STRATEGIES — use "
                f"--respect-enabled-strategies para simular o bloqueio "
                f"real de produção."
            )

        # --ensemble na CLI tem prioridade; se omitida, usa o valor do
        # .env (cfg.ensemble_mode) como default.
        use_ensemble = args.ensemble or cfg.ensemble_mode
        if use_ensemble:
            log.info(
                f"[BACKTEST] Modo ENSEMBLE ativo — combinando estratégias "
                f"'{cfg.enabled_strategies}' (min_confluencia="
                f"{cfg.ensemble_min_confluence}, min_confianca_media="
                f"{cfg.ensemble_min_avg_confidence:.2f})"
            )
        else:
            log.info(f"[BACKTEST] Modo SINGLE-STRATEGY ({cfg.strategy.value})")

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
            bypass_enabled_strategies=bypass,
            ensemble_mode=use_ensemble,
        )
        result = engine.run(df)

        print()
        print(result.summary())
        print()

        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        strategy_label = "ensemble" if use_ensemble else cfg.strategy.value
        csv_path = (
            f"backtest_{safe_symbol}_{strategy_label}_"
            f"{cfg.timeframe.value}.csv"
        )
        result.export_csv(csv_path)

        log.info(f"[BACKTEST] Resultado exportado para {csv_path}")

    except Exception as e:
        log.error(f"[BACKTEST] Erro fatal: {e}")
        sys.exit(1)


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

        # FIX (achado na análise de logs/backtest): loga explicitamente
        # qual modo de gate está em uso, para não haver ambiguidade
        # sobre por que total_trades deu 0 (ou não) em uma estratégia
        # fora de ENABLED_STRATEGIES.
        bypass = not args.respect_enabled_strategies
        if bypass and not cfg.is_strategy_enabled():
            log.info(
                f"[BACKTEST] Estratégia '{cfg.strategy.value}' será testada "
                f"mesmo NÃO estando em ENABLED_STRATEGIES — use "
                f"--respect-enabled-strategies para simular o bloqueio "
                f"real de produção."
            )

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
            bypass_enabled_strategies=bypass,
        )
        result = engine.run(df, symbol=symbol)

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
    """Função principal."""
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