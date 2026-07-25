"""
Módulo de Configuração — Hyperliquid Production Bot v3.0
========================================================
Gerencia toda a configuração do bot usando Pydantic v2 com:
- Validação de inputs com tipos fortes
- Carregamento via .env + JSON
- Suporte multi-symbol
- Perfis de estratégia por timeframe
- Logs estruturados
"""

from __future__ import annotations

import json
import os
import re
from dotenv import dotenv_values
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from loguru import logger as log
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class StrategyType(str, Enum):
    """Estratégias de trading disponíveis."""

    TREND_FOLLOW = "trend_follow"
    MEAN_REVERSION = "mean_reversion"
    ADAPTIVE_TREND = "adaptive_trend"
    HYBRID_REGIME = "hybrid_regime"
    ORDERFLOW_DELTA = "orderflow_delta"
    SCALPING_GRID = "scalping_grid"
    FUNDING_ARBITRAGE = "funding_arbitrage"


class Timeframe(str, Enum):
    """Timeframes suportados."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


# ──────────────────────────────────────────────
# Modelos de Configuração
# ──────────────────────────────────────────────


class SymbolConfig(BaseModel):
    """Configuração por símbolo (ativo)."""

    symbol: str = Field(..., description="Par de trading (ex: BTC/USDC)")
    coin: str = Field(..., description="Moeda base (ex: BTC)")
    enabled: bool = Field(True, description="Se o símbolo está ativo para trading")
    leverage: int = Field(3, ge=1, le=50, description="Alavancagem para este símbolo")
    isolated_margin: bool = Field(True, description="Usar margem isolada")
    strategy: Optional[StrategyType] = Field(None, description="Estratégia específica para este símbolo (opcional)")
    timeframe: Optional[Timeframe] = Field(None, description="Timeframe específico para este símbolo (opcional)")
    max_position_pct: float = Field(20.0, ge=0.1, le=100.0, description="Percentual máximo do capital para este símbolo")
    risk_per_trade_pct: float = Field(1.0, ge=0.1, le=10.0, description="Risco por trade para este símbolo")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Valida formato do símbolo."""
        if not re.match(r"^[A-Z0-9]{2,10}/USDC$", v):
            raise ValueError(f"Símbolo inválido: {v}. Formato esperado: BTC/USDC")
        return v

    @field_validator("coin")
    @classmethod
    def validate_coin(cls, v: str) -> str:
        """Valida nome da moeda."""
        if not re.match(r"^[A-Z0-9]{2,10}$", v):
            raise ValueError(f"Moeda inválida: {v}")
        return v


class RiskConfig(BaseModel):
    """Configuração de gerenciamento de risco."""

    max_open_trades: int = Field(3, ge=1, le=20, description="Máximo de trades simultâneos")
    max_drawdown_pct: float = Field(15.0, ge=1.0, le=50.0, description="Drawdown máximo permitido (%)")
    daily_loss_limit_pct: float = Field(5.0, ge=0.5, le=20.0, description="Limite de perda diária (%)")
    max_consecutive_losses: int = Field(3, ge=1, le=10, description="Perdas consecutivas máximas")
    cooldown_after_loss_sec: int = Field(300, ge=0, le=86400, description="Cooldown após perda (segundos)")
    stop_loss_pct: float = Field(2.0, ge=0.1, le=20.0, description="Stop Loss padrão (%)")
    take_profit_pct: float = Field(5.0, ge=0.1, le=50.0, description="Take Profit padrão (%)")
    trailing_stop: bool = Field(False, description="Habilitar trailing stop")
    trailing_stop_activation_pct: float = Field(3.0, ge=0.5, le=30.0, description="Ativação do trailing stop (%)")
    trailing_stop_distance_pct: float = Field(0.5, ge=0.1, le=10.0, description="Distância do trailing stop (%)")
    max_exposure_pct: float = Field(80.0, ge=10.0, le=100.0, description="Exposição máxima total (%)")
    circuit_breaker_loss_pct: float = Field(15.0, ge=1.0, le=50.0, description="Perda que aciona circuit breaker (%)")
    circuit_breaker_cooldown_sec: int = Field(3600, ge=60, le=86400, description="Cooldown do circuit breaker (segundos)")
    trade_hour_start_utc: int = Field(0, ge=0, le=23, description="Hora UTC de início para operar (0=desligado)")
    trade_hour_end_utc: int = Field(0, ge=0, le=23, description="Hora UTC de fim para operar (0=desligado)")
    max_spread_pct: float = Field(0.5, ge=0.01, le=5.0, description="Spread máximo permitido (%)")
    max_funding_rate: float = Field(0.1, ge=0.001, le=1.0, description="Funding rate máximo permitido (%)")
    taker_fee: float = Field(0.0005, ge=0.0, le=0.01, description="Taxa taker da exchange")

    # REMOVIDOS (não implementados em nenhum cálculo — ver auditoria item 4):
    # min_edge_vs_costs_mult, correlation_hedge_threshold, kelly_fraction,
    # anti_martingale_mult, btc_crash_filter_pct, liquidation_safety_buffer_pct
    # (este último tinha equivalente hardcoded em risk.py:
    # LIQUIDATION_SAFETY_BUFFER_PCT = 0.20, nunca lido de cfg.risk)


class StakingConfig(BaseModel):
    """Configuração de staking HYPE e vaults."""

    enabled: bool = Field(False, description="Ativar staking")
    validator_address: str = Field("", description="Endereço do validador para delegar HYPE")
    stake_pct: float = Field(0.0, ge=0.0, le=100.0, description="Percentual do saldo HYPE para staking")
    vault_address: str = Field("", description="Endereço do vault para yield passivo")
    vault_deposit_pct: float = Field(0.0, ge=0.0, le=100.0, description="Percentual para depositar no vault")
    auto_compound: bool = Field(True, description="Reinvestir recompensas automaticamente")
    min_stake_interval_hours: int = Field(24, ge=1, le=720, description="Intervalo mínimo entre staking (horas)")


class NotificationConfig(BaseModel):
    """Configuração de notificações."""

    telegram_bot_token: str = Field("", description="Token do bot Telegram")
    telegram_chat_id: str = Field("", description="Chat ID do Telegram")
    discord_webhook_url: str = Field("", description="Webhook URL do Discord")
    smtp_host: str = Field("", description="Host SMTP")
    smtp_port: int = Field(587, ge=1, le=65535, description="Porta SMTP")
    smtp_user: str = Field("", description="Usuário SMTP")
    smtp_password: str = Field("", description="Senha SMTP")
    email_from: str = Field("", description="Email remetente")
    email_to: str = Field("", description="Email destinatário")
    notify_on_trade: bool = Field(True, description="Notificar em cada trade")
    notify_on_error: bool = Field(True, description="Notificar em erros")
    notify_on_drawdown: bool = Field(True, description="Notificar em drawdown")
    notify_daily_summary: bool = Field(True, description="Resumo diário")


class DashboardConfig(BaseModel):
    """Configuração do dashboard web."""

    host: str = Field("0.0.0.0", description="Host do dashboard")
    port: int = Field(8080, ge=1024, le=65535, description="Porta do dashboard")
    user: str = Field("admin", description="Usuário do dashboard")
    password: str = Field("changeme", description="Senha do dashboard", min_length=4)
    session_secret: str = Field("", description="Chave secreta para sessões")


class MonitoringConfig(BaseModel):
    """Configuração de monitoramento."""

    prometheus_port: int = Field(9090, ge=1024, le=65535, description="Porta do Prometheus")
    health_check_port: int = Field(8081, ge=1024, le=65535, description="Porta do health check")
    log_level: str = Field("INFO", description="Nível de log (DEBUG, INFO, WARNING, ERROR)")
    log_to_file: bool = Field(True, description="Salvar logs em arquivo")
    log_retention_days: int = Field(30, ge=1, le=365, description="Retenção de logs (dias)")
    metrics_collection_interval: int = Field(60, ge=5, le=3600, description="Intervalo de coleta de métricas (s)")


class BacktestConfig(BaseModel):
    """Configuração de backtesting."""

    start_date: str = Field("2024-01-01", description="Data inicial do backtest")
    end_date: str = Field("2024-12-31", description="Data final do backtest")
    initial_capital: float = Field(10000.0, ge=100.0, description="Capital inicial do backtest")
    commission_pct: float = Field(0.01, ge=0.0, le=1.0, description="Comissão por trade (%)")
    slippage_pct: float = Field(0.05, ge=0.0, le=1.0, description="Slippage estimado (%)")
    walk_forward_windows: int = Field(4, ge=1, le=52, description="Janelas de walk-forward optimization")
    enable_multi_symbol: bool = Field(True, description="Backtest multi-symbol")


# ──────────────────────────────────────────────
# Config Principal
# ──────────────────────────────────────────────

def _cast_env_value(raw: str, annotation: Any) -> Any:
    """Converte string de env var para o tipo declarado do campo Pydantic."""
    if annotation is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    return raw


def _apply_prefixed_env_overrides(
    sub_model: BaseModel, prefix: str, env_values: Dict[str, Optional[str]]
) -> None:
    """
    Popula campos de um sub-model a partir de env vars com prefixo dado,
    contornando o parser automático de env_nested_delimiter do
    pydantic-settings.

    FIX (confirmado empiricamente): mesmo após remover default_factory
    das sub-configs, valores como NOTIFICATIONS_TELEGRAM_BOT_TOKEN
    continuavam chegando vazios em cfg.notifications.telegram_bot_token,
    apesar do .env estar correto (validado via dotenv_values() lendo o
    token corretamente). Causa: nomes de campo com múltiplos underscores
    (ex: telegram_bot_token) tornam ambígua, para o parser automático,
    a fronteira entre prefixo do sub-model e nome do campo — especialmente
    com 6 prefixos concorrentes e case_sensitive=False.

    Esta função lê o .env diretamente (independente da resolução interna
    do pydantic-settings) e aplica manualmente cada campo, usando
    correspondência exata de "{prefix}{NOME_DO_CAMPO_EM_MAIUSCULO}" —
    sem ambiguidade de underscore.
    """
    for field_name, field_info in sub_model.model_fields.items():
        env_key = f"{prefix}{field_name.upper()}"
        raw_value = env_values.get(env_key)
        if raw_value is None or raw_value == "":
            continue
        try:
            casted = _cast_env_value(raw_value, field_info.annotation)
            setattr(sub_model, field_name, casted)
        except (ValueError, TypeError) as e:
            log.warning(
                f"[CONFIG] Não foi possível converter {env_key}="
                f"'{raw_value}' para o tipo esperado: {e}"
            )


class BotConfig(BaseSettings):
    """
    Configuração principal do bot Hyperliquid v3.0.

    Carrega valores de variáveis de ambiente (.env) com fallback para defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="_",
    )

    # ── Credenciais ──
    hyperliquid_private_key: str = Field("", description="Private key da carteira Ethereum")
    hyperliquid_account_address: str = Field("", description="Endereço da carteira Ethereum")

    # ── Modo ──
    testnet: bool = Field(True, description="Usar testnet (true) ou mainnet (false)")

    # ── Símbolos ──
    symbols: str = Field("BTC/USDC,ETH/USDC,SOL/USDC", description="Símbolos separados por vírgula")

    # ── Estratégia ──
    strategy: StrategyType = Field(StrategyType.HYBRID_REGIME, description="Estratégia principal")
    timeframe: Timeframe = Field(Timeframe.H1, description="Timeframe padrão")

    # ── Estratégias habilitadas ──
    enabled_strategies: str = Field(
        "trend_follow,adaptive_trend,hybrid_regime",
        description="Estratégias liberadas para operar ao vivo (CSV). "
                     "mean_reversion/orderflow_delta/scalping_grid/funding_arbitrage "
                     "requerem validação walk-forward antes de habilitar."
    )

    @field_validator("enabled_strategies")
    @classmethod
    def validate_enabled_strategies(cls, v: str) -> str:
        valid = {s.value for s in StrategyType}
        strategies = [s.strip() for s in v.split(",") if s.strip()]
        for s in strategies:
            if s not in valid:
                raise ValueError(f"Estratégia desconhecida em enabled_strategies: {s}")
        return v

    def is_strategy_enabled(self, strategy: Optional[StrategyType] = None) -> bool:
        """Verifica se a estratégia (atual ou informada) está liberada para operar."""
        strat = (strategy or self.strategy).value
        return strat in [s.strip() for s in self.enabled_strategies.split(",")]

    # ── Capital ──
    capital_usd: float = Field(1000.0, ge=10.0, le=1_000_000.0, description="Capital em USD")
    risk_per_trade_pct: float = Field(1.0, ge=0.1, le=10.0, description="Risco por trade (%)")
    max_position_pct: float = Field(20.0, ge=0.1, le=100.0, description="Tamanho máximo da posição (%)")
    leverage: int = Field(3, ge=1, le=50, description="Alavancagem")
    isolated_margin: bool = Field(True, description="Margem isolada")

    # ── Dashboard ──
    dashboard_enabled: bool = Field(True, description="Habilita o dashboard web (thread em background dentro do processo live)")


    # ── Sub-configs ──
    # FIX (auditoria — bug do pydantic-settings): default_factory=X faz o
    # sub-model ser instanciado ANTES da camada de resolução de env vars
    # (env_nested_delimiter) ter chance de popular seus campos internos,
    # em algumas versões de pydantic-settings. Resultado prático: valores
    # como NOTIFICATIONS_TELEGRAM_BOT_TOKEN no .env nunca chegavam a
    # cfg.notifications.telegram_bot_token — o campo ficava com o default
    # vazio "", silenciosamente (sem erro, sem log). Isso explicava alertas
    # de Telegram configurados mas nunca disparados.
    #
    # Correção: instanciar diretamente (RiskConfig() em vez de
    # Field(default_factory=RiskConfig)) — validado no próprio
    # experimento _test_config.py do repositório ("Solução 2").
    risk: RiskConfig = RiskConfig()
    staking: StakingConfig = StakingConfig()
    notifications: NotificationConfig = NotificationConfig()
    dashboard: DashboardConfig = DashboardConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    backtest: BacktestConfig = BacktestConfig()

    # ── Internos (não expostos via env) ──
    _symbol_configs: List[SymbolConfig] = []
    _parsed: bool = False

    # ── Mapa de estratégia -> timeframe -> parâmetros ──
    STRATEGY_DEFAULTS: ClassVar[Dict[str, Dict[str, Dict[str, Any]]]] = {
        "trend_follow": {
            "1h": {"ema_fast": 9, "ema_slow": 21, "ema_trend": 200, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30, "atr_mult_sl": 2.0, "atr_mult_tp": 3.0},
            "4h": {"ema_fast": 12, "ema_slow": 26, "ema_trend": 200, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30, "atr_mult_sl": 2.0, "atr_mult_tp": 3.5},
            "1d": {"ema_fast": 20, "ema_slow": 50, "ema_trend": 200, "rsi_period": 14, "rsi_overbought": 65, "rsi_oversold": 35, "atr_mult_sl": 2.5, "atr_mult_tp": 4.0},
        },
        "mean_reversion": {
            "1h": {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "atr_mult_sl": 1.5, "atr_mult_tp": 2.0},
            "15m": {"bb_period": 20, "bb_std": 2.5, "rsi_period": 7, "rsi_oversold": 25, "rsi_overbought": 75, "atr_mult_sl": 1.5, "atr_mult_tp": 2.0},
            "5m": {"bb_period": 20, "bb_std": 2.5, "rsi_period": 5, "rsi_oversold": 20, "rsi_overbought": 80, "atr_mult_sl": 1.2, "atr_mult_tp": 1.8},
        },
        "adaptive_trend": {
            "1h": {"ema_fast": 9, "ema_slow": 21, "adx_period": 14, "adx_threshold": 25, "rsi_period": 14, "atr_mult_sl": 2.0, "atr_mult_tp": 3.0},
            "4h": {"ema_fast": 12, "ema_slow": 26, "adx_period": 14, "adx_threshold": 25, "rsi_period": 14, "atr_mult_sl": 2.0, "atr_mult_tp": 3.5},
        },
        "hybrid_regime": {
            "1h": {"regime_ema_fast": 20, "regime_ema_slow": 50, "vwap_period": 20, "vwap_std": 2.0, "smc_swing_lookback": 10, "atr_mult_sl": 2.0, "atr_mult_tp": 3.0},
            "4h": {"regime_ema_fast": 20, "regime_ema_slow": 50, "vwap_period": 20, "vwap_std": 2.0, "smc_swing_lookback": 15, "atr_mult_sl": 2.0, "atr_mult_tp": 3.5},
        },
        "orderflow_delta": {
            "1h": {"delta_period": 20, "cvd_period": 20, "divergence_lookback": 10, "absorption_threshold": 1.5, "atr_mult_sl": 1.5, "atr_mult_tp": 2.5},
            "15m": {"delta_period": 10, "cvd_period": 10, "divergence_lookback": 5, "absorption_threshold": 1.5, "atr_mult_sl": 1.2, "atr_mult_tp": 2.0},
        },
        "scalping_grid": {
            "1m": {"grid_levels": 5, "grid_spread_pct": 0.1, "grid_size_pct": 20.0, "take_profit_pct": 0.15, "stop_loss_pct": 0.3, "max_grid_exposure_pct": 60.0},
            "5m": {"grid_levels": 5, "grid_spread_pct": 0.15, "grid_size_pct": 20.0, "take_profit_pct": 0.25, "stop_loss_pct": 0.5, "max_grid_exposure_pct": 60.0},
        },
        "funding_arbitrage": {
            "1h": {"min_funding_rate": 0.0001, "max_funding_rate": 0.01, "position_hold_hours": 8, "atr_mult_sl": 1.0, "atr_mult_tp": 2.0, "max_positions": 3},
        },
    }

    # ──────────────────────────────────────────────
    # Validators
    # ──────────────────────────────────────────────

    @field_validator("hyperliquid_private_key")
    @classmethod
    def validate_private_key(cls, v: str) -> str:
        """Valida private key (hex com ou sem 0x)."""
        if v and v.startswith("0x"):
            v = v[2:]
        if v and not re.match(r"^[0-9a-fA-F]{64}$", v):
            raise ValueError("Private key inválida: deve ser um hex de 64 caracteres")
        return v

    @field_validator("hyperliquid_account_address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Valida endereço Ethereum."""
        if v and not re.match(r"^0x[0-9a-fA-F]{40}$", v):
            raise ValueError("Endereço Ethereum inválido: deve começar com 0x seguido de 40 hex chars")
        return v

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: str) -> str:
        """Valida lista de símbolos."""
        symbols = [s.strip() for s in v.split(",") if s.strip()]
        if not symbols:
            raise ValueError("Pelo menos um símbolo deve ser fornecido")
        for sym in symbols:
            if not re.match(r"^[A-Z0-9]{2,10}/USDC$", sym):
                raise ValueError(f"Símbolo inválido: {sym}")
        return v

    @model_validator(mode="after")
    def validate_credentials(self) -> "BotConfig":
        """Valida credenciais no modo mainnet."""
        if not self.testnet:
            if not self.hyperliquid_private_key:
                log.warning("Mainnet detectado sem private key definida. O bot não conseguirá fazer ordens.")
            if not self.hyperliquid_account_address:
                log.warning("Mainnet detectado sem account address definida.")
        return self

    @model_validator(mode="after")
    def apply_nested_env_overrides(self) -> "BotConfig":
        """
        Sobrescreve manualmente as sub-configs aninhadas com os valores
        reais do .env, contornando o bug de env_nested_delimiter (ver
        _apply_prefixed_env_overrides). Roda DEPOIS que pydantic-settings
        já tentou sua própria resolução automática — esta função corrige
        qualquer campo que tenha ficado vazio incorretamente.
        """
        try:
            env_file = self.model_config.get("env_file", ".env")
            env_values = dotenv_values(env_file)

            if not env_values:
                return self

            _apply_prefixed_env_overrides(self.risk, "RISK_", env_values)
            _apply_prefixed_env_overrides(self.staking, "STAKING_", env_values)
            _apply_prefixed_env_overrides(self.notifications, "NOTIFICATIONS_", env_values)
            _apply_prefixed_env_overrides(self.dashboard, "DASHBOARD_", env_values)
            _apply_prefixed_env_overrides(self.monitoring, "MONITORING_", env_values)
            _apply_prefixed_env_overrides(self.backtest, "BACKTEST_", env_values)
        except Exception as e:
            log.error(f"[CONFIG] Erro ao aplicar overrides de env aninhados: {e}")

        return self

    # ──────────────────────────────────────────────
    # Métodos Públicos
    # ──────────────────────────────────────────────

    def parse_symbols(self) -> List[SymbolConfig]:
        """
        Parseia a string de símbolos em objetos SymbolConfig.

        Returns:
            List[SymbolConfig]: Lista de configurações por símbolo.
        """
        if self._parsed and self._symbol_configs:
            return self._symbol_configs

        symbols = [s.strip() for s in self.symbols.split(",") if s.strip()]
        configs: List[SymbolConfig] = []

        for sym in symbols:
            coin = sym.replace("/USDC", "")
            configs.append(
                SymbolConfig(
                    symbol=sym,
                    coin=coin,
                    enabled=True,
                    leverage=self.leverage,
                    isolated_margin=self.isolated_margin,
                    max_position_pct=self.max_position_pct,
                    risk_per_trade_pct=self.risk_per_trade_pct,
                )
            )

        self._symbol_configs = configs
        self._parsed = True
        log.info(f"Símbolos carregados: {[s.symbol for s in configs]}")
        return configs

    def get_strategy_params(self, strategy: Optional[StrategyType] = None, timeframe: Optional[Timeframe] = None) -> Dict[str, Any]:
        """
        Retorna parâmetros padrão para uma estratégia + timeframe.

        Args:
            strategy: Estratégia (usa a padrão se None)
            timeframe: Timeframe (usa o padrão se None)

        Returns:
            Dict com parâmetros da estratégia.
        """
        strat = (strategy or self.strategy).value
        tf = (timeframe or self.timeframe).value

        try:
            params = self.STRATEGY_DEFAULTS[strat].get(tf, {})
            if not params:
                # Fallback para o primeiro timeframe disponível
                first_tf = next(iter(self.STRATEGY_DEFAULTS[strat].values()))
                params = first_tf
                log.warning(f"Parâmetros não encontrados para {strat}/{tf}, usando fallback")
            return params
        except KeyError:
            log.error(f"Estratégia desconhecida: {strat}")
            return {}

    def get_hyperliquid_url(self) -> str:
        """Retorna URL da API Hyperliquid conforme modo."""
        if self.testnet:
            return "https://api.hyperliquid-testnet.xyz"
        return "https://api.hyperliquid.xyz"

    def get_ws_url(self) -> str:
        """Retorna URL do WebSocket Hyperliquid."""
        if self.testnet:
            return "wss://api.hyperliquid-testnet.xyz/ws"
        return "wss://api.hyperliquid.xyz/ws"

    def to_dict(self) -> Dict[str, Any]:
        """Converte config para dict (omitindo credenciais)."""
        d = self.model_dump()
        # Remove credenciais sensíveis
        d.pop("hyperliquid_private_key", None)
        return d

    def save_json(self, path: str = "bot_config.json") -> None:
        """Salva configuração em JSON."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            log.info(f"Configuração salva em {path}")
        except OSError as e:
            log.error(f"Erro ao salvar configuração: {e}")
            raise

    @classmethod
    def load_json(cls, path: str = "bot_config.json") -> "BotConfig":
        """Carrega configuração de JSON e mescla com .env.

        Suporta dois formatos de chave no JSON:
        1. Campos de nível raiz (ex: "capital_usd", "leverage")
        2. Sub-configs aninhadas como dict (ex: "risk": {"daily_loss_limit_pct": 5.0})

        FIX (auditoria item 10): a versão anterior usava apenas
        setattr(cfg, key, value) para cada chave do JSON, o que só
        funciona para campos de nível raiz. Chaves com nomes que não
        batem exatamente com um atributo de BotConfig (ex:
        "max_daily_loss_pct" quando o campo real é
        "risk.daily_loss_limit_pct") eram silenciosamente ignoradas
        via hasattr() == False, sem log de aviso — o operador podia
        editar o JSON acreditando que a configuração estava sendo
        aplicada, quando na prática não estava.

        Esta versão:
        - Aplica corretamente sub-configs aninhadas quando fornecidas
          como dict (risk, staking, notifications, dashboard,
          monitoring, backtest).
        - Loga um WARNING explícito para qualquer chave de nível raiz
          que não corresponda a um campo real de BotConfig, em vez de
          ignorar silenciosamente.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = cls()

            # Nomes das sub-configs aninhadas conhecidas
            nested_configs = {
                "risk", "staking", "notifications", "dashboard",
                "monitoring", "backtest",
            }

            unknown_keys: List[str] = []

            for key, value in data.items():
                if value is None:
                    continue

                if key in nested_configs and isinstance(value, dict):
                    # Aplica campo a campo dentro da sub-config
                    sub_cfg = getattr(cfg, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_cfg, sub_key):
                            setattr(sub_cfg, sub_key, sub_value)
                        else:
                            unknown_keys.append(f"{key}.{sub_key}")
                    continue

                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                else:
                    unknown_keys.append(key)

            if unknown_keys:
                log.warning(
                    f"[CONFIG] {path}: chave(s) desconhecida(s) ignorada(s) "
                    f"(não correspondem a nenhum campo de BotConfig): "
                    f"{unknown_keys}. Verifique nomes/estrutura do JSON — "
                    f"campos de risco ficam em 'risk': {{...}}, não no "
                    f"nível raiz."
                )

            log.info(f"Configuração carregada de {path}")
            return cfg
        except FileNotFoundError:
            log.warning(f"Arquivo {path} não encontrado, usando defaults")
            return cls()
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:   # ALTERADO: + UnicodeDecodeError
            log.error(f"Erro ao carregar {path}: {e}")
            return cls()

    def validate_all(self) -> List[str]:
        """
        Valida toda a configuração e retorna lista de erros.

        Returns:
            List[str]: Lista de mensagens de erro (vazia se tudo ok).
        """
        errors: List[str] = []

        # Valida capital mínimo
        if self.capital_usd < 10:
            errors.append("Capital mínimo é $10 USD")

        # Valida exposição total
        total_exposure = self.max_position_pct * len(self.parse_symbols())
        if total_exposure > self.risk.max_exposure_pct:
            errors.append(
                f"Exposição total potencial ({total_exposure:.0f}%) excede o limite "
                f"({self.risk.max_exposure_pct}%). Reduza max_position_pct ou símbolos."
            )

        # Valida TP > SL
        if self.risk.take_profit_pct <= self.risk.stop_loss_pct:
            errors.append("Take Profit deve ser maior que Stop Loss")

        # Valida trailing stop
        if self.risk.trailing_stop_activation_pct <= self.risk.trailing_stop_distance_pct:
            errors.append("Ativação do trailing stop deve ser maior que a distância")

        # Valida staking
        if self.staking.enabled:
            if self.staking.stake_pct > 0 and not self.staking.validator_address:
                errors.append("Staking ativado mas validator_address não definido")
            if self.staking.vault_deposit_pct > 0 and not self.staking.vault_address:
                errors.append("Vault ativado mas vault_address não definido")

        return errors


# ──────────────────────────────────────────────
# Factory / Singleton
# ──────────────────────────────────────────────

_config_instance: Optional[BotConfig] = None


def get_config(reload: bool = False) -> BotConfig:
    """
    Retorna instância singleton da configuração.

    Args:
        reload: Se True, recarrega do .env

    Returns:
        BotConfig: Instância de configuração.
    """
    global _config_instance
    if _config_instance is None or reload:
        try:
            _config_instance = BotConfig()
            errors = _config_instance.validate_all()
            if errors:
                for err in errors:
                    log.error(f"Config validation: {err}")
            log.info("Configuração carregada com sucesso")
        except Exception as e:
            log.error(f"Erro ao carregar configuração: {e}")
            raise
    return _config_instance
