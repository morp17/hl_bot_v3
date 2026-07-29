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
    VOLATILITY_SQUEEZE = "volatility_squeeze"          # NOVA
    FUNDING_WEIGHTED_TREND = "funding_weighted_trend"   # NOVA


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
    max_exposure_pct: float = Field(80.0, ge=10.0, le=100.0, description="Exposição bruta máxima total (%)")
    max_correlated_exposure_pct: float = Field(
        60.0, ge=10.0, le=100.0,
        description=(
            "Exposição direcional máxima agregada (%) entre símbolos "
            "tratados como correlacionados. Por padrão, trata TODOS os "
            "símbolos operados pelo bot como um único grupo de "
            "correlação. Defina 100.0 para desabilitar esta proteção."
        ),
    )
    circuit_breaker_loss_pct: float = Field(15.0, ge=1.0, le=50.0, description="Perda que aciona circuit breaker (%)")
    circuit_breaker_cooldown_sec: int = Field(3600, ge=60, le=86400, description="Cooldown do circuit breaker (segundos)")
    trade_hour_start_utc: int = Field(0, ge=0, le=23, description="Hora UTC de início para operar (0=desligado)")
    trade_hour_end_utc: int = Field(0, ge=0, le=23, description="Hora UTC de fim para operar (0=desligado)")
    max_spread_pct: float = Field(0.5, ge=0.01, le=5.0, description="Spread máximo permitido (%)")
    max_funding_rate: float = Field(
        0.1, ge=0.001, le=1.0,
        description=(
            "Funding rate máximo permitido, em PONTOS PERCENTUAIS "
            "(ex: 0.1 = 0.1%, não 10%). É convertido para fração "
            "(dividido por 100) antes de ser usado em CapitalProtection "
            "— use max_funding_rate_fraction (property abaixo) em vez "
            "de dividir manualmente por 100 em código novo. "
            "NOTA DE ESCALA: este campo usa unidade DIFERENTE da usada "
            "em STRATEGY_DEFAULTS['funding_arbitrage']['max_funding_rate'] "
            "(0.01), que já é uma fração direta (1%) comparada "
            "diretamente contra o funding_rate bruto da exchange — são "
            "dois parâmetros com propósitos distintos (filtro de "
            "proteção de capital vs. limiar de sinal de estratégia), "
            "não devem ser confundidos nem unificados sem revisar ambos "
            "os pontos de uso."
        ),
    )
    taker_fee: float = Field(0.0005, ge=0.0, le=0.01, description="Taxa taker da exchange")

    @property
    def max_funding_rate_fraction(self) -> float:
        """
        Retorna max_funding_rate convertido de pontos percentuais para
        fração (ex: 0.1 -> 0.001), pronto para uso direto em
        CapitalProtection.max_funding_rate ou qualquer comparação contra
        funding_rate bruto retornado pela exchange (que já vem em fração,
        ex: 0.0001 = 0.01%).

        FIX (item 2 da auditoria — inconsistência de escala): esta
        property centraliza a única conversão correta, eliminando a
        divisão manual "/100" que antes só existia inline em main.py
        (fácil de esquecer/duplicar incorretamente em código futuro).
        """
        return self.max_funding_rate / 100.0


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
    user: str = Field("admin", description="Usuário do dashboard (HTTP Basic Auth)")
    password: str = Field("changeme", description="Senha do dashboard (HTTP Basic Auth)", min_length=4)
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
    Popula campos de um sub-model a partir de valores de ambiente com
    prefixo dado, contornando o parser automático de
    env_nested_delimiter do pydantic-settings.

    FIX (confirmado empiricamente): mesmo após remover default_factory
    das sub-configs, valores como NOTIFICATIONS_TELEGRAM_BOT_TOKEN
    continuavam chegando vazios em cfg.notifications.telegram_bot_token,
    apesar do .env estar correto. Causa: nomes de campo com múltiplos
    underscores tornam ambígua, para o parser automático, a fronteira
    entre prefixo do sub-model e nome do campo.

    Esta função aplica manualmente cada campo, usando correspondência
    exata de "{prefix}{NOME_DO_CAMPO_EM_MAIUSCULO}" — sem ambiguidade.

    Args:
        sub_model: Instância da sub-config a popular (ex: self.risk).
        prefix: Prefixo das chaves (ex: "RISK_").
        env_values: Dict de valores de ambiente já resolvidos (ver
            _resolve_env_values_for_nested_overrides — pode vir de
            arquivo .env, de os.environ, ou da mescla de ambos).
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


def _resolve_env_values_for_nested_overrides(
    env_file: Optional[str],
) -> Dict[str, Optional[str]]:
    """
    Resolve a fonte de valores usada para popular as sub-configs
    aninhadas (risk, staking, notifications, dashboard, monitoring,
    backtest), combinando o arquivo .env (se aplicável) com o ambiente
    de processo (os.environ) — este último sempre com prioridade.

    FIX (achado na análise dos resultados de teste — pytest local):
    a versão anterior lia EXCLUSIVAMENTE dotenv_values(arquivo), nunca
    considerando os.environ. Isso tinha dois efeitos práticos:

    1. Variáveis de ambiente definidas diretamente no processo (ex:
       via `docker run -e RISK_STOP_LOSS_PCT=...`, sem um arquivo .env
       físico no container) eram silenciosamente ignoradas para as
       sub-configs aninhadas — mesmo funcionando perfeitamente para os
       campos de nível raiz (que o pydantic-settings resolve
       nativamente via os.environ).
    2. Testes que usam `monkeypatch.setenv(...)` para simular
       variáveis de ambiente (sem escrever um arquivo .env físico)
       nunca viam esse valor refletido nas sub-configs — o teste
       sempre lia o arquivo .env real do disco, quando presente.

    Args:
        env_file: Caminho do arquivo .env a considerar, ou None/"" para
            não ler nenhum arquivo (apenas os.environ é usado nesse
            caso).

    Returns:
        Dict[str, Optional[str]]: valores mesclados, com os.environ
            sobrepondo qualquer valor conflitante vindo do arquivo.
    """
    merged: Dict[str, Optional[str]] = {}

    if env_file:
        try:
            file_values = dotenv_values(env_file)
            if file_values:
                merged.update(file_values)
        except Exception as e:
            log.debug(f"[CONFIG] Não foi possível ler '{env_file}': {e}")

    # os.environ sempre tem prioridade — mesmo comportamento que
    # pydantic-settings já aplica nativamente para campos de nível raiz.
    merged.update(os.environ)

    return merged


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
    strategy: StrategyType = Field(StrategyType.HYBRID_REGIME, description="Estratégia principal (modo single-strategy)")
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

    # ── Ensemble de estratégias ──
    ensemble_mode: bool = Field(
        False,
        description=(
            "Se True, usa get_ensemble_signal() (combina todas as "
            "estratégias em enabled_strategies, ponderadas por "
            "confiança) em vez do modo single-strategy (cfg.strategy). "
            "Default False para não alterar o comportamento de "
            "instalações existentes."
        ),
    )
    ensemble_min_confluence: int = Field(
        2, ge=1, le=10,
        description=(
            "Número mínimo de estratégias habilitadas que precisam "
            "concordar na mesma direção (buy ou sell) para o ensemble "
            "emitir um sinal, em vez de hold."
        ),
    )
    ensemble_min_avg_confidence: float = Field(
        0.55, ge=0.0, le=1.0,
        description=(
            "Confiança média mínima (0.0-1.0) entre os votos de uma "
            "direção para o ensemble emitir um sinal nessa direção."
        ),
    )

    # ── Sub-configs ──
    # FIX (auditoria — bug do pydantic-settings): default_factory=X faz o
    # sub-model ser instanciado ANTES da camada de resolução de env vars
    # ter chance de popular seus campos internos, em algumas versões de
    # pydantic-settings. Correção: instanciar diretamente.
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
        "volatility_squeeze": {
            "1h": {"squeeze_lookback": 50, "squeeze_quantile": 0.25, "volume_confirm_mult": 1.2, "atr_mult_sl": 1.5, "atr_mult_tp": 3.0},
            "4h": {"squeeze_lookback": 50, "squeeze_quantile": 0.25, "volume_confirm_mult": 1.2, "atr_mult_sl": 1.8, "atr_mult_tp": 3.5},
            "15m": {"squeeze_lookback": 50, "squeeze_quantile": 0.2, "volume_confirm_mult": 1.3, "atr_mult_sl": 1.2, "atr_mult_tp": 2.2},
        },
        "funding_weighted_trend": {
            "1h": {"min_funding_rate": 0.0001, "max_funding_rate": 0.01, "rsi_exhaustion_high": 75, "rsi_exhaustion_low": 25, "atr_mult_sl": 2.0, "atr_mult_tp": 3.5},
        },
    }

    # ──────────────────────────────────────────────
    # Construtor (FIX — achado na análise de teste)
    # ──────────────────────────────────────────────

    def __init__(self, **data: Any) -> None:
        """
        FIX (achado na análise dos resultados de pytest): pydantic-settings
        resolve o parâmetro especial `_env_file` internamente para decidir
        qual arquivo alimenta os campos de NÍVEL RAIZ (symbols, capital_usd
        etc.) — essa parte já funcionava corretamente. Porém,
        apply_nested_env_overrides() (o model_validator que popula
        manualmente as sub-configs aninhadas) sempre lia
        model_config["env_file"] (o default de CLASSE, ".env"),
        ignorando completamente qualquer override de instância —
        inclusive `_env_file=None`, usado por testes para isolar de um
        .env real em disco.

        Resultado prático observado: `BotConfig(_env_file=None)` em testes
        ainda "vazava" valores de um .env real presente no diretório de
        trabalho (token do Telegram, portas, percentuais de risco reais)
        para dentro de cfg.risk/cfg.notifications/etc., mascarando os
        valores esperados pelos testes (defaults ou monkeypatch.setenv).

        Esta correção detecta quando `_env_file` foi passado
        explicitamente (incluindo None) e, nesse caso, refaz a aplicação
        das sub-configs aninhadas usando a fonte corretamente resolvida
        (ver _resolve_env_values_for_nested_overrides), IGNORANDO
        qualquer valor incorreto que o model_validator automático possa
        ter aplicado durante a construção padrão.

        Sub-configs passadas EXPLICITAMENTE como kwarg (ex:
        `BotConfig(risk=RiskConfig(...))`) são preservadas intactas —
        não são resetadas nem sobrescritas por este mecanismo.

        Construções normais sem `_env_file` (o caso comum em produção,
        `BotConfig()`) NÃO são afetadas por este método — apenas o
        comportamento do model_validator automático muda ligeiramente
        (ver apply_nested_env_overrides: agora também mescla os.environ,
        não apenas o arquivo).

        Args:
            **data: Argumentos do construtor, incluindo eventualmente o
                parâmetro especial `_env_file` do pydantic-settings.
        """
        env_file_explicit = "_env_file" in data
        env_file_value = data.get("_env_file", ".env")
        explicit_fields = set(data.keys())

        super().__init__(**data)

        if env_file_explicit:
            self._reapply_nested_env_overrides(env_file_value, explicit_fields)

    def _reapply_nested_env_overrides(
        self, env_file: Optional[str], explicit_fields: Set[str]
    ) -> None:
        """
        Corrige as sub-configs aninhadas para respeitar um `_env_file`
        explicitamente passado ao construtor (ver __init__), incluindo o
        caso `_env_file=None`.

        Reseta cada sub-config (exceto as passadas explicitamente pelo
        chamador, ver `explicit_fields`) para seus defaults declarados
        antes de reaplicar — o model_validator automático já pode ter
        populado valores incorretos (lidos do .env de classe) durante a
        validação padrão que ocorre dentro de super().__init__().

        Args:
            env_file: Caminho do .env a considerar, ou None para não
                carregar nenhum arquivo (usa apenas os.environ).
            explicit_fields: Nomes dos campos passados diretamente ao
                construtor (ex: {"risk", "strategy"}) — sub-configs
                nesta lista são preservadas sem reset/reaplicação.
        """
        try:
            nested_field_map = {
                "risk": RiskConfig,
                "staking": StakingConfig,
                "notifications": NotificationConfig,
                "dashboard": DashboardConfig,
                "monitoring": MonitoringConfig,
                "backtest": BacktestConfig,
            }
            prefix_map = {
                "risk": "RISK_",
                "staking": "STAKING_",
                "notifications": "NOTIFICATIONS_",
                "dashboard": "DASHBOARD_",
                "monitoring": "MONITORING_",
                "backtest": "BACKTEST_",
            }

            fields_to_process = [
                name for name in nested_field_map if name not in explicit_fields
            ]
            if not fields_to_process:
                return

            for name in fields_to_process:
                setattr(self, name, nested_field_map[name]())

            env_values = _resolve_env_values_for_nested_overrides(env_file)
            if not env_values:
                return

            for name in fields_to_process:
                _apply_prefixed_env_overrides(
                    getattr(self, name), prefix_map[name], env_values
                )

        except Exception as e:
            log.error(f"[CONFIG] Erro ao reaplicar overrides de env aninhados: {e}")

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
        reais do .env de classe + os.environ, contornando o bug de
        env_nested_delimiter (ver _apply_prefixed_env_overrides).

        FIX (achado na análise de teste): passou a mesclar os.environ
        com prioridade sobre o arquivo (antes lia SOMENTE o arquivo via
        dotenv_values). Isso corrige dois cenários:
        1. Produção: variáveis de ambiente definidas diretamente no
           processo (ex: Docker sem .env físico) agora são respeitadas
           para as sub-configs aninhadas, não só para os campos de
           nível raiz.
        2. Quando esta função roda durante a construção padrão de um
           BotConfig(_env_file=None) (antes do __init__ customizado
           corrigir o resultado via _reapply_nested_env_overrides),
           pelo menos os valores vindos de os.environ (ex:
           monkeypatch.setenv em testes) já ficam corretos nesta
           primeira passada — o __init__ customizado então apenas
           remove o vazamento remanescente do arquivo .env real.

        Esta função roda DEPOIS que pydantic-settings já tentou sua
        própria resolução automática.
        """
        try:
            env_file = self.model_config.get("env_file", ".env")
            env_values = _resolve_env_values_for_nested_overrides(env_file)

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
        """Carrega configuração de JSON e mescla com .env."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = cls()

            nested_configs = {
                "risk", "staking", "notifications", "dashboard",
                "monitoring", "backtest",
            }

            unknown_keys: List[str] = []

            for key, value in data.items():
                if value is None:
                    continue

                if key in nested_configs and isinstance(value, dict):
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
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            log.error(f"Erro ao carregar {path}: {e}")
            return cls()

    def validate_all(self) -> List[str]:
        """
        Valida toda a configuração e retorna lista de erros.

        Returns:
            List[str]: Lista de mensagens de erro (vazia se tudo ok).
        """
        errors: List[str] = []

        if self.capital_usd < 10:
            errors.append("Capital mínimo é $10 USD")

        total_exposure = self.max_position_pct * len(self.parse_symbols())
        if total_exposure > self.risk.max_exposure_pct:
            errors.append(
                f"Exposição total potencial ({total_exposure:.0f}%) excede o limite "
                f"({self.risk.max_exposure_pct}%). Reduza max_position_pct ou símbolos."
            )

        if self.risk.take_profit_pct <= self.risk.stop_loss_pct:
            errors.append("Take Profit deve ser maior que Stop Loss")

        if self.risk.trailing_stop_activation_pct <= self.risk.trailing_stop_distance_pct:
            errors.append("Ativação do trailing stop deve ser maior que a distância")

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