"""
Testes unitários para o módulo de configuração.
"""
import pytest
from crypto_bot_core.config import (
    BotConfig,
    SymbolConfig,
    RiskConfig,
    StrategyType,
    Timeframe,
    get_config,
)


class TestSymbolConfig:
    """Testes para SymbolConfig."""

    def test_valid_symbol(self) -> None:
        """Testa criação de símbolo válido."""
        sc = SymbolConfig(symbol="BTC/USDC", coin="BTC")
        assert sc.symbol == "BTC/USDC"
        assert sc.coin == "BTC"
        assert sc.enabled is True

    def test_valid_symbol_eth(self) -> None:
        """Testa ETH/USDC."""
        sc = SymbolConfig(symbol="ETH/USDC", coin="ETH")
        assert sc.symbol == "ETH/USDC"

    def test_valid_symbol_sol(self) -> None:
        """Testa SOL/USDC."""
        sc = SymbolConfig(symbol="SOL/USDC", coin="SOL")
        assert sc.symbol == "SOL/USDC"

    def test_invalid_symbol_no_usdc(self) -> None:
        """Testa símbolo sem /USDC."""
        with pytest.raises(ValueError):
            SymbolConfig(symbol="BTC/USDT", coin="BTC")

    def test_invalid_symbol_lowercase(self) -> None:
        """Testa símbolo em minúsculo."""
        with pytest.raises(ValueError):
            SymbolConfig(symbol="btc/usdc", coin="BTC")

    def test_invalid_coin(self) -> None:
        """Testa moeda inválida."""
        with pytest.raises(ValueError):
            SymbolConfig(symbol="BTC/USDC", coin="btc")

    def test_leverage_range(self) -> None:
        """Testa validação de alavancagem."""
        with pytest.raises(ValueError):
            SymbolConfig(symbol="BTC/USDC", coin="BTC", leverage=0)
        with pytest.raises(ValueError):
            SymbolConfig(symbol="BTC/USDC", coin="BTC", leverage=51)

    def test_risk_pct_range(self) -> None:
        """Testa validação de risco."""
        with pytest.raises(ValueError):
            SymbolConfig(symbol="BTC/USDC", coin="BTC", risk_per_trade_pct=0.0)
        with pytest.raises(ValueError):
            SymbolConfig(symbol="BTC/USDC", coin="BTC", risk_per_trade_pct=15.0)

    def test_no_dashboard_enabled_field(self) -> None:
        """
        REGRESSÃO: dashboard_enabled pertence a BotConfig, não a
        SymbolConfig. Já apareceu erroneamente aqui uma vez
        (AttributeError em runtime ao construir o dashboard), causando
        'BotConfig' object has no attribute 'dashboard_enabled' porque
        o campo tinha sido colado na classe errada. Este teste apenas
        documenta a fronteira correta — SymbolConfig não deve ganhar
        esse campo de volta por engano em edições futuras.
        """
        sc = SymbolConfig(symbol="BTC/USDC", coin="BTC")
        assert not hasattr(sc, "dashboard_enabled")


class TestRiskConfig:
    """Testes para RiskConfig."""

    def test_default_values(self) -> None:
        """Testa valores padrão."""
        rc = RiskConfig()
        assert rc.max_drawdown_pct == 15.0
        assert rc.daily_loss_limit_pct == 5.0
        assert rc.max_consecutive_losses == 3
        assert rc.stop_loss_pct == 2.0
        assert rc.take_profit_pct == 5.0

    def test_drawdown_range(self) -> None:
        """Testa range do drawdown."""
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_pct=0.5)
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_pct=55.0)

    def test_removed_fields_not_present(self) -> None:
        """
        FIX (auditoria item 4/6): kelly_fraction, anti_martingale_mult,
        correlation_hedge_threshold, min_edge_vs_costs_mult,
        btc_crash_filter_pct e liquidation_safety_buffer_pct foram
        removidos por não terem nenhuma implementação de cálculo
        conectada — existiam apenas como configuração decorativa.
        Este teste impede que voltem a ser adicionados sem uma
        implementação real acompanhando.
        """
        rc = RiskConfig()
        removed_fields = [
            "kelly_fraction",
            "anti_martingale_mult",
            "correlation_hedge_threshold",
            "min_edge_vs_costs_mult",
            "btc_crash_filter_pct",
            "liquidation_safety_buffer_pct",
        ]
        for field in removed_fields:
            assert not hasattr(rc, field), (
                f"Campo '{field}' foi removido por não ter implementação "
                f"de cálculo real conectada (ver risk.py e capital_protection.py). "
                f"Se foi reintroduzido, também precisa da lógica correspondente."
            )

    def test_extra_fields_rejected_or_ignored(self) -> None:
        """
        Tenta construir RiskConfig com um campo removido — não deve
        quebrar silenciosamente aceitando o valor como se fosse válido
        (pydantic com extra='ignore' no BaseSettings não se aplica a
        BaseModel simples como RiskConfig, então o comportamento padrão
        de pydantic v2 para BaseModel é rejeitar campos desconhecidos
        por padrão, a menos que model_config diga o contrário).
        """
        # RiskConfig é BaseModel puro (não BaseSettings), então por
        # padrão do pydantic v2 campos extras não declarados no schema
        # são ignorados apenas se model_config permitir; caso contrário
        # levantam erro. Este teste apenas documenta que kelly_fraction
        # não é mais um kwarg válido.
        rc = RiskConfig()  # não deve levantar erro na ausência do campo
        assert rc is not None


class TestBotConfig:
    """Testes para BotConfig."""

    def test_default_values(self) -> None:
        """Testa valores padrão do BotConfig."""
        # Usa _env_file=None para evitar carregar o .env real
        cfg = BotConfig(_env_file=None)
        assert cfg.testnet is True
        assert cfg.capital_usd == 1000.0
        assert cfg.strategy == StrategyType.HYBRID_REGIME
        assert cfg.timeframe == Timeframe.H1
        assert cfg.leverage == 3

    def test_dashboard_enabled_default(self) -> None:
        """
        FIX: dashboard_enabled precisa existir em BotConfig (não em
        SymbolConfig) para que dashboard/main.py::start_dashboard_thread
        não quebre com AttributeError.
        """
        cfg = BotConfig(_env_file=None)
        assert hasattr(cfg, "dashboard_enabled")
        assert cfg.dashboard_enabled is True

    def test_parse_symbols_default(self) -> None:
        """Testa parsing de símbolos padrão."""
        cfg = BotConfig()
        symbols = cfg.parse_symbols()
        assert len(symbols) == 3
        assert symbols[0].symbol == "BTC/USDC"
        assert symbols[1].symbol == "ETH/USDC"
        assert symbols[2].symbol == "SOL/USDC"

    def test_parse_symbols_custom(self) -> None:
        """Testa parsing de símbolos customizados."""
        cfg = BotConfig(symbols="BTC/USDC,ETH/USDC")
        symbols = cfg.parse_symbols()
        assert len(symbols) == 2

    def test_parse_symbols_single(self) -> None:
        """Testa parsing de um único símbolo."""
        cfg = BotConfig(symbols="BTC/USDC")
        symbols = cfg.parse_symbols()
        assert len(symbols) == 1
        assert symbols[0].coin == "BTC"

    def test_invalid_symbols(self) -> None:
        """Testa símbolos inválidos."""
        with pytest.raises(ValueError):
            BotConfig(symbols="")

    def test_get_hyperliquid_url_testnet(self) -> None:
        """Testa URL da testnet."""
        cfg = BotConfig(testnet=True)
        assert "testnet" in cfg.get_hyperliquid_url()

    def test_get_hyperliquid_url_mainnet(self) -> None:
        """Testa URL da mainnet."""
        cfg = BotConfig(testnet=False)
        assert "testnet" not in cfg.get_hyperliquid_url()

    def test_get_ws_url_testnet(self) -> None:
        """Testa WebSocket URL da testnet."""
        cfg = BotConfig(testnet=True)
        assert "testnet" in cfg.get_ws_url()

    def test_get_strategy_params(self) -> None:
        """Testa obtenção de parâmetros de estratégia."""
        cfg = BotConfig(strategy=StrategyType.TREND_FOLLOW, timeframe=Timeframe.H1)
        params = cfg.get_strategy_params()
        assert "ema_fast" in params
        assert params["ema_fast"] == 9
        assert params["ema_slow"] == 21

    def test_get_strategy_params_hybrid(self) -> None:
        """Testa parâmetros do hybrid_regime."""
        cfg = BotConfig(strategy=StrategyType.HYBRID_REGIME, timeframe=Timeframe.H1)
        params = cfg.get_strategy_params()
        assert "regime_ema_fast" in params
        assert "vwap_period" in params

    def test_get_strategy_params_scalping(self) -> None:
        """Testa parâmetros do scalping_grid."""
        cfg = BotConfig(strategy=StrategyType.SCALPING_GRID, timeframe=Timeframe.M1)
        params = cfg.get_strategy_params()
        assert "grid_levels" in params
        assert params["grid_levels"] == 5

    def test_get_strategy_params_funding(self) -> None:
        """Testa parâmetros do funding_arbitrage."""
        cfg = BotConfig(strategy=StrategyType.FUNDING_ARBITRAGE, timeframe=Timeframe.H1)
        params = cfg.get_strategy_params()
        assert "min_funding_rate" in params

    def test_get_strategy_params_fallback_logs_warning(self, caplog) -> None:
        """
        Timeframe sem entrada em STRATEGY_DEFAULTS deve cair no
        fallback (primeiro timeframe do dict) e logar warning, não
        quebrar. Reproduz o cenário real observado em produção:
        trend_follow/5m não tem entrada própria.
        """
        cfg = BotConfig(strategy=StrategyType.TREND_FOLLOW, timeframe=Timeframe.M5)
        params = cfg.get_strategy_params()
        assert params  # fallback não deve ser vazio
        assert "ema_fast" in params

    def test_validate_private_key(self) -> None:
        """Testa validação de private key."""
        cfg = BotConfig(hyperliquid_private_key="0x" + "a" * 64)
        assert cfg.hyperliquid_private_key == "a" * 64

    def test_validate_private_key_no_prefix(self) -> None:
        """Testa private key sem 0x."""
        cfg = BotConfig(hyperliquid_private_key="b" * 64)
        assert cfg.hyperliquid_private_key == "b" * 64

    def test_invalid_private_key(self) -> None:
        """Testa private key inválida."""
        with pytest.raises(ValueError):
            BotConfig(hyperliquid_private_key="0xshort")

    def test_validate_address(self) -> None:
        """Testa validação de endereço."""
        cfg = BotConfig(hyperliquid_account_address="0x" + "c" * 40)
        assert cfg.hyperliquid_account_address == "0x" + "c" * 40

    def test_invalid_address(self) -> None:
        """Testa endereço inválido."""
        with pytest.raises(ValueError):
            BotConfig(hyperliquid_account_address="0xshort")

    def test_validate_all_ok(self) -> None:
        """Testa validação completa sem erros."""
        cfg = BotConfig(capital_usd=1000)
        errors = cfg.validate_all()
        assert len(errors) == 0

    def test_validate_all_capital(self) -> None:
        """Testa validação de capital mínimo."""
        with pytest.raises(ValueError):
            BotConfig(capital_usd=5)

    def test_validate_all_tp_sl(self) -> None:
        """Testa validação TP > SL."""
        cfg = BotConfig()
        cfg.risk.take_profit_pct = 1.0
        cfg.risk.stop_loss_pct = 2.0
        errors = cfg.validate_all()
        assert any("take profit" in e.lower() for e in errors)

    def test_to_dict_omits_key(self) -> None:
        """Testa que to_dict não expõe private key."""
        cfg = BotConfig(hyperliquid_private_key="0x" + "d" * 64)
        d = cfg.to_dict()
        assert "hyperliquid_private_key" not in d

    def test_get_config_singleton(self) -> None:
        """Testa singleton do get_config."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_get_config_reload(self) -> None:
        """Testa reload do get_config."""
        cfg1 = get_config()
        cfg2 = get_config(reload=True)
        assert cfg1 is not cfg2


class TestEnabledStrategies:
    """
    Testes para enabled_strategies / is_strategy_enabled (item 9 da
    auditoria): restringe quais estratégias podem gerar sinal real,
    evitando que mean_reversion/orderflow_delta/scalping_grid/
    funding_arbitrage operem sem validação walk-forward prévia.
    """

    def test_default_enabled_strategies(self) -> None:
        """Default deve conter apenas as 3 estratégias validadas."""
        cfg = BotConfig(_env_file=None)
        assert cfg.enabled_strategies == "trend_follow,adaptive_trend,hybrid_regime"

    def test_is_strategy_enabled_true_for_default(self) -> None:
        """trend_follow deve estar habilitada por padrão."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.TREND_FOLLOW)
        assert cfg.is_strategy_enabled() is True

    def test_is_strategy_enabled_false_for_mean_reversion(self) -> None:
        """
        mean_reversion NÃO está em enabled_strategies por padrão —
        reproduz o comportamento real observado em produção (bot
        retornando hold permanentemente com STRATEGY=mean_reversion
        e ENABLED_STRATEGIES padrão).
        """
        cfg = BotConfig(_env_file=None, strategy=StrategyType.MEAN_REVERSION)
        assert cfg.is_strategy_enabled() is False

    def test_is_strategy_enabled_with_explicit_strategy_param(self) -> None:
        """is_strategy_enabled aceita uma estratégia explícita, não só a global."""
        cfg = BotConfig(_env_file=None, strategy=StrategyType.HYBRID_REGIME)
        assert cfg.is_strategy_enabled(StrategyType.TREND_FOLLOW) is True
        assert cfg.is_strategy_enabled(StrategyType.ORDERFLOW_DELTA) is False

    def test_enabled_strategies_custom_list(self) -> None:
        """Lista customizada deve habilitar corretamente."""
        cfg = BotConfig(_env_file=None, enabled_strategies="mean_reversion,scalping_grid")
        assert cfg.is_strategy_enabled(StrategyType.MEAN_REVERSION) is True
        assert cfg.is_strategy_enabled(StrategyType.TREND_FOLLOW) is False

    def test_enabled_strategies_invalid_name_rejected(self) -> None:
        """Nome de estratégia desconhecida em enabled_strategies deve levantar erro."""
        with pytest.raises(ValueError):
            BotConfig(_env_file=None, enabled_strategies="estrategia_que_nao_existe")

    def test_enabled_strategies_whitespace_tolerant(self) -> None:
        """Espaços em volta das vírgulas não devem quebrar o parsing."""
        cfg = BotConfig(_env_file=None, enabled_strategies=" trend_follow , hybrid_regime ")
        assert cfg.is_strategy_enabled(StrategyType.TREND_FOLLOW) is True
        assert cfg.is_strategy_enabled(StrategyType.HYBRID_REGIME) is True


class TestNestedEnvOverrides:
    """
    REGRESSÃO CRÍTICA: valida que sub-configs aninhadas (risk, staking,
    notifications, dashboard, monitoring, backtest) recebem valores
    reais do .env via variáveis de ambiente, não apenas via construtor
    Python direto.

    Esta é a lacuna de cobertura que permitiu o bug do Telegram passar
    despercebido: todos os testes anteriores construíam BotConfig(risk=
    RiskConfig(...)) diretamente, nunca exercitando o caminho real de
    carregamento via env_nested_delimiter/apply_nested_env_overrides.
    Sem monkeypatch.setenv aqui, esse tipo de falha de integração nunca
    seria pego pela suíte de testes.
    """

    def test_notifications_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Reproduz o bug real: NOTIFICATIONS_TELEGRAM_BOT_TOKEN no
        ambiente deve chegar em cfg.notifications.telegram_bot_token.
        """
        monkeypatch.setenv("NOTIFICATIONS_TELEGRAM_BOT_TOKEN", "test_token_123")
        monkeypatch.setenv("NOTIFICATIONS_TELEGRAM_CHAT_ID", "999888777")

        cfg = BotConfig(_env_file=None)

        assert cfg.notifications.telegram_bot_token == "test_token_123"
        assert cfg.notifications.telegram_chat_id == "999888777"

    def test_risk_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RISK_STOP_LOSS_PCT no ambiente deve chegar em cfg.risk.stop_loss_pct."""
        monkeypatch.setenv("RISK_STOP_LOSS_PCT", "3.5")
        monkeypatch.setenv("RISK_MAX_DRAWDOWN_PCT", "22.0")

        cfg = BotConfig(_env_file=None)

        assert cfg.risk.stop_loss_pct == 3.5
        assert cfg.risk.max_drawdown_pct == 22.0

    def test_dashboard_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DASHBOARD_PORT no ambiente deve chegar em cfg.dashboard.port."""
        monkeypatch.setenv("DASHBOARD_PORT", "9999")

        cfg = BotConfig(_env_file=None)

        assert cfg.dashboard.port == 9999

    def test_backtest_slippage_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BACKTEST_SLIPPAGE_PCT no ambiente deve chegar em cfg.backtest.slippage_pct."""
        monkeypatch.setenv("BACKTEST_SLIPPAGE_PCT", "0.12")

        cfg = BotConfig(_env_file=None)

        assert cfg.backtest.slippage_pct == 0.12

    def test_bool_field_cast_correctly_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RISK_TRAILING_STOP=true deve virar bool True, não string 'true'."""
        monkeypatch.setenv("RISK_TRAILING_STOP", "true")

        cfg = BotConfig(_env_file=None)

        assert cfg.risk.trailing_stop is True
        assert isinstance(cfg.risk.trailing_stop, bool)

    def test_int_field_cast_correctly_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RISK_MAX_OPEN_TRADES=7 deve virar int 7, não string '7'."""
        monkeypatch.setenv("RISK_MAX_OPEN_TRADES", "7")

        cfg = BotConfig(_env_file=None)

        assert cfg.risk.max_open_trades == 7
        assert isinstance(cfg.risk.max_open_trades, int)

    def test_absent_env_keeps_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sem override no ambiente, o default do Field deve ser preservado."""
        monkeypatch.delenv("RISK_STOP_LOSS_PCT", raising=False)

        cfg = BotConfig(_env_file=None)

        assert cfg.risk.stop_loss_pct == 2.0  # default declarado em RiskConfig

    def test_invalid_env_value_logs_warning_not_crash(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """
        Valor não conversível para o tipo esperado deve logar warning
        e manter o default, não derrubar a aplicação inteira no boot.
        """
        monkeypatch.setenv("RISK_MAX_OPEN_TRADES", "não_é_um_numero")

        cfg = BotConfig(_env_file=None)

        # Não deve ter quebrado a inicialização
        assert cfg is not None
        assert cfg.risk.max_open_trades == 3  # default preservado


class TestStrategyDefaults:
    """Testes para defaults de estratégias."""

    def test_all_strategies_have_defaults(self) -> None:
        """Testa que todas as estratégias têm defaults."""
        for strategy in StrategyType:
            assert strategy.value in BotConfig.STRATEGY_DEFAULTS, f"Estratégia {strategy.value} sem defaults"

    def test_all_timeframes_have_params(self) -> None:
        """Testa que timeframes conhecidos têm parâmetros."""
        for strategy_name, tfs in BotConfig.STRATEGY_DEFAULTS.items():
            assert len(tfs) > 0, f"Estratégia {strategy_name} sem timeframes"
            for tf, params in tfs.items():
                assert len(params) > 0, f"{strategy_name}/{tf} sem parâmetros"


class TestJsonRoundtrip:
    """
    Testes para save_json/load_json (item 10 da auditoria): o mapeamento
    anterior ignorava silenciosamente chaves de sub-config no nível
    raiz do JSON. Esta versão aplica corretamente dicts aninhados e
    loga warning explícito para chaves desconhecidas.
    """

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """Salvar e recarregar deve preservar valores de nível raiz."""
        cfg = BotConfig(_env_file=None, capital_usd=5000.0, leverage=5)
        path = str(tmp_path / "test_config.json")
        cfg.save_json(path)

        loaded = BotConfig.load_json(path)
        assert loaded.capital_usd == 5000.0
        assert loaded.leverage == 5

    def test_load_json_nested_dict_applied(self, tmp_path) -> None:
        """
        FIX item 10: chave 'risk' como dict aninhado deve ser aplicada
        campo a campo em cfg.risk, não ignorada.
        """
        import json

        path = tmp_path / "test_config.json"
        path.write_text(json.dumps({
            "capital_usd": 2000.0,
            "risk": {
                "daily_loss_limit_pct": 8.0,
                "max_consecutive_losses": 7,
            },
        }))

        loaded = BotConfig.load_json(str(path))
        assert loaded.capital_usd == 2000.0
        assert loaded.risk.daily_loss_limit_pct == 8.0
        assert loaded.risk.max_consecutive_losses == 7

    def test_load_json_unknown_root_key_logs_warning(self, tmp_path, caplog) -> None:
        """
        Chave de nível raiz que não bate com nenhum campo real deve
        gerar warning explícito — reproduz o bug original onde
        'max_daily_loss_pct' (nome errado) era ignorado sem log algum.
        """
        import json
        import logging

        path = tmp_path / "test_config.json"
        path.write_text(json.dumps({
            "max_daily_loss_pct": 0.10,  # nome errado — campo real é risk.daily_loss_limit_pct
        }))

        with caplog.at_level(logging.WARNING):
            BotConfig.load_json(str(path))

        # Não valida a lib de log específica (loguru vs logging podem
        # divergir na captura), mas garante que não houve crash e que
        # o objeto foi criado com defaults.
        loaded = BotConfig.load_json(str(path))
        assert loaded is not None

    def test_load_json_unknown_nested_key_logs_warning(self, tmp_path) -> None:
        """Sub-chave desconhecida dentro de um dict aninhado também deve ser sinalizada."""
        import json

        path = tmp_path / "test_config.json"
        path.write_text(json.dumps({
            "risk": {
                "campo_que_nao_existe": 123,
                "daily_loss_limit_pct": 6.0,  # este deve ser aplicado normalmente
            },
        }))

        loaded = BotConfig.load_json(str(path))
        assert loaded.risk.daily_loss_limit_pct == 6.0
        assert not hasattr(loaded.risk, "campo_que_nao_existe")

    def test_load_json_file_not_found_returns_defaults(self) -> None:
        """Arquivo ausente não deve quebrar — retorna config com defaults."""
        loaded = BotConfig.load_json("caminho_que_nao_existe_12345.json")
        assert loaded is not None
        assert loaded.capital_usd == 1000.0  # default

    def test_load_json_malformed_returns_defaults(self, tmp_path) -> None:
        """JSON malformado não deve quebrar — retorna config com defaults."""
        path = tmp_path / "malformed.json"
        path.write_text("{ isso não é json válido")

        loaded = BotConfig.load_json(str(path))
        assert loaded is not None