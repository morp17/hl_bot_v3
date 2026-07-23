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

    def test_kelly_fraction(self) -> None:
        """Testa fração de Kelly."""
        rc = RiskConfig(kelly_fraction=0.0)
        assert rc.kelly_fraction == 0.0
        rc = RiskConfig(kelly_fraction=0.5)
        assert rc.kelly_fraction == 0.5
        with pytest.raises(ValueError):
            RiskConfig(kelly_fraction=1.5)


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
