"""
Teste de regressão para a escala de max_funding_rate — auditoria item 2.

Confirma que RiskConfig.max_funding_rate (em pontos percentuais, ex:
0.1 = 0.1%) é corretamente convertido para fração (0.001) via a
property max_funding_rate_fraction, e que essa é a mesma fração que
CapitalProtection.max_funding_rate recebe no fluxo real de main.py.
"""
from __future__ import annotations

import pytest

from crypto_bot_core.config import BotConfig, RiskConfig
from crypto_bot_core.capital_protection import CapitalProtection


class TestMaxFundingRateFractionProperty:
    """Testes para RiskConfig.max_funding_rate_fraction."""

    def test_default_value_conversion(self) -> None:
        rc = RiskConfig()
        assert rc.max_funding_rate == 0.1
        assert rc.max_funding_rate_fraction == pytest.approx(0.001)

    def test_custom_value_conversion(self) -> None:
        rc = RiskConfig(max_funding_rate=0.5)
        assert rc.max_funding_rate_fraction == pytest.approx(0.005)

    def test_min_boundary(self) -> None:
        rc = RiskConfig(max_funding_rate=0.001)
        assert rc.max_funding_rate_fraction == pytest.approx(0.00001)

    def test_max_boundary(self) -> None:
        rc = RiskConfig(max_funding_rate=1.0)
        assert rc.max_funding_rate_fraction == pytest.approx(0.01)


class TestFundingRateScaleEndToEnd:
    """
    Confirma que o valor que chega em CapitalProtection.check_funding_rate
    usa a mesma unidade (fração) que o funding_rate bruto retornado pela
    exchange — evitando bloqueios/permissões incorretas por descasamento
    de escala.
    """

    def test_capital_protection_receives_fraction_not_percent(self) -> None:
        cfg = BotConfig(_env_file=None)
        cp = CapitalProtection(
            initial_balance=cfg.capital_usd,
            max_funding_rate=cfg.risk.max_funding_rate_fraction,
        )
        # Default RiskConfig.max_funding_rate=0.1 (0.1%) -> fração 0.001
        assert cp.max_funding_rate == pytest.approx(0.001)

        # Funding rate típico de exchange já vem como fração pequena
        # (ex: 0.0001 = 0.01% por período) — bem abaixo do limite.
        ok, _ = cp.check_funding_rate(0.0001)
        assert ok is True

        # Funding extremo (0.5% = 0.005 em fração) deve estourar o limite
        # de 0.1% (0.001) e ser bloqueado.
        ok, reason = cp.check_funding_rate(0.005)
        assert ok is False
        assert "funding_alto" in reason

    def test_strategy_default_funding_rate_uses_different_but_intentional_scale(self) -> None:
        """
        Documenta explicitamente que STRATEGY_DEFAULTS usa fração DIRETA
        (não pontos percentuais/100) — não deve ser confundido com
        RiskConfig.max_funding_rate. Este teste existe para travar essa
        diferença como intencional, não como bug, caso alguém tente
        "unificar" os dois no futuro sem entender o propósito de cada um.
        """
        cfg = BotConfig(_env_file=None)
        strategy_max_funding = cfg.STRATEGY_DEFAULTS["funding_arbitrage"]["1h"]["max_funding_rate"]
        # 0.01 = 1% em fração direta — NÃO passa por /100.
        assert strategy_max_funding == 0.01
        # Não deve ser igual à fração de RiskConfig por coincidência de
        # unidades — são propositalmente escalas diferentes.
        assert strategy_max_funding != cfg.risk.max_funding_rate_fraction