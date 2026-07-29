"""
Testes para a camada de segurança contra liquidação — auditoria item 1.

Cobre:
- calc_liquidation_price_estimate() — estimativa pré-trade do preço de
  liquidação para LONG e SHORT, variando leverage e margem de manutenção.
- validate_stop_loss_safety() — gate que bloqueia ordens cujo SL esteja
  perigosamente próximo (ou do lado errado) da liquidação estimada.
"""
from __future__ import annotations

import pytest

from crypto_bot_core.risk import (
    LIQUIDATION_SAFETY_BUFFER_PCT,
    DEFAULT_MAINTENANCE_MARGIN_PCT,
    calc_liquidation_price_estimate,
    validate_stop_loss_safety,
)


# ──────────────────────────────────────────────
# calc_liquidation_price_estimate
# ──────────────────────────────────────────────


class TestCalcLiquidationPriceEstimateLong:
    """Testes para LONG (side='buy')."""

    def test_basic_long_10x(self) -> None:
        """
        LONG 10x: liq = entry * (1 - 1/10 + 0.03) = entry * 0.93
        """
        liq = calc_liquidation_price_estimate(
            entry_price=50000.0, side="buy", leverage=10,
        )
        expected = 50000.0 * (1 - 0.1 + DEFAULT_MAINTENANCE_MARGIN_PCT)
        assert liq == pytest.approx(expected, rel=1e-9)
        assert liq < 50000.0  # liquidação de LONG sempre abaixo da entrada

    def test_higher_leverage_moves_liquidation_closer(self) -> None:
        """Leverage maior -> liquidação mais próxima do preço de entrada (LONG)."""
        liq_low_lev = calc_liquidation_price_estimate(entry_price=50000.0, side="buy", leverage=3)
        liq_high_lev = calc_liquidation_price_estimate(entry_price=50000.0, side="buy", leverage=20)

        dist_low = 50000.0 - liq_low_lev
        dist_high = 50000.0 - liq_high_lev
        assert dist_high < dist_low

    def test_leverage_1x_far_from_liquidation(self) -> None:
        """Com leverage 1x, a liquidação teórica fica bem abaixo (quase sem risco de liq.)."""
        liq = calc_liquidation_price_estimate(entry_price=50000.0, side="buy", leverage=1)
        # 1 - 1/1 + 0.03 = 0.03 -> liq = 1500
        assert liq == pytest.approx(50000.0 * 0.03, rel=1e-9)

    def test_custom_maintenance_margin(self) -> None:
        liq = calc_liquidation_price_estimate(
            entry_price=1000.0, side="buy", leverage=5, maintenance_margin_pct=0.01,
        )
        expected = 1000.0 * (1 - 0.2 + 0.01)
        assert liq == pytest.approx(expected, rel=1e-9)


class TestCalcLiquidationPriceEstimateShort:
    """Testes para SHORT (side='sell')."""

    def test_basic_short_10x(self) -> None:
        """
        SHORT 10x: liq = entry * (1 + 1/10 - 0.03) = entry * 1.07
        """
        liq = calc_liquidation_price_estimate(
            entry_price=50000.0, side="sell", leverage=10,
        )
        expected = 50000.0 * (1 + 0.1 - DEFAULT_MAINTENANCE_MARGIN_PCT)
        assert liq == pytest.approx(expected, rel=1e-9)
        assert liq > 50000.0  # liquidação de SHORT sempre acima da entrada

    def test_higher_leverage_moves_liquidation_closer_short(self) -> None:
        liq_low_lev = calc_liquidation_price_estimate(entry_price=50000.0, side="sell", leverage=3)
        liq_high_lev = calc_liquidation_price_estimate(entry_price=50000.0, side="sell", leverage=20)

        dist_low = liq_low_lev - 50000.0
        dist_high = liq_high_lev - 50000.0
        assert dist_high < dist_low


class TestCalcLiquidationPriceEstimateInvalidInputs:
    """Casos de borda / inputs inválidos — devem retornar 0.0 sem lançar exceção."""

    def test_zero_entry_price(self) -> None:
        assert calc_liquidation_price_estimate(entry_price=0.0, side="buy", leverage=10) == 0.0

    def test_negative_entry_price(self) -> None:
        assert calc_liquidation_price_estimate(entry_price=-100.0, side="buy", leverage=10) == 0.0

    def test_zero_leverage(self) -> None:
        assert calc_liquidation_price_estimate(entry_price=50000.0, side="buy", leverage=0) == 0.0

    def test_negative_leverage(self) -> None:
        assert calc_liquidation_price_estimate(entry_price=50000.0, side="buy", leverage=-5) == 0.0

    def test_invalid_side(self) -> None:
        assert calc_liquidation_price_estimate(entry_price=50000.0, side="hold", leverage=10) == 0.0

    def test_result_never_negative(self) -> None:
        """Com leverage muito alto (ex: 50x) e margem de manutenção baixa,
        a fórmula não deve produzir preço de liquidação negativo."""
        liq = calc_liquidation_price_estimate(
            entry_price=100.0, side="buy", leverage=50, maintenance_margin_pct=0.0,
        )
        assert liq >= 0.0


# ──────────────────────────────────────────────
# validate_stop_loss_safety
# ──────────────────────────────────────────────


class TestValidateStopLossSafetyLong:
    """Testes para validação de SL em posições LONG."""

    def test_sl_with_sufficient_buffer_is_safe(self) -> None:
        """SL bem acima da liquidação (folga >> buffer mínimo) deve ser seguro."""
        liq_price = 45000.0
        sl = 48000.0  # (48000-45000)/45000 = 6.67% > 20%? não, mas testamos abaixo com folga maior
        # Ajusta para garantir folga > LIQUIDATION_SAFETY_BUFFER_PCT (20%)
        sl_safe_value = liq_price * 1.30  # 30% de folga
        ok, reason = validate_stop_loss_safety(
            stop_loss=sl_safe_value, liquidation_price=liq_price, side="buy",
        )
        assert ok is True
        assert reason == "ok"

    def test_sl_too_close_to_liquidation_is_unsafe(self) -> None:
        """SL com folga menor que o buffer mínimo deve ser rejeitado."""
        liq_price = 45000.0
        sl_close = liq_price * 1.05  # apenas 5% de folga, buffer exige 20%
        ok, reason = validate_stop_loss_safety(
            stop_loss=sl_close, liquidation_price=liq_price, side="buy",
        )
        assert ok is False
        assert "sl_proximo_liquidacao" in reason

    def test_sl_below_liquidation_is_unsafe(self) -> None:
        """SL abaixo (ou igual) da liquidação para LONG é definitivamente inseguro."""
        liq_price = 45000.0
        ok, reason = validate_stop_loss_safety(
            stop_loss=44000.0, liquidation_price=liq_price, side="buy",
        )
        assert ok is False
        assert "abaixo_ou_igual_liquidacao" in reason

    def test_sl_equal_to_liquidation_is_unsafe(self) -> None:
        liq_price = 45000.0
        ok, reason = validate_stop_loss_safety(
            stop_loss=45000.0, liquidation_price=liq_price, side="buy",
        )
        assert ok is False

    def test_custom_buffer_pct(self) -> None:
        """Buffer customizado (menor) deve aceitar uma folga que o default rejeitaria."""
        liq_price = 45000.0
        sl_close = liq_price * 1.05  # 5% de folga

        ok_default, _ = validate_stop_loss_safety(
            stop_loss=sl_close, liquidation_price=liq_price, side="buy",
        )
        ok_custom, _ = validate_stop_loss_safety(
            stop_loss=sl_close, liquidation_price=liq_price, side="buy", buffer_pct=0.03,
        )
        assert ok_default is False
        assert ok_custom is True


class TestValidateStopLossSafetyShort:
    """Testes para validação de SL em posições SHORT."""

    def test_sl_with_sufficient_buffer_is_safe(self) -> None:
        liq_price = 55000.0
        sl_safe_value = liq_price * 0.70  # SL bem abaixo da liquidação (30% de folga)
        ok, reason = validate_stop_loss_safety(
            stop_loss=sl_safe_value, liquidation_price=liq_price, side="sell",
        )
        assert ok is True
        assert reason == "ok"

    def test_sl_too_close_to_liquidation_is_unsafe(self) -> None:
        liq_price = 55000.0
        sl_close = liq_price * 0.95  # apenas 5% de folga
        ok, reason = validate_stop_loss_safety(
            stop_loss=sl_close, liquidation_price=liq_price, side="sell",
        )
        assert ok is False
        assert "sl_proximo_liquidacao" in reason

    def test_sl_above_liquidation_is_unsafe(self) -> None:
        """SL acima (ou igual) da liquidação para SHORT é definitivamente inseguro."""
        liq_price = 55000.0
        ok, reason = validate_stop_loss_safety(
            stop_loss=56000.0, liquidation_price=liq_price, side="sell",
        )
        assert ok is False
        assert "acima_ou_igual_liquidacao" in reason

    def test_sl_equal_to_liquidation_is_unsafe(self) -> None:
        liq_price = 55000.0
        ok, reason = validate_stop_loss_safety(
            stop_loss=55000.0, liquidation_price=liq_price, side="sell",
        )
        assert ok is False


class TestValidateStopLossSafetyEdgeCases:
    """Casos de borda — fail-open quando dados são insuficientes."""

    def test_zero_liquidation_price_fails_open(self) -> None:
        """Quando a liquidação não pôde ser calculada (0.0), o gate não deve
        bloquear a ordem — fail-open, mas com motivo explícito rastreável."""
        ok, reason = validate_stop_loss_safety(
            stop_loss=48000.0, liquidation_price=0.0, side="buy",
        )
        assert ok is True
        assert reason == "liquidacao_nao_calculavel"

    def test_zero_stop_loss_fails_open(self) -> None:
        ok, reason = validate_stop_loss_safety(
            stop_loss=0.0, liquidation_price=45000.0, side="buy",
        )
        assert ok is True
        assert reason == "liquidacao_nao_calculavel"

    def test_negative_liquidation_price_fails_open(self) -> None:
        ok, reason = validate_stop_loss_safety(
            stop_loss=48000.0, liquidation_price=-100.0, side="buy",
        )
        assert ok is True


class TestLiquidationSafetyIntegration:
    """Testes de integração entre as duas funções — fluxo real usado em main.py."""

    def test_high_leverage_with_tight_sl_is_rejected(self) -> None:
        """
        Cenário de risco real: leverage alto (20x) combinado com um SL
        percentual apertado (2%) deve ser detectado como inseguro —
        reproduz o problema original que motivou a correção (item 1 da
        auditoria): SL nominal "razoável" pode estar mais perto da
        liquidação do que o esperado quando a leverage é alta.
        """
        entry_price = 50000.0
        leverage = 20
        side = "buy"

        liq_price = calc_liquidation_price_estimate(
            entry_price=entry_price, side=side, leverage=leverage,
        )
        # SL percentual de 2% abaixo da entrada (comum em trend_follow default)
        sl = entry_price * 0.98

        ok, reason = validate_stop_loss_safety(
            stop_loss=sl, liquidation_price=liq_price, side=side,
        )

        # Com 20x, liq ≈ entry*(1 - 0.05 + 0.03) = entry*0.98 -> SL colide
        # com (ou fica muito próximo de) a própria liquidação.
        assert ok is False

    def test_low_leverage_with_same_sl_is_safe(self) -> None:
        """O mesmo SL percentual de 2%, mas com leverage baixa (3x), deve
        ter folga suficiente e ser aprovado."""
        entry_price = 50000.0
        leverage = 3
        side = "buy"

        liq_price = calc_liquidation_price_estimate(
            entry_price=entry_price, side=side, leverage=leverage,
        )
        sl = entry_price * 0.98

        ok, reason = validate_stop_loss_safety(
            stop_loss=sl, liquidation_price=liq_price, side=side,
        )

        assert ok is True
        assert reason == "ok"

    def test_symmetry_between_buy_and_sell_distance(self) -> None:
        """A distância percentual entre entrada e liquidação deve ser
        simetricamente equivalente para buy e sell com os mesmos
        parâmetros (mesma leverage/margem de manutenção)."""
        entry_price = 1000.0
        leverage = 10

        liq_buy = calc_liquidation_price_estimate(entry_price, "buy", leverage)
        liq_sell = calc_liquidation_price_estimate(entry_price, "sell", leverage)

        dist_buy_pct = (entry_price - liq_buy) / entry_price
        dist_sell_pct = (liq_sell - entry_price) / entry_price

        assert dist_buy_pct == pytest.approx(dist_sell_pct, rel=1e-9)