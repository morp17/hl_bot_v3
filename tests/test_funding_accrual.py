"""
Testes para acúmulo de funding em posições abertas — auditoria item 6.

Cobre:
- Position.accrued_funding / last_funding_ts (inicialização via __post_init__)
- PositionManager.accrue_funding() — cálculo de custo/receita por lado
  (long paga quando funding>0, short recebe, e vice-versa)
- Integração com record_close() — dedução do funding acumulado do PnL líquido
"""
from __future__ import annotations

import time

import pytest

from crypto_bot_core.risk import Position, PositionManager


# ──────────────────────────────────────────────
# Position — inicialização de campos de funding
# ──────────────────────────────────────────────


class TestPositionFundingFields:
    """Testes para os campos accrued_funding/last_funding_ts em Position."""

    def test_default_accrued_funding_is_zero(self) -> None:
        pos = Position(symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1)
        assert pos.accrued_funding == 0.0

    def test_last_funding_ts_defaults_to_open_time(self) -> None:
        open_time = time.time() - 100
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1,
            open_time=open_time,
        )
        assert pos.last_funding_ts == open_time

    def test_last_funding_ts_defaults_to_now_without_open_time(self) -> None:
        before = time.time()
        pos = Position(symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1)
        after = time.time()
        assert before <= pos.last_funding_ts <= after

    def test_explicit_last_funding_ts_preserved(self) -> None:
        explicit_ts = time.time() - 500
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1,
            last_funding_ts=explicit_ts,
        )
        assert pos.last_funding_ts == explicit_ts


# ──────────────────────────────────────────────
# PositionManager.accrue_funding
# ──────────────────────────────────────────────


@pytest.fixture
def pm() -> PositionManager:
    return PositionManager(capital_usd=10000.0)


class TestAccrueFundingDirectionality:
    """Testes de sinal — long paga com funding positivo, short recebe (e vice-versa)."""

    def test_long_pays_when_funding_positive(self, pm: PositionManager) -> None:
        """funding_rate > 0 -> LONG paga -> accrued_funding fica POSITIVO
        (custo, reduz o PnL líquido no fechamento)."""
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600,  # 1h atrás
            last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert pos.accrued_funding > 0.0

    def test_short_receives_when_funding_positive(self, pm: PositionManager) -> None:
        """funding_rate > 0 -> SHORT recebe -> accrued_funding fica
        NEGATIVO (receita, aumenta o PnL líquido no fechamento)."""
        pos = Position(
            symbol="BTC/USDC", side="sell", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600,
            last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert pos.accrued_funding < 0.0

    def test_long_receives_when_funding_negative(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600,
            last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=-0.0001, mark_price=50000.0)

        assert pos.accrued_funding < 0.0

    def test_short_pays_when_funding_negative(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="sell", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600,
            last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=-0.0001, mark_price=50000.0)

        assert pos.accrued_funding > 0.0


class TestAccrueFundingMagnitude:
    """Testes de magnitude — proporcionalidade a rate, notional e tempo decorrido."""

    def test_magnitude_proportional_to_elapsed_time(self, pm: PositionManager) -> None:
        """1h de acúmulo deve gerar exatamente o dobro de 2 chamadas de 30min
        (mesma rate/notional) — teste indireto via duas posições distintas
        com last_funding_ts diferentes."""
        pos_1h = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pos_30min = Position(
            symbol="ETH/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 1800, last_funding_ts=time.time() - 1800,
        )
        pm.add(pos_1h)
        pm.add(pos_30min)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)
        pm.accrue_funding(symbol="ETH/USDC", funding_rate=0.0001, mark_price=50000.0)

        # 1h de acúmulo ~ 2x o de 30min (mesma rate/notional)
        assert pos_1h.accrued_funding == pytest.approx(pos_30min.accrued_funding * 2, rel=0.05)

    def test_magnitude_proportional_to_notional(self, pm: PositionManager) -> None:
        """Qty maior -> notional maior -> funding acumulado proporcionalmente maior."""
        pos_small = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pos_large = Position(
            symbol="ETH/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos_small)
        pm.add(pos_large)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)
        pm.accrue_funding(symbol="ETH/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert pos_large.accrued_funding == pytest.approx(pos_small.accrued_funding * 10, rel=1e-6)

    def test_exact_formula_matches_expected_value(self, pm: PositionManager) -> None:
        """
        Valida a fórmula exata documentada:
        cost = funding_rate * notional * elapsed_hours * direction
        """
        elapsed_hours = 2.0
        open_time = time.time() - (elapsed_hours * 3600)
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.5,
            open_time=open_time, last_funding_ts=open_time,
        )
        pm.add(pos)

        funding_rate = 0.0002
        mark_price = 51000.0
        pm.accrue_funding(symbol="BTC/USDC", funding_rate=funding_rate, mark_price=mark_price)

        notional = 0.5 * mark_price
        expected = funding_rate * notional * elapsed_hours * 1.0  # direction=+1 para buy
        assert pos.accrued_funding == pytest.approx(expected, rel=0.02)


class TestAccrueFundingAccumulation:
    """Testes de acúmulo em múltiplas chamadas — deve somar, não sobrescrever."""

    def test_multiple_calls_accumulate(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)
        first_value = pos.accrued_funding

        # Simula passagem de mais tempo antes da próxima chamada
        pos.last_funding_ts -= 1800  # mais 30min "decorridos"
        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert pos.accrued_funding > first_value

    def test_updates_last_funding_ts_after_call(self, pm: PositionManager) -> None:
        old_ts = time.time() - 3600
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=old_ts, last_funding_ts=old_ts,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert pos.last_funding_ts > old_ts


class TestAccrueFundingEdgeCases:
    """Casos de borda — não deve quebrar nem acumular indevidamente."""

    def test_zero_funding_rate_no_op(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0, mark_price=50000.0)

        assert pos.accrued_funding == 0.0

    def test_zero_mark_price_no_op(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=0.0)

        assert pos.accrued_funding == 0.0

    def test_no_matching_symbol_no_op(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)

        pm.accrue_funding(symbol="ETH/USDC", funding_rate=0.0001, mark_price=3000.0)

        assert pos.accrued_funding == 0.0

    def test_empty_positions_does_not_raise(self, pm: PositionManager) -> None:
        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)  # não deve lançar

    def test_immediate_call_negligible_accrual(self, pm: PositionManager) -> None:
        """Chamada logo após abertura (elapsed_hours ~0) deve gerar
        acúmulo desprezível (não zero exato, mas muito pequeno)."""
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time(), last_funding_ts=time.time(),
        )
        pm.add(pos)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert abs(pos.accrued_funding) < 0.01

    def test_only_matching_symbol_affected_with_multiple_positions(self, pm: PositionManager) -> None:
        pos_btc = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pos_eth = Position(
            symbol="ETH/USDC", side="buy", entry_price=3000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos_btc)
        pm.add(pos_eth)

        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0001, mark_price=50000.0)

        assert pos_btc.accrued_funding != 0.0
        assert pos_eth.accrued_funding == 0.0


# ──────────────────────────────────────────────
# Integração: accrue_funding + record_close
# ──────────────────────────────────────────────


class TestFundingIntegrationWithRecordClose:
    """Testes confirmando que o funding acumulado é de fato deduzido do
    PnL líquido no fechamento (record_close)."""

    def test_positive_funding_cost_reduces_net_pnl(self, pm: PositionManager) -> None:
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)
        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0005, mark_price=50000.0)
        funding_cost_before_close = pos.accrued_funding
        assert funding_cost_before_close > 0.0

        result = pm.record_close(pos, exit_price=51000.0)

        # gross = (51000-50000)*1.0 = 1000; fee = (50000+51000)*1.0*0.0005 = 50.5
        expected_gross = 1000.0
        expected_fee = (50000.0 + 51000.0) * 1.0 * pm.taker_fee
        expected_net = expected_gross - expected_fee - funding_cost_before_close

        assert result["funding_cost"] == pytest.approx(funding_cost_before_close)
        assert result["net_pnl"] == pytest.approx(expected_net, rel=1e-6)

    def test_negative_funding_cost_increases_net_pnl(self, pm: PositionManager) -> None:
        """Posição que RECEBE funding (custo negativo) deve ter PnL líquido
        maior que o PnL bruto menos taxas."""
        pos = Position(
            symbol="BTC/USDC", side="sell", entry_price=50000.0, qty=1.0,
            open_time=time.time() - 3600, last_funding_ts=time.time() - 3600,
        )
        pm.add(pos)
        pm.accrue_funding(symbol="BTC/USDC", funding_rate=0.0005, mark_price=50000.0)
        funding_cost_before_close = pos.accrued_funding
        assert funding_cost_before_close < 0.0  # short recebe -> custo negativo

        result = pm.record_close(pos, exit_price=49000.0)

        expected_gross = (49000.0 - 50000.0) * 1.0 * -1  # short: (entry-exit) proxy
        expected_fee = (50000.0 + 49000.0) * 1.0 * pm.taker_fee
        expected_net = result["gross_pnl"] - expected_fee - funding_cost_before_close

        assert result["net_pnl"] == pytest.approx(expected_net, rel=1e-6)
        # net_pnl deve ser MAIOR do que seria sem considerar o funding (que é receita aqui)
        net_without_funding = result["gross_pnl"] - expected_fee
        assert result["net_pnl"] > net_without_funding

    def test_zero_accrued_funding_matches_legacy_behavior(self, pm: PositionManager) -> None:
        """Sem chamar accrue_funding, o comportamento deve ser idêntico ao
        legado (funding_cost=0, net_pnl = gross - fee)."""
        pos = Position(symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1)
        pm.add(pos)

        result = pm.record_close(pos, exit_price=51000.0)

        assert result["funding_cost"] == 0.0
        expected_gross = (51000.0 - 50000.0) * 0.1
        expected_fee = (50000.0 + 51000.0) * 0.1 * pm.taker_fee
        assert result["net_pnl"] == pytest.approx(expected_gross - expected_fee, rel=1e-6)

    def test_result_dict_includes_open_time(self, pm: PositionManager) -> None:
        """FIX (item 12 integração): record_close deve incluir open_time no
        resultado, usado pelo TradeLedger para calcular hold_seconds."""
        open_time = time.time() - 7200
        pos = Position(
            symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0.1,
            open_time=open_time,
        )
        pm.add(pos)

        result = pm.record_close(pos, exit_price=51000.0)

        assert result["open_time"] == open_time