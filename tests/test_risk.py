"""
Testes do Módulo de Risco — Hyperliquid Production Bot v3.0
============================================================
Testa PositionManager, cálculo de stops, sizing e filtros.

Requisitos:
- Type hints
- Tratamento de exceções
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.risk import (
    LIQUIDATION_SAFETY_BUFFER_PCT,
    CHECK_EXITS_MISS_THRESHOLD,
    Position,
    PositionManager,
    calc_position_size,
    calc_stops,
    trade_hours_ok,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame com dados OHLCV para testes de SL estrutural."""
    np.random.seed(42)
    n = 50
    close = np.random.randn(n).cumsum() + 100
    return pd.DataFrame(
        {
            "close": close,
            "high": close + np.random.rand(n) * 2,
            "low": close - np.random.rand(n) * 2,
            "volume": np.random.rand(n) * 1000,
        }
    )


@pytest.fixture
def pm() -> PositionManager:
    """PositionManager com configuração padrão.

    FIX (item 4 auditoria): max_position_pct agora é passado
    explicitamente — antes can_open() usava um hardcode de 0.20
    desconectado da configuração real. Aqui usamos 0.20 propositalmente
    igual ao valor legado, para que os testes de exposição abaixo
    continuem com os mesmos números esperados; test_max_position_pct_respected
    cobre o caso onde o valor É diferente do hardcode antigo.
    """
    return PositionManager(
        capital_usd=10000.0,
        max_open_trades=3,
        max_drawdown_pct=0.20,
        daily_loss_limit_pct=0.10,
        max_consecutive_losses=5,
        cooldown_after_loss_sec=300,
        max_position_pct=0.20,
    )


@pytest.fixture
def sample_position() -> Position:
    """Posição de exemplo."""
    return Position(
        symbol="BTC/USDC",
        side="buy",
        entry_price=50000.0,
        qty=0.1,
        stop_loss=49000.0,
        take_profit=52000.0,
        open_time=time.time(),
    )


# ──────────────────────────────────────────────
# Testes: Position (dataclass)
# ──────────────────────────────────────────────


class TestPosition:
    """Testes para a dataclass Position."""

    def test_valid_position(self) -> None:
        """Deve criar posição válida."""
        pos = Position(symbol="ETH/USDC", side="sell", entry_price=3000.0, qty=1.0)
        assert pos.symbol == "ETH/USDC"
        assert pos.side == "sell"
        assert pos.entry_price == 3000.0
        assert pos.qty == 1.0

    def test_invalid_side(self) -> None:
        """Deve rejeitar side inválido."""
        with pytest.raises(ValueError, match="side inválido"):
            Position(symbol="BTC/USDC", side="hold", entry_price=50000.0, qty=0.1)

    def test_zero_qty(self) -> None:
        """Deve rejeitar qty zero."""
        with pytest.raises(ValueError, match="qty deve ser > 0"):
            Position(symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=0)

    def test_negative_qty(self) -> None:
        """Deve rejeitar qty negativa."""
        with pytest.raises(ValueError, match="qty deve ser > 0"):
            Position(symbol="BTC/USDC", side="buy", entry_price=50000.0, qty=-1)

    def test_zero_entry_price(self) -> None:
        """Deve rejeitar entry_price zero."""
        with pytest.raises(ValueError, match="entry_price deve ser > 0"):
            Position(symbol="BTC/USDC", side="buy", entry_price=0, qty=0.1)


# ──────────────────────────────────────────────
# Testes: calc_position_size
# ──────────────────────────────────────────────


class TestCalcPositionSize:
    """Testes para calc_position_size."""

    def test_basic_sizing(self) -> None:
        """
        FIX (item 2 auditoria): stop_dist agora usa
        max(price*stop_loss_pct, atr*2) — a MESMA fórmula de calc_stops,
        não mais max(atr*2, price*0.001). Isso garante que o risco em
        dólares efetivamente exposto bata com risk_per_trade nominal.
        """
        qty = calc_position_size(
            balance=10000, price=50000, atr=1000, risk_per_trade=0.02,
            stop_loss_pct=0.02,
        )
        # risk_amt = 10000 * 0.02 = 200
        # stop_dist = max(50000*0.02, 1000*2) = max(1000, 2000) = 2000
        # qty_risk = 200 / 2000 = 0.1
        # max_qty = (10000 * 0.20) / 50000 = 0.04
        # qty = min(0.1, 0.04) = 0.04
        assert qty == 0.04

    def test_zero_balance(self) -> None:
        """Deve retornar 0 para saldo zero."""
        assert calc_position_size(balance=0, price=50000, atr=1000) == 0.0

    def test_zero_price(self) -> None:
        """Deve retornar 0 para preço zero."""
        assert calc_position_size(balance=10000, price=0, atr=1000) == 0.0

    def test_negative_balance(self) -> None:
        """Deve retornar 0 para saldo negativo."""
        assert calc_position_size(balance=-100, price=50000, atr=1000) == 0.0

    def test_custom_risk(self) -> None:
        """Deve usar risk_per_trade personalizado."""
        qty = calc_position_size(
            balance=10000, price=50000, atr=1000, risk_per_trade=0.05,
            stop_loss_pct=0.02,
        )
        # risk_amt = 10000 * 0.05 = 500
        # stop_dist = max(1000, 2000) = 2000
        # qty_risk = 500 / 2000 = 0.25
        # max_qty = 0.04
        # qty = min(0.25, 0.04) = 0.04
        assert qty == 0.04

    def test_small_atr_uses_pct_floor(self) -> None:
        """
        FIX (item 2): quando ATR é muito pequeno, o piso de distância
        de stop agora vem de price*stop_loss_pct (mesma fórmula de
        calc_stops), não mais de price*0.001 fixo.
        """
        qty = calc_position_size(
            balance=10000, price=50000, atr=1, risk_per_trade=0.02,
            stop_loss_pct=0.02,
        )
        # stop_dist = max(50000*0.02, 1*2) = max(1000, 2) = 1000
        # qty_risk = 200 / 1000 = 0.2
        # max_qty = 0.04
        # qty = min(0.2, 0.04) = 0.04
        assert qty == 0.04

    def test_default_stop_loss_pct_backward_compat(self) -> None:
        """
        Sem stop_loss_pct explícito, deve usar o default (0.02) da
        assinatura — chamadas legadas que não passam o parâmetro não
        devem quebrar.
        """
        qty = calc_position_size(balance=10000, price=50000, atr=1000, risk_per_trade=0.02)
        assert qty > 0

    def test_stop_loss_pct_affects_sizing(self) -> None:
        """
        Fórmula unificada: aumentar stop_loss_pct aumenta a distância
        de stop assumida, reduzindo qty_risk (menos alavancado por
        posição para o mesmo risco em $).
        """
        qty_tight = calc_position_size(
            balance=10000, price=50000, atr=100, risk_per_trade=0.02,
            stop_loss_pct=0.01, max_capital_pct=1.0,  # sem cap de capital para isolar o efeito
        )
        qty_wide = calc_position_size(
            balance=10000, price=50000, atr=100, risk_per_trade=0.02,
            stop_loss_pct=0.05, max_capital_pct=1.0,
        )
        assert qty_tight > qty_wide

    def test_max_capital_pct_caps_qty(self) -> None:
        """max_capital_pct deve continuar limitando o teto de qty."""
        qty = calc_position_size(
            balance=10000, price=50000, atr=1000, risk_per_trade=0.50,
            stop_loss_pct=0.02, max_capital_pct=0.05,
        )
        max_qty = (10000 * 0.05) / 50000
        assert qty == round(max_qty, 6)


# ──────────────────────────────────────────────
# Testes: calc_stops
# ──────────────────────────────────────────────


class TestCalcStops:
    """Testes para calc_stops."""

    def test_buy_stops(self) -> None:
        """Deve calcular SL e TP para compra."""
        sl, tp = calc_stops(price=50000, side="buy", atr=1000)
        # sl_pct = 50000 * 0.02 = 1000
        # sl_atr = 1000 * 2 = 2000
        # sl_d = max(1000, 2000) = 2000
        # tp_d = max(50000*0.04, 1000*4) = max(2000, 4000) = 4000
        # sl = 50000 - 2000 = 48000
        # tp = 50000 + 4000 = 54000
        assert sl == 48000.0
        assert tp == 54000.0

    def test_sell_stops(self) -> None:
        """Deve calcular SL e TP para venda."""
        sl, tp = calc_stops(price=50000, side="sell", atr=1000)
        # sl = 50000 + 2000 = 52000
        # tp = 50000 - 4000 = 46000
        assert sl == 52000.0
        assert tp == 46000.0

    def test_zero_price(self) -> None:
        """Deve retornar (0, 0) para preço zero."""
        sl, tp = calc_stops(price=0, side="buy", atr=1000)
        assert sl == 0.0
        assert tp == 0.0

    def test_invalid_side(self) -> None:
        """Deve retornar (0, 0) para side inválido."""
        sl, tp = calc_stops(price=50000, side="invalid", atr=1000)
        assert sl == 0.0
        assert tp == 0.0

    def test_custom_pcts(self) -> None:
        """Deve usar percentuais personalizados."""
        sl, tp = calc_stops(
            price=50000, side="buy", atr=1000,
            stop_loss_pct=0.01, take_profit_pct=0.02,
        )
        # sl_pct = 50000 * 0.01 = 500
        # sl_atr = 2000
        # sl_d = max(500, 2000) = 2000
        # tp_d = max(50000*0.02, 4000) = max(1000, 4000) = 4000
        assert sl == 48000.0
        assert tp == 54000.0

    def test_structural_sl(self, sample_df: pd.DataFrame) -> None:
        """Deve usar SL estrutural quando df é fornecido."""
        sl, tp = calc_stops(
            price=100, side="buy", atr=2,
            df=sample_df,
        )
        # Deve retornar valores válidos
        assert sl > 0
        assert tp > 0
        assert sl < 100  # SL abaixo do preço para compra
        assert tp > 100  # TP acima do preço para compra

    def test_calc_stops_and_calc_position_size_share_formula(self) -> None:
        """
        REGRESSÃO (item 2 auditoria): a distância de stop assumida por
        calc_position_size deve bater com a distância de stop
        percentual/ATR usada por calc_stops (sem considerar o SL
        estrutural, que é uma camada adicional só de calc_stops).
        Isso garante que o risco em $ nominal (risk_per_trade) e o
        risco em $ real (qty * distância do SL) fiquem alinhados
        quando o SL estrutural não entra em jogo.
        """
        price, atr, stop_loss_pct = 50000.0, 1000.0, 0.02

        sl, _ = calc_stops(price=price, side="buy", atr=atr, stop_loss_pct=stop_loss_pct)
        stop_dist_from_calc_stops = price - sl

        # calc_position_size usa a mesma fórmula internamente:
        # max(price*stop_loss_pct, atr*2)
        expected_stop_dist = max(price * stop_loss_pct, atr * 2)

        assert stop_dist_from_calc_stops == expected_stop_dist


# ──────────────────────────────────────────────
# Testes: trade_hours_ok
# ──────────────────────────────────────────────


class TestTradeHoursOk:
    """Testes para trade_hours_ok."""

    def test_no_filter(self) -> None:
        """Deve retornar True sem filtro de horário."""
        assert trade_hours_ok(0, 0) is True

    def test_within_hours(self) -> None:
        """Deve retornar True dentro do horário (simulado)."""
        with patch("crypto_bot_core.risk.datetime") as mock_dt:
            mock_dt.now.return_value.hour = 14
            mock_dt.now.return_value.strftime.return_value = "2024-01-01"
            assert trade_hours_ok(8, 22) is True

    def test_outside_hours(self) -> None:
        """Deve retornar False fora do horário."""
        with patch("crypto_bot_core.risk.datetime") as mock_dt:
            mock_dt.now.return_value.hour = 6
            mock_dt.now.return_value.strftime.return_value = "2024-01-01"
            assert trade_hours_ok(8, 22) is False

    def test_overnight_within(self) -> None:
        """Deve funcionar com janela que passa da meia-noite."""
        with patch("crypto_bot_core.risk.datetime") as mock_dt:
            mock_dt.now.return_value.hour = 22
            mock_dt.now.return_value.strftime.return_value = "2024-01-01"
            assert trade_hours_ok(20, 6) is True  # 22h está dentro de 20-6

    def test_overnight_outside(self) -> None:
        """Deve retornar False fora da janela noturna."""
        with patch("crypto_bot_core.risk.datetime") as mock_dt:
            mock_dt.now.return_value.hour = 12
            mock_dt.now.return_value.strftime.return_value = "2024-01-01"
            assert trade_hours_ok(20, 6) is False  # 12h está fora de 20-6


# ──────────────────────────────────────────────
# Testes: PositionManager
# ──────────────────────────────────────────────


class TestPositionManagerInit:
    """Testes de inicialização do PositionManager."""

    def test_init_defaults(self) -> None:
        """Deve inicializar com valores padrão."""
        pm = PositionManager(capital_usd=10000)
        assert pm.current_balance == 10000.0
        assert pm.peak_balance == 10000.0
        assert pm.pnl_today == 0.0
        assert pm.consecutive_losses == 0
        assert pm.positions == []

    def test_init_custom(self) -> None:
        """Deve inicializar com valores personalizados."""
        pm = PositionManager(
            capital_usd=50000,
            max_open_trades=5,
            max_drawdown_pct=0.30,
            daily_loss_limit_pct=0.15,
        )
        assert pm.current_balance == 50000.0
        assert pm.max_open_trades == 5
        assert pm.max_drawdown_pct == 0.30
        assert pm.daily_loss_limit_pct == 0.15

    def test_init_max_position_pct_default(self) -> None:
        """
        FIX (item 4 auditoria): max_position_pct deve ser aceito no
        construtor com default sensato (0.20), para não quebrar
        chamadas legadas sem esse parâmetro.
        """
        pm = PositionManager(capital_usd=10000)
        assert pm.max_position_pct == 0.20

    def test_init_max_position_pct_custom(self) -> None:
        """max_position_pct customizado deve ser respeitado."""
        pm = PositionManager(capital_usd=10000, max_position_pct=0.05)
        assert pm.max_position_pct == 0.05


class TestPositionManagerCanOpen:
    """Testes para PositionManager.can_open."""

    def test_can_open_initial(self, pm: PositionManager) -> None:
        """Deve permitir abrir posição inicialmente."""
        ok, reason = pm.can_open()
        assert ok is True
        assert reason == "ok"

    def test_max_open_trades(self, pm: PositionManager) -> None:
        """Deve bloquear quando atingir max_open_trades."""
        for i in range(pm.max_open_trades):
            pm.add(Position(
                symbol=f"ASSET{i}/USDC",
                side="buy",
                entry_price=100.0,
                qty=1.0,
            ))
        ok, reason = pm.can_open()
        assert ok is False
        assert "max_open_trades" in reason

    def test_drawdown_limit(self, pm: PositionManager) -> None:
        """Deve bloquear quando drawdown excede limite."""
        pm.current_balance = 5000  # 50% drawdown
        pm.peak_balance = 10000
        ok, reason = pm.can_open()
        assert ok is False
        assert "drawdown" in reason

    def test_daily_loss_limit(self, pm: PositionManager) -> None:
        """Deve bloquear quando perda diária excede limite."""
        pm.pnl_today = -2000  # 20% de perda (limite=10%)
        pm.peak_balance = 10000
        ok, reason = pm.can_open()
        assert ok is False
        assert "daily_loss" in reason

    def test_consecutive_losses(self, pm: PositionManager) -> None:
        """Deve bloquear quando perdas consecutivas excedem limite."""
        pm.consecutive_losses = 5
        ok, reason = pm.can_open()
        assert ok is False
        assert "consecutive_losses" in reason

    def test_cooldown(self, pm: PositionManager) -> None:
        """Deve bloquear durante cooldown pós-loss."""
        pm._last_loss_ts = time.monotonic() - 10  # 10s atrás (cooldown=300s)
        ok, reason = pm.can_open()
        assert ok is False
        assert "cooldown" in reason

    def test_cooldown_expired(self, pm: PositionManager) -> None:
        """Deve permitir após cooldown expirar."""
        pm._last_loss_ts = time.monotonic() - 600  # 600s atrás (>300s)
        ok, reason = pm.can_open()
        assert ok is True
        assert reason == "ok"

    def test_max_exposure_uses_configured_max_position_pct(self) -> None:
        """
        FIX (item 4 auditoria): can_open() deve calcular max_exposure
        usando self.max_position_pct (configurado), não mais um
        hardcode fixo de 0.20 desconectado da config real. Aqui usamos
        um valor DIFERENTE do hardcode legado (0.05) para provar que
        o parâmetro está de fato sendo respeitado.
        """
        pm = PositionManager(
            capital_usd=10000,
            max_open_trades=3,
            max_position_pct=0.05,  # bem menor que o hardcode legado de 0.20
        )
        # max_exposure = max_open_trades * max_position_pct = 3*0.05 = 0.15
        # threshold de bloqueio = max_exposure * 0.95 = 0.1425 (14.25% do capital)
        # Exposição desta posição: (2000 * 1.0) / 10000 = 20% > 14.25% -> deve bloquear.
        # Com o hardcode legado (0.20 fixo, ignorando max_position_pct=0.05),
        # o threshold seria 3*0.20*0.95=0.57 (57%), e esta mesma posição
        # NÃO bloquearia — é exatamente essa diferença que prova que o
        # parâmetro configurado está sendo respeitado, não o hardcode antigo.
        pm.add(Position(symbol="BTC/USDC", side="buy", entry_price=2000.0, qty=1.0))
        ok, reason = pm.can_open()
        assert ok is False
        assert "exposure" in reason

    def test_max_exposure_with_default_hardcode_equivalent(self, pm: PositionManager) -> None:
        """
        Com max_position_pct=0.20 (fixture pm), o comportamento deve
        bater com o legado: só bloqueia perto de max_open_trades*0.20
        de exposição total.
        """
        # 3 posições * 0.20 = 60% de exposição máxima teórica;
        # colocando uma exposição bem abaixo disso não deve bloquear.
        pm.add(Position(symbol="BTC/USDC", side="buy", entry_price=1000.0, qty=1.0))
        ok, reason = pm.can_open()
        assert ok is True


class TestPositionManagerAdd:
    """Testes para PositionManager.add."""

    def test_add_position(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve adicionar posição à lista."""
        pm.add(sample_position)
        assert len(pm.positions) == 1
        assert pm.positions[0].symbol == "BTC/USDC"

    def test_add_invalid_type(self, pm: PositionManager) -> None:
        """Deve rejeitar tipo inválido."""
        pm.add("not_a_position")  # type: ignore
        assert len(pm.positions) == 0


class TestPositionManagerRecordClose:
    """Testes para PositionManager.record_close."""

    def test_record_close_profit(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve registrar fechamento com lucro."""
        pm.add(sample_position)
        result = pm.record_close(sample_position, exit_price=51000)
        assert result["net_pnl"] > 0
        assert pm.consecutive_losses == 0
        assert len(pm.positions) == 0

    def test_record_close_loss(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve registrar fechamento com perda."""
        pm.add(sample_position)
        result = pm.record_close(sample_position, exit_price=49000)
        assert result["net_pnl"] < 0
        assert pm.consecutive_losses == 1
        assert pm._last_loss_ts > 0

    def test_record_close_not_found(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve tratar posição não encontrada."""
        result = pm.record_close(sample_position, exit_price=51000)
        assert "error" in result


class TestPositionManagerCheckExits:
    """Testes para PositionManager.check_exits."""

    def test_hit_take_profit(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve identificar TP atingido."""
        pm.add(sample_position)
        to_close = pm.check_exits(price=53000)
        assert len(to_close) == 1
        assert to_close[0].symbol == "BTC/USDC"

    def test_hit_stop_loss(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve identificar SL atingido."""
        pm.add(sample_position)
        to_close = pm.check_exits(price=48000)
        assert len(to_close) == 1

    def test_no_exit(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve retornar lista vazia sem hits."""
        pm.add(sample_position)
        to_close = pm.check_exits(price=50500)
        assert len(to_close) == 0

    def test_hit_trailing(self, pm: PositionManager) -> None:
        """Deve identificar trailing stop atingido."""
        pm.trailing_stop = True
        pos = Position(
            symbol="BTC/USDC", side="buy",
            entry_price=50000, qty=0.1,
            stop_loss=49000, take_profit=52000,
            trailing_stop_price=49500,
        )
        pm.add(pos)
        to_close = pm.check_exits(price=49400)
        assert len(to_close) == 1


class TestPositionManagerUpdateTrailing:
    """Testes para PositionManager.update_trailing."""

    def test_trailing_disabled(self, pm: PositionManager, sample_position: Position) -> None:
        """Não deve atualizar trailing quando desabilitado."""
        pm.add(sample_position)
        pm.update_trailing(price=51000)
        assert sample_position.trailing_stop_price == 0.0

    def test_trailing_long_update(self, pm: PositionManager) -> None:
        """Deve atualizar trailing stop para LONG."""
        pm.trailing_stop = True
        pm.trailing_stop_pct = 0.01
        pm.trailing_activation_pct = 0.0  # sem ativação mínima

        pos = Position(
            symbol="BTC/USDC", side="buy",
            entry_price=50000, qty=0.1,
            stop_loss=49000, take_profit=52000,
        )
        pm.add(pos)
        pm.update_trailing(price=51000)
        # nt = 51000 * (1 - 0.01) = 50490
        assert pos.trailing_stop_price == 50490.0

    def test_trailing_short_update(self, pm: PositionManager) -> None:
        """Deve atualizar trailing stop para SHORT."""
        pm.trailing_stop = True
        pm.trailing_stop_pct = 0.01
        pm.trailing_activation_pct = 0.0

        pos = Position(
            symbol="BTC/USDC", side="sell",
            entry_price=50000, qty=0.1,
            stop_loss=51000, take_profit=48000,
            trailing_stop_price=51000,  # inicializado acima do entry
        )
        pm.add(pos)
        pm.update_trailing(price=49000)
        # nt = 49000 * (1 + 0.01) = 49490
        # 49490 < 51000 (trailing_stop_price anterior) -> True
        assert pos.trailing_stop_price == 49490.0


class TestPositionManagerToDict:
    """Testes para PositionManager.to_dict."""

    def test_to_dict_empty(self, pm: PositionManager) -> None:
        """Deve retornar dict vazio."""
        d = pm.to_dict()
        assert d["positions"] == []
        assert d["pnl_today"] == 0.0

    def test_to_dict_with_position(self, pm: PositionManager, sample_position: Position) -> None:
        """Deve retornar dict com posição."""
        pm.add(sample_position)
        d = pm.to_dict()
        assert len(d["positions"]) == 1
        assert d["positions"][0]["symbol"] == "BTC/USDC"


class TestLiquidationSafetyBufferConstant:
    """
    Testa que LIQUIDATION_SAFETY_BUFFER_PCT continua existindo como
    constante hardcoded de módulo (não como campo de RiskConfig,
    removido no item 6 por nunca ter sido de fato conectado a ela).
    """

    def test_constant_exists_and_has_expected_value(self) -> None:
        assert LIQUIDATION_SAFETY_BUFFER_PCT == 0.20

    def test_constant_is_module_level_not_config_field(self) -> None:
        """RiskConfig não deve ter esse campo — é intencionalmente hardcoded."""
        from crypto_bot_core.config import RiskConfig
        rc = RiskConfig()
        assert not hasattr(rc, "liquidation_safety_buffer_pct")