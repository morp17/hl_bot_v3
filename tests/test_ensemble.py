"""
Testes para o modo Ensemble de estratégias — get_ensemble_signal()
(crypto_bot_core/strategies/signals.py).

Usa estratégias STUB (monkeypatch em STRATEGY_MAP) em vez de depender
do cálculo real de indicadores — isola a lógica de agregação/confluência
do ensemble, que é o que esta função realmente precisa garantir.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from crypto_bot_core.strategies import signals as signals_mod
from crypto_bot_core.strategies.signals import get_ensemble_signal


def _make_stub(signal: str, confidence: float, reason: str = "stub"):
    """Cria uma função de estratégia stub que ignora df/cfg e retorna sinal fixo."""
    def _stub(df: pd.DataFrame, cfg: Any) -> Any:
        return signal, {"confidence": confidence, "reason": reason}
    return _stub


def _make_raising_stub(exc: Exception):
    """Cria uma função de estratégia stub que sempre lança uma exceção."""
    def _stub(df: pd.DataFrame, cfg: Any) -> Any:
        raise exc
    return _stub


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """DataFrame mínimo — as estratégias são stubadas e não leem colunas reais."""
    return pd.DataFrame({"close": [100.0, 101.0, 102.0]})


@pytest.fixture
def mock_cfg() -> MagicMock:
    """Config mínima necessária para get_ensemble_signal."""
    cfg = MagicMock()
    cfg.enabled_strategies = "trend_follow,adaptive_trend,hybrid_regime"
    cfg.ensemble_min_confluence = 2
    cfg.ensemble_min_avg_confidence = 0.5
    return cfg


class TestEnsembleConfluence:
    """Testes de confluência básica (buy/sell)."""

    def test_buy_confluence_reached(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        """2 votos de buy (>= min_confluence=2) com confiança média acima
        do mínimo devem resultar em sinal BUY."""
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.6))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("buy", 0.7))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("hold", 0.0))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "buy"
        assert params["buy_count"] == 2
        assert params["sell_count"] == 0
        assert params["confidence"] == pytest.approx((0.6 + 0.7) / 2, abs=1e-6)
        assert params["reason"] == "ensemble_buy_confluencia"

    def test_sell_confluence_reached(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("sell", 0.8))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("sell", 0.6))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("hold", 0.0))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "sell"
        assert params["sell_count"] == 2
        assert params["confidence"] == pytest.approx((0.8 + 0.6) / 2, abs=1e-6)

    def test_insufficient_confluence_returns_hold(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        """Apenas 1 voto de buy — abaixo de ensemble_min_confluence=2 — deve retornar hold."""
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.9))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("hold", 0.0))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("hold", 0.0))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "hold"
        assert params["reason"] == "sem_confluencia_minima"

    def test_low_average_confidence_returns_hold(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        """2 votos de buy (confluência OK), mas confiança média abaixo de
        ensemble_min_avg_confidence=0.5 — deve retornar hold."""
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.3))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("buy", 0.2))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("hold", 0.0))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "hold"

    def test_conflicting_directions_higher_score_wins(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        """Quando buy E sell atingem o mínimo simultaneamente, vence a
        direção com maior confiança média."""
        mock_cfg.enabled_strategies = "trend_follow,adaptive_trend,hybrid_regime,orderflow_delta"

        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.55))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("buy", 0.55))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("sell", 0.9))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "orderflow_delta", _make_stub("sell", 0.9))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "sell"
        assert params["confidence"] == pytest.approx(0.9, abs=1e-6)


class TestEnsembleEdgeCases:
    """Testes de casos de borda e tratamento de erros."""

    def test_no_enabled_strategies(self, dummy_df: pd.DataFrame, mock_cfg: MagicMock) -> None:
        mock_cfg.enabled_strategies = ""
        signal, params = get_ensemble_signal(dummy_df, mock_cfg)
        assert signal == "hold"
        assert params["reason"] == "nenhuma_estrategia_habilitada"

    def test_unknown_strategy_name_skipped(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        """Nome de estratégia desconhecido na lista não deve derrubar o
        ensemble — apenas é ignorado (com warning)."""
        mock_cfg.enabled_strategies = "trend_follow,estrategia_inexistente"
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.9))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert isinstance(signal, str)
        # Apenas 1 voto válido (trend_follow) — abaixo do min_confluence=2 -> hold
        assert signal == "hold"

    def test_strategy_exception_is_skipped_not_propagated(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        """Uma estratégia que lança exceção não deve derrubar o ensemble
        inteiro — as demais continuam sendo avaliadas normalmente."""
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_raising_stub(ValueError("boom")))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("buy", 0.9))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("buy", 0.9))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "buy"
        assert params["buy_count"] == 2

    def test_votes_list_contains_all_evaluated_strategies(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.6, reason="r1"))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("hold", 0.0, reason="r2"))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("sell", 0.4, reason="r3"))

        _, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert len(params["votes"]) == 3
        strategies_voted = {v["strategy"] for v in params["votes"]}
        assert strategies_voted == {"trend_follow", "adaptive_trend", "hybrid_regime"}


class TestEnsembleConfluenceThreshold:
    """Testes variando o parâmetro de confluência mínima."""

    def test_min_confluence_1_allows_single_vote(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        mock_cfg.ensemble_min_confluence = 1
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.8))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("hold", 0.0))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("hold", 0.0))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "buy"
        assert params["buy_count"] == 1

    def test_min_confluence_3_requires_all(
        self, monkeypatch: pytest.MonkeyPatch, dummy_df: pd.DataFrame, mock_cfg: MagicMock
    ) -> None:
        mock_cfg.ensemble_min_confluence = 3
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "trend_follow", _make_stub("buy", 0.9))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "adaptive_trend", _make_stub("buy", 0.9))
        monkeypatch.setitem(signals_mod.STRATEGY_MAP, "hybrid_regime", _make_stub("hold", 0.0))

        signal, params = get_ensemble_signal(dummy_df, mock_cfg)

        assert signal == "hold"