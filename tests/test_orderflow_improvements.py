"""
Testes para as melhorias de orderflow_delta: delta CLV-ponderado
(indicators.add_cvd), absorption_threshold funcional
(indicators.add_orderflow) e persistência de sinal
(signals.signal_orderflow_delta).

FIX (achado em execução real de teste): todos os DataFrames aqui agora
respeitam MIN_CANDLES_REQUIRED=50 (crypto_bot_core/indicators.py) via
o helper _pad_neutral() — a versão anterior usava DataFrames de 1-3
linhas, o que fazia _validate_dataframe() rejeitar silenciosamente
(fail-open, retorna df inalterado) e todas as asserções de coluna
falharem com KeyError. Isso era um bug nos testes, não no código de
produção.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.config import BotConfig, StrategyType, Timeframe
from crypto_bot_core.indicators import MIN_CANDLES_REQUIRED, add_cvd, add_orderflow
from crypto_bot_core.strategies.signals import signal_orderflow_delta


@pytest.fixture
def mock_cfg() -> MagicMock:
    cfg = MagicMock(spec=BotConfig)
    cfg.strategy = StrategyType.ORDERFLOW_DELTA
    cfg.timeframe = Timeframe.H1
    cfg.get_strategy_params.return_value = {
        "divergence_lookback": 10,
        "absorption_threshold": 1.5,
    }
    return cfg


def _pad_neutral(tail_df: pd.DataFrame, min_rows: int = MIN_CANDLES_REQUIRED + 5) -> pd.DataFrame:
    """
    Constrói um DataFrame com `min_rows` barras neutras (high=110,
    low=100, close=105 -> clv=0, sem delta) seguidas pelas linhas de
    `tail_df` (o cenário real sob teste), satisfazendo
    MIN_CANDLES_REQUIRED sem alterar a semântica das asserções, que
    sempre olham para as ÚLTIMAS linhas (iloc[-1], iloc[-2] etc.).

    Args:
        tail_df: DataFrame com as linhas que representam o cenário
            testado (serão preservadas no final do resultado).
        min_rows: Total mínimo de linhas de preenchimento neutro antes
            de `tail_df`.

    Returns:
        pd.DataFrame: preenchimento + tail_df, índice resetado.
    """
    n_pad = min_rows
    pad = pd.DataFrame({
        "high": [110.0] * n_pad,
        "low": [100.0] * n_pad,
        "close": [105.0] * n_pad,
        "volume": [100.0] * n_pad,
    })
    # Garante que tail_df tenha as mesmas colunas (preenche ausentes com NaN -> depois dropna se necessário)
    combined = pd.concat([pad, tail_df], ignore_index=True)
    return combined


# ──────────────────────────────────────────────
# indicators.add_cvd — delta CLV-ponderado
# ──────────────────────────────────────────────


class TestAddCvdClvWeighting:
    def test_close_at_high_gives_max_positive_clv(self) -> None:
        """Fechamento na máxima da barra -> clv=+1 -> delta = +volume."""
        tail = pd.DataFrame({
            "high": [110.0], "low": [100.0], "close": [110.0], "volume": [1000.0],
        })
        df = _pad_neutral(tail)
        df = add_cvd(df)
        assert df["delta"].iloc[-1] == pytest.approx(1000.0)

    def test_close_at_low_gives_max_negative_clv(self) -> None:
        """Fechamento na mínima da barra -> clv=-1 -> delta = -volume."""
        tail = pd.DataFrame({
            "high": [110.0], "low": [100.0], "close": [100.0], "volume": [1000.0],
        })
        df = _pad_neutral(tail)
        df = add_cvd(df)
        assert df["delta"].iloc[-1] == pytest.approx(-1000.0)

    def test_close_at_midpoint_gives_zero_clv(self) -> None:
        tail = pd.DataFrame({
            "high": [110.0], "low": [100.0], "close": [105.0], "volume": [1000.0],
        })
        df = _pad_neutral(tail)
        df = add_cvd(df)
        assert df["delta"].iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_upper_wick_reduces_bullish_delta_vs_naive_sign(self) -> None:
        """
        Regressão do problema original: candle de alta (close > close
        anterior) mas com pavio superior grande (fechou longe da
        máxima) deve gerar delta claramente MENOR que o volume total —
        a proxy antiga (sign-based) atribuiria +volume cheio nesse caso.
        """
        tail = pd.DataFrame({
            "high": [120.0, 130.0],
            "low": [100.0, 100.0],
            "close": [105.0, 106.0],  # fechou perto da mínima na 2ª barra
            "volume": [1000.0, 1000.0],
        })
        df = _pad_neutral(tail)
        df = add_cvd(df)
        # clv da última barra = ((106-100)-(130-106))/(130-100) = (6-24)/30 = -0.6
        assert df["delta"].iloc[-1] == pytest.approx(-600.0, rel=1e-6)

    def test_zero_range_bar_does_not_crash_and_gives_zero_delta(self) -> None:
        """high == low (sem range, ex: dados corrompidos) -> delta=0, sem exceção."""
        tail = pd.DataFrame({
            "high": [100.0], "low": [100.0], "close": [100.0], "volume": [1000.0],
        })
        df = _pad_neutral(tail)
        df = add_cvd(df)
        assert df["delta"].iloc[-1] == 0.0

    def test_missing_high_low_columns_returns_df_unchanged(self) -> None:
        """Sem 'high'/'low' (requisito novo), a função deve falhar de
        forma segura (fail-open) e não adicionar a coluna delta."""
        df = pd.DataFrame({
            "close": [100.0] * (MIN_CANDLES_REQUIRED + 5),
            "volume": [1000.0] * (MIN_CANDLES_REQUIRED + 5),
        })
        result = add_cvd(df)
        assert "delta" not in result.columns

    def test_cvd_is_cumulative_sum_of_delta(self) -> None:
        tail = pd.DataFrame({
            "high": [110.0, 110.0, 110.0],
            "low": [100.0, 100.0, 100.0],
            "close": [110.0, 100.0, 105.0],
            "volume": [100.0, 100.0, 100.0],
        })
        df = _pad_neutral(tail)
        df = add_cvd(df)
        expected_cumsum = df["delta"].cumsum()
        pd.testing.assert_series_equal(df["cvd"], expected_cumsum, check_names=False)


# ──────────────────────────────────────────────
# indicators.add_orderflow — absorption_threshold parametrizável
# ──────────────────────────────────────────────


class TestAddOrderflowAbsorptionParametrized:
    def _make_df(self, n: int = MIN_CANDLES_REQUIRED + 5) -> pd.DataFrame:
        np.random.seed(1)
        close = 100 + np.cumsum(np.random.randn(n) * 0.2)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n)) * 100 + 50
        return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})

    def test_default_parameters_preserve_legacy_behavior(self) -> None:
        """Sem passar os novos parâmetros, o resultado deve ser idêntico
        ao comportamento anterior (percentis 0.8/0.2 hardcoded)."""
        df = self._make_df()
        result_default = add_orderflow(df.copy())
        result_explicit = add_orderflow(
            df.copy(), absorption_volume_percentile=0.8, absorption_range_percentile=0.2,
        )
        pd.testing.assert_series_equal(
            result_default["of_absorption"], result_explicit["of_absorption"],
        )

    def test_stricter_percentile_reduces_absorption_signals(self) -> None:
        """Um percentil de volume mais exigente (0.95 em vez de 0.5) deve
        gerar NO MÁXIMO a mesma quantidade de sinais de absorção."""
        df = self._make_df()
        lenient = add_orderflow(df.copy(), absorption_volume_percentile=0.5)
        strict = add_orderflow(df.copy(), absorption_volume_percentile=0.95)
        assert strict["of_absorption"].sum() <= lenient["of_absorption"].sum()

    def test_columns_present(self) -> None:
        df = self._make_df()
        result = add_orderflow(df)
        for col in ["of_bearish_div", "of_bullish_div", "of_absorption"]:
            assert col in result.columns


# ──────────────────────────────────────────────
# signals.signal_orderflow_delta — persistência + absorption_threshold
# ──────────────────────────────────────────────
# NOTA: estes testes já operavam sobre DataFrames "prontos" (indicadores
# calculados manualmente, não via add_cvd/add_orderflow), então NÃO são
# afetados pelo MIN_CANDLES_REQUIRED — mantidos como no original.


class TestOrderflowDeltaPersistence:
    def test_persistent_positive_delta_generates_buy(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
            "delta": [1, 2, 3],  # 3 barras consecutivas positivas
            "cvd": [10, 12, 15],
            "cvd_ma": [9, 10, 11],
            "of_bearish_div": [0, 0, 0],
            "of_bullish_div": [0, 0, 1],
            "of_absorption": [0, 0, 0],
        })
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "buy"
        assert params["persistence_bars"] == 3

    def test_single_bar_flip_does_not_generate_buy(self, mock_cfg: MagicMock) -> None:
        """Delta positivo só na última barra (as anteriores negativas)
        NÃO deve gerar sinal — regressão que a persistência corrige."""
        df = pd.DataFrame({
            "close": [100, 99, 102],
            "volume": [1000, 1100, 1200],
            "delta": [-1, -2, 3],  # sem persistência: 2 negativas + 1 positiva
            "cvd": [10, 8, 11],
            "cvd_ma": [9, 9, 9],
            "of_bearish_div": [0, 0, 0],
            "of_bullish_div": [0, 0, 1],
            "of_absorption": [0, 0, 0],
        })
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal != "buy"

    def test_persistent_negative_delta_generates_sell(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [102, 101, 100],
            "volume": [1000, 1100, 1200],
            "delta": [-1, -2, -3],
            "cvd": [15, 12, 10],
            "cvd_ma": [16, 14, 13],
            "of_bearish_div": [0, 0, 1],
            "of_bullish_div": [0, 0, 0],
            "of_absorption": [0, 0, 0],
        })
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "sell"


class TestOrderflowDeltaAbsorptionThresholdUsage:
    def test_absorption_signal_includes_volume_ratio_and_uses_threshold(self, mock_cfg: MagicMock) -> None:
        n = 25
        df = pd.DataFrame({
            "close": [100.0] * n,
            "volume": [1000.0] * (n - 1) + [3000.0],  # pico de volume na última barra
            "delta": [0.5] * n,
            "cvd": [10.0] * n,
            "cvd_ma": [10.0] * n,
            "of_bearish_div": [0] * n,
            "of_bullish_div": [0] * n,
            "of_absorption": [0] * (n - 1) + [1],
        })
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "buy"
        assert params["reason"] == "orderflow_absorption_buy"
        assert "volume_ratio" in params
        assert params["volume_ratio"] > 1.0

    def test_higher_absorption_threshold_reduces_confidence(self, mock_cfg: MagicMock) -> None:
        """Threshold configurado mais alto (exige mais volume relativo
        para confiança máxima) deve resultar em confiança MENOR para o
        mesmo volume_ratio observado."""
        n = 25
        df = pd.DataFrame({
            "close": [100.0] * n,
            "volume": [1000.0] * (n - 1) + [1500.0],
            "delta": [0.5] * n,
            "cvd": [10.0] * n,
            "cvd_ma": [10.0] * n,
            "of_bearish_div": [0] * n,
            "of_bullish_div": [0] * n,
            "of_absorption": [0] * (n - 1) + [1],
        })

        mock_cfg.get_strategy_params.return_value = {"divergence_lookback": 10, "absorption_threshold": 1.2}
        _, params_low_threshold = signal_orderflow_delta(df, mock_cfg)

        mock_cfg.get_strategy_params.return_value = {"divergence_lookback": 10, "absorption_threshold": 5.0}
        _, params_high_threshold = signal_orderflow_delta(df, mock_cfg)

        assert params_high_threshold["confidence"] <= params_low_threshold["confidence"]


class TestOrderflowDeltaBackwardCompatibility:
    """Confirma que os testes/fixtures originais (3 barras, delta já
    consistente) continuam funcionando com a exigência de persistência."""

    def test_original_buy_fixture_still_works(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
            "delta": [1, 2, 3],
            "cvd": [10, 12, 15],
            "cvd_ma": [9, 10, 11],
            "of_bearish_div": [0, 0, 0],
            "of_bullish_div": [0, 0, 1],
            "of_absorption": [0, 0, 0],
        })
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "buy"
        assert "orderflow_buy" in params.get("reason", "")

    def test_original_sell_fixture_still_works(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [102, 101, 100],
            "volume": [1000, 1100, 1200],
            "delta": [-1, -2, -3],
            "cvd": [15, 12, 10],
            "cvd_ma": [16, 14, 13],
            "of_bearish_div": [0, 0, 1],
            "of_bullish_div": [0, 0, 0],
            "of_absorption": [0, 0, 0],
        })
        signal, params = signal_orderflow_delta(df, mock_cfg)
        assert signal == "sell"
        assert "orderflow_sell" in params.get("reason", "")