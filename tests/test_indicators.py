"""
Testes unitários para o módulo de indicadores.
"""
import numpy as np
import pandas as pd
import pytest

from crypto_bot_core.config import BotConfig
from crypto_bot_core.indicators import (
    add_ema,
    add_rsi,
    add_atr,
    add_bollinger_bands,
    add_adx,
    add_macd,
    add_ichimoku,
    add_volume_profile,
    add_cvd,
    add_orderflow,
    add_all_indicators,
    get_latest_values,
    _validate_dataframe,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Cria DataFrame de exemplo com dados OHLCV."""
    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    data = {
        "open": close - np.random.rand(n) * 50,
        "high": close + np.random.rand(n) * 100,
        "low": close - np.random.rand(n) * 100,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
    }
    return pd.DataFrame(data)


class TestValidateDataFrame:
    """Testes para _validate_dataframe."""

    def test_valid_df(self, sample_df: pd.DataFrame) -> None:
        assert _validate_dataframe(sample_df, ["close"]) is True

    def test_empty_df(self) -> None:
        assert _validate_dataframe(pd.DataFrame()) is False

    def test_missing_column(self, sample_df: pd.DataFrame) -> None:
        assert _validate_dataframe(sample_df, ["nonexistent"]) is False

    def test_small_df(self) -> None:
        small = pd.DataFrame({"close": [1, 2, 3]})
        assert _validate_dataframe(small) is False


class TestEMA:
    """Testes para EMA."""

    def test_add_ema(self, sample_df: pd.DataFrame) -> None:
        df = add_ema(sample_df, 9)
        assert "ema_9" in df.columns
        assert not df["ema_9"].isna().all()

    def test_add_ema_custom_name(self, sample_df: pd.DataFrame) -> None:
        df = add_ema(sample_df, 21, "ema_fast")
        assert "ema_fast" in df.columns

    def test_ema_empty_df(self) -> None:
        df = add_ema(pd.DataFrame(), 9)
        assert df.empty


class TestRSI:
    """Testes para RSI."""

    def test_add_rsi(self, sample_df: pd.DataFrame) -> None:
        df = add_rsi(sample_df, 14)
        assert "rsi" in df.columns
        rsi_values = df["rsi"].dropna()
        assert all(0 <= v <= 100 for v in rsi_values)

    def test_rsi_empty_df(self) -> None:
        df = add_rsi(pd.DataFrame(), 14)
        assert df.empty


class TestATR:
    """Testes para ATR."""

    def test_add_atr(self, sample_df: pd.DataFrame) -> None:
        df = add_atr(sample_df, 14)
        assert "atr" in df.columns
        assert df["atr"].iloc[-1] > 0

    def test_atr_empty_df(self) -> None:
        df = add_atr(pd.DataFrame(), 14)
        assert df.empty


class TestBollingerBands:
    """Testes para Bollinger Bands."""

    def test_add_bb(self, sample_df: pd.DataFrame) -> None:
        df = add_bollinger_bands(sample_df, 20, 2.0)
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_middle" in df.columns
        # Upper deve ser > lower
        assert df["bb_upper"].iloc[-1] > df["bb_lower"].iloc[-1]


class TestADX:
    """Testes para ADX."""

    def test_add_adx(self, sample_df: pd.DataFrame) -> None:
        df = add_adx(sample_df, 14)
        assert "adx" in df.columns
        assert "plus_di" in df.columns
        assert "minus_di" in df.columns


class TestMACD:
    """Testes para MACD."""

    def test_add_macd(self, sample_df: pd.DataFrame) -> None:
        df = add_macd(sample_df, 12, 26, 9)
        assert "macd_line" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns


class TestIchimoku:
    """Testes para Ichimoku."""

    def test_add_ichimoku(self, sample_df: pd.DataFrame) -> None:
        df = add_ichimoku(sample_df)
        assert "tenkan_sen" in df.columns
        assert "kijun_sen" in df.columns
        assert "senkou_span_a" in df.columns
        assert "senkou_span_b" in df.columns


class TestVolumeProfile:
    """Testes para Volume Profile."""

    def test_add_volume_profile(self, sample_df: pd.DataFrame) -> None:
        df = add_volume_profile(sample_df, 20)
        assert "vp_avg" in df.columns
        assert "vp_ratio" in df.columns
        assert "vp_vwap" in df.columns


class TestCVD:
    """Testes para CVD."""

    def test_add_cvd(self, sample_df: pd.DataFrame) -> None:
        df = add_cvd(sample_df)
        assert "cvd" in df.columns
        assert "delta" in df.columns


class TestOrderFlow:
    """Testes para OrderFlow."""

    def test_add_orderflow(self, sample_df: pd.DataFrame) -> None:
        df = add_orderflow(sample_df, 10)
        assert "of_bearish_div" in df.columns
        assert "of_bullish_div" in df.columns
        assert "of_absorption" in df.columns


class TestAllIndicators:
    """Testes para add_all_indicators."""

    def test_add_all(self, sample_df: pd.DataFrame) -> None:
        cfg = BotConfig()
        df = add_all_indicators(sample_df, cfg)
        # Deve ter adicionado muitas colunas
        assert len(df.columns) > 10
        assert "ema_fast" in df.columns
        assert "rsi" in df.columns
        assert "atr" in df.columns
        assert "bb_upper" in df.columns
        assert "adx" in df.columns
        assert "macd_line" in df.columns
        assert "cvd" in df.columns

    def test_get_latest_values(self, sample_df: pd.DataFrame) -> None:
        cfg = BotConfig()
        df = add_all_indicators(sample_df, cfg)
        latest = get_latest_values(df)
        assert isinstance(latest, dict)
        assert len(latest) > 0
        assert "close" in latest
        assert isinstance(latest["close"], float)
