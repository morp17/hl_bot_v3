"""
Testes para a versão reestruturada de signal_scalping_grid (multi-nível,
espaçamento ATR-adaptativo, confirmação de volume opcional) —
crypto_bot_core/strategies/signals.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from crypto_bot_core.config import BotConfig, StrategyType, Timeframe
from crypto_bot_core.strategies.signals import signal_scalping_grid


@pytest.fixture
def mock_cfg() -> MagicMock:
    cfg = MagicMock(spec=BotConfig)
    cfg.strategy = StrategyType.SCALPING_GRID
    cfg.timeframe = Timeframe.M1
    cfg.get_strategy_params.return_value = {
        "grid_levels": 5,
        "grid_spread_pct": 0.1,
        "grid_size_pct": 20.0,
    }
    return cfg


class TestScalpingGridFallbackPct:
    """Sem coluna 'atr' — deve usar exatamente o cálculo percentual legado
    (garante retrocompatibilidade com o comportamento anterior à reestruturação)."""

    def test_buy_uses_pct_mode(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100, 99.9, 99.5],
            "ema_fast": [100, 100, 100],
            "ema_slow": [100, 100, 100],
            "rsi": [50, 48, 44],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "buy", f"Esperado buy, obtido {signal}. Params: {params}"
        assert params["grid_mode"] == "pct"
        assert "confidence" in params
        assert 0.0 <= params["confidence"] <= 1.0

    def test_sell_uses_pct_mode(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100, 100.1, 100.2],
            "ema_fast": [100, 100, 100],
            "ema_slow": [100, 100, 100],
            "rsi": [50, 52, 56],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "sell"
        assert params["grid_mode"] == "pct"

    def test_hold_outside_grid_pct_mode(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100, 100, 100],
            "ema_fast": [100, 100, 100],
            "ema_slow": [100, 100, 100],
            "rsi": [50, 50, 50],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "hold"
        assert params["reason"] == "fora_do_grid"
        assert params["confidence"] == 0.0


class TestScalpingGridAtrMode:
    """Com coluna 'atr' presente (>0) — deve usar espaçamento dinâmico
    baseado em ATR em vez do percentual fixo."""

    def test_atr_mode_selected_when_available(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100.0, 100.0, 100.0],
            "ema_fast": [100.0, 100.0, 100.0],
            "ema_slow": [100.0, 100.0, 100.0],
            "rsi": [50.0, 48.0, 42.0],
            "atr": [0.5, 0.5, 0.5],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert params["grid_mode"] == "atr"
        assert signal in ("buy", "sell", "hold")

    def test_atr_zero_falls_back_to_pct(self, mock_cfg: MagicMock) -> None:
        """ATR presente mas <= 0 deve cair no modo percentual — proteção
        contra grid_step inválido/zero."""
        df = pd.DataFrame({
            "close": [100, 99.9, 99.5],
            "ema_fast": [100, 100, 100],
            "ema_slow": [100, 100, 100],
            "rsi": [50, 48, 44],
            "atr": [0.0, 0.0, 0.0],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert params["grid_mode"] == "pct"

    def test_grid_depth_varies_with_atr_and_affects_confidence(self, mock_cfg: MagicMock) -> None:
        """
        Espaçamento ATR maior (menor granularidade) toca menos níveis do
        grid; espaçamento ATR menor (maior granularidade) toca mais
        níveis para o mesmo preço/RSI — a confiança deve refletir essa
        profundidade (mais níveis tocados = maior confiança).

        Valores calibrados analiticamente a partir da fórmula de toque de
        nível: nível i é tocado quando i <= (close*0.01) / (grid_step*1.01).
        - atr=1.2 -> grid_step=0.6 -> apenas nível 1 tocado.
        - atr=0.3 -> grid_step=0.15 -> todos os 5 níveis tocados.
        """
        base = {
            "close": [100.0, 100.0, 100.0],
            "ema_fast": [100.0, 100.0, 100.0],
            "ema_slow": [100.0, 100.0, 100.0],
            "rsi": [50.0, 45.0, 40.0],
        }

        df_shallow = pd.DataFrame({**base, "atr": [1.2, 1.2, 1.2]})
        signal_shallow, params_shallow = signal_scalping_grid(df_shallow, mock_cfg)

        df_deep = pd.DataFrame({**base, "atr": [0.3, 0.3, 0.3]})
        signal_deep, params_deep = signal_scalping_grid(df_deep, mock_cfg)

        assert signal_shallow == "buy"
        assert signal_deep == "buy"
        assert params_shallow["grid_level"] == 1
        assert params_deep["grid_level"] == 5
        assert params_deep["confidence"] > params_shallow["confidence"]

    def test_levels_touched_is_list_and_matches_grid_level(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100.0, 100.0, 100.0],
            "ema_fast": [100.0, 100.0, 100.0],
            "ema_slow": [100.0, 100.0, 100.0],
            "rsi": [50.0, 45.0, 40.0],
            "atr": [0.3, 0.3, 0.3],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "buy"
        assert isinstance(params["levels_touched"], list)
        assert params["grid_level"] == max(params["levels_touched"])
        assert params["levels_touched"] == [1, 2, 3, 4, 5]


class TestScalpingGridVolumeConfirmation:
    """Confirmação de volume opcional — não deve quebrar quando a coluna
    está ausente, e deve influenciar (ou bloquear) o sinal quando presente."""

    def test_no_volume_column_does_not_block_signal(self, mock_cfg: MagicMock) -> None:
        df = pd.DataFrame({
            "close": [100, 99.9, 99.5],
            "ema_fast": [100, 100, 100],
            "ema_slow": [100, 100, 100],
            "rsi": [50, 48, 44],
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "buy"

    def test_low_volume_blocks_signal(self, mock_cfg: MagicMock) -> None:
        """Volume do candle atual abaixo da média móvel de 20 períodos
        deve bloquear o sinal, mesmo com preço/RSI favoráveis ao grid."""
        n = 25
        close = [100.0] * (n - 3) + [99.9, 99.7, 99.5]
        rsi = [50.0] * (n - 3) + [48.0, 46.0, 44.0]
        volume = [1000.0] * (n - 1) + [200.0]  # último candle bem abaixo da média

        df = pd.DataFrame({
            "close": close,
            "ema_fast": [100.0] * n,
            "ema_slow": [100.0] * n,
            "rsi": rsi,
            "volume": volume,
        })
        signal, params = signal_scalping_grid(df, mock_cfg)
        assert signal == "hold"

    def test_high_volume_confirms_signal_with_higher_confidence(self, mock_cfg: MagicMock) -> None:
        """Volume acima da média deve confirmar o sinal e aumentar a
        confiança em relação a um caso de volume neutro (na média)."""
        n = 25
        close = [100.0] * (n - 3) + [99.9, 99.7, 99.5]
        rsi = [50.0] * (n - 3) + [48.0, 46.0, 44.0]

        volume_neutral = [500.0] * n  # último valor == média -> score de volume ~0
        df_neutral = pd.DataFrame({
            "close": close, "ema_fast": [100.0] * n, "ema_slow": [100.0] * n,
            "rsi": rsi, "volume": volume_neutral,
        })
        signal_neutral, params_neutral = signal_scalping_grid(df_neutral, mock_cfg)

        volume_high = [500.0] * (n - 1) + [5000.0]  # último candle bem acima da média
        df_high = pd.DataFrame({
            "close": close, "ema_fast": [100.0] * n, "ema_slow": [100.0] * n,
            "rsi": rsi, "volume": volume_high,
        })
        signal_high, params_high = signal_scalping_grid(df_high, mock_cfg)

        assert signal_neutral == "buy"
        assert signal_high == "buy"
        assert params_high["confidence"] >= params_neutral["confidence"]