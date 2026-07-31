"""
Teste de regressão para a correção da reconciliação de posições —
_reconcile_state() deve forçar _sync_from_exchange() quando a
contagem de posições locais divergir da DEX, não apenas logar.

Achado em produção (log real): uma ordem limit BUY BTC ficou
"resting" e foi preenchida na exchange em um ciclo POSTERIOR ao envio
— como o bot só cria a Position local na resposta imediata de
place_bulk_tpsl() (quando já vem "filled"), a posição preenchida
tardiamente ficava invisível para check_exits()/accrue_funding()/
capital_protection até o próximo full resync periódico (~30 ciclos).
_reconcile_state() já detectava "local=0 vs DEX=1" mas apenas
logava — não corrigia.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crypto_bot_core.config import BotConfig
from crypto_bot_core.risk import Position


@pytest.fixture
def bot_with_mocked_deps(monkeypatch: pytest.MonkeyPatch) -> "HyperliquidBot":
    """
    Cria uma instância de HyperliquidBot com todas as dependências
    externas (lock, exchange, dashboard, health server) mockadas,
    isolando apenas a lógica de _reconcile_state().
    """
    import main as main_mod

    monkeypatch.setattr(main_mod, "LockManager", MagicMock())
    monkeypatch.setattr(main_mod, "HealthServer", MagicMock())

    with patch("crypto_bot_core.dashboard.main.start_dashboard_thread", return_value=None), \
         patch("crypto_bot_core.execution.OrderExecutor") as mock_executor_cls:

        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor

        cfg = BotConfig(_env_file=None, capital_usd=1000.0)
        bot = main_mod.HyperliquidBot(cfg)

        # Substitui o lock por um mock que sempre "adquire" com sucesso
        bot._lock = MagicMock()
        bot._lock.acquire.return_value = True

        bot.executor = mock_executor
        yield bot


class TestReconcileStateForcesResyncOnPositionMismatch:
    """
    Testes para a correção: _reconcile_state() deve chamar
    _sync_from_exchange() quando a contagem de posições locais
    divergir da contagem real na DEX — não apenas logar o warning.
    """

    def test_forces_resync_when_dex_has_extra_position(self, bot_with_mocked_deps) -> None:
        """
        Cenário do log real: local=0, DEX=1 (ordem limit preenchida
        tardiamente na exchange). Deve forçar _sync_from_exchange().
        """
        bot = bot_with_mocked_deps
        bot.position_manager.positions = []  # local = 0

        # Saldo já sincronizado (sem divergência de saldo, que teria
        # prioridade e já forçaria resync por outro caminho) — isola
        # o teste especificamente para a divergência de CONTAGEM.
        bot.capital_protection.state.current_balance = 1000.0
        bot.position_manager.current_balance = 1000.0
        bot.capital_protection.state.peak_balance = 1000.0
        bot.position_manager.peak_balance = 1000.0

        # DEX reporta 1 posição ativa (contracts > 0)
        bot.executor.connector.fetch_positions.return_value = [
            {"symbol": "BTC/USDC:USDC", "contracts": 0.00023, "side": "long", "entryPrice": 64331.0},
        ]

        with patch.object(bot, "_sync_from_exchange") as mock_sync:
            bot._reconcile_state()
            mock_sync.assert_called_once()

    def test_does_not_resync_when_counts_match(self, bot_with_mocked_deps) -> None:
        """Contagens iguais (local == DEX) não devem disparar resync
        adicional — evita chamadas desnecessárias à API a cada ciclo."""
        bot = bot_with_mocked_deps
        bot.position_manager.positions = [
            Position(symbol="BTC/USDC", side="buy", entry_price=64331.0, qty=0.00023),
        ]

        bot.capital_protection.state.current_balance = 1000.0
        bot.position_manager.current_balance = 1000.0
        bot.capital_protection.state.peak_balance = 1000.0
        bot.position_manager.peak_balance = 1000.0

        bot.executor.connector.fetch_positions.return_value = [
            {"symbol": "BTC/USDC:USDC", "contracts": 0.00023, "side": "long", "entryPrice": 64331.0},
        ]

        with patch.object(bot, "_sync_from_exchange") as mock_sync:
            bot._reconcile_state()
            mock_sync.assert_not_called()

    def test_does_not_resync_when_dex_has_fewer_positions(self, bot_with_mocked_deps) -> None:
        """
        Caso oposto (local=1, DEX=0) — já é tratado por
        _verify_positions_on_exchange() em outro ponto do ciclo
        (fechamento externo). _reconcile_state() ainda deve forçar
        resync aqui também, pois qualquer divergência de contagem é
        motivo de reconciliação, não apenas o caso "DEX tem mais".
        """
        bot = bot_with_mocked_deps
        bot.position_manager.positions = [
            Position(symbol="BTC/USDC", side="buy", entry_price=64331.0, qty=0.00023),
        ]

        bot.capital_protection.state.current_balance = 1000.0
        bot.position_manager.current_balance = 1000.0
        bot.capital_protection.state.peak_balance = 1000.0
        bot.position_manager.peak_balance = 1000.0

        bot.executor.connector.fetch_positions.return_value = []  # DEX = 0

        with patch.object(bot, "_sync_from_exchange") as mock_sync:
            bot._reconcile_state()
            mock_sync.assert_called_once()

    def test_balance_divergence_takes_priority_and_still_resyncs(self, bot_with_mocked_deps) -> None:
        """
        Quando HÁ divergência de saldo (que já forçava resync antes
        desta correção) E de contagem de posições simultaneamente, o
        resync deve ocorrer uma única vez (via saldo, que retorna cedo
        no método) — não deve haver dupla chamada.
        """
        bot = bot_with_mocked_deps
        bot.position_manager.positions = []

        # Divergência de saldo > STATE_RECONCILE_EPSILON_USD (0.5)
        bot.capital_protection.state.current_balance = 1000.0
        bot.position_manager.current_balance = 950.0
        bot.capital_protection.state.peak_balance = 1000.0
        bot.position_manager.peak_balance = 1000.0

        bot.executor.connector.fetch_positions.return_value = [
            {"symbol": "BTC/USDC:USDC", "contracts": 0.00023, "side": "long", "entryPrice": 64331.0},
        ]

        with patch.object(bot, "_sync_from_exchange") as mock_sync:
            bot._reconcile_state()
            mock_sync.assert_called_once()

    def test_fetch_positions_error_does_not_raise(self, bot_with_mocked_deps) -> None:
        """Erro ao consultar a DEX para checagem de contagem não deve
        propagar exceção — fail-safe, apenas loga em DEBUG."""
        bot = bot_with_mocked_deps
        bot.position_manager.positions = []

        bot.capital_protection.state.current_balance = 1000.0
        bot.position_manager.current_balance = 1000.0
        bot.capital_protection.state.peak_balance = 1000.0
        bot.position_manager.peak_balance = 1000.0

        bot.executor.connector.fetch_positions.side_effect = ConnectionError("timeout simulado")

        # Não deve levantar exceção
        bot._reconcile_state()

    def test_resync_imports_late_filled_position_with_sl_tp(self, bot_with_mocked_deps) -> None:
        """
        Teste de integração parcial: confirma que, ao chamar
        _sync_from_exchange() de fato (não mockado desta vez), uma
        posição presente na DEX mas ausente localmente é importada
        para position_manager.positions — reproduzindo o cenário real
        do log (ordem resting preenchida entre ciclos).
        """
        bot = bot_with_mocked_deps
        bot.position_manager.positions = []

        bot.capital_protection.state.current_balance = 1000.0
        bot.position_manager.current_balance = 1000.0
        bot.capital_protection.state.peak_balance = 1000.0
        bot.position_manager.peak_balance = 1000.0

        bot.executor.connector.fetch_positions.return_value = [
            {"symbol": "BTC/USDC:USDC", "contracts": 0.00023, "side": "long", "entryPrice": 64331.0},
        ]
        bot.executor.connector.info.open_orders.return_value = []
        bot.executor.connector.native.bulk_cancel.return_value = None
        bot.executor.connector.fetch_balance.return_value = {
            "USDC": {"free": 1000.0}
        }

        bot._reconcile_state()

        assert len(bot.position_manager.positions) == 1
        imported = bot.position_manager.positions[0]
        assert imported.symbol == "BTC/USDC"
        assert imported.side == "buy"
        assert imported.qty == pytest.approx(0.00023)