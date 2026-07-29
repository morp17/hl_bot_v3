"""
Testes para o lock por símbolo em OrderExecutor — proteção contra
reentrância entre threads no mesmo símbolo (crypto_bot_core/execution.py).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from crypto_bot_core.execution import OrderExecutor


@pytest.fixture
def mock_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.hyperliquid_private_key = "0x" + "ab" * 32
    cfg.hyperliquid_account_address = "0x" + "cd" * 20
    cfg.sandbox = True
    return cfg


@pytest.fixture
def executor(mock_cfg: MagicMock) -> OrderExecutor:
    with patch("crypto_bot_core.execution.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector._get_sz_decimals.return_value = 5
        mock_get_connector.return_value = mock_connector
        exe = OrderExecutor(mock_cfg)
        exe._connector = mock_connector
        yield exe


class TestSymbolLockCreation:
    """Testes para _get_symbol_lock — criação e reuso do lock por símbolo."""

    def test_same_symbol_returns_same_lock_instance(self, executor: OrderExecutor) -> None:
        lock1 = executor._get_symbol_lock("BTC/USDC")
        lock2 = executor._get_symbol_lock("BTC/USDC")
        assert lock1 is lock2

    def test_different_symbols_get_different_locks(self, executor: OrderExecutor) -> None:
        lock_btc = executor._get_symbol_lock("BTC/USDC")
        lock_eth = executor._get_symbol_lock("ETH/USDC")
        assert lock_btc is not lock_eth

    def test_lock_creation_is_thread_safe_for_same_symbol(self, executor: OrderExecutor) -> None:
        """
        Dispara N threads simultaneamente pedindo o lock do MESMO símbolo
        pela primeira vez — todas devem receber a mesma instância, sem
        criar locks duplicados por corrida na criação da entrada do dict.
        """
        symbol = "SOL/USDC"
        results: list = []
        barrier = threading.Barrier(10)

        def worker() -> None:
            barrier.wait()  # maximiza a chance de colisão real na criação
            results.append(executor._get_symbol_lock(symbol))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(lock is results[0] for lock in results)
        # Apenas uma entrada deve ter sido criada no dicionário interno.
        assert len(executor._symbol_locks) == 1


class TestSymbolLockSerializesSameSymbolOperations:
    """
    Testes de regressão: confirmam que operações concorrentes no MESMO
    símbolo nunca se sobrepõem — o cenário de risco original era
    place_bulk_tpsl cancelando ordens que outra chamada acabara de criar.
    """

    def test_place_bulk_tpsl_serializes_same_symbol(self, executor: OrderExecutor) -> None:
        symbol = "BTC/USDC"
        state: Dict[str, Any] = {"in_critical_section": False, "violations": 0}
        section_lock = threading.Lock()  # protege o próprio state de teste

        def fake_place_bulk_tpsl(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            with section_lock:
                if state["in_critical_section"]:
                    state["violations"] += 1
                state["in_critical_section"] = True
            # Janela deliberadamente ampla para dar chance real de colisão
            # caso o lock por símbolo não estivesse funcionando.
            time.sleep(0.05)
            with section_lock:
                state["in_critical_section"] = False
            return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}}}

        executor.connector.place_bulk_tpsl.side_effect = fake_place_bulk_tpsl

        def worker() -> None:
            executor.place_bulk_tpsl(
                symbol=symbol,
                side="buy",
                entry_qty=0.1,
                entry_price=50000.0,
                tp_price=52000.0,
                sl_price=48000.0,
            )

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["violations"] == 0

    def test_different_symbols_do_not_wait_for_each_other(self, executor: OrderExecutor) -> None:
        """
        Símbolos diferentes usam locks distintos — uma operação lenta em
        um símbolo não deve bloquear outro símbolo. Teste baseado em
        timing com margem generosa para evitar flakiness.
        """
        slow_symbol = "BTC/USDC"
        fast_symbol = "ETH/USDC"

        def slow_side_effect(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            time.sleep(0.2)
            return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}}}

        def fast_side_effect(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 2}}]}}}

        def call_slow() -> None:
            executor.connector.place_bulk_tpsl.side_effect = slow_side_effect
            executor.place_bulk_tpsl(
                symbol=slow_symbol, side="buy", entry_qty=0.1,
                entry_price=50000.0, tp_price=52000.0, sl_price=48000.0,
            )

        t_slow = threading.Thread(target=call_slow)
        t_slow.start()
        time.sleep(0.02)  # garante que a chamada lenta já adquiriu seu lock

        executor.connector.place_bulk_tpsl.side_effect = fast_side_effect
        start = time.monotonic()
        executor.place_bulk_tpsl(
            symbol=fast_symbol, side="buy", entry_qty=0.1,
            entry_price=3000.0, tp_price=3100.0, sl_price=2900.0,
        )
        elapsed = time.monotonic() - start

        t_slow.join()

        # Com locks por símbolo, a chamada para o símbolo diferente NÃO
        # deveria esperar os ~0.2s do símbolo lento.
        assert elapsed < 0.15


class TestSymbolLockCoversAllMutatingMethods:
    """Confirma que cada método relevante adquire o lock do símbolo correto."""

    def test_place_order_uses_symbol_lock(self, executor: OrderExecutor) -> None:
        executor.connector.place_order.return_value = {"oid": "1", "status": "resting"}
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.place_order("BTC/USDC", "buy", 0.1, 50000.0)
            spy.assert_called_once_with("BTC/USDC")

    def test_place_tpsl_order_uses_symbol_lock(self, executor: OrderExecutor) -> None:
        executor.connector.place_tpsl_order.return_value = {"oid": "tp_1"}
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.place_tpsl_order("BTC/USDC", "sell", 0.1, 52000.0, 51000.0, "tp")
            spy.assert_called_once_with("BTC/USDC")

    def test_place_bulk_tpsl_uses_symbol_lock(self, executor: OrderExecutor) -> None:
        executor.connector.place_bulk_tpsl.return_value = {
            "status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}}
        }
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.place_bulk_tpsl(
                symbol="BTC/USDC", side="buy", entry_qty=0.1,
                entry_price=50000.0, tp_price=52000.0, sl_price=48000.0,
            )
            spy.assert_called_once_with("BTC/USDC")

    def test_cancel_order_uses_symbol_lock(self, executor: OrderExecutor) -> None:
        executor.connector.cancel_order.return_value = True
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.cancel_order("BTC/USDC", "12345")
            spy.assert_called_once_with("BTC/USDC")

    def test_cancel_all_orders_uses_symbol_lock(self, executor: OrderExecutor) -> None:
        executor.connector.cancel_all_orders.return_value = 2
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.cancel_all_orders("BTC/USDC")
            spy.assert_called_once_with("BTC/USDC")

    def test_close_position_uses_symbol_lock(self, executor: OrderExecutor) -> None:
        executor.connector.close_position.return_value = {"oid": "close_1"}
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.close_position("BTC/USDC", 0.1, "buy")
            spy.assert_called_once_with("BTC/USDC")

    def test_update_leverage_does_not_use_symbol_lock(self, executor: OrderExecutor) -> None:
        """update_leverage é intencionalmente EXCLUÍDO do lock por
        símbolo (ver docstring do método em execution.py) — não interage
        com o ciclo cancel-then-create de place_bulk_tpsl e é chamado
        tipicamente uma única vez no startup."""
        executor.connector.update_leverage.return_value = True
        with patch.object(executor, "_get_symbol_lock", wraps=executor._get_symbol_lock) as spy:
            executor.update_leverage("BTC/USDC", 5)
            spy.assert_not_called()