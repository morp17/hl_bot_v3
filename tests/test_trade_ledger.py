"""
Testes para o módulo de trilha de auditoria de trades (TradeLedger) —
auditoria item 12: registro append-only e persistente de trades
fechados em modo live.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict

import pytest

from crypto_bot_core.trade_ledger import TradeLedger, get_ledger


@pytest.fixture
def ledger_path(tmp_path: Any) -> str:
    """Caminho temporário isolado por teste."""
    return str(tmp_path / "trades_test.jsonl")


@pytest.fixture
def ledger(ledger_path: str) -> TradeLedger:
    return TradeLedger(path=ledger_path)


def _sample_trade_kwargs(**overrides: Any) -> Dict[str, Any]:
    base = dict(
        symbol="BTC/USDC",
        side="buy",
        entry_price=50000.0,
        exit_price=51000.0,
        qty=0.1,
        gross_pnl=100.0,
        fee=5.0,
        funding_cost=0.5,
        net_pnl=94.5,
        exit_reason="take_profit",
        strategy="trend_follow",
        open_time=time.time() - 3600,
    )
    base.update(overrides)
    return base


class TestTradeLedgerInit:
    """Testes de inicialização — criação de diretório."""

    def test_creates_parent_directory(self, tmp_path: Any) -> None:
        nested_path = str(tmp_path / "nested" / "dir" / "trades.jsonl")
        ledger = TradeLedger(path=nested_path)
        assert os.path.isdir(os.path.dirname(nested_path))

    def test_does_not_create_file_until_first_record(self, ledger_path: str) -> None:
        TradeLedger(path=ledger_path)
        assert not os.path.exists(ledger_path)


class TestTradeLedgerRecord:
    """Testes para record() — escrita append-only em JSONL."""

    def test_record_creates_file(self, ledger: TradeLedger, ledger_path: str) -> None:
        ledger.record(**_sample_trade_kwargs())
        assert os.path.exists(ledger_path)

    def test_record_writes_valid_json_line(self, ledger: TradeLedger, ledger_path: str) -> None:
        ledger.record(**_sample_trade_kwargs())
        with open(ledger_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["symbol"] == "BTC/USDC"
        assert parsed["side"] == "buy"
        assert parsed["net_pnl"] == 94.5

    def test_record_includes_all_expected_fields(self, ledger: TradeLedger, ledger_path: str) -> None:
        ledger.record(**_sample_trade_kwargs())
        entries = ledger.read_all()
        assert len(entries) == 1
        entry = entries[0]
        expected_keys = {
            "closed_at", "closed_at_iso", "symbol", "side", "entry_price",
            "exit_price", "qty", "gross_pnl", "fee", "funding_cost",
            "net_pnl", "exit_reason", "strategy", "open_time", "hold_seconds",
        }
        assert expected_keys.issubset(entry.keys())

    def test_record_computes_hold_seconds(self, ledger: TradeLedger) -> None:
        open_time = time.time() - 120  # 2 minutos atrás
        ledger.record(**_sample_trade_kwargs(open_time=open_time))
        entries = ledger.read_all()
        assert entries[0]["hold_seconds"] == pytest.approx(120, abs=2)

    def test_record_without_open_time_has_none_hold_seconds(self, ledger: TradeLedger) -> None:
        ledger.record(**_sample_trade_kwargs(open_time=None))
        entries = ledger.read_all()
        assert entries[0]["hold_seconds"] is None

    def test_multiple_records_are_appended_not_overwritten(self, ledger: TradeLedger) -> None:
        ledger.record(**_sample_trade_kwargs(symbol="BTC/USDC"))
        ledger.record(**_sample_trade_kwargs(symbol="ETH/USDC"))
        ledger.record(**_sample_trade_kwargs(symbol="SOL/USDC"))

        entries = ledger.read_all()
        assert len(entries) == 3
        symbols = [e["symbol"] for e in entries]
        assert symbols == ["BTC/USDC", "ETH/USDC", "SOL/USDC"]

    def test_record_close_external_reason(self, ledger: TradeLedger) -> None:
        ledger.record(**_sample_trade_kwargs(exit_reason="close_external"))
        entries = ledger.read_all()
        assert entries[0]["exit_reason"] == "close_external"

    def test_record_negative_pnl(self, ledger: TradeLedger) -> None:
        ledger.record(**_sample_trade_kwargs(
            exit_price=49000.0, gross_pnl=-100.0, net_pnl=-105.5, exit_reason="stop_loss",
        ))
        entries = ledger.read_all()
        assert entries[0]["net_pnl"] == -105.5

    def test_record_does_not_raise_on_internal_error(self, ledger: TradeLedger, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        FIX documentado no módulo: falha no ledger NUNCA deve interromper
        o fluxo de trading — record() deve engolir a exceção e apenas logar.
        """
        def broken_open(*args: Any, **kwargs: Any) -> Any:
            raise OSError("disco cheio (simulado)")

        monkeypatch.setattr("builtins.open", broken_open)

        # Não deve levantar exceção mesmo com falha de I/O.
        ledger.record(**_sample_trade_kwargs())


class TestTradeLedgerReadAll:
    """Testes para read_all()."""

    def test_read_all_empty_when_file_missing(self, ledger: TradeLedger) -> None:
        assert ledger.read_all() == []

    def test_read_all_preserves_order(self, ledger: TradeLedger) -> None:
        for i in range(5):
            ledger.record(**_sample_trade_kwargs(symbol=f"COIN{i}/USDC"))
        entries = ledger.read_all()
        assert [e["symbol"] for e in entries] == [f"COIN{i}/USDC" for i in range(5)]

    def test_read_all_skips_corrupted_lines(self, ledger: TradeLedger, ledger_path: str) -> None:
        """Uma linha corrompida no meio do arquivo não deve invalidar as
        demais — apenas é ignorada com warning."""
        ledger.record(**_sample_trade_kwargs(symbol="BTC/USDC"))
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write("{isso nao e json valido\n")
        ledger.record(**_sample_trade_kwargs(symbol="ETH/USDC"))

        entries = ledger.read_all()
        assert len(entries) == 2
        assert entries[0]["symbol"] == "BTC/USDC"
        assert entries[1]["symbol"] == "ETH/USDC"

    def test_read_all_skips_blank_lines(self, ledger: TradeLedger, ledger_path: str) -> None:
        ledger.record(**_sample_trade_kwargs())
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write("\n\n")
        entries = ledger.read_all()
        assert len(entries) == 1


class TestTradeLedgerSummary:
    """Testes para summary() — agregação simples."""

    def test_summary_empty_ledger(self, ledger: TradeLedger) -> None:
        s = ledger.summary()
        assert s == {"total_trades": 0, "total_net_pnl": 0.0, "win_rate_pct": 0.0}

    def test_summary_all_wins(self, ledger: TradeLedger) -> None:
        ledger.record(**_sample_trade_kwargs(net_pnl=100.0))
        ledger.record(**_sample_trade_kwargs(net_pnl=50.0))
        s = ledger.summary()
        assert s["total_trades"] == 2
        assert s["total_net_pnl"] == 150.0
        assert s["win_rate_pct"] == 100.0

    def test_summary_mixed_results(self, ledger: TradeLedger) -> None:
        ledger.record(**_sample_trade_kwargs(net_pnl=100.0))
        ledger.record(**_sample_trade_kwargs(net_pnl=-40.0))
        ledger.record(**_sample_trade_kwargs(net_pnl=-10.0))
        ledger.record(**_sample_trade_kwargs(net_pnl=30.0))

        s = ledger.summary()
        assert s["total_trades"] == 4
        assert s["total_net_pnl"] == pytest.approx(80.0)
        assert s["win_rate_pct"] == 50.0

    def test_summary_zero_pnl_counts_as_win(self, ledger: TradeLedger) -> None:
        """net_pnl == 0.0 é contado como 'win' (>= 0) — consistente com a
        convenção usada em PositionManager.record_close()."""
        ledger.record(**_sample_trade_kwargs(net_pnl=0.0))
        s = ledger.summary()
        assert s["win_rate_pct"] == 100.0


class TestTradeLedgerConcurrency:
    """Testes de thread-safety — múltiplas threads gravando simultaneamente."""

    def test_concurrent_writes_do_not_corrupt_file(self, ledger: TradeLedger) -> None:
        n_threads = 10
        n_records_per_thread = 5

        def worker(thread_id: int) -> None:
            for i in range(n_records_per_thread):
                ledger.record(**_sample_trade_kwargs(symbol=f"T{thread_id}-{i}/USDC"))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = ledger.read_all()
        assert len(entries) == n_threads * n_records_per_thread
        # Todas as linhas devem ser JSON válido (nenhuma entrelaçada/corrompida)
        symbols = {e["symbol"] for e in entries}
        assert len(symbols) == n_threads * n_records_per_thread


class TestGetLedgerSingleton:
    """Testes para o factory get_ledger()."""

    def test_returns_same_instance(self, tmp_path: Any) -> None:
        # Reset do singleton para isolamento do teste
        import crypto_bot_core.trade_ledger as ledger_mod
        ledger_mod._ledger_instance = None

        path = str(tmp_path / "singleton_test.jsonl")
        l1 = get_ledger(path)
        l2 = get_ledger(path)
        assert l1 is l2

        ledger_mod._ledger_instance = None  # cleanup para não vazar entre testes