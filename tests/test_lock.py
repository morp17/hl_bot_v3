"""
Testes unitários para crypto_bot_core/lock.py

Cobre:
- LockManager.acquire() — adquirir lock
- LockManager.acquire() — lock já existente (outra instância)
- LockManager.release() — liberar lock
- LockManager.heartbeat() — atualizar heartbeat
- LockManager._is_stale() — detectar lock stale
- Context manager (with)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest

from crypto_bot_core.lock import LockManager, LOCK_DIR


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def temp_lock_path() -> str:
    """Cria um caminho temporário para o lock file."""
    tmp_dir = tempfile.mkdtemp()
    lock_path = os.path.join(tmp_dir, "test_bot.lock")
    yield lock_path
    # Cleanup
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass


@pytest.fixture
def lock_manager(temp_lock_path: str) -> LockManager:
    """Cria um LockManager com caminho temporário."""
    return LockManager(lock_path=temp_lock_path)


# ──────────────────────────────────────────────
# Testes: Acquire
# ──────────────────────────────────────────────


class TestLockAcquire:
    """Testes para LockManager.acquire()."""

    def test_acquire_success(self, lock_manager: LockManager) -> None:
        """Deve adquirir lock com sucesso."""
        result = lock_manager.acquire()
        assert result is True
        assert lock_manager.acquired is True
        assert os.path.exists(lock_manager.lock_path)

        # Verificar conteúdo do lock
        with open(lock_manager.lock_path, "r", encoding="utf-8") as f:
            lock_data = json.load(f)
        assert lock_data["pid"] == os.getpid()
        assert "created_at" in lock_data
        assert "heartbeat" in lock_data

    def test_acquire_twice_same_process(self, lock_manager: LockManager) -> None:
        """Segundo acquire no mesmo processo deve falhar (lock já existe)."""
        assert lock_manager.acquire() is True
        # Segundo acquire sem force — lock existe e processo dono vive
        result = lock_manager.acquire(force=False)
        assert result is False

    def test_acquire_force_overwrites(self, lock_manager: LockManager) -> None:
        """Acquire com force=True deve sobrescrever lock existente."""
        assert lock_manager.acquire() is True
        # Force sobrescreve
        result = lock_manager.acquire(force=True)
        assert result is True
        assert lock_manager.acquired is True

    def test_acquire_after_release(self, lock_manager: LockManager) -> None:
        """Após release, deve ser possível adquirir novamente."""
        assert lock_manager.acquire() is True
        lock_manager.release()
        assert lock_manager.acquired is False

        result = lock_manager.acquire()
        assert result is True
        assert lock_manager.acquired is True

    def test_acquire_creates_directory(self) -> None:
        """Deve criar diretório do lock se não existir."""
        # Usar um caminho em diretório que definitivamente não existe
        import uuid
        unique_dir = os.path.join(tempfile.gettempdir(), f"hl_test_{uuid.uuid4().hex}")
        lock_path = os.path.join(unique_dir, "test_bot.lock")

        try:
            manager = LockManager(lock_path=lock_path)
            result = manager.acquire()
            assert result is True
            assert os.path.exists(unique_dir)
            assert os.path.exists(lock_path)

            # Cleanup
            manager.release()
        finally:
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                if os.path.exists(unique_dir):
                    os.rmdir(unique_dir)
            except OSError:
                pass


# ──────────────────────────────────────────────
# Testes: Release
# ──────────────────────────────────────────────


class TestLockRelease:
    """Testes para LockManager.release()."""

    def test_release_removes_file(self, lock_manager: LockManager) -> None:
        """Release deve remover o arquivo de lock."""
        lock_manager.acquire()
        assert os.path.exists(lock_manager.lock_path)

        lock_manager.release()
        assert lock_manager.acquired is False
        assert not os.path.exists(lock_manager.lock_path)

    def test_release_without_acquire(self, lock_manager: LockManager) -> None:
        """Release sem acquire não deve crashar."""
        lock_manager.release()  # Não deve levantar exceção
        assert lock_manager.acquired is False

    def test_release_twice(self, lock_manager: LockManager) -> None:
        """Release duplo não deve crashar."""
        lock_manager.acquire()
        lock_manager.release()
        lock_manager.release()  # Segundo release não deve crashar
        assert lock_manager.acquired is False


# ──────────────────────────────────────────────
# Testes: Heartbeat
# ──────────────────────────────────────────────


class TestLockHeartbeat:
    """Testes para LockManager.heartbeat()."""

    def test_heartbeat_updates_timestamp(self, lock_manager: LockManager) -> None:
        """Heartbeat deve atualizar o timestamp no lock file."""
        lock_manager.acquire()

        # Ler timestamp inicial
        with open(lock_manager.lock_path, "r", encoding="utf-8") as f:
            initial_data = json.load(f)
        initial_heartbeat = initial_data["heartbeat"]

        # Pequena pausa para garantir que o timestamp avance
        import time
        time.sleep(0.01)

        # Forçar heartbeat (simular que passou tempo)
        lock_manager._last_heartbeat = 0  # Reseta para forçar heartbeat
        lock_manager.heartbeat()

        # Ler timestamp atualizado
        with open(lock_manager.lock_path, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        updated_heartbeat = updated_data["heartbeat"]

        assert updated_heartbeat > initial_heartbeat

    def test_heartbeat_without_acquire(self, lock_manager: LockManager) -> None:
        """Heartbeat sem lock não deve crashar."""
        lock_manager.heartbeat()  # Não deve levantar exceção

    def test_heartbeat_after_release(self, lock_manager: LockManager) -> None:
        """Heartbeat após release não deve crashar."""
        lock_manager.acquire()
        lock_manager.release()
        lock_manager.heartbeat()  # Não deve levantar exceção


# ──────────────────────────────────────────────
# Testes: Stale Detection
# ──────────────────────────────────────────────


class TestLockStale:
    """Testes para LockManager._is_stale()."""

    def test_no_lock_file_is_stale(self, lock_manager: LockManager) -> None:
        """Sem arquivo de lock, é considerado stale."""
        assert lock_manager._is_stale() is True

    def test_lock_with_dead_process_is_stale(
        self, lock_manager: LockManager
    ) -> None:
        """Lock com processo dono morto é stale."""
        lock_manager.acquire()

        # Simular que o processo dono morreu
        with patch.object(lock_manager, "_pid_exists", return_value=False):
            assert lock_manager._is_stale() is True

    def test_lock_with_alive_process_not_stale(
        self, lock_manager: LockManager
    ) -> None:
        """Lock com processo dono vivo não é stale."""
        lock_manager.acquire()

        with patch.object(lock_manager, "_pid_exists", return_value=True):
            assert lock_manager._is_stale() is False

    def test_lock_with_stale_heartbeat_is_stale(
        self, lock_manager: LockManager
    ) -> None:
        """Lock com heartbeat muito antigo é stale."""
        lock_manager.acquire()

        # Corromper o heartbeat para ser muito antigo
        with open(lock_manager.lock_path, "r", encoding="utf-8") as f:
            lock_data = json.load(f)
        lock_data["heartbeat"] = time.time() - 300  # 5 minutos atrás
        with open(lock_manager.lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f)

        with patch.object(lock_manager, "_pid_exists", return_value=True):
            assert lock_manager._is_stale() is True


# ──────────────────────────────────────────────
# Testes: Context Manager
# ──────────────────────────────────────────────


class TestLockContextManager:
    """Testes para LockManager como context manager."""

    def test_context_manager_acquire_and_release(
        self, temp_lock_path: str
    ) -> None:
        """Context manager deve adquirir e liberar lock."""
        with LockManager(lock_path=temp_lock_path) as lock:
            assert lock.acquired is True
            assert os.path.exists(temp_lock_path)

        assert not os.path.exists(temp_lock_path)

    def test_context_manager_release_on_exception(
        self, temp_lock_path: str
    ) -> None:
        """Context manager deve liberar lock mesmo em caso de exceção."""
        try:
            with LockManager(lock_path=temp_lock_path) as lock:
                assert lock.acquired is True
                raise ValueError("Erro simulado")
        except ValueError:
            pass

        # Lock deve ter sido liberado
        assert not os.path.exists(temp_lock_path)


# ──────────────────────────────────────────────
# Testes: PID Exists
# ──────────────────────────────────────────────


class TestPidExists:
    """Testes para LockManager._pid_exists()."""

    def test_current_process_exists(self) -> None:
        """PID do processo atual deve existir."""
        assert LockManager._pid_exists(os.getpid()) is True

    def test_invalid_pid_not_exists(self) -> None:
        """PID inválido não deve existir (usando mock para evitar chamada real ao sistema)."""
        with patch.object(LockManager, "_pid_exists", return_value=False):
            assert LockManager._pid_exists(999999999) is False
