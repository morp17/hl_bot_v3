"""
Módulo de Lock File / Heartbeat — Hyperliquid Production Bot v3.0
=================================================================
Previne que duas instâncias do bot rodem simultaneamente com a
mesma API key, evitando race conditions e ordens conflitantes.

Usa um lock file no sistema com PID e heartbeat timestamp.
Se o processo dono do lock não existir mais, o lock é considerado
stale e pode ser sobrescrito.

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Optional

from loguru import logger as log


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

LOCK_DIR = os.path.join(tempfile.gettempdir(), "hl_bot")
LOCK_FILE = os.path.join(LOCK_DIR, "bot.lock")
HEARTBEAT_INTERVAL = 30  # segundos entre heartbeats
STALE_TIMEOUT = 120  # segundos sem heartbeat = lock stale


# ──────────────────────────────────────────────
# LockManager
# ──────────────────────────────────────────────


class LockManager:
    """Gerenciador de lock file para prevenir múltiplas instâncias.

    Cria um arquivo de lock com PID e timestamp. Se o lock já existir
    e o processo dono ainda estiver vivo, impede a inicialização.
    Se o processo dono morreu, o lock é considerado stale e substituído.

    Attributes:
        lock_path: Caminho do arquivo de lock.
        acquired: Se o lock foi adquirido com sucesso.
        _last_heartbeat: Timestamp do último heartbeat.
    """

    def __init__(self, lock_path: str = LOCK_FILE) -> None:
        """Inicializa o LockManager.

        Args:
            lock_path: Caminho do arquivo de lock.
        """
        self.lock_path = lock_path
        self.acquired = False
        self._last_heartbeat: float = 0.0

    def acquire(self, force: bool = False) -> bool:
        """Tenta adquirir o lock.

        Args:
            force: Se True, sobrescreve lock existente sem verificar.

        Returns:
            bool: True se o lock foi adquirido, False se outra instância
                  está rodando.

        Raises:
            OSError: Se não for possível criar o diretório do lock.
        """
        try:
            # Criar diretório do lock se não existir
            lock_dir = os.path.dirname(self.lock_path)
            if lock_dir:
                os.makedirs(lock_dir, exist_ok=True)

            if not force and os.path.exists(self.lock_path):
                # Verificar se o lock é stale ou se o processo dono ainda vive
                if not self._is_stale():
                    log.error(
                        f"[LOCK] Lock file existe em {self.lock_path} e "
                        f"processo dono ainda está vivo. "
                        f"Outra instância já está rodando!"
                    )
                    return False

                # Lock stale — logar warning e sobrescrever
                log.warning(
                    f"[LOCK] Lock file stale encontrado em {self.lock_path}. "
                    f"Sobrescrevendo..."
                )

            # Escrever lock file
            lock_data = {
                "pid": os.getpid(),
                "created_at": time.time(),
                "heartbeat": time.time(),
                "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            }

            with open(self.lock_path, "w", encoding="utf-8") as f:
                json.dump(lock_data, f, indent=2)

            self.acquired = True
            self._last_heartbeat = time.time()
            log.info(
                f"[LOCK] Lock adquirido: PID={lock_data['pid']} "
                f"em {self.lock_path}"
            )
            return True

        except OSError as e:
            log.error(f"[LOCK] Erro de sistema ao adquirir lock: {e}")
            return False
        except Exception as e:
            log.error(f"[LOCK] Erro ao adquirir lock: {e}")
            return False

    def release(self) -> None:
        """Libera o lock file.

        Remove o arquivo de lock se ele pertence a este processo.
        """
        try:
            if not self.acquired:
                return

            if os.path.exists(self.lock_path):
                # Verificar se o lock ainda é nosso antes de remover
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        lock_data = json.load(f)
                    if lock_data.get("pid") == os.getpid():
                        os.remove(self.lock_path)
                        log.info(f"[LOCK] Lock liberado: {self.lock_path}")
                    else:
                        log.debug(
                            f"[LOCK] Lock pertence a outro processo "
                            f"(PID {lock_data.get('pid')}), não removendo"
                        )
                except (json.JSONDecodeError, OSError):
                    # Lock corrompido ou inacessível — tentar remover mesmo assim
                    try:
                        os.remove(self.lock_path)
                    except OSError:
                        pass

            self.acquired = False

        except Exception as e:
            log.debug(f"[LOCK] Erro ao liberar lock: {e}")

    def heartbeat(self) -> None:
        """Atualiza o heartbeat no lock file.

        Deve ser chamado periodicamente para indicar que o processo
        dono do lock ainda está vivo.
        """
        try:
            if not self.acquired:
                return

            now = time.time()
            if now - self._last_heartbeat < HEARTBEAT_INTERVAL:
                return

            if os.path.exists(self.lock_path):
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        lock_data = json.load(f)

                    # Só atualiza se o lock ainda é nosso
                    if lock_data.get("pid") == os.getpid():
                        lock_data["heartbeat"] = now
                        with open(self.lock_path, "w", encoding="utf-8") as f:
                            json.dump(lock_data, f, indent=2)
                        self._last_heartbeat = now
                        log.debug(f"[LOCK] Heartbeat atualizado: {now}")
                except (json.JSONDecodeError, OSError):
                    log.debug("[LOCK] Erro ao atualizar heartbeat")
            else:
                # Lock foi removido externamente — tentar readquirir
                log.warning("[LOCK] Lock file não encontrado para heartbeat")
                self.acquired = False

        except Exception as e:
            log.debug(f"[LOCK] Erro no heartbeat: {e}")

    def _is_stale(self) -> bool:
        """Verifica se o lock existente é stale (processo dono morreu).

        Returns:
            bool: True se o lock é stale (pode ser sobrescrito).
        """
        try:
            if not os.path.exists(self.lock_path):
                return True

            with open(self.lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)

            pid = lock_data.get("pid")
            heartbeat = lock_data.get("heartbeat", 0)
            now = time.time()

            # Verificar se o processo dono ainda existe
            if pid and self._pid_exists(pid):
                # Processo existe — verificar heartbeat
                if now - heartbeat > STALE_TIMEOUT:
                    log.warning(
                        f"[LOCK] Processo PID={pid} existe mas heartbeat "
                        f"está stale ({now - heartbeat:.0f}s sem atualização)"
                    )
                    return True
                return False

            # Processo não existe mais — lock é stale
            log.info(f"[LOCK] Processo dono PID={pid} não existe mais")
            return True

        except (json.JSONDecodeError, OSError, FileNotFoundError):
            # Lock corrompido ou não encontrado — considerar stale
            return True
        except Exception as e:
            log.debug(f"[LOCK] Erro ao verificar stale: {e}")
            return True  # fail-safe: permite sobrescrever

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        """Verifica se um processo com o PID dado existe.

        Args:
            pid: PID do processo.

        Returns:
            bool: True se o processo existe.
        """
        try:
            if hasattr(os, "kill"):
                os.kill(pid, 0)
                return True
            # Windows fallback
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except (OSError, ImportError, AttributeError):
            # No Unix: ESRCH = processo não existe
            return False
        except Exception:
            return False

    def __enter__(self) -> "LockManager":
        """Context manager: acquire na entrada."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager: release na saída."""
        self.release()
