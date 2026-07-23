"""
Servidor HTTP de Healthcheck — Hyperliquid Production Bot v3.0
==============================================================
Endpoint leve na porta 8081 para healthcheck do Docker.
Não depende do dashboard (FastAPI) — roda em modo live também.

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Logs estruturados
- Thread separada para não bloquear o loop principal
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

from loguru import logger as log


# ──────────────────────────────────────────────
# Estado compartilhado do healthcheck
# ──────────────────────────────────────────────

_health_state: Dict[str, Any] = {
    "status": "starting",
    "last_step": 0,
    "positions": 0,
    "errors": 0,
}


def update_health(state: Dict[str, Any]) -> None:
    """Atualiza o estado compartilhado do healthcheck."""
    global _health_state
    _health_state.update(state)


# ──────────────────────────────────────────────
# Handler HTTP
# ──────────────────────────────────────────────


class HealthHandler(BaseHTTPRequestHandler):
    """Handler minimalista para healthcheck."""

    def do_GET(self) -> None:
        """Responde a requisições GET."""
        try:
            if self.path == "/health" or self.path == "/":
                status_code = 200 if _health_state.get("status") == "online" else 503
                body = json.dumps(_health_state).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suprime logs do HTTP server para não poluir."""
        pass


# ──────────────────────────────────────────────
# Server Thread
# ──────────────────────────────────────────────


class HealthServer:
    """Servidor HTTP de healthcheck em thread separada."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        """
        Args:
            host: Endereço para bind.
            port: Porta para healthcheck.
        """
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Inicia o servidor em thread separada."""
        try:
            self._server = HTTPServer((self.host, self.port), HealthHandler)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="health-server",
            )
            self._thread.start()
            log.info(f"[HEALTH] Servidor HTTP em http://{self.host}:{self.port}")
        except Exception as e:
            log.error(f"[HEALTH] Erro ao iniciar servidor: {e}")

    def stop(self) -> None:
        """Para o servidor."""
        try:
            if self._server:
                self._server.shutdown()
                log.info("[HEALTH] Servidor parado")
        except Exception as e:
            log.debug(f"[HEALTH] Erro ao parar servidor: {e}")
