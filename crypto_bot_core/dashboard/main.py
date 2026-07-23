"""
Módulo de Dashboard Web — Hyperliquid Production Bot v3.0
==========================================================
Servidor web local para monitoramento do bot via FastAPI.

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Logs estruturados
"""

from __future__ import annotations

import os
from typing import Any, Dict

import jinja2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from loguru import logger as log
import uvicorn

from ..config import BotConfig


# ──────────────────────────────────────────────
# Templates (Jinja2 direto, sem Starlette)
# ──────────────────────────────────────────────

_templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_templates_dir),
    autoescape=True,
)


def _render_template(name: str, **context: Any) -> str:
    """Renderiza template Jinja2."""
    template = _jinja_env.get_template(name)
    return template.render(**context)


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(title="Hyperliquid Bot Dashboard")

# Estado global do dashboard (compartilhado com o bot)
_dashboard_state: Dict[str, Any] = {
    "status": "offline",
    "positions": [],
    "metrics": {},
    "last_update": 0,
}


def update_state(state: Dict[str, Any]) -> None:
    """Atualiza o estado compartilhado do dashboard."""
    global _dashboard_state
    _dashboard_state.update(state)


# ──────────────────────────────────────────────
# Rotas
# ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Página principal do dashboard."""
    html = _render_template(
        "index.html",
        request=request,
        status=_dashboard_state.get("status", "offline"),
        positions=_dashboard_state.get("positions", []),
        metrics=_dashboard_state.get("metrics", {}),
    )
    return HTMLResponse(content=html)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Health check da API."""
    return {"status": "ok", "state": _dashboard_state}


@app.get("/api/metrics")
async def metrics() -> Dict[str, Any]:
    """Métricas em tempo real."""
    return _dashboard_state.get("metrics", {})


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────


def start_dashboard(cfg: BotConfig) -> None:
    """
    Inicia o servidor do dashboard.

    Args:
        cfg: Configuração do bot.
    """
    try:
        host = cfg.dashboard.host
        port = cfg.dashboard.port

        log.info(f"Dashboard iniciando em http://{host}:{port}")
        log.info(f"Credenciais: {cfg.dashboard.user} / {cfg.dashboard.password}")

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
    except Exception as e:
        log.error(f"Erro ao iniciar dashboard: {e}")
        raise
