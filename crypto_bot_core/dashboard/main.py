"""
Módulo de Dashboard Web — Hyperliquid Production Bot v3.0
==========================================================
Servidor web local para monitoramento e configuração do bot via FastAPI.

Roda como thread em background DENTRO do processo do bot live (mesmo
padrão do HealthServer), não mais como modo CLI separado — isso é
necessário porque _dashboard_state é uma variável de módulo em memória:
rodar o dashboard em um processo separado do bot fazia com que o estado
nunca fosse compartilhado, e a página sempre mostrasse "offline".

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Logs estruturados
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import jinja2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from loguru import logger as log
import uvicorn

from ..config import BotConfig


# ──────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────

_templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_templates_dir),
    autoescape=True,
)


def _render_template(name: str, **context: Any) -> str:
    template = _jinja_env.get_template(name)
    return template.render(**context)


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(title="Hyperliquid Bot Dashboard")

_dashboard_state: Dict[str, Any] = {
    "status": "offline",
    "positions": [],
    "metrics": {},
    "last_update": 0,
}

# Referência ao BotConfig ativo do processo — setada por start_dashboard_thread().
# Necessária para que o endpoint de edição altere a config REAL em uso pelo
# bot (não uma cópia), e para persistir via save_json.
_active_cfg: Optional[BotConfig] = None
_config_lock = threading.Lock()


def update_state(state: Dict[str, Any]) -> None:
    """Atualiza o estado compartilhado do dashboard."""
    global _dashboard_state
    _dashboard_state.update(state)


# ──────────────────────────────────────────────
# Schema de edição de config (campos seguros para hot-reload)
# ──────────────────────────────────────────────

class RiskUpdateRequest(BaseModel):
    """
    Campos de risco editáveis via dashboard em tempo de execução.

    NÃO inclui: credenciais, símbolos, leverage (requer chamada à
    exchange, não é "hot"), timeframe/strategy globais (trocar em
    runtime tem efeitos colaterais em indicadores que exigem reprocessar
    o DataFrame do zero — mais seguro exigir restart para esses).
    """
    stop_loss_pct: Optional[float] = Field(None, ge=0.1, le=20.0)
    take_profit_pct: Optional[float] = Field(None, ge=0.1, le=50.0)
    max_drawdown_pct: Optional[float] = Field(None, ge=1.0, le=50.0)
    daily_loss_limit_pct: Optional[float] = Field(None, ge=0.5, le=20.0)
    max_open_trades: Optional[int] = Field(None, ge=1, le=20)
    max_consecutive_losses: Optional[int] = Field(None, ge=1, le=10)
    enabled_strategies: Optional[str] = None


# ──────────────────────────────────────────────
# Rotas — leitura
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
        risk=_active_cfg.risk.model_dump() if _active_cfg else {},
        enabled_strategies=_active_cfg.enabled_strategies if _active_cfg else "",
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


@app.get("/api/config")
async def get_config_endpoint() -> Dict[str, Any]:
    """Retorna a configuração de risco ATUALMENTE em uso pelo bot."""
    if _active_cfg is None:
        raise HTTPException(status_code=503, detail="Config não disponível (bot não iniciado)")
    return {
        "risk": _active_cfg.risk.model_dump(),
        "enabled_strategies": _active_cfg.enabled_strategies,
    }


# ──────────────────────────────────────────────
# Rota — escrita (item novo: torna save_json/load_json úteis de fato)
# ──────────────────────────────────────────────


@app.post("/api/config")
async def update_config_endpoint(update: RiskUpdateRequest) -> JSONResponse:
    """
    Aplica alterações de risco em tempo real e persiste em bot_config.json.

    Fluxo:
    1. Valida os campos recebidos (Pydantic já rejeita fora de range).
    2. Aplica em memória em _active_cfg.risk (efeito IMEDIATO no bot
       ao vivo, sem restart — a próxima chamada de can_open()/calc_stops()
       já usa o novo valor).
    3. Persiste em bot_config.json via BotConfig.save_json(), para que
       o valor sobreviva a um restart do processo.

    Campos não incluídos no request permanecem inalterados.
    """
    global _active_cfg

    if _active_cfg is None:
        raise HTTPException(status_code=503, detail="Config não disponível (bot não iniciado)")

    try:
        with _config_lock:
            updates = update.model_dump(exclude_unset=True, exclude_none=True)

            if not updates:
                return JSONResponse({"status": "no_changes"}, status_code=200)

            applied: Dict[str, Any] = {}

            for key, value in updates.items():
                if key == "enabled_strategies":
                    _active_cfg.enabled_strategies = value
                    applied[key] = value
                elif hasattr(_active_cfg.risk, key):
                    setattr(_active_cfg.risk, key, value)
                    applied[key] = value
                else:
                    log.warning(f"[DASHBOARD] Campo desconhecido ignorado: {key}")

            # Persistir em disco para sobreviver a restart
            _active_cfg.save_json("bot_config.json")

            log.info(f"[DASHBOARD] Config atualizada via API: {applied}")

        return JSONResponse({"status": "applied", "updated_fields": applied}, status_code=200)

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"[DASHBOARD] Erro ao aplicar config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Startup — thread em background (substitui o antigo start_dashboard bloqueante)
# ──────────────────────────────────────────────


def start_dashboard_thread(cfg: BotConfig) -> Optional[threading.Thread]:
    """
    Inicia o dashboard como thread em background DENTRO do processo
    do bot live — mesmo padrão de HealthServer.

    FIX arquitetural: a versão anterior (start_dashboard) rodava
    uvicorn.run() de forma bloqueante e só era invocada via
    `--mode dashboard`, um processo Python SEPARADO do bot live. Como
    _dashboard_state é uma variável de módulo em memória, o processo do
    dashboard nunca via as atualizações feitas pelo processo do bot —
    a página sempre mostrava estado vazio/offline. Rodar como thread no
    mesmo processo corrige isso na raiz.

    Args:
        cfg: Configuração ativa do bot (será referenciada por
            _active_cfg para leitura/escrita via /api/config).

    Returns:
        Thread do servidor, ou None se dashboard_enabled=False ou erro.
    """
    global _active_cfg

    try:
        if not cfg.dashboard_enabled:
            log.info("[DASHBOARD] Desabilitado via config (dashboard_enabled=false)")
            return None

        _active_cfg = cfg

        def _run() -> None:
            try:
                log.info(
                    f"[DASHBOARD] Iniciando em "
                    f"http://{cfg.dashboard.host}:{cfg.dashboard.port} "
                    f"(thread em background)"
                )
                uvicorn.run(
                    app,
                    host=cfg.dashboard.host,
                    port=cfg.dashboard.port,
                    log_level="warning",  # evita duplicar verbosidade do loguru
                )
            except Exception as e:
                log.error(f"[DASHBOARD] Erro na thread do servidor: {e}")

        thread = threading.Thread(target=_run, daemon=True, name="dashboard-server")
        thread.start()
        return thread

    except Exception as e:
        log.error(f"[DASHBOARD] Erro ao iniciar dashboard: {e}")
        return None


def start_dashboard(cfg: BotConfig) -> None:
    """
    MANTIDO POR COMPATIBILIDADE apenas para uso standalone/debug
    (ex: `python main.py --mode dashboard` continua funcionando, mas
    mostrará estado vazio, pois não é o mesmo processo do bot live —
    use apenas para inspecionar a UI/rotas isoladamente, não para
    monitorar um bot ao vivo real).
    """
    try:
        log.warning(
            "[DASHBOARD] Rodando em modo standalone (--mode dashboard). "
            "Este modo NÃO reflete o estado do bot ao vivo — para "
            "monitoramento real, o dashboard já sobe automaticamente "
            "como thread dentro do processo --mode live."
        )
        uvicorn.run(app, host=cfg.dashboard.host, port=cfg.dashboard.port, log_level="info")
    except Exception as e:
        log.error(f"Erro ao iniciar dashboard: {e}")
        raise