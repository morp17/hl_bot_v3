# ============================================================
# Hyperliquid Production Bot v3.0 — Dockerfile
# ============================================================
# Estágio 1: Build
FROM python:3.12-slim AS builder

WORKDIR /app

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Estágio 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Timezone
ENV TZ=America/Sao_Paulo
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Cria usuário não-root para segurança
RUN useradd -m -u 1000 botuser && \
    mkdir -p /app/data /app/logs && \
    chown -R botuser:botuser /app

# Copia dependências do estágio builder
COPY --from=builder /root/.local /home/botuser/.local

# Copia código da aplicação
COPY . .

# Ajusta permissões
RUN chown -R botuser:botuser /app

# Ajusta PATH para incluir pacotes do user
ENV PATH=/home/botuser/.local/bin:$PATH

# Usuário não-root
USER botuser

# Portas
EXPOSE 8080 9090 8081

# FIX (auditoria — item 8): antes o HEALTHCHECK só confirmava que o
# PROCESSO existia (pgrep), não que o bot estava operacional (conectado
# à exchange, sem erros consecutivos, ciclo rodando). O próprio
# health_server.py já expõe /health com status estruturado
# ("online"/"degraded"/"error" — ver crypto_bot_core/health_server.py e
# HyperliquidBot.step()/update_health()). Agora o HEALTHCHECK consome
# esse endpoint real via curl, retornando código de saída != 0 (e
# portanto Docker marcando o container como "unhealthy") sempre que
# /health não responder 200 — o que só ocorre quando
# _health_state["status"] == "online".
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Comando padrão
CMD ["python", "main.py"]