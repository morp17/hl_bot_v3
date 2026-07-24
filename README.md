# Hyperliquid Production Bot v3.0

Bot de trading automatizado para **Hyperliquid DEX** com 7 estratégias, proteção de capital em 4 níveis, staking HYPE, dashboard local e deploy Docker.

## 📋 Índice

- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Uso](#-uso)
- [Estratégias](#-estratégias)
- [Proteção de Capital](#-proteção-de-capital)
- [Staking](#-staking)
- [Notificações](#-notificações)
- [Dashboard](#-dashboard)
- [Docker](#-docker)
- [Testes](#-testes)
- [Estrutura do Projeto](#-estrutura-do-projeto)

---

## 🔧 Requisitos

- **Python 3.12+**
- **Pip** (gerenciador de pacotes)
- **Conta na Hyperliquid** (testnet ou mainnet)
- **Carteira Ethereum** com private key

### Dependências

Todas as dependências estão listadas em [`requirements.txt`](hl_bot_v3/requirements.txt):

```
pydantic>=2.0
pydantic-settings>=2.0
ccxt>=4.0
hyperliquid-python-sdk>=0.0.1
pandas>=2.0
numpy>=1.24
loguru>=0.7
requests>=2.31
eth-account>=0.5
python-dotenv>=1.0
pytest>=8.0
```

---

## 🚀 Instalação

### 1. Clone o repositório

```bash
cd e:/BOTS/VPS/C_BOT/hl_bot_v3
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o arquivo `.env`

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas informações:

```ini
# Credenciais da Hyperliquid (testnet)
HYPERLIQUID_PRIVATE_KEY=0x_sua_private_key_aqui
HYPERLIQUID_ACCOUNT_ADDRESS=0x_seu_endereco_aqui
TESTNET=true

# Símbolos para trading
SYMBOLS=BTC/USDC,ETH/USDC,SOL/USDC

# Estratégia
STRATEGY=hybrid_regime
TIMEFRAME=1h

# Capital
CAPITAL_USD=1000
RISK_PER_TRADE_PCT=1.0
```

### 5. Verifique a instalação

```bash
python -m pytest tests/ -v
```

Todos os **271 testes** devem passar.

---

## ⚙️ Configuração

### Variáveis de Ambiente (`.env`)

O arquivo `.env` usa **prefixos** para organizar sub-modelos do Pydantic:

| Prefixo | Sub-modelo | Exemplo |
|---------|-----------|---------|
| *(raiz)* | `BotConfig` | `CAPITAL_USD=1000` |
| `RISK_` | `RiskConfig` | `RISK_STOP_LOSS_PCT=2.0` |
| `NOTIFICATIONS_` | `NotificationConfig` | `NOTIFICATIONS_TELEGRAM_BOT_TOKEN=...` |
| `STAKING_` | `StakingConfig` | `STAKING_ENABLED=false` |
| `DASHBOARD_` | `DashboardConfig` | `DASHBOARD_PORT=8080` |
| `MONITORING_` | `MonitoringConfig` | `MONITORING_PROMETHEUS_PORT=9090` |
| `BACKTEST_` | `BacktestConfig` | `BACKTEST_START_DATE=2024-01-01` |

### Parâmetros de Risco

| Variável | Default | Descrição |
|----------|---------|-----------|
| `RISK_MAX_OPEN_TRADES` | 3 | Máximo de trades simultâneos |
| `RISK_MAX_DRAWDOWN_PCT` | 15.0 | Drawdown máximo (%) |
| `RISK_DAILY_LOSS_LIMIT_PCT` | 5.0 | Limite de perda diária (%) |
| `RISK_MAX_CONSECUTIVE_LOSSES` | 3 | Perdas consecutivas máximas |
| `RISK_COOLDOWN_AFTER_LOSS_SEC` | 300 | Cooldown após perda (s) |
| `RISK_STOP_LOSS_PCT` | 2.0 | Stop Loss padrão (%) |
| `RISK_TAKE_PROFIT_PCT` | 5.0 | Take Profit padrão (%) |
| `RISK_TRAILING_STOP` | false | Habilitar trailing stop |
| `RISK_CIRCUIT_BREAKER_LOSS_PCT` | 15.0 | Perda que aciona circuit breaker (%) |

---

## 🎮 Uso

### Modo Live (Trading Real)

```bash
cd hl_bot_v3
python main.py --mode live --interval 60
```

- `--interval`: Intervalo entre ciclos em segundos (default: 60)
- O bot executa um loop infinito até `Ctrl+C`

### Modo Backtest

```bash
python main.py --mode backtest --symbol BTC/USDC
```

Executa o `BacktestEngine` sobre dados históricos reais buscados da
exchange. Para validação de robustez temporal (múltiplas janelas
sequenciais, sem reotimização de parâmetros entre elas), use
`BacktestEngine.run_walk_forward()` programaticamente — ainda não
exposto via CLI.

> ⚠️ **Leia antes de confiar nos resultados**: o motor aplica
> slippage configurável e checa SL/TP contra o range high/low de
> cada barra (não apenas o close), mas quando ambos SL e TP são
> tocados na mesma barra, o SL é assumido como o primeiro a ser
> atingido — uma premissa conservadora, não um fato garantido sem
> dados intrabar (tick/M1). Trate os resultados como estimativa,
> não como garantia de performance futura.

### Modo Dashboard

```bash
python main.py --mode dashboard
```

Inicia o servidor web local para monitoramento (porta 8080 por padrão).

### Parâmetros da CLI

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--mode` | `live` | Modo: `live`, `backtest`, `dashboard` |
| `--symbol` | `""` | Símbolo para backtest |
| `--interval` | `60` | Intervalo entre ciclos (s) |
| `--config` | `.env` | Caminho do arquivo de configuração |

---

## 📈 Estratégias

O bot implementa **7 estratégias** de trading, selecionáveis via `STRATEGY` no `.env`.

> ⚠️ **Nem todas estão liberadas para operar por padrão.** O controle
> `ENABLED_STRATEGIES` (ver `.env.example`) restringe quais estratégias
> podem gerar sinais reais. Por padrão, apenas `trend_follow`,
> `adaptive_trend` e `hybrid_regime` estão habilitadas — as demais
> (`mean_reversion`, `orderflow_delta`, `scalping_grid`,
> `funding_arbitrage`) mostraram expectância negativa ou insuficiente
> nos backtests de referência do repositório, ou dependem de dados
> (funding rate) que exigem configuração adicional. Alterar
> `STRATEGY=mean_reversion` sem também adicionar `mean_reversion` a
> `ENABLED_STRATEGIES` resulta no bot retornando `hold` permanentemente.

### 1. Trend Follow (`trend_follow`)
Segue tendência usando EMAs (9/21/200) + RSI + ADX.
- **Entrada long:** EMA fast > EMA slow, preço > EMA 200, RSI < 70, ADX > 25
- **Entrada short:** EMA fast < EMA slow, preço < EMA 200, RSI > 30, ADX > 25

### 2. Mean Reversion (`mean_reversion`)
Reversão à média com Bollinger Bands + RSI.
- **Entrada long:** Preço ≤ banda inferior, RSI < 30
- **Entrada short:** Preço ≥ banda superior, RSI > 70

### 3. Adaptive Trend (`adaptive_trend`)
Alterna entre trend following e range trading baseado no ADX.
- **ADX > 25:** Modo tendência (segue EMAs)
- **ADX < 20:** Modo range (RSI extremos)
- **ADX 20-25:** Modo híbrido (cruzamento de EMAs)

### 4. Hybrid Regime (`hybrid_regime`) — **Recomendada**
3 camadas de análise:
- **Layer 1:** Regime macro (EMAs 50/200 → bull/bear/sideways)
- **Layer 2:** VWAP sweep (preço varrendo VWAP com desvio padrão)
- **Layer 3:** SMC Structure (Break of Structure / Change of Character)

### 5. OrderFlow Delta (`orderflow_delta`)
Baseada em delta, CVD e divergências.
- **Entrada long:** Delta positivo, CVD > média, divergência bullish
- **Entrada short:** Delta negativo, CVD < média, divergência bearish

### 6. Scalping Grid (`scalping_grid`)
Grid trading para timeframes curtos (1m/5m).
- Cria níveis de compra/venda ao redor do preço
- Compra em suportes com RSI baixo, vende em resistências com RSI alto

### 7. Funding Arbitrage (`funding_arbitrage`)
Arbitragem de funding rate.
- **Funding muito positivo (+1%):** Vender (mercado comprado)
- **Funding muito negativo (-1%):** Comprar (mercado vendido)

---

## 🛡️ Proteção de Capital

Sistema em **4 níveis**:

### Nível 1 — Filtros de Mercado
- **Horário:** Opera apenas na janela configurada (`trade_hour_start_utc`/`end_utc`)
- **Spread:** Bloqueia se spread > `max_spread_pct`
- **Funding:** Bloqueia se funding rate > `max_funding_rate`

### Nível 2 — Drawdown e Perda
- **Drawdown máximo:** Pausa se drawdown > `max_drawdown_pct`
- **Perda diária:** Pausa se perda do dia > `daily_loss_limit_pct`
- **Perdas consecutivas:** Pausa após N perdas seguidas
- **Cooldown:** Aguarda X segundos após perda

### Nível 3 — Exposição
- **Exposição total:** Limita % do capital em posições
- **Posição individual:** Limita % do capital por trade

### Nível 4 — Circuit Breaker
- Aciona automaticamente se a perda desde o pico exceder o limite
- Pausa todas as operações por X segundos
- Requer reinicialização manual ou expiração do tempo

---

## 💰 Staking

O bot suporta staking de HYPE e operações com vaults.

### Configuração

```ini
STAKING_ENABLED=true
STAKING_VALIDATOR_ADDRESS=0x_endereco_do_validador
STAKING_STAKE_PCT=10
STAKING_VAULT_ADDRESS=0x_endereco_do_vault
STAKING_VAULT_DEPOSIT_PCT=5
STAKING_AUTO_COMPOUND=true
```

### Funcionalidades

- **Delegar stake:** Aloca HYPE para um validador
- **Remover stake:** Retira HYPE de um validador
- **Consultar recompensas:** Obtém histórico de recompensas
- **Vault transfer:** Deposita/retira fundos de vaults
- **Auto-compound:** Reinveste recompensas automaticamente

---

## 🔔 Notificações

Suporte a **3 canais** de notificação:

### Telegram

```ini
NOTIFICATIONS_TELEGRAM_BOT_TOKEN=seu_token_aqui
NOTIFICATIONS_TELEGRAM_CHAT_ID=seu_chat_id_aqui
```

### Discord

```ini
NOTIFICATIONS_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Email (SMTP)

```ini
NOTIFICATIONS_SMTP_HOST=smtp.gmail.com
NOTIFICATIONS_SMTP_PORT=587
NOTIFICATIONS_SMTP_USER=seu_email@gmail.com
NOTIFICATIONS_SMTP_PASSWORD=sua_senha
NOTIFICATIONS_EMAIL_FROM=bot@exemplo.com
NOTIFICATIONS_EMAIL_TO=admin@exemplo.com
```

### Eventos Notificados

- ✅ Abertura de posição
- ✅ Fechamento de posição (com PnL)
- 🚨 Erros críticos
- 📉 Drawdown excessivo
- 📊 Resumo diário

---

## 📊 Dashboard

O dashboard web local permite monitorar o bot em tempo real.

### Acesso

```
http://localhost:8080
```

### Credenciais Padrão

- **Usuário:** `admin`
- **Senha:** `changeme`

### Funcionalidades

- Visualização de posições abertas
- Gráficos de PnL
- Histórico de trades
- Métricas de performance
- Health check do bot

---

## 🐳 Docker

### Construir a imagem

```bash
cd hl_bot_v3
docker build -t hyperliquid-bot-v3 .
```

### Executar com Docker Compose

```bash
docker-compose up -d
```

O `docker-compose.yml` inclui:
- **Bot:** Serviço principal
- **Prometheus:** Coleta de métricas (porta 9090)
- **Grafana:** Visualização de métricas (porta 3000)

### Parar

```bash
docker-compose down
```

---

## 🧪 Testes

### Executar todos os testes

```bash
cd hl_bot_v3
python -m pytest tests/ -v
```

### Executar testes de um módulo específico

```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_strategies.py -v
python -m pytest tests/test_risk.py -v
```

### Cobertura

```bash
python -m pytest tests/ --cov=crypto_bot_core --cov-report=term
```

---

## 📁 Estrutura do Projeto

```
hl_bot_v3/
├── main.py                          # Ponto de entrada principal
├── .env                             # Configuração (NÃO commitar)
├── .env.example                     # Template de configuração
├── .gitignore                       # Arquivos ignorados
├── requirements.txt                 # Dependências Python
├── pyproject.toml                   # Config do projeto (pytest, mypy)
├── Dockerfile                       # Build multi-stage
├── docker-compose.yml               # Bot + Prometheus + Grafana
├── README.md                        # Este arquivo
│
├── crypto_bot_core/                 # Código principal
│   ├── __init__.py
│   ├── config.py                    # Configuração Pydantic v2
│   ├── indicators.py                # Indicadores técnicos
│   ├── risk.py                      # Gerenciamento de risco
│   ├── execution.py                 # Execução de ordens
│   ├── capital_protection.py        # Proteção de capital
│   ├── staking.py                   # Staking HYPE + vaults
│   ├── notifications.py             # Notificações multicanal
│   ├── monitoring.py                # Métricas e health check
│   │
│   ├── exchanges/                   # Conexão com exchanges
│   │   ├── __init__.py
│   │   └── hyperliquid.py           # Connector Hyperliquid
│   │
│   ├── strategies/                  # Estratégias de trading
│   │   ├── __init__.py
│   │   └── signals.py               # 7 estratégias
│   │
│   └── dashboard/                   # Dashboard web
│       ├── __init__.py
│       ├── static/                  # Arquivos estáticos
│       └── templates/               # Templates HTML
│
├── tests/                           # Testes unitários
│   ├── __init__.py
│   ├── test_config.py               # 36 testes
│   ├── test_exchange.py             # 21 testes
│   ├── test_indicators.py           # 20 testes
│   ├── test_strategies.py           # 60 testes
│   ├── test_risk.py                 # 45 testes
│   ├── test_execution.py            # 31 testes
│   ├── test_capital_protection.py   # 29 testes
│   ├── test_staking.py              # 9 testes
│   ├── test_notifications.py        # 8 testes
│   └── test_monitoring.py           # 12 testes
│
├── data/                            # Dados (logs, cache)
├── scripts/                         # Scripts auxiliares
└── venv/                            # Ambiente virtual (opcional)
```

---

## 🔒 Segurança

- **NUNCA** commite o arquivo `.env` com credenciais reais
- Use **testnet** para testes (`TESTNET=true`)
- A private key da Hyperliquid é a mesma da sua carteira Ethereum
- Mantenha o bot em um ambiente seguro (VPS com firewall)
- Ative notificações para ser alertado de eventos críticos

---

## 📚 Referências

- [Hyperliquid DEX](https://hyperliquid.xyz)
- [Hyperliquid Docs](https://hyperliquid.gitbook.io)
- [Hyperliquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
- [Hyperliquid Testnet](https://app.hyperliquid-testnet.xyz)

---

## 📄 Licença

Este projeto é fornecido apenas para fins educacionais. Use por sua conta e risco em produção.
