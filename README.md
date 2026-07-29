# 🤖 Hyperliquid Production Bot v3.0

> Bot de trading algorítmico para a **Hyperliquid DEX** (perpétuos), com arquitetura modular, 9 estratégias, ensemble ponderado e sistema de risco em 4 níveis.

---

## 📋 Índice

1. [Visão Geral](#-visão-geral)
2. [Instalação e Configuração](#-instalação-e-configuração)
3. [Modos de Operação](#-modos-de-operação)
4. [Configuração — `.env` (Referência Completa)](#-configuração--env-referência-completa)
5. [Arquitetura do Núcleo](#-arquitetura-do-núcleo-de-negociação)
6. [Sistema de Risco](#-sistema-de-risco)
7. [Modo Ensemble](#-modo-ensemble)
8. [Catálogo de Estratégias](#-estratégias--catálogo-completo)
9. [Backtest](#-backtest--uso-correto)
10. [CI/CD e Qualidade](#-cicd-e-qualidade-de-código)
11. [Limitações Conhecidas](#-limitações-conhecidas)
12. [Roadmap e Próximos Passos](#-roadmap-e-próximos-passos)

---

## 🎯 Visão Geral

| Recurso | Descrição |
|---------|-----------|
| **Estratégias** | 9 estratégias (7 originais + 2 novas) com modo single ou ensemble ponderado |
| **Risco** | 4 níveis de proteção + validação pré-trade de liquidação |
| **Sincronização** | Reconciliação cruzada periódica com a exchange |
| **Dashboard** | Web autenticado, health check real, trilha de auditoria persistente (JSONL) |
| **Backtest** | Bypass configurável de `enabled_strategies`, suporte a ensemble |

---

## 🚀 Instalação e Configuração

```bash
# 1. Criar ambiente virtual
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate           # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente
cp .env.example .env
# → Edite .env com suas credenciais (NUNCA commite o .env real!)
```

### ✅ Testes

```bash
# Suíte completa (Windows: test_lock.py roda separado)
python -m pytest tests/ -v --ignore=tests/test_lock.py

# Teste de locks isolado
python -m pytest tests/test_lock.py -v
```

---

## ⚙️ Modos de Operação

```bash
# 🟢 Trading real
python main.py --mode live

# 📊 Backtest single-strategy
python main.py --mode backtest --symbol BTC/USDC

# 📊 Backtest modo ensemble
python main.py --mode backtest --symbol ETH/USDC --ensemble

# 🔒 Simula fielmente o comportamento de produção
python main.py --mode backtest --symbol BTC/USDC --respect-enabled-strategies

# 🖥️ Dashboard standalone (debug apenas)
python main.py --mode dashboard
```

> ⚠️ **Atenção:** Em modo `live`, o dashboard já sobe automaticamente como thread em background. **Não use** `--mode dashboard` para monitorar um bot ao vivo real.

---

## 🔐 Configuração — `.env` (Referência Completa)

### Credenciais e Modo

```env
HYPERLIQUID_PRIVATE_KEY=
HYPERLIQUID_ACCOUNT_ADDRESS=
TESTNET=true
```

> 💡 **Sempre valide em testnet antes de mainnet.** `TESTNET=false` sem credenciais gera apenas warning no boot — não bloqueia. Confirme manualmente antes de rodar `--mode live` em produção.

### Símbolos e Estratégia

```env
SYMBOLS=BTC/USDC,ETH/USDC,SOL/USDC
STRATEGY=hybrid_regime
ENABLED_STRATEGIES=trend_follow,adaptive_trend,hybrid_regime
TIMEFRAME=1h
```

> 🛡️ `ENABLED_STRATEGIES` é o **gate de segurança** contra operar estratégias não validadas. Em modo ensemble, essa lista vira os participantes do ensemble, não apenas um filtro.

### Ensemble

```env
ENSEMBLE_MODE=false
ENSEMBLE_MIN_CONFLUENCE=2
ENSEMBLE_MIN_AVG_CONFIDENCE=0.55
```

> 📖 Ver [seção 7](#-modo-ensemble) para detalhes.

### Capital e Risco (Nível Raiz)

```env
CAPITAL_USD=1000
RISK_PER_TRADE_PCT=1.0
MAX_POSITION_PCT=20.0
LEVERAGE=3
ISOLATED_MARGIN=true
```

### Bloco `RISK_*`

```env
RISK_MAX_OPEN_TRADES=3
RISK_MAX_DRAWDOWN_PCT=15.0
RISK_DAILY_LOSS_LIMIT_PCT=5.0
RISK_MAX_CONSECUTIVE_LOSSES=3
RISK_COOLDOWN_AFTER_LOSS_SEC=300
RISK_STOP_LOSS_PCT=2.0
RISK_TAKE_PROFIT_PCT=5.0
RISK_TRAILING_STOP=false
RISK_TRAILING_STOP_ACTIVATION_PCT=3.0
RISK_TRAILING_STOP_DISTANCE_PCT=0.5
RISK_MAX_EXPOSURE_PCT=80.0
RISK_MAX_CORRELATED_EXPOSURE_PCT=60.0
RISK_TAKER_FEE=0.0005
RISK_CIRCUIT_BREAKER_LOSS_PCT=15.0
RISK_CIRCUIT_BREAKER_COOLDOWN_SEC=3600
RISK_MAX_SPREAD_PCT=0.5
RISK_MAX_FUNDING_RATE=0.1
```

> ⚠️ `RISK_MAX_FUNDING_RATE` está em **pontos percentuais** (`0.1 = 0.1%`), convertido internamente via `RiskConfig.max_funding_rate_fraction`. Não confundir com o parâmetro de mesmo nome em `STRATEGY_DEFAULTS['funding_arbitrage']`, que é **fração direta** (`0.01 = 1%`) — propósitos e escalas diferentes por design.

### Dashboard

```env
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_USER=admin
DASHBOARD_PASSWORD=changeme
```

> 🔒 **Troque a senha padrão** antes de expor a porta além de localhost. Todas as rotas exigem HTTP Basic Auth. O `POST /api/config` permite alterar SL/TP/drawdown/estratégias habilitadas em produção.

### Outros Blocos

Os demais blocos — `NOTIFICATIONS_*`, `STAKING_*`, `MONITORING_*`, `BACKTEST_*` — seguem o padrão documentado inline no arquivo `.env.example`.

---

## 🏗️ Arquitetura do Núcleo de Negociação

```
Config → Estratégia(s) → Sinal (+ confidence) → Proteções (N1-N4) → Sizing → 
Validação de liquidação → Execução → Monitor/Ledger/Dashboard
```

### Ciclo `step()` (a cada `--interval` segundos)

1. **A cada 30 ciclos** (`RECONCILE_FULL_SYNC_INTERVAL_CYCLES`): resincroniza posições, ordens e saldo com a DEX (não apenas no boot).
2. **Para cada símbolo habilitado:** busca OHLCV + funding rate → indicadores → sinal → filtros de proteção → sizing → validação de liquidação → execução.
3. Verifica **fechamentos externos** (posições fechadas fora do bot).
4. **Reconcilia estado** entre `CapitalProtection`, `PositionManager` e a DEX real — força resync se saldo divergir acima de $0.5.
5. Atualiza saldo, dashboard e health check.

---

## 🛡️ Sistema de Risco

### Camadas de Proteção (`capital_protection.py`)

| Nível | O quê | Configuração |
|-------|-------|--------------|
| **N1 — Mercado** | Horário, spread, funding rate | `RISK_TRADE_HOUR_*`, `RISK_MAX_SPREAD_PCT`, `RISK_MAX_FUNDING_RATE` |
| **N2 — Drawdown** | Drawdown máximo, perda diária, perdas consecutivas, cooldown | `RISK_MAX_DRAWDOWN_PCT`, `RISK_DAILY_LOSS_LIMIT_PCT`, `RISK_MAX_CONSECUTIVE_LOSSES` |
| **N3 — Exposição Bruta** | Limite por posição individual e total | `RISK_MAX_EXPOSURE_PCT`, `MAX_POSITION_PCT` |
| **N3b — Exposição Correlacionada** | Trata todos os símbolos como um grupo de correlação único | `RISK_MAX_CORRELATED_EXPOSURE_PCT` |
| **N4 — Circuit Breaker** | Pausa automática após perda severa | `RISK_CIRCUIT_BREAKER_LOSS_PCT`, `RISK_CIRCUIT_BREAKER_COOLDOWN_SEC` |

### Validação Pré-Trade de Liquidação (`risk.py`)

Antes de cada ordem:

- `calc_liquidation_price_estimate()` estima o preço de liquidação (aproximação, não o valor exato da exchange).
- `validate_stop_loss_safety()` **aborta a ordem** se o SL estiver a menos de `LIQUIDATION_SAFETY_BUFFER_PCT` (**20%**, hardcoded) de folga da liquidação estimada.

> ⚠️ Sempre mais crítico com **leverage alto + SL percentual apertado**.

### Funding Acumulado no PnL

- `PositionManager.accrue_funding()` acumula custo/receita de funding proporcional ao tempo em posição.
- `record_close()` deduz isso do PnL líquido — evita PnL sistematicamente otimista em posições de longa duração.

### Trilha de Auditoria

Todo trade fechado (interno ou externo) é gravado em `data/trades_live.jsonl` (append-only) via `TradeLedger` — sobrevive a restart, consultável via `read_all()` / `summary()`.

---

## 🧠 Modo Ensemble

Combina todas as estratégias de `ENABLED_STRATEGIES`, cada uma com um **score de confidence** (0–1). Só emite `buy`/`sell` quando:

1. Nº de estratégias concordando ≥ `ENSEMBLE_MIN_CONFLUENCE`
2. Confiança média dessas ≥ `ENSEMBLE_MIN_AVG_CONFIDENCE`

> 📈 Mais conservador (menos trades, potencialmente maior qualidade) que single-strategy. **Recomendação:** valide via `--mode backtest --ensemble` antes de ligar em produção — não há dados históricos de performance ainda para este modo.

---

## 📈 Estratégias — Catálogo Completo

> ⚠️ **Regra geral:** nenhuma estratégia deve ser habilitada em `ENABLED_STRATEGIES` para produção sem antes rodar `--mode backtest` (idealmente `run_walk_forward()`) no símbolo/timeframe alvo.

### 1. `trend_follow` ✅

**Lógica:** EMA fast/slow + filtro de tendência (EMA 200) + RSI + ADX.

| Parâmetro | Recomendação |
|-----------|--------------|
| `RISK_STOP_LOSS_PCT` | 2–3% |
| `RISK_TAKE_PROFIT_PCT` | ≥ 2x o SL |
| Timeframe | ≥ 1h |

> 💡 Funciona melhor em timeframes ≥1h; em timeframes curtos gera muito ruído. Em mercado lateral (ADX baixo persistente) tende a operar contra a mão — considere combinar com `adaptive_trend` via ensemble.

---

### 2. `mean_reversion` ⚠️ (desabilitada por padrão)

**Lógica:** Bollinger Bands + RSI em extremos.

| Parâmetro | Recomendação |
|-----------|--------------|
| `RISK_STOP_LOSS_PCT` | 1.5% (mais apertado que o padrão) |
| `RISK_MAX_CONSECUTIVE_LOSSES` | 2–3 |

> ⚠️ Cripto tem **caudas gordas** — quedas "esticadas" continuam caindo com frequência maior que ativos tradicionais. Não recomendada como estratégia isolada em tendência forte — melhor em ranges confirmados (ADX baixo).

---

### 3. `adaptive_trend` ✅

**Lógica:** Alterna trend/range conforme ADX.

| Parâmetro | Recomendação |
|-----------|--------------|
| Padrão | Adequados |
| Atenção | Monitore o campo `mode` retornado (`trend`/`range`/`hybrid`) |

> 💡 Sinais em modo `hybrid` têm confiança fixa mais baixa (0.35) — considere um filtro adicional de tamanho de posição reduzido para esses.

---

### 4. `hybrid_regime` ✅ (recomendada para produção)

**Lógica:** 3 camadas — regime macro (EMA 50/200) + VWAP sweep + estrutura (BOS).

| Parâmetro | Recomendação |
|-----------|--------------|
| `RISK_STOP_LOSS_PCT` | 2.5–3% (um pouco mais largo) |

> 💡 A mais robusta do conjunto. SL muito apertado tende a ser stopado por ruído antes do movimento de regime se confirmar.

---

### 5. `orderflow_delta` ⚠️ (desabilitada por padrão)

**Lógica:** Delta CLV-ponderado (posição do fechamento no range da barra), com persistência de 3 barras e `absorption_threshold` funcional.

> ⚠️ Ainda depende de **proxy de fluxo** (não trades reais tick-a-tick) — trate como sinal de confirmação, não como sinal primário isolado. **Recomendado apenas em ensemble**, nunca como única estratégia em produção real.

---

### 6. `scalping_grid` ⚠️ (desabilitada por padrão)

**Lógica:** Grid multi-nível, espaçamento ATR-adaptativo com fallback percentual.

| Parâmetro | Recomendação |
|-----------|--------------|
| `RISK_MAX_OPEN_TRADES` | 1–2 |

> ⚠️ **Limitação:** não gerencia múltiplos níveis simultâneos abertos — cada sinal ainda abre uma única posição, não acumula por nível. Dado o timeframe curto e frequência de sinais, mantenha `RISK_MAX_OPEN_TRADES` baixo.

---

### 7. `funding_arbitrage` ⚠️ (desabilitada por padrão)

**Lógica:** Opera contra funding extremo, sem considerar tendência.

> ⚠️ Pode brigar contra tendências saudáveis (funding alto é normal em bull run). **Prefira `funding_weighted_trend`** em vez desta para uso em produção — mantida disponível apenas para quem quiser o comportamento de reversão pura.

---

### 8. `volatility_squeeze` 🆕 (não validada)

**Lógica:** Detecta compressão de Bollinger Width seguida de rompimento com confirmação de volume.

| Parâmetro | Recomendação |
|-----------|--------------|
| `RISK_STOP_LOSS_PCT` | Dinâmico via ATR (`atr_mult_sl` em `STRATEGY_DEFAULTS`) |
| `RISK_TRAILING_STOP` | `true` |

> ⚠️ Rompimentos falsos (fakeouts) são comuns logo após squeeze. **Validação obrigatória:** rode backtest em pelo menos 2 regimes de mercado (tendência e lateral) antes de habilitar.

---

### 9. `funding_weighted_trend` 🆕 (não validada)

**Lógica:** Segue tendência apenas com carry favorável; reverte só em exaustão (funding extremo + RSI esticado).

> 💡 Mais seletiva que `funding_arbitrage` — espere menos sinais. Adequada como componente do ensemble junto com `trend_follow`/`hybrid_regime` para reforçar confluência em vez de operar isolada.

---

## 📊 Gerenciamento de Risco — Recomendações Gerais

| Regra | Detalhe |
|-------|---------|
| 🚫 **Nunca** desabilite proteção correlacionada | `RISK_MAX_CORRELATED_EXPOSURE_PCT=100` em portfólio multi-símbolo cripto é perigoso — BTC/ETH/SOL se movem em conjunto na maioria dos regimes. |
| 🔗 **Leverage e SL andam juntos** | Com leverage ≥10x, confirme que `validate_stop_loss_safety()` não está abortando ordens silenciosamente. Se ocorrer com frequência, **reduza a leverage** do símbolo. |
| ⏸️ **Circuit breaker não é opcional** | `RISK_CIRCUIT_BREAKER_LOSS_PCT` deve ser **sempre menor** que `RISK_MAX_DRAWDOWN_PCT` — o circuit breaker deve disparar **antes** do drawdown máximo ser atingido. |
| 💰 **Funding em posições de swing** | Confirme que `accrue_funding()` está sendo chamado a cada ciclo (log `[FUNDING]` em modo DEBUG). Em posições longas, funding acumulado pode ser uma fração relevante do PnL. |
| 🔐 **Dashboard = superfície de risco** | Trate a senha do dashboard como credencial de mesmo nível que a chave privada — quem acessa pode alterar SL/TP/drawdown em tempo real. |
| 📋 **Auditoria contínua** | Revise `data/trades_live.jsonl` periodicamente (`TradeLedger.summary()`) para confirmar que o PnL líquido reportado bate com o extrato real da exchange. Divergência indica bug de sincronização não capturado pela reconciliação automática. |

---

## 🔬 Backtest — Uso Correto

```bash
# Testar estratégia NÃO habilitada em produção (bypass automático por padrão)
python main.py --mode backtest --symbol BTC/USDC   # usa STRATEGY do .env

# Simular fielmente o comportamento de produção
python main.py --mode backtest --symbol BTC/USDC --respect-enabled-strategies

# Testar o modo ensemble sem alterar o .env
python main.py --mode backtest --symbol ETH/USDC --ensemble
```

> ✅ O campo **Símbolo** no summary/CSV agora reflete corretamente o símbolo passado via `--symbol` (bug corrigido — antes sempre mostrava o primeiro item de `SYMBOLS`).

Para validação de robustez temporal (múltiplas janelas), use `BacktestEngine.run_walk_forward()` programaticamente — ainda não exposto via CLI.

---

## 🔧 CI/CD e Qualidade de Código

| Ferramenta | Escopo |
|------------|--------|
| **flake8** | Sintaxe e complexidade |
| **mypy** | Bloqueante para `risk.py`, `capital_protection.py`, `trade_ledger.py`, `execution.py` (módulos auditados linha a linha); resto do projeto é informativo até auditoria completa. |
| **pytest** | 483 testes, cobrindo estratégias, risco, execução, ledger, config, locks de concorrência, backtest. |

---

## ⚠️ Limitações Conhecidas

> Não resolvidas nesta auditoria.

| # | Limitação | Impacto |
|---|-----------|---------|
| 1 | `scalping_grid` não gerencia múltiplos níveis simultâneos abertos | Cada sinal abre posição única |
| 2 | `mean_reversion` sem filtro de regime de tendência | Pode operar contra tendência forte |
| 3 | Estimativa de liquidação usa margem de manutenção fixa (3%), não os tiers reais por ativo da Hyperliquid | Aproximação, não valor exato |
| 4 | `ensemble_mode` e as duas novas estratégias carecem de validação histórica de performance | Apenas testadas unitariamente |
| 5 | `_test_config.py` na raiz é código morto | Pode ser removido a qualquer momento sem impacto |

---

## 🗺️ Roadmap e Próximos Passos

- [ ] Validação histórica de `ensemble_mode` e estratégias novas (`volatility_squeeze`, `funding_weighted_trend`)
- [ ] Suporte a múltiplos níveis simultâneos no `scalping_grid`
- [ ] Filtro de regime de tendência no `mean_reversion`
- [ ] Tiers reais de liquidação por ativo da Hyperliquid
- [ ] Expôr `run_walk_forward()` via CLI
- [ ] Auditoria completa do restante do projeto (mypy bloqueante em todos os módulos)

---

## 📄 Licença

Este projeto é de uso privado e educacional. Não compartilhe credenciais ou `.env` em repositórios públicos.

---

> *Manual gerado para Hyperliquid Production Bot v3.0 — última atualização: 2026-07-29*
