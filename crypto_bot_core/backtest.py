"""
Módulo de Backtesting — Hyperliquid Production Bot v3.0
=======================================================
Engine de backtesting que reutiliza as 7+ estratégias existentes,
o sistema de risco e os indicadores do bot real.

Uso:
    from crypto_bot_core.backtest import BacktestEngine

    engine = BacktestEngine(cfg, initial_capital=10000.0)
    result = engine.run(df, symbol="BTC/USDC")  # df com indicadores já calculados

    print(result.summary())
    result.export_csv("backtest_result.csv")

    # Modo ensemble (combina todas as estratégias de enabled_strategies):
    engine = BacktestEngine(cfg, ensemble_mode=True)
    result = engine.run(df, symbol="BTC/USDC")

    # Walk-forward (validação de robustez temporal, sem reotimização
    # de parâmetros entre janelas):
    results = engine.run_walk_forward(df)

FIX (achado em uso real — símbolo incorreto no resultado): por padrão,
BacktestEngine agora IGNORA o gate de cfg.enabled_strategies ao gerar
sinais (bypass_enabled_strategies=True por default). Motivo: o próprio
propósito do backtest é validar uma estratégia ANTES de liberá-la para
operar ao vivo via ENABLED_STRATEGIES — com o gate ativo por padrão no
backtest, testar qualquer estratégia fora da lista padrão sempre
resultava em total_trades=0 silenciosamente, e gerava um WARNING por
barra processada (spam de milhares de linhas em backtests longos).

Se você quer simular EXATAMENTE o que aconteceria em produção com o
.env atual (ou seja, respeitando o bloqueio de estratégias não
habilitadas), instancie com bypass_enabled_strategies=False.

Suporta também o modo ENSEMBLE (ensemble_mode=True), que usa
get_ensemble_signal() em vez de get_signal() — combina todas as
estratégias de cfg.enabled_strategies ponderadas por confiança,
permitindo validar via dados históricos o comportamento que
ENSEMBLE_MODE=true teria em produção, sem precisar alterar o .env.

FIX (achado em uso real — item de bug reportado pelo usuário):
BacktestResult.symbol antes SEMPRE usava
self.cfg.symbols.split(",")[0].strip() — o primeiro símbolo da lista
SYMBOLS do .env — independentemente de qual símbolo foi de fato
buscado/testado (ex: via `python main.py --mode backtest --symbol
ETH/USDC`). Rodar o backtest para qualquer símbolo diferente do
primeiro da lista exibia o símbolo ERRADO no summary/CSV. Corrigido:
run() agora aceita um parâmetro `symbol` explícito, propagado por
main.py::_run_backtest(). Mantém fallback para o comportamento antigo
quando `symbol` não é informado, por compatibilidade com chamadores
existentes (ex: alguns testes).

NOTA METODOLÓGICA (ver auditoria — itens 6 e 7):
- Slippage é aplicado tanto na entrada quanto na saída (config
  backtest.slippage_pct), o que antes era lido do BotConfig mas
  nunca usado no loop de simulação.
- TP/SL são checados contra high/low da barra (não apenas close).
  Quando ambos SL e TP estão dentro do range da mesma barra, o SL é
  assumido como tocado primeiro — premissa CONSERVADORA arbitrária,
  não um fato: sem dados intrabar (tick/M1) não há como saber a
  ordem real dos toques. Resultados devem ser lidos com essa
  ressalva.
- run_walk_forward() particiona os dados em janelas sequenciais e
  roda cada uma isoladamente, SEM reotimizar STRATEGY_DEFAULTS por
  janela — é uma checagem de robustez/consistência temporal, não uma
  walk-forward optimization completa.

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
- Testes unitários em tests/test_backtest.py, tests/test_backtest_gate.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger as log

from .config import BotConfig
from .risk import calc_position_size, calc_stops
from .strategies.signals import get_ensemble_signal, get_signal


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────


@dataclass
class BacktestTrade:
    """Representa um trade simulado no backtest.

    Attributes:
        entry_time: Timestamp de entrada.
        exit_time: Timestamp de saída.
        side: "buy" ou "sell".
        entry_price: Preço de entrada (já com slippage aplicado).
        exit_price: Preço de saída (já com slippage aplicado).
        qty: Quantidade do ativo.
        stop_loss: Preço do stop loss.
        take_profit: Preço do take profit.
        gross_pnl: PnL bruto (sem taxas).
        fees: Taxas pagas.
        net_pnl: PnL líquido (após taxas).
        pnl_pct: Retorno percentual.
        bars_held: Número de velas que a posição ficou aberta.
        exit_reason: Motivo da saída (take_profit, stop_loss,
                     trailing_stop, signal_reversal, end_of_data).
    """

    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    stop_loss: float
    take_profit: float
    gross_pnl: float
    fees: float
    net_pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResult:
    """Resultado completo de uma execução de backtest.

    Attributes:
        strategy: Nome da estratégia utilizada (ou "ensemble").
        symbol: Símbolo efetivamente testado.
        timeframe: Timeframe dos dados.
        start_date: Data de início da simulação.
        end_date: Data de fim da simulação.
        total_bars: Número de velas processadas.
        initial_capital: Capital inicial.
        final_capital: Capital final.
        total_trades: Total de trades executados.
        winning_trades: Trades com lucro.
        losing_trades: Trades com prejuízo.
        win_rate: Percentual de acertos.
        total_gross_pnl: PnL bruto total.
        total_net_pnl: PnL líquido total.
        total_fees: Taxas totais pagas.
        max_drawdown: Drawdown máximo em USDC.
        max_drawdown_pct: Drawdown máximo percentual.
        sharpe_ratio: Índice de Sharpe anualizado.
        profit_factor: Razão lucro/prejuízo.
        avg_win: Lucro médio por trade vencedor.
        avg_loss: Prejuízo médio por trade perdedor.
        avg_bars_held: Média de velas por trade.
        expectancy: Expectativa matemática por trade.
        trades: Lista de trades executados.
        equity_curve: Curva de equity ao longo do tempo.
    """

    strategy: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_bars: int
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_gross_pnl: float
    total_net_pnl: float
    total_fees: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_bars_held: float
    expectancy: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Retorna um resumo formatado do backtest.

        Returns:
            str: String formatada com todas as métricas.
        """
        try:
            total_return_pct = (
                (self.final_capital / self.initial_capital) - 1
            ) * 100 if self.initial_capital > 0 else 0.0

            lines = [
                "=" * 55,
                f"  BACKTEST RESULT — {self.strategy} @ {self.timeframe}",
                "=" * 55,
                f"  Símbolo:      {self.symbol}",
                f"  Período:      {self.start_date} → {self.end_date}",
                f"  Velas:        {self.total_bars}",
                f"  Capital:      ${self.initial_capital:.2f} → ${self.final_capital:.2f}",
                f"  Retorno:      {total_return_pct:+.2f}%",
                "",
                f"  Trades:       {self.total_trades}",
                f"  Wins:         {self.winning_trades}",
                f"  Losses:       {self.losing_trades}",
                f"  Win Rate:     {self.win_rate:.1f}%",
                f"  Profit Factor: {self.profit_factor:.3f}",
                f"  Expectancy:   {self.expectancy:+.4f} USDT",
                "",
                f"  Avg Win:      ${self.avg_win:+.4f}",
                f"  Avg Loss:     ${self.avg_loss:+.4f}",
                f"  Avg Bars:     {self.avg_bars_held:.1f}",
                "",
                f"  Max Drawdown: ${self.max_drawdown:.2f} ({self.max_drawdown_pct:.2%})",
                f"  Sharpe Ratio: {self.sharpe_ratio:.3f}",
                f"  Total Fees:   ${self.total_fees:.4f}",
                "=" * 55,
                "  NOTA: SL/TP checados via high/low intrabar; quando ambos",
                "  são tocados na mesma barra, SL é assumido primeiro (premissa",
                "  conservadora, não garantida). Slippage aplicado conforme",
                "  backtest.slippage_pct. Ver documentação do módulo.",
                "=" * 55,
            ]
            return "\n".join(lines)
        except Exception as e:
            log.error(f"[BACKTEST] Erro ao gerar summary: {e}")
            return "Erro ao gerar resumo do backtest."

    def to_dict(self) -> Dict[str, Any]:
        """Exporta resultado como dicionário.

        Returns:
            Dict com todas as métricas (sem a lista de trades completa).
        """
        try:
            total_return_pct = (
                (self.final_capital / self.initial_capital) - 1
            ) * 100 if self.initial_capital > 0 else 0.0

            return {
                "strategy": self.strategy,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "total_bars": self.total_bars,
                "initial_capital": self.initial_capital,
                "final_capital": self.final_capital,
                "total_return_pct": round(total_return_pct, 4),
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": round(self.win_rate, 2),
                "profit_factor": round(self.profit_factor, 4),
                "expectancy": round(self.expectancy, 6),
                "max_drawdown_pct": round(self.max_drawdown_pct, 6),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "total_fees": round(self.total_fees, 4),
                "total_net_pnl": round(self.total_net_pnl, 4),
                "avg_win": round(self.avg_win, 6),
                "avg_loss": round(self.avg_loss, 6),
                "avg_bars_held": round(self.avg_bars_held, 2),
                "trades_count": len(self.trades),
            }
        except Exception as e:
            log.error(f"[BACKTEST] Erro ao gerar to_dict: {e}")
            return {"error": str(e)}

    def export_csv(self, path: str) -> None:
        """Exporta trades para CSV.

        Args:
            path: Caminho do arquivo CSV.
        """
        try:
            if not self.trades:
                log.warning("[BACKTEST] Nenhum trade para exportar")
                return

            rows = []
            for t in self.trades:
                rows.append({
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "gross_pnl": round(t.gross_pnl, 6),
                    "fees": round(t.fees, 6),
                    "net_pnl": round(t.net_pnl, 6),
                    "pnl_pct": round(t.pnl_pct, 6),
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason,
                })

            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
            log.info(f"[BACKTEST] {len(rows)} trades exportados para {path}")
        except Exception as e:
            log.error(f"[BACKTEST] Erro ao exportar CSV para {path}: {e}")


# ──────────────────────────────────────────────
# BacktestEngine
# ──────────────────────────────────────────────


class BacktestEngine:
    """Engine de backtesting.

    Simula a execução de uma estratégia (ou do ensemble de estratégias)
    sobre dados históricos OHLCV, respeitando as mesmas regras de risco
    do bot real.

    Attributes:
        cfg: Configuração do bot.
        initial_capital: Capital inicial para a simulação.
        bypass_enabled_strategies: Se True (default), a estratégia
            configurada em cfg.strategy é executada mesmo que não
            esteja em cfg.enabled_strategies (modo single-strategy
            apenas — não se aplica em modo ensemble).
        ensemble_mode: Se True, usa get_ensemble_signal() em vez de
            get_signal() — combina todas as estratégias de
            cfg.enabled_strategies ponderadas por confiança.
    """

    # Mapeamento de timeframe para minutos (para anualização do Sharpe)
    TF_MINUTES_MAP: Dict[str, int] = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    def __init__(
        self,
        cfg: BotConfig,
        initial_capital: float = 10000.0,
        bypass_enabled_strategies: bool = True,
        ensemble_mode: bool = False,
    ) -> None:
        """Inicializa o BacktestEngine.

        Args:
            cfg: Configuração do bot.
            initial_capital: Capital inicial para o backtest.
            bypass_enabled_strategies: Se True (default), ignora o
                gate de cfg.enabled_strategies — a estratégia
                configurada em cfg.strategy sempre executa sua lógica
                real de sinal, independente de estar habilitada para
                live. Defina False para simular fielmente o
                comportamento que ocorreria em produção com o .env
                atual (estratégias não habilitadas retornam hold em
                toda barra). Não se aplica quando ensemble_mode=True
                (o ensemble usa enabled_strategies como lista de
                participantes, sem gate de bloqueio adicional).
            ensemble_mode: Se True, usa get_ensemble_signal() em vez
                de get_signal() — combina todas as estratégias de
                cfg.enabled_strategies ponderadas por confiança,
                exigindo confluência mínima (cfg.ensemble_min_confluence)
                e confiança média mínima (cfg.ensemble_min_avg_confidence)
                antes de emitir buy/sell. Permite validar via backtest
                o comportamento que ENSEMBLE_MODE=true teria em
                produção, sem precisar alterar o .env. Default False
                (mantém o comportamento single-strategy pré-existente).

        Raises:
            ValueError: Se a estratégia não for suportada (verificado
                apenas quando ensemble_mode=False).
        """
        try:
            self.cfg = cfg
            self.initial_capital = initial_capital
            self.bypass_enabled_strategies = bypass_enabled_strategies
            self.ensemble_mode = ensemble_mode

            if not ensemble_mode:
                self._validate_config()

                if not cfg.is_strategy_enabled() and bypass_enabled_strategies:
                    log.info(
                        f"[BACKTEST] Estratégia '{cfg.strategy.value}' NÃO está em "
                        f"enabled_strategies ('{cfg.enabled_strategies}'), mas "
                        f"bypass_enabled_strategies=True (default) — o backtest "
                        f"executará a lógica real da estratégia mesmo assim, "
                        f"pois este é justamente o mecanismo de validação "
                        f"walk-forward que precede a habilitação em produção."
                    )
                elif not cfg.is_strategy_enabled() and not bypass_enabled_strategies:
                    log.warning(
                        f"[BACKTEST] Estratégia '{cfg.strategy.value}' NÃO está em "
                        f"enabled_strategies e bypass_enabled_strategies=False — "
                        f"o backtest retornará hold em TODAS as barras, resultando "
                        f"em total_trades=0. Isso simula fielmente o comportamento "
                        f"em produção com o .env atual."
                    )
            else:
                log.info(
                    f"[BACKTEST] Modo ENSEMBLE: combinando estratégias "
                    f"'{cfg.enabled_strategies}' (min_confluencia="
                    f"{cfg.ensemble_min_confluence}, min_confianca_media="
                    f"{cfg.ensemble_min_avg_confidence:.2f}). "
                    f"bypass_enabled_strategies não se aplica neste modo — "
                    f"enabled_strategies já funciona como lista de "
                    f"participantes do ensemble, sem gate de bloqueio "
                    f"adicional."
                )

            log.info(
                f"[BACKTEST] Engine inicializado: "
                f"estratégia={'ensemble' if ensemble_mode else cfg.strategy.value}, "
                f"timeframe={cfg.timeframe.value}, "
                f"capital={initial_capital:.2f}, "
                f"bypass_enabled_strategies={bypass_enabled_strategies}, "
                f"ensemble_mode={ensemble_mode}"
            )
        except Exception as e:
            log.error(f"[BACKTEST] Erro ao inicializar engine: {e}")
            raise

    def _validate_config(self) -> None:
        """Valida se a configuração é compatível com backtest (modo
        single-strategy apenas — não chamado em modo ensemble).

        Raises:
            ValueError: Se a estratégia não for reconhecida.
        """
        valid_strategies = {
            "trend_follow",
            "mean_reversion",
            "adaptive_trend",
            "hybrid_regime",
            "orderflow_delta",
            "scalping_grid",
            "funding_arbitrage",
            "volatility_squeeze",
            "funding_weighted_trend",
        }

        strategy_name = self.cfg.strategy.value
        if strategy_name not in valid_strategies:
            raise ValueError(
                f"Estratégia desconhecida ou não suportada: {strategy_name}. "
                f"Válidas: {', '.join(sorted(valid_strategies))}"
            )

    def _tf_minutes(self) -> int:
        """Converte timeframe para minutos (para anualização do Sharpe).

        Returns:
            int: Minutos do timeframe atual. Default 60 se não encontrado.
        """
        return self.TF_MINUTES_MAP.get(self.cfg.timeframe.value, 60)

    def run(self, df: pd.DataFrame, symbol: Optional[str] = None) -> BacktestResult:
        """Executa o backtest sobre o DataFrame com indicadores.

        O DataFrame deve conter colunas OHLCV (open, high, low, close, volume)
        e os indicadores já calculados (via add_all_indicators).

        Args:
            df: DataFrame com colunas OHLCV + indicadores.
            symbol: Símbolo efetivamente testado (ex: "ETH/USDC"), usado
                para popular BacktestResult.symbol corretamente. Se
                None, cai no fallback legado (primeiro símbolo de
                cfg.symbols) — mantido por compatibilidade com
                chamadores que não passam este argumento (ex: alguns
                testes existentes).

                FIX (achado em uso real): antes, BacktestResult.symbol
                sempre usava self.cfg.symbols.split(",")[0], ignorando
                qual símbolo foi de fato buscado via --symbol em
                main.py — rodar `--symbol ETH/USDC` exibia "BTC/USDC"
                no summary/CSV sempre que SYMBOLS no .env tivesse
                BTC/USDC como primeiro item da lista.

        Returns:
            BacktestResult com todos os trades e métricas.

        Raises:
            ValueError: Se dados forem insuficientes (< 50 velas).
        """
        try:
            # ── Validação inicial ──────────────────────────
            if df is None or len(df) < 50:
                raise ValueError(
                    f"Dados insuficientes: {len(df) if df is not None else 0} "
                    f"velas (mínimo 50)"
                )

            required_cols = {"open", "high", "low", "close", "volume"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(
                    f"Colunas obrigatórias ausentes no DataFrame: {missing}"
                )

            log.info(
                f"[BACKTEST] Iniciando "
                f"{'ensemble' if self.ensemble_mode else self.cfg.strategy.value} @ "
                f"{self.cfg.timeframe.value} — {len(df)} velas"
            )

            # ── Estado da simulação ────────────────────────
            balance = float(self.initial_capital)
            peak_balance = balance
            positions: List[Dict[str, Any]] = []
            trades: List[BacktestTrade] = []
            equity_curve: List[float] = [balance]

            # Parâmetros de risco
            risk_per_trade = self.cfg.risk_per_trade_pct / 100.0
            max_capital_pct = self.cfg.max_position_pct / 100.0
            stop_loss_pct = self.cfg.risk.stop_loss_pct / 100.0
            take_profit_pct = self.cfg.risk.take_profit_pct / 100.0
            taker_fee = self.cfg.risk.taker_fee
            # FIX (item 7 auditoria): slippage_pct existia na config mas nunca
            # era lido/aplicado no loop de simulação — agora é.
            slippage_pct = self.cfg.backtest.slippage_pct / 100.0
            trailing_stop_enabled = self.cfg.risk.trailing_stop
            trailing_activation_pct = self.cfg.risk.trailing_stop_activation_pct / 100.0
            trailing_stop_pct = self.cfg.risk.trailing_stop_distance_pct / 100.0

            # Índice inicial (pular warm-up dos indicadores)
            start_idx = max(50, len(df) - min(2000, len(df)))

            # ── Loop principal ─────────────────────────────
            for i in range(start_idx, len(df)):
                try:
                    bar = df.iloc[:i + 1]
                    current = df.iloc[i]
                    price = float(current["close"])
                    high = float(current["high"])
                    low = float(current["low"])
                    atr = float(current["atr"]) if pd.notna(current.get("atr", np.nan)) else 0.0

                    if price <= 0 or atr <= 0:
                        equity_curve.append(balance)
                        continue

                    # ── Gerenciar posições abertas ──────────
                    for pos in list(positions):
                        try:
                            exit_reason: Optional[str] = None
                            # Preço de execução do fechamento antes do slippage.
                            # Default = close da barra; sobrescrito abaixo se
                            # o fechamento ocorreu por TP/SL/trailing tocado
                            # via high/low (mais realista que usar sempre close).
                            exit_price_raw: float = price

                            # FIX (item 6 auditoria): checagem via high/low
                            # intrabar em vez de apenas close. Quando SL e TP
                            # estão ambos dentro do range [low, high] da mesma
                            # barra, SL é assumido tocado primeiro — premissa
                            # CONSERVADORA arbitrária (não factual sem dados
                            # tick/M1), adotada para não superestimar performance.
                            if pos["side"] == "buy":
                                if low <= pos["stop_loss"]:
                                    exit_reason = "stop_loss"
                                    exit_price_raw = pos["stop_loss"]
                                elif high >= pos["take_profit"]:
                                    exit_reason = "take_profit"
                                    exit_price_raw = pos["take_profit"]
                            else:  # sell
                                if high >= pos["stop_loss"]:
                                    exit_reason = "stop_loss"
                                    exit_price_raw = pos["stop_loss"]
                                elif low <= pos["take_profit"]:
                                    exit_reason = "take_profit"
                                    exit_price_raw = pos["take_profit"]

                            # Trailing Stop — também checado via high/low
                            if trailing_stop_enabled and exit_reason is None:
                                if pos["side"] == "buy" and low <= pos["trailing_stop"]:
                                    exit_reason = "trailing_stop"
                                    exit_price_raw = pos["trailing_stop"]
                                elif pos["side"] == "sell" and high >= pos["trailing_stop"]:
                                    exit_reason = "trailing_stop"
                                    exit_price_raw = pos["trailing_stop"]

                                # Atualizar trailing stop (baseado em close,
                                # comportamento original preservado)
                                if pos["side"] == "buy" and price > pos["entry_price"]:
                                    activation_price = pos["entry_price"] * (1 + trailing_activation_pct)
                                    if price >= activation_price:
                                        new_trail = price * (1 - trailing_stop_pct)
                                        if new_trail > pos["trailing_stop"]:
                                            pos["trailing_stop"] = new_trail
                                elif pos["side"] == "sell" and price < pos["entry_price"]:
                                    activation_price = pos["entry_price"] * (1 - trailing_activation_pct)
                                    if price <= activation_price:
                                        new_trail = price * (1 + trailing_stop_pct)
                                        if new_trail < pos["trailing_stop"]:
                                            pos["trailing_stop"] = new_trail

                            if exit_reason:
                                # FIX (item 7 auditoria): aplicar slippage no
                                # preço de saída. Fechamento sempre é contra o
                                # trader (pior preço na direção do fechamento).
                                if pos["side"] == "buy":
                                    fill_exit_price = exit_price_raw * (1 - slippage_pct)
                                else:
                                    fill_exit_price = exit_price_raw * (1 + slippage_pct)

                                mult = 1 if pos["side"] == "buy" else -1
                                gross = (fill_exit_price - pos["entry_price"]) * pos["qty"] * mult
                                fees = (pos["entry_price"] + fill_exit_price) * pos["qty"] * taker_fee
                                net = gross - fees
                                cost_basis = pos["entry_price"] * pos["qty"]
                                pnl_pct = (net / cost_basis * 100) if cost_basis > 0 else 0.0

                                trade = BacktestTrade(
                                    entry_time=str(df.index[pos["entry_idx"]]),
                                    exit_time=str(df.index[i]),
                                    side=pos["side"],
                                    entry_price=pos["entry_price"],
                                    exit_price=fill_exit_price,
                                    qty=pos["qty"],
                                    stop_loss=pos["stop_loss"],
                                    take_profit=pos["take_profit"],
                                    gross_pnl=gross,
                                    fees=fees,
                                    net_pnl=net,
                                    pnl_pct=pnl_pct,
                                    bars_held=i - pos["entry_idx"],
                                    exit_reason=exit_reason,
                                )
                                trades.append(trade)
                                balance += net
                                peak_balance = max(peak_balance, balance)
                                positions.remove(pos)
                        except Exception as e:
                            log.error(
                                f"[BACKTEST] Erro ao gerenciar posição no índice {i}: {e}"
                            )
                            continue

                    # ── Verificar sinais de entrada ─────────
                    try:
                        has_position = any(
                            p["side"] in ("buy", "sell") for p in positions
                        )

                        if not has_position and len(positions) < self.cfg.risk.max_open_trades:
                            # FIX (suporte a ensemble_mode): usa
                            # get_ensemble_signal() quando o modo ensemble
                            # está ativo, propagando o mesmo comportamento
                            # de agregação/confluência que main.py usaria
                            # em produção com ENSEMBLE_MODE=true. Caso
                            # contrário, mantém get_signal() single-strategy
                            # com o bypass de enabled_strategies conforme
                            # configurado no construtor.
                            if self.ensemble_mode:
                                signal, params = get_ensemble_signal(bar, self.cfg)
                            else:
                                signal, params = get_signal(
                                    bar,
                                    self.cfg,
                                    enforce_enabled_gate=not self.bypass_enabled_strategies,
                                )

                            if signal in ("buy", "sell"):
                                sl, tp = calc_stops(
                                    price=price,
                                    side=signal,
                                    atr=atr,
                                    stop_loss_pct=stop_loss_pct,
                                    take_profit_pct=take_profit_pct,
                                    df=bar,
                                )

                                # FIX (item 1+2 auditoria): stop_loss_pct
                                # agora é passado explicitamente, pois
                                # calc_position_size deixou de usar uma
                                # estimativa de distância de stop desacoplada
                                # de calc_stops() — ver crypto_bot_core/risk.py
                                qty = calc_position_size(
                                    balance=balance,
                                    price=price,
                                    atr=atr,
                                    risk_per_trade=risk_per_trade,
                                    max_capital_pct=max_capital_pct,
                                    stop_loss_pct=stop_loss_pct,
                                )

                                if qty > 0 and sl > 0 and tp > 0:
                                    # FIX (item 7 auditoria): aplicar slippage
                                    # na entrada. Entrada sempre é contra o
                                    # trader (pior preço na direção da entrada).
                                    if signal == "buy":
                                        fill_entry_price = price * (1 + slippage_pct)
                                    else:
                                        fill_entry_price = price * (1 - slippage_pct)

                                    # Trailing stop inicial = stop loss
                                    trail_initial = sl

                                    positions.append({
                                        "side": signal,
                                        "entry_price": fill_entry_price,
                                        "qty": qty,
                                        "stop_loss": sl,
                                        "take_profit": tp,
                                        "trailing_stop": trail_initial,
                                        "entry_idx": i,
                                    })

                                    log.debug(
                                        f"[BACKTEST] Entrada {signal.upper()} @ {fill_entry_price:.2f} "
                                        f"(raw={price:.2f}) qty={qty:.6f} SL={sl:.2f} TP={tp:.2f} "
                                        f"conf={params.get('confidence', 0.0):.2f}"
                                    )
                    except Exception as e:
                        log.error(
                            f"[BACKTEST] Erro ao processar sinal no índice {i}: {e}"
                        )

                    equity_curve.append(balance)

                except Exception as e:
                    log.error(
                        f"[BACKTEST] Erro no índice {i} do loop principal: {e}"
                    )
                    equity_curve.append(balance)
                    continue

            # ── Fechar posições remanescentes no final ─────
            try:
                for pos in list(positions):
                    # Fechamento forçado por fim de dados — sem checagem de
                    # high/low (não há "toque" de nível, é fechamento a
                    # mercado no close da última barra), mas slippage ainda
                    # se aplica pois é uma execução real.
                    if pos["side"] == "buy":
                        fill_exit_price = price * (1 - slippage_pct)
                    else:
                        fill_exit_price = price * (1 + slippage_pct)

                    mult = 1 if pos["side"] == "buy" else -1
                    gross = (fill_exit_price - pos["entry_price"]) * pos["qty"] * mult
                    fees = (pos["entry_price"] + fill_exit_price) * pos["qty"] * taker_fee
                    net = gross - fees
                    cost_basis = pos["entry_price"] * pos["qty"]
                    pnl_pct = (net / cost_basis * 100) if cost_basis > 0 else 0.0

                    trade = BacktestTrade(
                        entry_time=str(df.index[pos["entry_idx"]]),
                        exit_time=str(df.index[len(df) - 1]),
                        side=pos["side"],
                        entry_price=pos["entry_price"],
                        exit_price=fill_exit_price,
                        qty=pos["qty"],
                        stop_loss=pos["stop_loss"],
                        take_profit=pos["take_profit"],
                        gross_pnl=gross,
                        fees=fees,
                        net_pnl=net,
                        pnl_pct=pnl_pct,
                        bars_held=len(df) - 1 - pos["entry_idx"],
                        exit_reason="end_of_data",
                    )
                    trades.append(trade)
                    balance += net
                    positions.remove(pos)
            except Exception as e:
                log.error(f"[BACKTEST] Erro ao fechar posições remanescentes: {e}")

            # ── Calcular métricas ──────────────────────────
            try:
                winning = [t for t in trades if t.net_pnl >= 0]
                losing = [t for t in trades if t.net_pnl < 0]
                total_trades = len(trades)

                win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0.0
                total_net = sum(t.net_pnl for t in trades)
                total_gross = sum(t.gross_pnl for t in trades)
                total_fees = sum(t.fees for t in trades)
                avg_win = sum(t.net_pnl for t in winning) / len(winning) if winning else 0.0
                avg_loss = sum(t.net_pnl for t in losing) / len(losing) if losing else 0.0
                avg_bars = (
                    sum(t.bars_held for t in trades) / total_trades
                    if total_trades > 0 else 0.0
                )

                # Profit Factor
                gross_wins = sum(t.gross_pnl for t in winning) if winning else 0.0
                gross_losses = abs(sum(t.gross_pnl for t in losing)) if losing else 1.0
                profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

                # Expectancy
                expectancy = total_net / total_trades if total_trades > 0 else 0.0

                # Max Drawdown
                equity = np.array(equity_curve, dtype=np.float64)
                peak = np.maximum.accumulate(equity)
                drawdown = peak - equity
                max_dd = float(np.max(drawdown))
                max_dd_pct = float(np.max(drawdown / peak)) if peak[-1] > 0 else 0.0

                # Sharpe Ratio (assumindo taxa livre de risco = 0)
                sharpe = 0.0
                if len(equity_curve) > 1:
                    returns = np.diff(equity) / equity[:-1]
                    returns = returns[~np.isnan(returns)]
                    if len(returns) > 0 and np.std(returns) > 0:
                        tf_min = self._tf_minutes()
                        annualization_factor = np.sqrt(365 * 24 * 60 / tf_min) if tf_min > 0 else 1.0
                        sharpe = float(
                            np.mean(returns) / np.std(returns) * annualization_factor
                        )

                # FIX (achado em uso real): usa o símbolo explicitamente
                # passado a run() quando disponível — antes sempre caía
                # no fallback (primeiro símbolo de cfg.symbols),
                # independentemente do símbolo realmente testado.
                effective_symbol = symbol or (
                    self.cfg.symbols.split(",")[0].strip() if self.cfg.symbols else "UNKNOWN"
                )

                result = BacktestResult(
                    strategy="ensemble" if self.ensemble_mode else self.cfg.strategy.value,
                    symbol=effective_symbol,
                    timeframe=self.cfg.timeframe.value,
                    start_date=str(df.index[start_idx]),
                    end_date=str(df.index[-1]),
                    total_bars=len(df) - start_idx,
                    initial_capital=self.initial_capital,
                    final_capital=balance,
                    total_trades=total_trades,
                    winning_trades=len(winning),
                    losing_trades=len(losing),
                    win_rate=win_rate,
                    total_gross_pnl=total_gross,
                    total_net_pnl=total_net,
                    total_fees=total_fees,
                    max_drawdown=max_dd,
                    max_drawdown_pct=max_dd_pct,
                    sharpe_ratio=sharpe,
                    profit_factor=profit_factor,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    avg_bars_held=avg_bars,
                    expectancy=expectancy,
                    trades=trades,
                    equity_curve=equity_curve,
                )

                log.info(
                    f"[BACKTEST] Concluído: {total_trades} trades, "
                    f"Win Rate {win_rate:.1f}%, "
                    f"Retorno {((balance / self.initial_capital) - 1) * 100:+.2f}%"
                )

                return result

            except Exception as e:
                log.error(f"[BACKTEST] Erro ao calcular métricas finais: {e}")
                raise

        except ValueError:
            raise
        except Exception as e:
            log.error(f"[BACKTEST] Erro fatal na execução: {e}")
            raise

    def run_walk_forward(
        self, df: pd.DataFrame, windows: Optional[int] = None, symbol: Optional[str] = None
    ) -> List[BacktestResult]:
        """
        Executa walk-forward validation: particiona df em N janelas
        sequenciais e roda o backtest em cada uma isoladamente.

        Diferente de simplesmente rodar em todo o período de uma vez,
        isso permite observar a consistência (ou instabilidade) do
        resultado ao longo de sub-períodos distintos — um resultado
        agregado positivo pode mascarar 1 janela excelente e 3 ruins,
        o que indicaria dependência forte de regime de mercado
        específico em vez de edge estrutural da estratégia.

        NOTA: esta é uma walk-forward SEM reotimização de parâmetros
        entre janelas (os STRATEGY_DEFAULTS de config.py permanecem
        fixos em todas as janelas). É uma validação de ROBUSTEZ
        temporal/consistência, não uma walk-forward optimization
        completa (que reajustaria hiperparâmetros por janela com
        dados in-sample e testaria out-of-sample). Não apresentar
        o resultado como algo mais rigoroso do que isso.

        Args:
            df: DataFrame completo com indicadores já calculados.
            windows: Número de janelas (default:
                cfg.backtest.walk_forward_windows).
            symbol: Símbolo efetivamente testado, propagado para cada
                BacktestResult de janela (ver run()). Se None, usa o
                fallback legado.

        Returns:
            List[BacktestResult]: um resultado por janela processada
            (janelas menores que 50 velas são puladas e logadas).

        Raises:
            ValueError: se windows < 1 ou dados insuficientes para
                o número de janelas solicitado.
        """
        try:
            n_windows = windows or self.cfg.backtest.walk_forward_windows
            if n_windows < 1:
                raise ValueError(f"windows deve ser >= 1: {n_windows}")

            if df is None:
                raise ValueError("df não pode ser None")

            total_len = len(df)
            if total_len < 50 * n_windows:
                raise ValueError(
                    f"Dados insuficientes para {n_windows} janelas: "
                    f"{total_len} velas (mínimo {50 * n_windows})"
                )

            window_size = total_len // n_windows
            results: List[BacktestResult] = []

            log.info(
                f"[WALK-FORWARD] Iniciando {n_windows} janelas de "
                f"~{window_size} velas cada (total {total_len} velas)"
            )

            for w in range(n_windows):
                start = w * window_size
                end = total_len if w == n_windows - 1 else (w + 1) * window_size
                window_df = df.iloc[start:end]

                if len(window_df) < 50:
                    log.warning(
                        f"[WALK-FORWARD] Janela {w + 1}/{n_windows} pequena "
                        f"demais ({len(window_df)} velas), pulando"
                    )
                    continue

                log.info(
                    f"[WALK-FORWARD] Janela {w + 1}/{n_windows}: "
                    f"{window_df.index[0]} → {window_df.index[-1]} "
                    f"({len(window_df)} velas)"
                )

                try:
                    result = self.run(window_df, symbol=symbol)
                    results.append(result)
                    log.info(
                        f"[WALK-FORWARD] Janela {w + 1}/{n_windows} concluída: "
                        f"{result.total_trades} trades, "
                        f"win_rate={result.win_rate:.1f}%, "
                        f"retorno={((result.final_capital / result.initial_capital) - 1) * 100:+.2f}%"
                    )
                except ValueError as e:
                    log.warning(
                        f"[WALK-FORWARD] Janela {w + 1}/{n_windows} falhou "
                        f"na validação, pulando: {e}"
                    )
                    continue

            if not results:
                log.warning(
                    "[WALK-FORWARD] Nenhuma janela produziu resultado válido"
                )

            # Resumo de consistência entre janelas (log apenas — não altera
            # o retorno, mas ajuda a leitura rápida sem reprocessar tudo)
            if len(results) > 1:
                win_rates = [r.win_rate for r in results]
                returns_pct = [
                    ((r.final_capital / r.initial_capital) - 1) * 100
                    for r in results
                ]
                positive_windows = sum(1 for r in returns_pct if r > 0)
                log.info(
                    f"[WALK-FORWARD] Resumo: {positive_windows}/{len(results)} "
                    f"janelas positivas | win_rate min={min(win_rates):.1f}% "
                    f"max={max(win_rates):.1f}% | retorno min={min(returns_pct):+.2f}% "
                    f"max={max(returns_pct):+.2f}%"
                )
                if positive_windows < len(results):
                    log.warning(
                        f"[WALK-FORWARD] {len(results) - positive_windows} "
                        f"janela(s) negativa(s) — resultado agregado pode "
                        f"mascarar inconsistência temporal. Revisar antes de "
                        f"usar em produção."
                    )

            return results

        except ValueError:
            raise
        except Exception as e:
            log.error(f"[WALK-FORWARD] Erro fatal: {e}")
            raise