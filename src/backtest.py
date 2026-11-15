"""Historical-only strategy testing with backtesting.py.

This module has no exchange connectivity and cannot place live orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

BASE = Path(__file__).resolve().parents[1]


def ema(values, period: int):
    return pd.Series(values).ewm(span=period, adjust=False).mean().to_numpy()


def rsi(values, period: int):
    series = pd.Series(values)
    delta = series.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.fillna(50).to_numpy()


class EmaRsiEducationalStrategy(Strategy):
    fast_ema = 12
    slow_ema = 26
    rsi_period = 14
    rsi_high = 65
    rsi_low = 35
    position_fraction = 0.10

    def init(self):
        close = self.data.Close
        self.fast = self.I(ema, close, self.fast_ema, name="EMA fast")
        self.slow = self.I(ema, close, self.slow_ema, name="EMA slow")
        self.momentum = self.I(rsi, close, self.rsi_period, name="RSI")
        # Separate educational audit log; it does not change trading decisions.
        self.signal_log = []

    def next(self):
        # Detect a fresh crossover, not merely the current EMA ordering.
        crossed_up = self.fast[-2] <= self.slow[-2] and self.fast[-1] > self.slow[-1]
        crossed_down = self.fast[-2] >= self.slow[-2] and self.fast[-1] < self.slow[-1]

        if not crossed_up and not crossed_down:
            return

        signal_type = "ENTRY" if crossed_up else "EXIT"
        rsi_value = float(self.momentum[-1])
        fast_value = float(self.fast[-1])
        slow_value = float(self.slow[-1])
        signal_bar = len(self.data) - 1
        signal_time = self.data.index[-1]
        signal_close = float(self.data.Close[-1])

        if crossed_up:
            filter_passed = rsi_value < self.rsi_high
            if self.position:
                decision = "IGNORED_ALREADY_IN_POSITION"
            elif not filter_passed:
                decision = "REJECTED_RSI_FILTER"
            else:
                decision = "SIMULATED_ENTRY"
                self.buy(size=self.position_fraction)
        else:
            filter_passed = rsi_value > self.rsi_low
            if not self.position:
                decision = "IGNORED_NO_OPEN_POSITION"
            elif not filter_passed:
                decision = "REJECTED_RSI_FILTER"
            else:
                decision = "SIMULATED_EXIT"
                self.position.close()

        self.signal_log.append({
            "SignalType": signal_type,
            "SignalBar": signal_bar,
            "SignalTime": signal_time,
            "SignalClose": signal_close,
            "Signal_EMA_fast": fast_value,
            "Signal_EMA_slow": slow_value,
            "Signal_RSI": rsi_value,
            "RSI_Filter_Passed": filter_passed,
            "Decision": decision,
            # Market orders are simulated at the next bar's open by this engine.
            "ExecutionBar": signal_bar + 1 if decision in ("SIMULATED_ENTRY", "SIMULATED_EXIT") else None,
        })


def load_ohlcv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")

    for column in ("open", "high", "low", "close", "volume"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")

    price_columns = raw[["open", "high", "low", "close"]]
    if (price_columns <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (raw["high"] < raw[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Each High must be >= Open, Close and Low.")
    if (raw["low"] > raw[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Each Low must be <= Open, Close and High.")
    if (raw["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    index = pd.to_datetime(raw.pop("timestamp"), utc=True, errors="raise")
    data = raw.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    data.index = index
    data.index.name = "Timestamp"
    return data[["Open", "High", "Low", "Close", "Volume"]]


def calculate_trade_metrics(trades: pd.DataFrame) -> dict:
    """Summarize closed simulated trades; monetary values use the backtest currency."""
    if trades.empty:
        return {
            "average_profit": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_losses": 0,
        }

    pnl = pd.to_numeric(trades["PnL"], errors="coerce").fillna(0)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_profit = float(winners.sum())
    gross_loss = float(abs(losers.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    max_consecutive_losses = 0
    current_streak = 0
    for result in pnl:
        if result < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    return {
        "average_profit": float(winners.mean()) if not winners.empty else 0.0,
        "average_loss": float(losers.mean()) if not losers.empty else 0.0,
        "profit_factor": profit_factor,
        "max_consecutive_losses": max_consecutive_losses,
    }



def analyze_signal_outcomes(data: pd.DataFrame, signals: pd.DataFrame):
    """Measure what happened after each signal; descriptive, not predictive."""
    if signals.empty:
        return signals.copy(), {}

    analyzed = signals.copy()
    horizons = (3, 6, 12, 24)

    for horizon in horizons:
        future_closes, forward_returns, outcomes = [], [], []
        for _, signal in analyzed.iterrows():
            future_bar = int(signal["SignalBar"]) + horizon
            if future_bar >= len(data):
                future_closes.append(np.nan)
                forward_returns.append(np.nan)
                outcomes.append("INSUFFICIENT_FUTURE_DATA")
                continue

            future_close = float(data["Close"].iloc[future_bar])
            signal_close = float(signal["SignalClose"])
            change_pct = (future_close / signal_close - 1.0) * 100.0
            future_closes.append(future_close)
            forward_returns.append(change_pct)

            if abs(change_pct) < 1e-9:
                outcomes.append("FLAT")
            elif signal["SignalType"] == "ENTRY":
                outcomes.append("FAVORABLE" if change_pct > 0 else "UNFAVORABLE")
            else:
                outcomes.append("FAVORABLE_EXIT" if change_pct < 0 else "MISSED_UPSIDE")

        analyzed[f"FutureClose_{horizon}h"] = future_closes
        analyzed[f"ForwardReturnPct_{horizon}h"] = forward_returns
        analyzed[f"Outcome_{horizon}h"] = outcomes

    summary = {}
    for horizon in horizons:
        return_column = f"ForwardReturnPct_{horizon}h"
        valid = analyzed[analyzed[return_column].notna()]
        for signal_type in ("ENTRY", "EXIT"):
            subset = valid[valid["SignalType"] == signal_type]
            values = subset[return_column]
            summary[f"{signal_type}_{horizon}h"] = {
                "count": int(len(subset)),
                "positive": int((values > 0).sum()),
                "negative": int((values < 0).sum()),
                "flat": int((values.abs() < 1e-9).sum()),
            }
    return analyzed, summary


def print_signal_analysis(summary: dict):
    print("\n=== SIGNAL FOLLOW-THROUGH (HISTORICAL) ===")
    print("Forward price change from the signal candle close; counts are descriptive only.")
    for horizon in (3, 6, 12, 24):
        entry = summary.get(f"ENTRY_{horizon}h", {})
        exit_ = summary.get(f"EXIT_{horizon}h", {})
        print(
            f"{horizon:>2}h | ENTRY: +{entry.get('positive', 0)} / "
            f"-{entry.get('negative', 0)} / ={entry.get('flat', 0)} "
            f"(n={entry.get('count', 0)}) | "
            f"EXIT: +{exit_.get('positive', 0)} / "
            f"-{exit_.get('negative', 0)} / ={exit_.get('flat', 0)} "
            f"(n={exit_.get('count', 0)})"
        )
    print("For ENTRY, positive change is favorable; for EXIT, negative change is favorable.")

def run_backtest(data_path: Path, config_path: Path, make_plot: bool = False):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_ohlcv(data_path)

    position_fraction = float(config["position_fraction"])
    if not 0 < position_fraction < 1:
        raise ValueError("position_fraction must be greater than 0 and less than 1.")

    # Set strategy parameters from the project's config file.
    strategy_type = type(
        "ConfiguredEducationalStrategy",
        (EmaRsiEducationalStrategy,),
        {
            "fast_ema": int(config["fast_ema"]),
            "slow_ema": int(config["slow_ema"]),
            "rsi_period": int(config["rsi_period"]),
            "rsi_low": float(config["rsi_low"]),
            "rsi_high": float(config["rsi_high"]),
            "position_fraction": position_fraction,
        },
    )

    engine = Backtest(
        data,
        strategy_type,
        cash=float(config["starting_balance"]),
        commission=float(config["fee_rate"]),
        exclusive_orders=True,
    )
    stats = engine.run()

    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)
    trades = stats["_trades"]
    trades_path = reports_dir / "backtest_trades.csv"
    trades.to_csv(trades_path, index=False)

    # Export crossover decisions and, where applicable, link them to the
    # simulated fill on the following bar. This is diagnostic only.
    signal_log = pd.DataFrame(stats["_strategy"].signal_log)
    if not signal_log.empty:
        signal_log["ExecutionTime"] = pd.Series(
            [pd.NaT] * len(signal_log), index=signal_log.index, dtype="datetime64[ns, UTC]"
        )
        signal_log["ExecutionOpen"] = np.nan
        signal_log["MatchedTradePnL"] = np.nan
        signal_log["MatchedTradeReturnPct"] = np.nan

        for row_index, signal in signal_log.iterrows():
            execution_bar = signal["ExecutionBar"]
            if pd.isna(execution_bar):
                continue
            execution_bar = int(execution_bar)
            if execution_bar < len(data):
                signal_log.at[row_index, "ExecutionTime"] = data.index[execution_bar]
                signal_log.at[row_index, "ExecutionOpen"] = float(data["Open"].iloc[execution_bar])

            if signal["SignalType"] == "ENTRY" and not trades.empty:
                matched = trades[trades["EntryBar"] == execution_bar]
            elif signal["SignalType"] == "EXIT" and not trades.empty:
                matched = trades[trades["ExitBar"] == execution_bar]
            else:
                matched = trades.iloc[0:0]

            if not matched.empty:
                trade = matched.iloc[0]
                signal_log.at[row_index, "MatchedTradePnL"] = float(trade["PnL"])
                signal_log.at[row_index, "MatchedTradeReturnPct"] = float(trade["ReturnPct"])

    signals_path = reports_dir / "backtest_signals.csv"
    signal_log.to_csv(signals_path, index=False)

    analyzed_signals, signal_summary = analyze_signal_outcomes(data, signal_log)
    analysis_path = reports_dir / "signal_analysis.csv"
    analyzed_signals.to_csv(analysis_path, index=False)

    trade_metrics = calculate_trade_metrics(trades)

    print("\n=== MARKET LAB: HISTORICAL BACKTEST ===")
    print(f"Bars processed: {len(data)}")
    for metric in (
        "Return [%]",
        "Buy & Hold Return [%]",
        "Max. Drawdown [%]",
        "# Trades",
        "Win Rate [%]",
        "Equity Final [$]",
    ):
        if metric in stats:
            print(f"{metric}: {stats[metric]}")

    print(f"Average winning trade: {trade_metrics['average_profit']:.2f}")
    print(f"Average losing trade: {trade_metrics['average_loss']:.2f}")
    profit_factor = trade_metrics["profit_factor"]
    profit_factor_text = "∞ (no losing trades)" if np.isinf(profit_factor) else f"{profit_factor:.2f}"
    print(f"Profit Factor: {profit_factor_text}")
    print(f"Maximum consecutive losses: {trade_metrics['max_consecutive_losses']}")
    print(f"Trade report: {trades_path}")
    print(f"Signal audit report: {signals_path}")
    print(f"Signal outcome analysis: {analysis_path}")
    print_signal_analysis(signal_summary)
    print("\nSimulation only. Historical results do not predict future outcomes.")

    if make_plot:
        plot_path = reports_dir / "backtest_chart.html"
        engine.plot(filename=str(plot_path), open_browser=False)
        print(f"Interactive chart: {plot_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run a historical-only Market Lab backtest.")
    parser.add_argument("--data", type=Path, default=BASE / "data" / "sample_market.csv")
    parser.add_argument("--plot", action="store_true", help="Also create an interactive HTML chart.")
    args = parser.parse_args()
    run_backtest(args.data, BASE / "config.json", make_plot=args.plot)


if __name__ == "__main__":
    main()
