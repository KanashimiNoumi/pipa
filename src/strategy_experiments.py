"""Compare historical-only EMA/RSI strategy variants.

This module is an offline educational experiment. It has no exchange
connectivity and cannot place live orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from src.backtest import BASE, PRICE_SCALE, calculate_trade_metrics, ema, load_ohlcv, rsi


def atr_percent(high, low, close, period: int = 14):
    """ATR as a percentage of close, using only current and past candles."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return (atr / close.replace(0, np.nan) * 100).fillna(0).to_numpy()


def atr_percent_limit(values, lookback: int = 200, quantile: float = 0.90):
    """Prior-window volatility ceiling; shift prevents using the current bar."""
    series = pd.Series(values)
    return (
        series.rolling(lookback, min_periods=lookback)
        .quantile(quantile)
        .shift(1)
        .to_numpy()
    )


class ExperimentStrategy(Strategy):
    fast_ema = 12
    slow_ema = 26
    trend_ema = 200
    rsi_period = 14
    rsi_high = 65
    rsi_low = 35
    position_fraction = 0.10
    use_trend_filter = False
    use_volatility_filter = False

    def init(self):
        close = self.data.Close
        self.fast = self.I(ema, close, self.fast_ema, name="EMA fast")
        self.slow = self.I(ema, close, self.slow_ema, name="EMA slow")
        self.momentum = self.I(rsi, close, self.rsi_period, name="RSI")

        # Register only indicators actually used by this variant. This keeps
        # the baseline warm-up identical to the original EMA/RSI backtest.
        self.trend = None
        self.volatility = None
        self.volatility_ceiling = None

        if self.use_trend_filter:
            self.trend = self.I(ema, close, self.trend_ema, name="EMA 200")

        if self.use_volatility_filter:
            self.volatility = self.I(
                atr_percent,
                self.data.High,
                self.data.Low,
                close,
                14,
                name="ATR %",
            )
            self.volatility_ceiling = self.I(
                atr_percent_limit,
                self.volatility,
                200,
                0.90,
                name="ATR 90th percentile",
            )

        self.signal_log = []

    def next(self):
        crossed_up = self.fast[-2] <= self.slow[-2] and self.fast[-1] > self.slow[-1]
        crossed_down = self.fast[-2] >= self.slow[-2] and self.fast[-1] < self.slow[-1]

        if not crossed_up and not crossed_down:
            return

        rsi_value = float(self.momentum[-1])
        decision = "WAIT"
        reasons = []

        if crossed_up:
            if self.position:
                decision = "IGNORED_ALREADY_IN_POSITION"
            elif rsi_value >= self.rsi_high:
                decision = "REJECTED_RSI_FILTER"
                reasons.append("RSI")
            else:
                if self.use_trend_filter and self.data.Close[-1] <= self.trend[-1]:
                    reasons.append("TREND")

                if self.use_volatility_filter:
                    ceiling = self.volatility_ceiling[-1]
                    current_volatility = self.volatility[-1]
                    if not np.isfinite(ceiling) or current_volatility > ceiling:
                        reasons.append("VOLATILITY")

                if reasons:
                    decision = "REJECTED_" + "_AND_".join(reasons) + "_FILTER"
                else:
                    decision = "SIMULATED_ENTRY"
                    self.buy(size=self.position_fraction)

        elif crossed_down:
            if not self.position:
                decision = "IGNORED_NO_OPEN_POSITION"
            elif rsi_value <= self.rsi_low:
                decision = "REJECTED_RSI_FILTER"
            else:
                decision = "SIMULATED_EXIT"
                self.position.close()

        self.signal_log.append({
            "SignalTime": self.data.index[-1],
            "SignalType": "ENTRY" if crossed_up else "EXIT",
            "RSI": rsi_value,
            "Close": float(self.data.Close[-1]) * PRICE_SCALE,
            "EMA200": (
                float(self.trend[-1]) * PRICE_SCALE
                if self.trend is not None else np.nan
            ),
            "ATR_pct": (
                float(self.volatility[-1])
                if self.volatility is not None else np.nan
            ),
            "Decision": decision,
        })


def build_monthly_report(stats, variant: str, starting_balance: float) -> pd.DataFrame:
    """Build monthly mark-to-market returns and closed-trade statistics."""
    equity_curve = stats["_equity_curve"].copy()
    equity_curve["Month"] = equity_curve.index.strftime("%Y-%m")

    month_end_equity = equity_curve.groupby("Month")["Equity"].last()
    monthly_drawdown = equity_curve.groupby("Month")["DrawdownPct"].min() * 100

    trades = stats["_trades"].copy()
    if not trades.empty:
        trades["Month"] = pd.to_datetime(trades["ExitTime"]).dt.strftime("%Y-%m")
    rows = []
    previous_month_equity = float(starting_balance)

    for month, ending_equity in month_end_equity.items():
        month_trades = trades[trades["Month"] == month] if not trades.empty else trades
        trade_count = len(month_trades)

        if trade_count:
            pnl = month_trades["PnL"].astype(float)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            win_rate = float((pnl > 0).mean() * 100)
            profit_factor = (
                float(wins.sum() / abs(losses.sum()))
                if len(losses) else (float("inf") if len(wins) else np.nan)
            )
        else:
            win_rate = np.nan
            profit_factor = np.nan

        ending_equity = float(ending_equity)
        monthly_return = (
            (ending_equity / previous_month_equity - 1) * 100
            if previous_month_equity else np.nan
        )

        rows.append({
            "Variant": variant,
            "Month": month,
            "ReturnPct": monthly_return,
            "EndingEquity": ending_equity,
            "MaxDrawdownPct": float(monthly_drawdown.loc[month]),
            "Trades": trade_count,
            "WinRatePct": win_rate,
            "ProfitFactor": profit_factor,
        })
        previous_month_equity = ending_equity

    return pd.DataFrame(rows)


def run_variant(data: pd.DataFrame, config: dict, name: str,
                use_trend: bool, use_volatility: bool):
    engine_data = data.copy()
    engine_data[["Open", "High", "Low", "Close"]] /= PRICE_SCALE

    strategy_type = type(
        f"{name}Strategy",
        (ExperimentStrategy,),
        {
            "fast_ema": int(config["fast_ema"]),
            "slow_ema": int(config["slow_ema"]),
            "rsi_period": int(config["rsi_period"]),
            "rsi_low": float(config["rsi_low"]),
            "rsi_high": float(config["rsi_high"]),
            "position_fraction": float(config["position_fraction"]),
            "use_trend_filter": use_trend,
            "use_volatility_filter": use_volatility,
        },
    )

    starting_balance = float(config["starting_balance"])
    engine = Backtest(
        engine_data,
        strategy_type,
        cash=starting_balance,
        commission=float(config["fee_rate"]),
        exclusive_orders=True,
    )
    stats = engine.run()
    trades = stats["_trades"]
    trade_metrics = calculate_trade_metrics(trades)

    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)
    safe_name = name.lower()

    trade_path = reports_dir / f"experiment_{safe_name}_trades.csv"
    trade_report = trades.copy()
    for column in ("EntryPrice", "ExitPrice", "SL", "TP"):
        if column in trade_report.columns:
            trade_report[column] *= PRICE_SCALE
    if "Size" in trade_report.columns:
        trade_report["Size"] /= PRICE_SCALE
    trade_report.to_csv(trade_path, index=False)

    signal_path = reports_dir / f"experiment_{safe_name}_signals.csv"
    pd.DataFrame(stats["_strategy"].signal_log).to_csv(signal_path, index=False)

    monthly = build_monthly_report(stats, name, starting_balance)

    summary = {
        "Variant": name,
        "ReturnPct": float(stats["Return [%]"]),
        "BuyHoldReturnPct": float(stats["Buy & Hold Return [%]"]),
        "MaxDrawdownPct": float(stats["Max. Drawdown [%]"]),
        "Trades": int(stats["# Trades"]),
        "WinRatePct": float(stats["Win Rate [%]"]) if pd.notna(stats["Win Rate [%]"]) else 0.0,
        "FinalEquity": float(stats["Equity Final [$]"]),
        "AverageWin": trade_metrics["average_profit"],
        "AverageLoss": trade_metrics["average_loss"],
        "ProfitFactor": trade_metrics["profit_factor"],
        "MaxConsecutiveLosses": trade_metrics["max_consecutive_losses"],
        "TradeReport": str(trade_path),
        "SignalReport": str(signal_path),
    }
    return summary, monthly


def main():
    parser = argparse.ArgumentParser(
        description="Compare offline historical EMA/RSI strategy experiments."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE / "data" / "sample_market.csv",
        help="OHLCV CSV; defaults to the sample dataset.",
    )
    args = parser.parse_args()

    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    data = load_ohlcv(args.data)

    variants = [
        ("A_BASELINE", False, False),
        ("B_TREND_FILTER", True, False),
        ("C_TREND_VOLATILITY", True, True),
    ]

    results = []
    monthly_results = []
    print("\n=== MARKET LAB: OFFLINE STRATEGY EXPERIMENTS ===")
    print(f"Dataset: {args.data}")
    print(f"Bars: {len(data)}")
    print("All variants use the same data, fees, position sizing and fill convention.")

    for name, use_trend, use_volatility in variants:
        print(f"\nRunning {name}...")
        summary, monthly = run_variant(data, config, name, use_trend, use_volatility)
        results.append(summary)
        monthly_results.append(monthly)

    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)

    summary = pd.DataFrame(results)
    summary_path = reports_dir / "strategy_experiments_summary.csv"
    summary.to_csv(summary_path, index=False)

    monthly_summary = pd.concat(monthly_results, ignore_index=True)
    monthly_path = reports_dir / "strategy_experiments_monthly.csv"
    monthly_summary.to_csv(monthly_path, index=False)

    display_columns = [
        "Variant", "ReturnPct", "MaxDrawdownPct", "Trades",
        "WinRatePct", "ProfitFactor", "MaxConsecutiveLosses",
    ]
    print("\n=== COMPARISON ===")
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    monthly_columns = [
        "Variant", "Month", "ReturnPct", "MaxDrawdownPct",
        "Trades", "WinRatePct", "ProfitFactor",
    ]
    print("\n=== MONTHLY COMPARISON ===")
    print(monthly_summary[monthly_columns].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"
    ))
    print(f"\nSummary report: {summary_path}")
    print(f"Monthly report: {monthly_path}")
    print("\nSimulation only. Historical results are descriptive and do not predict future outcomes.")


if __current__ == "__main__":
    main()
