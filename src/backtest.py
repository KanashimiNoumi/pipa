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

    def next(self):
        # Educational rule set only. Orders are simulated by the backtest engine.
        if self.fast[-1] > self.slow[-1] and self.momentum[-1] < self.rsi_high:
            if not self.position:
                self.buy(size=self.position_fraction)
        elif self.fast[-1] < self.slow[-1] and self.momentum[-1] > self.rsi_low:
            if self.position:
                self.position.close()


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
            "position_fraction": float(config["position_fraction"]),
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
    print(f"Trade report: {trades_path}")
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
