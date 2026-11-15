from pathlib import Path
import json
import pandas as pd

from .indicators import add_indicators
from .strategy import get_signal
from .paper_trader import PaperTrader, save_log

BASE = Path(__file__).resolve().parents[1]

def main():
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    data_path = BASE / "data" / "sample_market.csv"
    df = pd.read_csv(data_path)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(sorted(missing))}")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Цены должны быть положительными.")

    df = add_indicators(
        df,
        fast=config["fast_ema"],
        slow=config["slow_ema"],
        rsi_period=config["rsi_period"],
    )
    trader = PaperTrader(
        cash=float(config["starting_balance"]),
        fee_rate=float(config["fee_rate"]),
        position_fraction=float(config["position_fraction"]),
    )

    log = []
    for _, row in df.iterrows():
        signal = get_signal(row, config["rsi_low"], config["rsi_high"])
        log.append(trader.process(row["timestamp"], row["close"], signal))

    save_log(log, BASE / "paper_trading_log.csv")
    last = log[-1]
    initial = float(config["starting_balance"])
    change = (last["equity"] / initial - 1) * 100

    print("\\n=== MARKET LAB: PAPER TRADING ===")
    print(f"Свечей обработано: {len(df)}")
    print(f"Последняя цена в файле: {last['price']}")
    print(f"Виртуальный баланс: {last['cash']:.2f}")
    print(f"Стоимость портфеля: {last['equity']:.2f}")
    print(f"Изменение к старту: {change:+.2f}%")
    print(f"Событий виртуальных сделок: {sum(r['event'] != 'NO_ACTION' for r in log)}")
    print("\\nПодробный журнал: paper_trading_log.csv")
    print("Это симуляция на исторических данных, не прогноз и не финансовая рекомендация.")

if __name__ == "__main__":
    main()
