from dataclasses import dataclass, asdict
import csv
from pathlib import Path

@dataclass
class PaperTrader:
    cash: float
    position_qty: float = 0.0
    entry_price: float = 0.0
    fee_rate: float = 0.001
    position_fraction: float = 0.10

    def process(self, timestamp, price, signal):
        event = "NO_ACTION"
        if signal == "SIMULATED_LONG" and self.position_qty == 0:
            budget = self.cash * self.position_fraction
            fee = budget * self.fee_rate
            spend = budget - fee
            if spend > 0 and price > 0:
                self.position_qty = spend / price
                self.entry_price = price
                self.cash -= budget
                event = "VIRTUAL_BUY"
        elif signal == "SIMULATED_EXIT" and self.position_qty > 0:
            gross = self.position_qty * price
            fee = gross * self.fee_rate
            self.cash += gross - fee
            self.position_qty = 0.0
            self.entry_price = 0.0
            event = "VIRTUAL_SELL"

        equity = self.cash + self.position_qty * price
        return {
            "timestamp": timestamp,
            "price": round(float(price), 4),
            "signal": signal,
            "event": event,
            "cash": round(self.cash, 2),
            "position_qty": round(self.position_qty, 8),
            "equity": round(equity, 2),
        }

def save_log(rows, path="paper_trading_log.csv"):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
