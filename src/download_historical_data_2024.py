"""Download and normalize Binance public BTC/USDT hourly candles for 2024.

Historical data only. This script does not connect to an account or place orders.
Uses Python's standard library; no extra packages are required.
"""
from __future__ import annotations

import csv
import io
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUTPUT_FILE = DATA_DIR / "BTCUSDT_1h_2024.csv"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
YEAR = 2024
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
RAW_COLUMNS = 12


def month_url(month: int) -> str:
    filename = f"{SYMBOL}-{INTERVAL}-{YEAR}-{month:02d}.zip"
    return f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{filename}"


def timestamp_to_iso(raw_timestamp: str) -> str:
    # Binance Spot archive timestamps in 2024 are milliseconds; 2025+ uses microseconds.
    value = int(raw_timestamp)
    seconds = value / 1_000 if value < 10**15 else value / 1_000_000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def download_month(month: int) -> list[tuple[int, list[str]]]:
    url = month_url(month)
    request = Request(url, headers={"User-Agent": "MarketLab-Historical-Data/1.0"})
    print(f"[{month:02d}/12] Downloading {url}")

    try:
        with urlopen(request, timeout=45) as response:
            archive_bytes = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while downloading month {month:02d}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for month {month:02d}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Download timed out for month {month:02d}.") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise RuntimeError(
                    f"Expected one CSV in month {month:02d} ZIP, found {len(csv_names)}."
                )
            with archive.open(csv_names[0]) as csv_file:
                text_stream = io.TextIOWrapper(csv_file, encoding="utf-8", newline="")
                reader = csv.reader(text_stream)
                rows: list[tuple[int, list[str]]] = []
                for line_number, row in enumerate(reader, start=1):
                    if not row:
                        continue
                    if len(row) < RAW_COLUMNS:
                        raise RuntimeError(
                            f"Unexpected CSV format in month {month:02d}, row {line_number}: "
                            f"expected at least {RAW_COLUMNS} columns, got {len(row)}."
                        )
                    open_time = int(row[0])
                    # Keep only open time, OHLC and base-asset volume.
                    normalized = [
                        timestamp_to_iso(row[0]),
                        row[1], row[2], row[3], row[4], row[5],
                    ]
                    rows.append((open_time, normalized))
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded file for month {month:02d} is not a valid ZIP.") from exc

    if not rows:
        raise RuntimeError(f"No candle rows found for month {month:02d}.")
    print(f"       {len(rows)} hourly candles")
    return rows


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[tuple[int, list[str]]] = []

    print("MARKET LAB — HISTORICAL DATA PREPARATION")
    print(f"Pair: {SYMBOL} | Interval: {INTERVAL} | Period: {YEAR}")
    print("Source: Binance public historical archive; no account or API key is used.")
    print()

    for month in range(1, 13):
        all_rows.extend(download_month(month))
        # A short pause keeps the sequence gentle on the public archive.
        time.sleep(0.25)

    all_rows.sort(key=lambda item: item[0])

    unique_rows: list[tuple[int, list[str]]] = []
    seen: set[int] = set()
    duplicate_count = 0
    for item in all_rows:
        if item[0] in seen:
            duplicate_count += 1
            continue
        seen.add(item[0])
        unique_rows.append(item)

    expected_rows = (366 if YEAR % 4 == 0 and (YEAR % 100 != 0 or YEAR % 400 == 0) else 365) * 24
    missing_hours = max(0, expected_rows - len(unique_rows))

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(row for _, row in unique_rows)

    print()
    print("=== DATA CHECK ===")
    print(f"Rows written: {len(unique_rows)}")
    print(f"Expected for 2024: {expected_rows}")
    print(f"Duplicate timestamps removed: {duplicate_count}")
    print(f"Hours absent compared with a complete year: {missing_hours}")
    if unique_rows:
        print(f"First candle: {unique_rows[0][1][0]}")
        print(f"Last candle:  {unique_rows[-1][1][0]}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    if len(unique_rows) != expected_rows:
        print(
            "NOTE: The archive did not contain exactly 8,760 unique hourly rows. "
            "Review the counts before using this file."
        )
    print("Historical dataset prepared. This file is for offline educational analysis only.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
