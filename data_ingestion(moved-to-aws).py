#!/usr/bin/env python3
"""
Real-Time Data Ingestion Module for Binance Spot Market.

This module ingests:
- 1-minute OHLCV data
- Top 10-level Order Book snapshots

All data is stored in PostgreSQL using psycopg2.
Supports multiple trading pairs (symbols) concurrently via threading.
"""

import os
import time
import yaml
import logging
import threading
import requests
import datetime
import psycopg2

# ------------------------------------------------------------------------------
# CONFIGURATION LOADING
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

DB_CONF = config['database']
SYMBOLS = config['symbols']  # e.g., ["BTC/USDC", "ETH/USDC"]
API_KEY = config['binance']['api_key']
API_SECRET = config['binance']['api_secret']
USE_TESTNET = config['binance'].get('use_testnet', False)

BASE_URL = "https://testnet.binance.vision" if USE_TESTNET else "https://api.binance.com"

# ------------------------------------------------------------------------------
# LOGGER CONFIGURATION
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataIngestion")

# ------------------------------------------------------------------------------
# DATABASE CONNECTION (PostgreSQL via psycopg2)
# ------------------------------------------------------------------------------

conn = psycopg2.connect(
    host=DB_CONF['host'],
    port=DB_CONF['port'],
    user=DB_CONF['user'],
    password=DB_CONF['password'],
    dbname=DB_CONF['name']
)
conn.autocommit = True

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

# Convert a trading pair into its corresponding OHLC table name
def get_ohlc_table_name(symbol: str) -> str:
    """Convert a trading pair into its corresponding OHLC table name."""
    return f"{symbol.split('/')[0].lower()}_data"

# ------------------------------------------------------------------------------
# OHLCV INGESTION
# ------------------------------------------------------------------------------

# Fetch the latest 1-minute OHLCV data from Binance
def fetch_latest_ohlc(symbol: str):
    """Fetch the latest 1-minute OHLCV data from Binance."""
    endpoint = "/api/v3/klines"
    params = {"symbol": symbol.replace("/", ""), "interval": "1m", "limit": 1}
    try:
        resp = requests.get(BASE_URL + endpoint, params=params)
        resp.raise_for_status()
        kline = resp.json()[-1]
        timestamp = datetime.datetime.fromtimestamp(kline[0] / 1000, datetime.timezone.utc).replace(tzinfo=None)
        return {
            "timestamp": timestamp,
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5])
        }
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

# Backfill historical OHLCV data from Binance
def backfill_ohlcv(symbol: str, minutes: int = 1440):
    """Backfill historical OHLCV data from Binance and store in DB."""
    logger.info(f"[{symbol}] Starting backfill of last {minutes} minutes...")
    endpoint = "/api/v3/klines"
    params = {
        "symbol": symbol.replace("/", ""),
        "interval": "1m",
        "limit": min(minutes, 1000)  # Binance max = 1000
    }

    end_ts = int(datetime.datetime.utcnow().timestamp() * 1000)
    while minutes > 0:
        params["endTime"] = end_ts
        try:
            resp = requests.get(BASE_URL + endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break

            rows = []
            for kline in data:
                ts = datetime.datetime.fromtimestamp(kline[0] / 1000, datetime.timezone.utc).replace(tzinfo=None)
                rows.append({
                    "timestamp": ts,
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })
                end_ts = kline[0] - 1

            for row in reversed(rows):  # inserisci in ordine crescente
                store_ohlc_to_db(symbol, row)

            minutes -= len(rows)
            time.sleep(0.25)
        except Exception as e:
            logger.error(f"[{symbol}] Error in backfill: {e}")
            break

    logger.info(f"[{symbol}] Backfill completed.")

# Store OHLCV data into the appropriate table
def store_ohlc_to_db(symbol: str, ohlc: dict):
    """Insert OHLCV data into the appropriate table. Ignores duplicates by timestamp."""
    table = get_ohlc_table_name(symbol)
    query = f"""
        INSERT INTO {table} (timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (
                ohlc["timestamp"], ohlc["open"], ohlc["high"],
                ohlc["low"], ohlc["close"], ohlc["volume"]
            ))
        logger.info(f"[{symbol}] OHLCV saved at {ohlc['timestamp']}")
    except Exception as e:
        logger.error(f"Error inserting OHLCV into DB: {e}")

# ------------------------------------------------------------------------------
# ORDER BOOK INGESTION
# ------------------------------------------------------------------------------

# Fetch top-level order book snapshot from Binance
def fetch_orderbook(symbol: str, depth: int = 10):
    """Fetch top-level order book snapshot from Binance (default: top 10 levels)."""
    endpoint = "/api/v3/depth"
    params = {"symbol": symbol.replace("/", ""), "limit": depth}
    try:
        resp = requests.get(BASE_URL + endpoint, params=params)
        resp.raise_for_status()
        ob = resp.json()
        snapshot = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            "symbol": symbol
        }
        bids = ob.get("bids", [])[:depth] + [["0", "0"]] * (depth - len(ob.get("bids", [])))
        asks = ob.get("asks", [])[:depth] + [["0", "0"]] * (depth - len(ob.get("asks", [])))
        for i in range(depth):
            snapshot[f"bid_price_{i+1}"] = float(bids[i][0])
            snapshot[f"bid_vol_{i+1}"] = float(bids[i][1])
            snapshot[f"ask_price_{i+1}"] = float(asks[i][0])
            snapshot[f"ask_vol_{i+1}"] = float(asks[i][1])
        return snapshot
    except Exception as e:
        logger.error(f"Error fetching orderbook for {symbol}: {e}")
        return None

# Store order book snapshot into the database
def store_orderbook_to_db(snapshot: dict):
    """Insert a full snapshot of the order book into the DB."""
    bid_cols = [f"bid_price_{i+1}" for i in range(10)] + [f"bid_vol_{i+1}" for i in range(10)]
    ask_cols = [f"ask_price_{i+1}" for i in range(10)] + [f"ask_vol_{i+1}" for i in range(10)]
    columns = ["timestamp", "symbol"] + bid_cols + ask_cols
    values = [snapshot["timestamp"], snapshot["symbol"]]
    for i in range(10):
        values += [snapshot[f"bid_price_{i+1}"], snapshot[f"bid_vol_{i+1}"]]
    for i in range(10):
        values += [snapshot[f"ask_price_{i+1}"], snapshot[f"ask_vol_{i+1}"]]
    placeholders = ", ".join(["%s"] * len(values))
    query = f"INSERT INTO orderbook_snapshots ({', '.join(columns)}) VALUES ({placeholders});"
    try:
        with conn.cursor() as cur:
            cur.execute(query, values)
        logger.info(f"[{snapshot['symbol']}] Orderbook snapshot saved at {snapshot['timestamp']}")
    except Exception as e:
        logger.error(f"Error inserting orderbook snapshot: {e}")

# ------------------------------------------------------------------------------
# INGESTION LOOPS
# ------------------------------------------------------------------------------

# Ingest OHLCV data in a loop
def ingest_ohlc_loop(symbol: str):
    """Background thread for continuously ingesting OHLCV data per minute."""
    last_ts = None
    table = get_ohlc_table_name(symbol)
    logger.info(f"[{symbol}] OHLCV loop started (table: {table})")

    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(timestamp) FROM {table};")
            res = cur.fetchone()
            if res: last_ts = res[0]
    except Exception as e:
        logger.error(f"Error fetching last OHLCV timestamp for {symbol}: {e}")

    logger.info(f"[{symbol}] Resuming from: {last_ts}" if last_ts else f"[{symbol}] Starting fresh.")

    while True:
        ohlc = fetch_latest_ohlc(symbol)
        if ohlc and (last_ts is None or ohlc["timestamp"] > last_ts):
            store_ohlc_to_db(symbol, ohlc)
            last_ts = ohlc["timestamp"]
        time.sleep(max(1, 60 - datetime.datetime.utcnow().second))

# Ingest order book snapshots in a loop
def ingest_orderbook_loop(symbol: str):
    """Background thread for continuously ingesting order book snapshots every second."""
    logger.info(f"[{symbol}] Orderbook loop started")
    while True:
        snapshot = fetch_orderbook(symbol)
        if snapshot:
            store_orderbook_to_db(snapshot)
        time.sleep(1)

# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    threads = []
    for sym in SYMBOLS:
        t1 = threading.Thread(target=ingest_ohlc_loop, args=(sym,), daemon=True)
        t2 = threading.Thread(target=ingest_orderbook_loop, args=(sym,), daemon=True)
        threads.extend([t1, t2])
        t1.start()
        t2.start()
    for t in threads:
        t.join()
