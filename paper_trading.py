# paper_trading.py
"""
Simulated trading module based on model signals.
Executes paper trades (LONG only), tracks P&L, and saves results to CSV.
Integrates with Binance testnet if enabled.
"""

# ------------------------------------------------------------------------------
# IMPORTS & INITIAL SETUP
# ------------------------------------------------------------------------------

import os
import yaml
import time
import glob
import pandas as pd
import argparse
from datetime import datetime, timezone
import ccxt
import psycopg2

os.makedirs("logs", exist_ok=True)
os.makedirs("csv", exist_ok=True)

# ------------------------------------------------------------------------------
# CONFIGURATION LOADING
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

DB_AWS_CONF = config['database-online']
SYMBOL = config['model'].get('target_symbol')
USE_TESTNET = config['binance'].get('use_testnet', False)
API_KEY = config['binance']['api_key']
API_SECRET = config['binance']['api_secret']
TRADING_STATE_CSV = "csv/paper_trading_state.csv"
threshold_buy = 0.6
threshold_sell = 0.6

threshold_path = sorted(glob.glob("optuna_experiments/study_*/best_trial.txt"))
if threshold_path:
    last_best_trial = threshold_path[-1]
    with open(last_best_trial, "r") as f:
        in_params_section = False
        for line in f:
            line = line.strip()
            if line.startswith("Params:"):
                in_params_section = True
                continue
            if in_params_section:
                if "threshold_buy" in line:
                    threshold_buy = float(line.split(":")[1].strip())
                elif "threshold_sell" in line:
                    threshold_sell = float(line.split(":")[1].strip())
else:
    print("[WARN] Non ho trovato best_trial.txt, userò threshold_buy=0.6 e threshold_sell=0.6 di default")

portfolio = {
    "USDC": config['trading'].get('initial_usdc', 1000),
    "SOL": config['trading'].get('initial_sol', 0)
}
in_position = False

# ------------------------------------------------------------------------------
# DATABASE CONNECTION (AWS only for price retrieval)
# ------------------------------------------------------------------------------

aws_conn = psycopg2.connect(
    host=DB_AWS_CONF['host'],
    port=DB_AWS_CONF['port'],
    user=DB_AWS_CONF['user'],
    password=DB_AWS_CONF['password'],
    dbname=DB_AWS_CONF['name']
)
aws_conn.autocommit = True

# ------------------------------------------------------------------------------
# BINANCE CLIENT (TESTNET MODE IF ENABLED)
# ------------------------------------------------------------------------------

exchange = None
if USE_TESTNET:
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
    })
    exchange.set_sandbox_mode(True)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def load_trading_state(symbol):
    if not os.path.exists(TRADING_STATE_CSV):
        return None
    df = pd.read_csv(TRADING_STATE_CSV)
    df = df[df["symbol"] == symbol]
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "in_position": bool(row["in_position"]),
        "entry_price": float(row["entry_price"]),
        "entry_time": pd.to_datetime(row["entry_time"]),
        "model_version": int(row["model_version"]),
        "threshold_used": float(row["threshold_used"])
    }

def update_trading_state(symbol, in_position, entry_price, entry_time, model_version, threshold_used):
    df = pd.DataFrame([{
        "symbol": symbol,
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time": entry_time,
        "model_version": model_version,
        "threshold_used": threshold_used,
        "last_update": datetime.now(timezone.utc).isoformat()
    }])
    df.to_csv(TRADING_STATE_CSV, index=False)

def get_latest_prediction(run_id):
    inference_file = os.path.join("csv", run_id, "inference_preds.csv")
    if not os.path.exists(inference_file):
        return None, None, None

    try:
        df = pd.read_csv(inference_file)
        df = df[df["symbol"] == SYMBOL]
        if df.empty:
            return None, None, None
        latest = df.sort_values(by="timestamp", ascending=False).iloc[0]
        ts = pd.to_datetime(latest["timestamp"])
        return ts, int(latest["predicted_class"]), float(latest["prediction_prob"])
    except Exception as e:
        print(f"[ERROR] Failed to read prediction: {e}")
        return None, None, None


def get_current_price():
    table_name = f"{SYMBOL.split('/')[0].lower()}_data"
    with aws_conn.cursor() as cur:
        cur.execute(f"SELECT close FROM {table_name} ORDER BY timestamp DESC LIMIT 1;")
        row = cur.fetchone()
        return float(row[0]) if row else None

def log_trade(side, entry_time, entry_price, exit_time, exit_price, sol_quantity, trade_amount_usdc, fee, total_value, pnl, confidence, exit_reason, holding_minutes=None, profit_pct=None):
    try:
        file_exists = os.path.isfile(TRADES_CSV)
        row = {
            "symbol": SYMBOL,
            "side": side,
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "exit_price": exit_price if exit_price else None,
            "sol_quantity": sol_quantity,
            "trade_amount_usdc": trade_amount_usdc,
            "fee": fee,
            "total_value": total_value,
            "pnl": pnl,
            "confidence": confidence,
            "exit_reason": exit_reason,
            "holding_minutes": holding_minutes,
            "profit_pct": profit_pct,
            }
        
        with open(TRADES_CSV, mode="a", newline="") as f:
            writer = pd.DataFrame([row])
            if not file_exists:
                writer.to_csv(f, index=False)
            else:
                writer.to_csv(f, index=False, header=False)
    except Exception as e:
        print(f"[ERROR] Failed to log trade to CSV: {e}")
# ------------------------------------------------------------------------------
# TRADE EXECUTION LOGIC
# ------------------------------------------------------------------------------

def execute_trade(signal: int, timestamp: datetime, conf: float):
    price = get_current_price()
    if price is None:
        print("Cannot retrieve current price for trade execution.")
        return

    try:
        model_version_path = os.path.join("csv", run_id, "model_version.txt")
        with open(model_version_path, "r") as f:
            model_version = f.read().strip()
    except Exception as e:
        print(f"[ERROR] Failed to load model version: {e}")
        model_version = "unknown"

    state = load_trading_state(SYMBOL)


    if signal == 2 and conf >= threshold_buy:
       # Calcola l'importo in USDC da investire
        risk_percentage = min(config['trading'].get('risk_percentage', 0.10) * (conf / threshold_buy), 0.30)
        trade_amount_usdc = portfolio["USDC"] * risk_percentage * conf
        # Calcola le fee
        fee_percent = config['trading'].get('fee_percent', 0.2)
        fee = trade_amount_usdc * (fee_percent / 100)

        # Calcola l'importo netto e i SOL da acquistare
        net_trade_amount_usdc = trade_amount_usdc - fee
        sol_to_buy = net_trade_amount_usdc / price

        # Aggiorna il portafoglio
        portfolio["USDC"] -= trade_amount_usdc
        portfolio["SOL"] += sol_to_buy

        # Calcola il valore totale del portafoglio
        total_value = portfolio["USDC"] + (portfolio["SOL"] * price)

        # Log dell'operazione BUY
        print(f"[{timestamp}] BUY: Acquistati {sol_to_buy:.4f} SOL a {price:.4f} USDC")
        print(f"Importo investito: {trade_amount_usdc:.2f} USDC (Fee: {fee:.2f} USDC)")
        print(f"Portafoglio aggiornato: {portfolio['SOL']:.4f} SOL, {portfolio['USDC']:.2f} USDC")
        print(f"Valore totale: {total_value:.2f} USDC, Confidenza: {conf:.4f}")

        # Qui dovresti chiamare la funzione log_trade con i nuovi parametri
        log_trade("BUY", timestamp, price, None, None, sol_to_buy, trade_amount_usdc, fee, total_value, None, conf, "ENTRY")
                
        # Aggiorna lo stato della posizione
        update_trading_state(SYMBOL, True, price, timestamp, model_version, threshold_buy)
        # Se usi il testnet, qui potresti inviare l'ordine, ma per la simulazione non è strettamente necessario
        return
            
    if state and state['in_position']:
        entry_price = state['entry_price']
        entry_time = state['entry_time']
        holding_minutes = (timestamp - entry_time).total_seconds() / 60
        profit_pct = (price - entry_price) / entry_price

        exit_reason = None
        if profit_pct >= 0.015:
            exit_reason = "TP"        # Take Profit
        elif profit_pct <= -0.01:
            exit_reason = "SL"        # Stop Loss
        elif holding_minutes >= 180:
            exit_reason = "TIMEOUT"   # Timeout di detenzione
        elif signal == 0 and conf >= threshold_sell:
            exit_reason = "SELL_SIGNAL"

        if exit_reason:
            # Vendi l'intera posizione in SOL (o una frazione, se preferisci)
            sol_to_sell = portfolio["SOL"]
            trade_amount_usdc = sol_to_sell * price  # Valore lordo in USDC della vendita
            fee_percent = config['trading'].get('fee_percent', 0.2)
            fee = trade_amount_usdc * (fee_percent / 100)
            net_trade_amount_usdc = trade_amount_usdc - fee

            # Aggiorna il portafoglio: vendi tutti i SOL, aggiungi il netto in USDC
            portfolio["SOL"] = 0
            portfolio["USDC"] += net_trade_amount_usdc

            total_value = portfolio["USDC"]  # Ora, senza SOL, il valore totale è solo in USDC
            # Calcola il PnL dell'operazione basato sul prezzo di ingresso
            pnl = (price - entry_price) * sol_to_sell

            exit_time = timestamp
            holding_minutes = (exit_time - entry_time).total_seconds() / 60
            profit_pct = ((price - entry_price) / entry_price) * 100

            print(f"[{timestamp}] SELL: Venduti {sol_to_sell:.4f} SOL a {price:.4f} USDC")
            print(f"Importo ricevuto: {net_trade_amount_usdc:.2f} USDC (Fee: {fee:.2f} USDC)")
            print(f"Portafoglio aggiornato: {portfolio['SOL']:.4f} SOL, {portfolio['USDC']:.2f} USDC")
            print(f"Valore totale: {total_value:.2f} USDC, PnL: {pnl:.2f}")
            
            # Registra il trade SELL con tutti i dettagli
            log_trade("SELL", entry_time, entry_price, exit_time, price, sol_to_sell,
            trade_amount_usdc, fee, total_value, pnl, conf, exit_reason,
            holding_minutes=holding_minutes,
            profit_pct=profit_pct)
            update_trading_state(SYMBOL, False, None, None, model_version, threshold_sell)
        
            return
        
# ------------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="Unique run identifier")
    args = parser.parse_args()

    if args.run_id is not None:
        run_id = args.run_id
    else:
        # Se non ci arriva un run_id da fuori, creiamone uno di base
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"paper_{timestamp_str}"

    os.makedirs(f"csv/{run_id}", exist_ok=True)
    with open(f"csv/{run_id}/model_version.txt", "r") as f:
        model_version = f.read().strip()

    # Ora definisci i path CSV all’interno della cartella dedicata
    TRADES_CSV = f"csv/{run_id}/paper_trades.csv"
    TRADING_STATE_CSV = f"csv/{run_id}/paper_trading_state.csv"

    print(f"=== Starting paper trading simulation for run_id={run_id} ===")
    print(f"[CONFIG] Trading symbol: {SYMBOL}")
    print(f"[CONFIG] Using Binance Testnet: {USE_TESTNET}")
    print("=========================================")

    last_pred_time = None

    try:
        while True:
            pred_time, signal, conf = get_latest_prediction(run_id)
            if pred_time is not None and (last_pred_time is None or pred_time > last_pred_time):
                print(f"\n[{pred_time}] New prediction received — Signal: {signal}, Confidence: {conf:.4f}")
                execute_trade(signal, pred_time, conf)
                last_pred_time = pred_time
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Paper trading simulation stopped by user.")