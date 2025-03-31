# feature_engineering.py
"""
Feature Engineering Module for Trading Models.

Provides:
- Technical indicators (SMA, RSI, MACD, Bollinger Band Width, momentum, volatility)
- Higher timeframe aggregations (15-min, 1-hour)
- Order book features (spread, imbalance, mid-price)
Used in both training and real-time inference pipelines.
"""

import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine

# ------------------------------------------------------------------------------
# CONFIGURATION & DB CONNECTION
# ------------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_AWS_CONF = config['database-online']
FEATURE_CONF = config['model']['features']
TARGET_HORIZON = FEATURE_CONF.get('target_horizon', 10)

engine = create_engine(
    f"postgresql://{DB_AWS_CONF['user']}:{DB_AWS_CONF['password']}@"
    f"{DB_AWS_CONF['host']}:{DB_AWS_CONF['port']}/{DB_AWS_CONF['name']}"
)

# ------------------------------------------------------------------------------
# INDICATOR FUNCTIONS
# ------------------------------------------------------------------------------

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on OHLCV data.
    Also computes 'future_return' for the next TARGET_HORIZON minutes,
    but DOES NOT drop rows. The dropna finale sarà eseguito dopo tutti i merge.
    """
    df = df.copy().reset_index(drop=True)

    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['momentum_10'] = df['close'].diff(10)

    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_width'] = (2 * std_20) / sma_20

    df['rsi_14'] = compute_rsi(df['close'], 14)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

    df['close_lag1'] = df['close'].shift(1)
    df['pct_change_1'] = (df['close'] - df['close_lag1']) / df['close_lag1'] * 100
    df['volatility_10'] = df['close'].rolling(10).std()

    # Calcolo del future_return: come variazione percentuale tra close attuale e close tra TARGET_HORIZON minuti
    df["future_return"] = df["close"].shift(-TARGET_HORIZON) / df["close"] - 1

    # In questa fase NON facciamo dropna, così da non perdere righe che potrebbero
    # riacquisire validità dopo i vari merge_asof. Lo faremo più avanti.
    return df

def compute_orderbook_features(ob_df: pd.DataFrame) -> pd.DataFrame:
    """Compute order book features: spread, imbalance, and mid-price."""
    ob_df = ob_df.copy().reset_index(drop=True)

    ob_df['spread'] = ob_df['ask_price_1'] - ob_df['bid_price_1']

    bid_sum = sum(ob_df[f'bid_vol_{i}'] for i in range(1, FEATURE_CONF['orderbook_levels'] + 1))
    ask_sum = sum(ob_df[f'ask_vol_{i}'] for i in range(1, FEATURE_CONF['orderbook_levels'] + 1))
    ob_df['ob_imbalance'] = (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-9)
    ob_df['mid_price'] = (ob_df['bid_price_1'] + ob_df['ask_price_1']) / 2

    return ob_df

# ------------------------------------------------------------------------------
# TRAINING FEATURE PREPARATION
# ------------------------------------------------------------------------------

def prepare_training_data(symbol: str, threshold_buy: float, threshold_sell: float, horizon: int):
    """
    Generate training feature matrix X and target y.
    Steps:
    1) Load OHLCV, compute tech indicators (including future_return).
    2) Merge 15-min and 1-hour aggregates.
    3) Merge order book data.
    4) Drop NaN rows (final).
    5) Create target (0/1/2) from future_return.
    6) Drop NaN se necessario, log delle dimensioni e classi.
    7) Return X, y, feature_cols.
    """

    # 1) Carico OHLCV e calcolo indicatori
    ohlc_table = f"{symbol.lower().split('/')[0]}_data"
    df_ohlc = pd.read_sql(f"SELECT * FROM {ohlc_table} ORDER BY timestamp ASC;", engine, parse_dates=['timestamp'])
    if df_ohlc.empty:
        raise ValueError("No OHLCV data available for training.")

    df_ohlc = compute_technical_indicators(df_ohlc)
    print(f"[DEBUG] After compute_technical_indicators: {df_ohlc.shape[0]} rows")

    # 2) Creiamo aggregazioni 15min e 1h e uniamo con merge_asof
    df_15min = df_ohlc.resample('15min', on='timestamp').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum'
    }).reset_index()

    df_15min['sma_15min'] = df_15min['close'].rolling(3).mean()
    df_15min['rsi_14_15m'] = compute_rsi(df_15min['close'], 14)
    df_15min['volatility_15m'] = df_15min['close'].rolling(3).std()

    df_1h = df_ohlc.resample('1h', on='timestamp').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum'
    }).reset_index()

    df_1h['sma_1h'] = df_1h['close'].rolling(6).mean()
    df_1h['rsi_14_1h'] = compute_rsi(df_1h['close'], 14)
    df_1h['volatility_1h'] = df_1h['close'].rolling(6).std()

    # Merge 15min
    df = pd.merge_asof(
        df_ohlc.sort_values('timestamp'),
        df_15min[['timestamp', 'sma_15min', 'rsi_14_15m', 'volatility_15m']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    # Merge 1h
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        df_1h[['timestamp', 'sma_1h', 'rsi_14_1h', 'volatility_1h']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    print(f"[DEBUG] After merging 15min & 1h: {df.shape[0]} rows")

    # 3) Merge con order book
    df_ob = pd.read_sql(
        f"SELECT * FROM orderbook_snapshots WHERE symbol = '{symbol}' ORDER BY timestamp ASC;",
        engine, parse_dates=['timestamp']
    )
    df_ob = df_ob.rename(columns={'timestamp': 'ob_timestamp'}).drop(columns=['symbol'])

    df = pd.merge_asof(
        df.sort_values('timestamp'),
        df_ob.sort_values('ob_timestamp'),
        left_on='timestamp',
        right_on='ob_timestamp',
        direction='backward'
    )
    print(f"[DEBUG] After merging order book: {df.shape[0]} rows")

    # 4) Drop NaN finale (prima di definire il target, qui potresti anche invertire l'ordine,
    #    ma tipicamente calcoli future_return sugli OHLCV, quindi lui c'è già)
    print("[DEBUG] Shape prima del dropna finale:", df.shape)
    df.dropna(inplace=True)
    print("[DEBUG] Shape dopo dropna finale:", df.shape)

    # 5) Calcoliamo le feature di order book (spread, imbalance, mid_price)
    df = compute_orderbook_features(df)

    # 6) Creiamo la colonna target in base a future_return e la soglia THRESH
    df["target"] = 1
    df.loc[df["future_return"] > threshold_buy, "target"] = 2
    df.loc[df["future_return"] < -threshold_sell, "target"] = 0

    # Eventuale dropna post-target (se future_return ha generato nuovi NaN in shift)
    # df.dropna(inplace=True)  # <-- se proprio necessario

    # Log finali
    print("[DEBUG] future_return stats:", df["future_return"].describe())
    print("[DEBUG] target distribution:", df["target"].value_counts())

    drop_cols = [
        'timestamp', 'ob_timestamp', 'open', 'high', 'low', 'close', 'volume',
        'future_return', 'target'
    ]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    print(f"[INFO] Feature columns: {feature_cols}")
    print("Classe 0 (SELL):", sum(df["target"] == 0))
    print("Classe 1 (HOLD):", sum(df["target"] == 1))
    print("Classe 2 (BUY) :", sum(df["target"] == 2))

    # Return
    X = df[feature_cols].values
    y = df["target"].values
    return X, y, feature_cols

# ------------------------------------------------------------------------------
# INFERENCE FEATURE PREPARATION
# ------------------------------------------------------------------------------

def prepare_live_features(symbol: str, return_df=False):
    ohlc_table = f"{symbol.lower().split('/')[0]}_data"
    window = FEATURE_CONF['ohlcv_window']
    df_recent = pd.read_sql(
        f"SELECT * FROM {ohlc_table} ORDER BY timestamp DESC LIMIT {window};",
        engine, parse_dates=['timestamp']
    ).sort_values('timestamp')

    if df_recent.empty or len(df_recent) < window:
        raise ValueError("Not enough OHLCV data for inference.")

    # Calcola gli indicatori sugli ultimi 'window' record
    df_recent_feat = compute_technical_indicators(df_recent)
    if "future_return" in df_recent_feat.columns:
        df_recent_feat.drop(columns=["future_return"], inplace=True)

    # Aggregazioni higher timeframe
    df_15min = df_recent.resample('15min', on='timestamp').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum'
    }).reset_index()
    df_1h = df_recent.resample('1h', on='timestamp').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum'
    }).reset_index()

    # Calcolo dei valori per il timeframe 15min
    df_15min['sma_15min'] = df_15min['close'].rolling(3).mean()
    df_15min['rsi_14_15m'] = compute_rsi(df_15min['close'], 14)
    df_15min['volatility_15m'] = df_15min['close'].rolling(3).std()
    
    # Calcolo dei valori per il timeframe 1h
    df_1h['sma_1h'] = df_1h['close'].rolling(6).mean()
    df_1h['rsi_14_1h'] = compute_rsi(df_1h['close'], 14)
    df_1h['volatility_1h'] = df_1h['close'].rolling(6).std()

    # Prendi l'ultima riga dei dati recenti (snapshot più recente)
    latest = df_recent_feat.iloc[-1:].copy()
    latest['sma_15min'] = df_15min['sma_15min'].iloc[-1]
    latest['rsi_14_15m'] = df_15min['rsi_14_15m'].iloc[-1]
    latest['volatility_15m'] = df_15min['volatility_15m'].iloc[-1]
    
    latest['sma_1h'] = df_1h['sma_1h'].iloc[-1]
    latest['rsi_14_1h'] = df_1h['rsi_14_1h'].iloc[-1]
    latest['volatility_1h'] = df_1h['volatility_1h'].iloc[-1]

    # Carica l'ultimo snapshot dell'order book e uniscilo
    df_ob = pd.read_sql(
        f"SELECT * FROM orderbook_snapshots WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 1;",
        engine
    )
    if df_ob.empty:
        raise ValueError("No order book snapshot available.")

    for col in df_ob.columns:
        if col not in ['timestamp', 'symbol']:
            latest[col] = df_ob.iloc[0][col]

    latest = compute_orderbook_features(latest)

    drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in latest.columns if c not in drop_cols]
    X_live = latest[feature_cols].values

    if return_df:
        return X_live, feature_cols, latest
    else:
        return X_live, feature_cols