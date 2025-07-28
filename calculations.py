# ema_calculations.py

def add_indicators(df):
    df = df.copy()
    df["ema6"]   = df["close"].ewm(span=6, adjust=False).mean()
    df["ema9"]   = df["close"].ewm(span=9, adjust=False).mean()
    df["ema10"]  = df["close"].ewm(span=10, adjust=False).mean()
    df["ema13"]  = df["close"].ewm(span=13, adjust=False).mean()
    df["ema20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema12"]  = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"]  = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df['adx'], df['plus_di'], df['minus_di'] = compute_adx(df)
    return df
