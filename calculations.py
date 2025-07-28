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

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iat[-1]

def compute_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def add_bollinger_bands(df, price_col='close', window=20, num_std=2):
    df['bb_middle'] = df[price_col].rolling(window).mean()
    df['bb_std'] = df[price_col].rolling(window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df

def add_keltner_channels(df, price_col='close', atr_col='atr', window=20, atr_mult=1.5):
    df['kc_middle'] = df[price_col].ewm(span=window, adjust=False).mean()
    # ATR calculation (if not already present)
    if atr_col not in df.columns:
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window).mean()
    df['kc_upper'] = df['kc_middle'] + atr_mult * df['atr']
    df['kc_lower'] = df['kc_middle'] - atr_mult * df['atr']
    return df
