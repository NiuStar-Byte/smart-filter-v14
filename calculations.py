import numpy as np
import pandas as pd

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute Exponential Moving Average (EMA) for a given pandas Series and span.
    """
    return series.ewm(span=span, adjust=False).mean()

def add_ema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds commonly used EMA columns to the DataFrame.
    """
    ema_spans = [6, 9, 10, 12, 13, 20, 21, 26, 50, 200]
    for span in ema_spans:
        df[f'ema{span}'] = compute_ema(df['close'], span)
    return df

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes MACD and MACD Signal line, adds them to the DataFrame.
    """
    df['ema12'] = compute_ema(df['close'], 12)
    df['ema26'] = compute_ema(df['close'], 26)
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Computes VWAP for the DataFrame.
    """
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes ATR using the true range method.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(period).mean()
    return atr

def compute_adx(df: pd.DataFrame, period: int = 14):
    """
    Computes ADX, +DI, -DI as Series and adds them to the DataFrame.
    Returns adx, plus_di, minus_di.
    """
    df = df.copy()
    up_move = df['high'].diff()
    down_move = df['low'].diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    return adx, plus_di, minus_di

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes RSI using the EMA smoothing method.
    """
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute Commodity Channel Index (CCI).
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - ma) / (0.015 * md)
    return cci

def calculate_stochrsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """
    Compute Stochastic RSI (StochRSI), and optional smoothed K/D lines.
    Returns stochrsi_k, stochrsi_d as Series.
    """
    rsi = compute_rsi(df, rsi_period)
    min_rsi = rsi.rolling(stoch_period).min()
    max_rsi = rsi.rolling(stoch_period).max()
    stochrsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    stochrsi_k = stochrsi.rolling(smooth_k).mean()
    stochrsi_d = stochrsi_k.rolling(smooth_d).mean()
    return stochrsi_k, stochrsi_d

def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Adds Bollinger Bands columns to DataFrame.
    """
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['bb_upper'] = ma + num_std * std
    df['bb_lower'] = ma - num_std * std
    return df

def add_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Adds Keltner Channel columns to DataFrame.
    """
    if 'atr' not in df.columns:
        df['atr'] = compute_atr(df, period)
    ma = df['close'].rolling(window=period).mean()
    df['kc_upper'] = ma + atr_mult * df['atr']
    df['kc_lower'] = ma - atr_mult * df['atr']
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all standard indicators and EMAs to the DataFrame.
    """
    df = df.copy()
    df = add_ema_columns(df)
    df = compute_macd(df)
    df['vwap'] = compute_vwap(df)
    df['RSI'] = compute_rsi(df)
    df['atr'] = compute_atr(df)
    adx, plus_di, minus_di = compute_adx(df)
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['cci'] = calculate_cci(df)
    stochrsi_k, stochrsi_d = calculate_stochrsi(df)
    df['stochrsi_k'] = stochrsi_k
    df['stochrsi_d'] = stochrsi_d
    df = add_bollinger_bands(df)
    df = add_keltner_channels(df)
    return df
