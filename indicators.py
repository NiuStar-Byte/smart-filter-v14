import pandas as pd
import numpy as np

# 1. RSI Calculation
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 2. Bollinger Bands Calculation
def calculate_bollinger_bands(df, window=20):
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * 2)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * 2)
    return df

# 3. Stochastic Oscillator Calculation
def calculate_stochastic_oscillator(df, window=14):
    df['stochastic'] = ((df['close'] - df['low'].rolling(window=window).min()) /
                        (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())) * 100
    return df

# 4. SuperTrend Calculation
def calculate_supertrend(df, period=7, multiplier=3):
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    df['upper_band'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['lower_band'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
    return df

# 5. Average True Range (ATR) Calculation
def calculate_atr(df, period=14):
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    return df

# 6. Parabolic SAR Calculation
def calculate_parabolic_sar(df, acceleration=0.02, maximum=0.2):
    df['sar'] = df['close'].copy()
    up_trend = True
    ep = df['high'][0]
    af = acceleration
    sar = df['sar'][0]
    
    for i in range(1, len(df)):
        if up_trend:
            sar = sar + af * (ep - sar)
            if df['low'][i] < sar:
                up_trend = False
                sar = ep
                ep = df['low'][i]
                af = acceleration
        else:
            sar = sar + af * (ep - sar)
            if df['high'][i] > sar:
                up_trend = True
                sar = ep
                ep = df['high'][i]
                af = acceleration
        
        df['sar'][i] = sar
    return df

# 7. Average Directional Index (ADX) Calculation
def calculate_adx(df, period=14):
    df['+DI'] = df['high'].diff()
    df['-DI'] = df['low'].diff()
    df['ADX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    return df

# 8. Market Structure Filter: Higher Highs, Higher Lows (or Lower Lows, Lower Highs)
def calculate_market_structure(df):
    df['market_structure'] = 'None'
    for i in range(2, len(df)):
        if df['high'][i] > df['high'][i-1] and df['low'][i] > df['low'][i-1]:
            df['market_structure'][i] = 'Uptrend'
        elif df['high'][i] < df['high'][i-1] and df['low'][i] < df['low'][i-1]:
            df['market_structure'][i] = 'Downtrend'
        else:
            df['market_structure'][i] = 'Sideways'
    return df

# 9. Support and Resistance Zones Calculation
def calculate_support_resistance(df, period=20):
    df['support'] = df['low'].rolling(window=period).min()
    df['resistance'] = df['high'].rolling(window=period).max()
    return df

# 10. Pivot Points Calculation
def calculate_pivot_points(df):
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['support_1'] = (2 * df['pivot']) - df['high']
    df['resistance_1'] = (2 * df['pivot']) - df['low']
    return df

# 11. Composite Trend Indicator (CTI)
def calculate_composite_trend_indicator(df):
    df['CTI'] = (df['close'] - df['open']) / (df['high'] - df['low']) * 100
    return df
