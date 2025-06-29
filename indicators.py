# Function_ID_01_v1: get_local_wib
def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').strftime('%H:%M WIB')

# Function_ID_02_v1: get_resting_order_density - REMOVE or refactor as per main.py's logic.
# Already handled in main.py, if needed only log here.
# Refactored logging mechanism can be moved to main.py.

# Function_ID_03_v1: log_orderbook_and_density - REMOVE
# The functionality is now incorporated directly into main.py, specifically after signal analysis.

# Function_ID_04_v1: calculate_rsi
def calculate_rsi_04(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function_ID_05_v1: calculate_bollinger_bands
def calculate_bollinger_bands_05(df, window=20):
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * 2)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * 2)
    return df

# Function_ID_06_v1: calculate_stochastic_oscillator
def calculate_stochastic_oscillator_06(df, window=14):
    df['stochastic'] = ((df['close'] - df['low'].rolling(window=window).min()) /
                        (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())) * 100
    return df

# Function_ID_07_v1: calculate_supertrend
def calculate_supertrend_07(df, period=7, multiplier=3):
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    df['upper_band'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['lower_band'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
    return df

# Function_ID_08_v1: calculate_atr
def calculate_atr_08(df, period=14):
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    return df

# Function_ID_09_v1: calculate_parabolic_sar
def calculate_parabolic_sar_09(df, acceleration=0.02, maximum=0.2):
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

# Function_ID_10_v1: calculate_adx
def calculate_adx_10(df, period=14):
    df['+DI'] = df['high'].diff()
    df['-DI'] = df['low'].diff()
    df['ADX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    return df

# Function_ID_11_v1: calculate_market_structure
def calculate_market_structure_11(df):
    df['market_structure'] = 'None'
    for i in range(2, len(df)):
        if df['high'][i] > df['high'][i-1] and df['low'][i] > df['low'][i-1]:
            df['market_structure'][i] = 'Uptrend'
        elif df['high'][i] < df['high'][i-1] and df['low'][i] < df['low'][i-1]:
            df['market_structure'][i] = 'Downtrend'
        else:
            df['market_structure'][i] = 'Sideways'
    return df

# Function_ID_12_v1: calculate_support_resistance
def calculate_support_resistance_12(df, period=20):
    df['support'] = df['low'].rolling(window=period).min()
    df['resistance'] = df['high'].rolling(window=period).max()
    return df

# Function_ID_13_v1: calculate_pivot_points
def calculate_pivot_points_13(df):
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['support_1'] = (2 * df['pivot']) - df['high']
    df['resistance_1'] = (2 * df['pivot']) - df['low']
    return df

# Function_ID_14_v1: calculate_composite_trend_indicator
def calculate_composite_trend_indicator_14(df):
    df['CTI'] = (df['close'] - df['open']) / (df['high'] - df['low']) * 100
    return df
