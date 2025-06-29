# Function_ID_01_v1: get_local_wib
def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').strftime('%H:%M WIB')

# Function_ID_02_v1: get_resting_order_density
def get_resting_order_density(symbol, depth=100, band_pct=0.005):
    try:
        from kucoin_orderbook import fetch_orderbook
        bids, asks = fetch_orderbook(symbol, depth)
        if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
            return {'bid_density': 0.0, 'ask_density': 0.0, 'bid_levels': 0, 'ask_levels': 0, 'midprice': None}
        best_bid = bids['price'].iloc[0]
        best_ask = asks['price'].iloc[0]
        midprice = (best_bid + best_ask) / 2
        low, high = midprice * (1 - band_pct), midprice * (1 + band_pct)
        bids_in_band = bids[bids['price'] >= low]
        asks_in_band = asks[asks['price'] <= high]
        bid_density = bids_in_band['size'].sum() / max(len(bids_in_band), 1)
        ask_density = asks_in_band['size'].sum() / max(len(asks_in_band), 1)
        return {'bid_density': float(bid_density), 'ask_density': float(ask_density),
                'bid_levels': len(bids_in_band), 'ask_levels': len(asks_in_band), 'midprice': float(midprice)}
    except Exception:
        return {'bid_density': 0.0, 'ask_density': 0.0, 'bid_levels': 0, 'ask_levels': 0, 'midprice': None}

# Function_ID_03_v1: log_orderbook_and_density
def log_orderbook_and_density(symbol):
    try:
        result = get_order_wall_delta(symbol)
        print(
            f"[OrderBookDeltaLog] {symbol} | "
            f"buy_wall={result['buy_wall']} | "
            f"sell_wall={result['sell_wall']} | "
            f"wall_delta={result['wall_delta']} | "
            f"midprice={result['midprice']}"
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}")
    try:
        dens = get_resting_order_density(symbol)
        print(
            f"[RestingOrderDensityLog] {symbol} | "
            f"bid_density={dens['bid_density']:.2f} | ask_density={dens['ask_density']:.2f} | "
            f"bid_levels={dens['bid_levels']} | ask_levels={dens['ask_levels']} | midprice={dens['midprice']}"
        )
    except Exception as e:
        print(f"[RestingOrderDensityLog] {symbol} ERROR: {e}")

# Function_ID_04_v1: calculate_rsi
def calculate_rsi_04(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function_ID_05_v1: calculate_bollinger_bands
def calculate_bollinger_bands(df, window=20):
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * 2)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * 2)
    return df

# Function_ID_06_v1: calculate_stochastic_oscillator
def calculate_stochastic_oscillator(df, window=14):
    df['stochastic'] = ((df['close'] - df['low'].rolling(window=window).min()) /
                        (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())) * 100
    return df

# Function_ID_07_v1: calculate_supertrend
def calculate_supertrend(df, period=7, multiplier=3):
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    df['upper_band'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['lower_band'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
    return df

# Function_ID_08_v1: calculate_atr
def calculate_atr(df, period=14):
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    return df

# Function_ID_09_v1: calculate_parabolic_sar
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

# Function_ID_10_v1: calculate_adx
def calculate_adx(df, period=14):
    df['+DI'] = df['high'].diff()
    df['-DI'] = df['low'].diff()
    df['ADX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    return df

# Function_ID_11_v1: calculate_market_structure
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

# Function_ID_12_v1: calculate_support_resistance
def calculate_support_resistance_12(df, period=20):
    df['support'] = df['low'].rolling(window=period).min()
    df['resistance'] = df['high'].rolling(window=period).max()
    return df

# Function_ID_13_v1: calculate_pivot_points
def calculate_pivot_points(df):
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['support_1'] = (2 * df['pivot']) - df['high']
    df['resistance_1'] = (2 * df['pivot']) - df['low']
    return df

# Function_ID_14_v1: calculate_composite_trend_indicator
def calculate_composite_trend_indicator(df):
    df['CTI'] = (df['close'] - df['open']) / (df['high'] - df['low']) * 100
    return df
