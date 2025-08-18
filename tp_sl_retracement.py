import pandas as pd

def calculate_tp_sl(df, entry_price, direction, atr_multiplier=1.5):
    """
    df: pd.DataFrame with columns ['high', 'low', 'close']
    entry_price: float
    direction: 'LONG' or 'SHORT'
    atr_multiplier: float, how many ATRs for TP/SL buffer
    Returns: dict with 'tp', 'sl', and optionally 'fib_levels'
    """
    # Calculate ATR
    df['tr'] = df[['high', 'low', 'close']].apply(lambda row: max(row['high']-row['low'], abs(row['high']-row['close']), abs(row['low']-row['close'])), axis=1)
    atr = df['tr'].rolling(14).mean().iat[-1] if len(df) >= 14 else df['tr'].mean()
    
    # Find swing highs/lows (last N bars)
    lookback = 20
    recent_high = df['high'][-lookback:].max()
    recent_low = df['low'][-lookback:].min()
    
    # Calculate Fib retracement levels
    fib_levels = {
        '0.382': recent_low + 0.382 * (recent_high - recent_low),
        '0.5': recent_low + 0.5 * (recent_high - recent_low),
        '0.618': recent_low + 0.618 * (recent_high - recent_low),
        '1.0': recent_high,
    }
    
    if direction.upper() == "LONG":
        sl = min(recent_low, entry_price - atr_multiplier * atr)
        tp = fib_levels['1.0'] if fib_levels['1.0'] > entry_price else entry_price + atr_multiplier * atr
    elif direction.upper() == "SHORT":
        sl = max(recent_high, entry_price + atr_multiplier * atr)
        tp = fib_levels['0.0'] if recent_low < entry_price else entry_price - atr_multiplier * atr
    else:
        sl = None
        tp = None
    
    return {'tp': round(tp, 6), 'sl': round(sl, 6), 'fib_levels': fib_levels}
