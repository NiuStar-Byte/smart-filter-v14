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
    ema_spans = [6, 9, 10, 12, 13, 20, 21, 26, 50, 100, 200]  # Added EMA100 for 30min gate (2026-02-27)
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
    Uses standard Wilder's smoothing for DM and ATR.
    """
    df = df.copy()
    # Calculate directional movement
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # True range
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

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

def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Williams %R indicator.
    Returns a pandas Series.
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    return williams_r

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

def compute_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Choppiness Index and returns it as a pandas Series.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = tr.rolling(period).mean()
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    choppiness = 100 * np.log10(atr.rolling(period).sum() / (highest_high - lowest_low)) / np.log10(period)
    return choppiness

# Ensure all indicator functions are imported or defined above this function.

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all standard indicators and EMAs to the DataFrame.
    Ensures all used columns are present for filter/detector logic.
    Diagnostic logs are now removed/commented out for production.
    """
    df = df.copy()
    # -- Diagnostics commented out for production --
    # print(f"[add_indicators] DataFrame shape: {df.shape}")
    # required_cols = ['high', 'low', 'close']
    # print(f"[add_indicators] Columns: {df.columns.tolist()}")
    # for col in required_cols:
    #    if col in df.columns:
    #        nan_count = df[col].isna().sum()
    #        print(f"[add_indicators] NaN count in '{col}': {nan_count}")
    #    else:
    #        print(f"[add_indicators] Missing column: {col}")

    df = add_ema_columns(df)
    df = compute_macd(df)
    df['vwap'] = compute_vwap(df)
    df['RSI'] = compute_rsi(df)
    df['atr'] = compute_atr(df)
    df['atr_ma'] = df['atr'].rolling(14).mean()

    # ADX calculation
    adx, plus_di, minus_di = compute_adx(df)
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # -- ADX diagnostics commented out for production --
    # adx_valid_count = df['adx'].notna().sum()
    # print(f"[add_indicators] ADX non-NaN count: {adx_valid_count} / {len(df)}")
    # if adx_valid_count < 5:
    #     print("[add_indicators] Warning: ADX has fewer than 5 valid values. Check input data length and format.")
    # print("[add_indicators] ADX last 20 values:")
    # print(df['adx'].tail(20))

    df['cci'] = calculate_cci(df)
    stochrsi_k, stochrsi_d = calculate_stochrsi(df)
    df['stochrsi_k'] = stochrsi_k
    df['stochrsi_d'] = stochrsi_d
    df = add_bollinger_bands(df)
    df = add_keltner_channels(df)
    df['chop_zone'] = compute_choppiness_index(df)
    df['williams_r'] = compute_williams_r(df)
    return df


# ==================== CONSOLIDATED CALCULATION FUNCTIONS ====================

def calculate_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate True Range for ATR and TP/SL calculations.
    TR = MAX(high - low, |high - close[prev]|, |low - close[prev]|)
    
    Used by both ATR (add_indicators) and TP/SL calculations.
    Consolidated here to avoid duplication.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period for rolling average (for ATR)
    
    Returns:
        pd.Series: True Range values
    """
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr
    except Exception as e:
        print(f"[calculate_true_range] Error: {e}", flush=True)
        return pd.Series(0, index=df.index)


def calculate_atr_for_tp_sl(df: pd.DataFrame, entry_price: float, lookback: int = 14) -> float:
    """
    Calculate ATR value for TP/SL calculations.
    Reuses True Range calculation to avoid duplication.
    
    Args:
        df: DataFrame with OHLC data
        entry_price: Entry price (used as fallback if ATR too small)
        lookback: ATR period (default 14)
    
    Returns:
        float: ATR value
    """
    try:
        tr = calculate_true_range(df, lookback)
        atr = tr.rolling(max(1, lookback), min_periods=1).mean()
        atr_val = float(atr.iat[-1]) if len(atr) > 0 else 0.0
        
        if np.isnan(atr_val) or atr_val <= 0:
            # Fallback: 1% of entry price
            atr_val = max(entry_price * 0.01, 0.0001)
        
        return atr_val
    except Exception as e:
        print(f"[calculate_atr_for_tp_sl] Error: {e}, using entry_price * 0.01 as fallback", flush=True)
        return max(entry_price * 0.01, 0.0001)


def calculate_tp_sl_from_atr(entry_price: float, atr_value: float, direction: str,
                             atr_mult_tp: float = 1.25, atr_mult_sl: float = 1.0,
                             regime: str = None) -> dict:
    """
    Calculate TP/SL from ATR value using 1.25:1 Risk:Reward ratio (FALLBACK - market-driven preferred).
    
    UNIFORM 1.25:1 RR (FALLBACK when market-driven not available):
    LONG:  TP = Entry + (1.25 × ATR), SL = Entry - (1.0 × ATR)
    SHORT: TP = Entry - (1.25 × ATR), SL = Entry + (1.0 × ATR)
    
    RULES (2026-03-19 FIXED):
    - Market-driven TP/SL PREFERRED: Uses actual support/resistance
    - If market-driven RR > 2.5:1: CAP at 2.5:1 max
    - FALLBACK (no market structure): Use 1.25:1 ATR ratio
    
    Rationale: User specified 1.25:1 fallback, 2.5:1 market-driven cap.
    
    Args:
        entry_price: Entry price
        atr_value: ATR value
        direction: "LONG" or "SHORT"
        atr_mult_tp: ATR multiplier for TP (default 1.5)
        atr_mult_sl: ATR multiplier for SL (default 1.0)
        regime: Market regime (not used for multiplier adjustment anymore)
    
    Returns:
        dict: {'tp': float, 'sl': float, 'achieved_rr': float, 'source': str}
    """
    try:
        # DISABLED: OPTION C was adding 1.5x multiplier for RANGE regime
        # User requested flat 1.5:1 RR across ALL regimes (no special RANGE adjustment)
        # if regime == "RANGE":
        #     atr_mult_tp = atr_mult_tp * 1.5  # 2.0 → 3.0
        
        dir_up = str(direction).strip().upper() == "LONG"
        dir_down = str(direction).strip().upper() == "SHORT"
        
        if dir_up:
            tp = entry_price + (atr_mult_tp * atr_value)
            sl = entry_price - (atr_mult_sl * atr_value)
            source = "atr_1_25_to_1_long"
        elif dir_down:
            tp = entry_price - (atr_mult_tp * atr_value)
            sl = entry_price + (atr_mult_sl * atr_value)
            source = "atr_1_25_to_1_short"
        else:
            # Default to LONG
            tp = entry_price + (atr_mult_tp * atr_value)
            sl = entry_price - (atr_mult_sl * atr_value)
            source = "atr_1_25_to_1_default_long"
        
        # Calculate achieved RR from ACTUAL TP/SL distances (not just multiplier ratio)
        # FIX #1: Properly calculate RR instead of hardcoding to 1.5
        try:
            reward = abs(tp - entry_price)
            risk = abs(entry_price - sl)
            if risk > 0:
                achieved_rr = round(reward / risk, 2)
            else:
                achieved_rr = 1.5
            print(f"[FIX1-ACTIVE] dir={direction} entry={entry_price:.8f} tp={tp:.8f} sl={sl:.8f} → reward={reward:.8f} risk={risk:.8f} rr={achieved_rr}", flush=True)
        except Exception as e:
            print(f"[FIX1-ERROR] Exception in RR calc: {e}", flush=True)
            achieved_rr = 1.5
        
        return {
            'tp': round(float(tp), 8),
            'sl': round(float(sl), 8),
            'achieved_rr': achieved_rr,
            'atr_value': float(atr_value),
            'source': source,
            'fib_levels': None,
            'chosen_ratio': None,
            'sl_capped': False
        }
    except Exception as e:
        print(f"[calculate_tp_sl_from_atr] Error: {e}", flush=True)
        return {
            'tp': entry_price + atr_value,
            'sl': entry_price - atr_value,
            'achieved_rr': 2.0,
            'atr_value': float(atr_value),
            'source': 'exception_fallback',
            'fib_levels': None,
            'chosen_ratio': None,
            'sl_capped': False
        }


def calculate_pnl(entry_price: float, exit_price: float, direction: str, 
                  notional_position: float = 100.0) -> dict:
    """
    Calculate P&L (USD and percentage) for a closed trade.
    
    Formula: 
      - P&L% = ((exit - entry) / entry) × 100
      - P&L USD = ((exit - entry) / entry) × notional_position
    
    Notional position represents leverage-adjusted capital:
      - Default: $10 margin × 10x leverage = $100 notional
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        direction: "LONG" or "SHORT"
        notional_position: Notional position size for P&L calc (default $100 = $10 margin × 10x leverage)
    
    Returns:
        dict: {
            'pnl_usd': float,
            'pnl_pct': float,
            'direction': str
        }
    """
    try:
        entry = float(entry_price)
        exit_val = float(exit_price)
        
        if entry == 0:
            print(f"[calculate_pnl] Warning: entry_price is 0, returning 0 P&L", flush=True)
            return {'pnl_usd': 0.0, 'pnl_pct': 0.0, 'direction': direction}
        
        dir_up = str(direction).strip().upper() == "LONG"
        dir_down = str(direction).strip().upper() == "SHORT"
        
        if dir_up:
            # LONG: profit if exit > entry
            pnl_pct = ((exit_val - entry) / entry) * 100
            pnl_usd = ((exit_val - entry) / entry) * notional_position
        elif dir_down:
            # SHORT: profit if exit < entry
            pnl_pct = ((entry - exit_val) / entry) * 100
            pnl_usd = ((entry - exit_val) / entry) * notional_position
        else:
            # Default to LONG
            pnl_pct = ((exit_val - entry) / entry) * 100
            pnl_usd = ((exit_val - entry) / entry) * notional_position
        
        return {
            'pnl_usd': round(float(pnl_usd), 4),
            'pnl_pct': round(float(pnl_pct), 2),
            'direction': direction
        }
    except Exception as e:
        print(f"[calculate_pnl] Error: {e}", flush=True)
        return {'pnl_usd': 0.0, 'pnl_pct': 0.0, 'direction': direction}


def detect_early_breakout(df: pd.DataFrame, lookback: int = 3) -> bool:
    """
    Detect if price has broken out above recent high.
    
    LONG: Current close > close from N candles ago (uptrend confirmation)
    
    Args:
        df: DataFrame with 'close' column
        lookback: Number of candles to look back (default 3)
    
    Returns:
        bool: True if breakout detected, False otherwise
    """
    try:
        if df is None or len(df) < lookback + 1:
            return False
        
        current_close = df['close'].iat[-1]
        past_close = df['close'].iat[-lookback-1]
        
        return bool(current_close > past_close)
    except Exception as e:
        print(f"[detect_early_breakout] Error: {e}", flush=True)
        return False


def calculate_tp_sl_from_df(df, entry_price, direction, regime=None):
    """
    MARKET-DRIVEN TP/SL using actual price structure (CORRECTED 2026-03-18).
    
    Strategy:
    1. MARKET-DRIVEN (PREFERRED): Use recent swing highs/lows as S&R
       - Calculate natural RR from actual price structure
       - Accept if RR is within reasonable bounds (0.5-4.0)
    
    2. FALLBACK (IF NO S&R): Use fixed 1.25:1 RR ratio
       - Risk = ATR-based distance
       - Reward = Risk × 1.25 (proper 1.25:1 ratio in distance)
       - This ensures risk/reward is properly calibrated
    
    Args:
        df: DataFrame with OHLCV data
        entry_price: Entry price
        direction: "LONG" or "SHORT"
        regime: Market regime (optional)
    
    Returns:
        dict: {'tp': float, 'sl': float, 'achieved_rr': float, 'source': str}
        or None: If no valid setup found (will use ATR fallback)
    """
    try:
        if df is None or len(df) < 5:
            return None
        
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        current_price = close.iloc[-1]
        
        # Find recent swing highs and lows (last 20 candles)
        lookback = min(20, len(df))
        recent_highs = high.tail(lookback).nlargest(3).values
        recent_lows = low.tail(lookback).nsmallest(3).values
        
        # Filter for actual support/resistance levels
        supports = sorted([x for x in recent_lows if x < current_price], reverse=True)
        resistances = sorted([x for x in recent_highs if x > current_price])
        
        # === MARKET-DRIVEN: Use S&R if both exist ===
        if direction.upper() == "LONG":
            if supports and resistances:
                # Both S&R exist - use natural market structure
                sl = supports[0]  # Nearest support
                tp = resistances[0]  # Nearest resistance
                # CRITICAL FIX: Use entry_price for RR calculation, not current_price!
                reward = tp - entry_price
                risk = entry_price - sl
                achieved_rr = round(reward / risk, 2) if risk > 0 else 0
                
                # Quality gate: reject if RR unrealistic (0.5-4.0) OR cap if RR > 2.5:1
                if achieved_rr < 0.5 or achieved_rr > 4.0:
                    print(f"[MARKET_DRIVEN] REJECTED {direction}: RR {achieved_rr} outside bounds (0.5-4.0)", flush=True)
                    return None
                
                # CAP at 2.5:1 max (2026-03-19 FIXED)
                if achieved_rr > 2.5:
                    tp_capped = current_price + (risk * 2.5)
                    achieved_rr = 2.5
                    sl_capped_flag = True
                    print(f"[MARKET_DRIVEN] CAP {direction}: RR reduced from {round(reward/risk,2)} to 2.5:1", flush=True)
                else:
                    tp_capped = float(tp)
                    sl_capped_flag = False
                
                return {
                    'tp': float(tp_capped),
                    'sl': float(sl),
                    'achieved_rr': achieved_rr,
                    'source': 'market_structure_long',
                    'reward': float(risk * achieved_rr),
                    'risk': float(risk),
                    'fib_levels': None,
                    'chosen_ratio': None,
                    'sl_capped': sl_capped_flag
                }
            
            # === FALLBACK: Use 1.25:1 RR with ATR ===
            elif supports:
                # Only support exists - use fixed 1.25:1 RR
                atr = calculate_atr_for_tp_sl(df, entry_price, 14)
                sl = supports[0]
                risk = current_price - sl
                reward = risk * 1.25  # Fixed 1.25:1 ratio
                tp = current_price + reward
                achieved_rr = 1.25
                
                return {
                    'tp': float(tp),
                    'sl': float(sl),
                    'achieved_rr': achieved_rr,
                    'source': 'fallback_125_long',
                    'reward': float(reward),
                    'risk': float(risk),
                    'fib_levels': None,
                    'chosen_ratio': 1.25,
                    'sl_capped': False
                }
        
        else:  # SHORT
            if supports and resistances:
                # Both S&R exist - use natural market structure
                sl = resistances[0]  # Nearest resistance
                tp = supports[0]  # Nearest support
                # CRITICAL FIX: Use entry_price for RR calculation, not current_price!
                reward = entry_price - tp
                risk = sl - entry_price
                achieved_rr = round(reward / risk, 2) if risk > 0 else 0
                
                # Quality gate: reject if RR unrealistic (0.5-4.0) OR cap if RR > 2.5:1
                if achieved_rr < 0.5 or achieved_rr > 4.0:
                    print(f"[MARKET_DRIVEN] REJECTED {direction}: RR {achieved_rr} outside bounds (0.5-4.0)", flush=True)
                    return None
                
                # CAP at 2.5:1 max (2026-03-19 FIXED)
                if achieved_rr > 2.5:
                    tp_capped = current_price - (risk * 2.5)
                    achieved_rr = 2.5
                    sl_capped_flag = True
                    print(f"[MARKET_DRIVEN] CAP {direction}: RR reduced from {round(reward/risk,2)} to 2.5:1", flush=True)
                else:
                    tp_capped = float(tp)
                    sl_capped_flag = False
                
                return {
                    'tp': float(tp_capped),
                    'sl': float(sl),
                    'achieved_rr': achieved_rr,
                    'source': 'market_structure_short',
                    'reward': float(risk * achieved_rr),
                    'risk': float(risk),
                    'fib_levels': None,
                    'chosen_ratio': None,
                    'sl_capped': sl_capped_flag
                }
            
            # === FALLBACK: Use 1.25:1 RR with ATR ===
            elif resistances:
                # Only resistance exists - use fixed 1.25:1 RR
                atr = calculate_atr_for_tp_sl(df, entry_price, 14)
                sl = resistances[0]
                risk = sl - current_price
                reward = risk * 1.25  # Fixed 1.25:1 ratio
                tp = current_price - reward
                achieved_rr = 1.25
                
                return {
                    'tp': float(tp),
                    'sl': float(sl),
                    'achieved_rr': achieved_rr,
                    'source': 'fallback_125_short',
                    'reward': float(reward),
                    'risk': float(risk),
                    'fib_levels': None,
                    'chosen_ratio': 1.25,
                    'sl_capped': False
                }
        
        # No valid setup found
        return None
    
    except Exception as e:
        print(f"[calculate_tp_sl_from_df] Error: {e}", flush=True)
        return None
