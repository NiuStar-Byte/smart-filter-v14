#!/usr/bin/env python3
"""
Analyze what DIFFERS between timeframes besides volatility
"""
import sys
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main')

from kucoin_data import get_ohlcv
import pandas as pd

symbol = "ETH-USDT"

print("=" * 100)
print(f"ANALYZING {symbol} - WHAT DIFFERS BETWEEN TIMEFRAMES?")
print("=" * 100)
print()

data = {}
for tf in ["15min", "30min", "1h", "2h", "4h"]:
    df = get_ohlcv(symbol, tf, limit=250)
    if df is None or df.empty:
        continue
    data[tf] = df

# 1. VOLATILITY METRICS
print("1️⃣ VOLATILITY METRICS")
print("-" * 100)
for tf, df in data.items():
    atr = df['high'].iloc[-20:].subtract(df['low'].iloc[-20:]).mean()
    volatility = df['close'].std()
    vol_pct = (df['close'].pct_change().abs().mean()) * 100
    print(f"{tf:8} | ATR: {atr:8.4f} | Close StdDev: {volatility:8.2f} | Avg % Change: {vol_pct:6.2f}%")

# 2. TREND CHARACTERISTICS
print("\n2️⃣ TREND CHARACTERISTICS (Do longer TFs have longer trends?)")
print("-" * 100)
for tf, df in data.items():
    # Calculate trend duration (how many consecutive up/down closes)
    returns = df['close'].pct_change()
    direction = (returns > 0).astype(int)
    changes = direction.diff().fillna(0)
    trend_runs = (changes != 0).cumsum()
    max_trend_length = trend_runs.value_counts().max()
    avg_trend_length = trend_runs.value_counts().mean()
    
    print(f"{tf:8} | Max consecutive trend bars: {max_trend_length:3.0f} | Avg trend length: {avg_trend_length:4.1f}")

# 3. SUPPORT/RESISTANCE EFFECTIVENESS
print("\n3️⃣ SUPPORT/RESISTANCE CHARACTERISTICS")
print("-" * 100)
for tf, df in data.items():
    # How often does price touch S/R?
    ranges = df['high'].iloc[-20:] - df['low'].iloc[-20:]
    avg_range = ranges.mean()
    range_pct = (avg_range / df['close'].iloc[-1]) * 100
    print(f"{tf:8} | Avg range per candle: {range_pct:6.2f}% | S/R proximity (2%) often hit: {range_pct > 2.0}")

# 4. MEAN REVERSION vs TREND FOLLOWING
print("\n4️⃣ MEAN REVERSION vs TREND FOLLOWING")
print("-" * 100)
for tf, df in data.items():
    # Autocorrelation test: do prices tend to revert or continue?
    returns = df['close'].pct_change().dropna()
    lag1_autocorr = returns.corr(returns.shift(1))
    
    if lag1_autocorr > 0.1:
        bias = "🔄 TRENDING (prices continue)"
    elif lag1_autocorr < -0.1:
        bias = "↩️  MEAN REVERTING (prices revert)"
    else:
        bias = "➡️  NEUTRAL (random walk)"
    
    print(f"{tf:8} | Autocorrelation: {lag1_autocorr:+7.3f} | {bias}")

# 5. NOISE LEVEL
print("\n5️⃣ NOISE LEVEL (Intrabar vs trend)")
print("-" * 100)
for tf, df in data.items():
    # Noise = (High-Low) vs (Close movement)
    hl_range = df['high'] - df['low']
    cl_movement = df['close'].diff().abs()
    noise_ratio = hl_range.mean() / cl_movement.mean() if cl_movement.mean() > 0 else 0
    
    print(f"{tf:8} | (High-Low) / Close_movement = {noise_ratio:5.2f} | More noise = higher ratio")

# 6. REVERSAL CHARACTERISTICS
print("\n6️⃣ REVERSAL PATTERN CHARACTERISTICS")
print("-" * 100)
for tf, df in data.items():
    # How often do we see reversals (3 consecutive direction changes)?
    returns = df['close'].pct_change()
    direction = (returns > 0).astype(int)
    changes = direction.diff().fillna(0)
    reversals = (changes.abs() == 1).sum()  # Direction change
    reversal_rate = (reversals / len(df)) * 100
    
    print(f"{tf:8} | Direction reversals: {reversal_rate:5.1f}% of candles | Pattern frequency")

# 7. LOOKBACK PERIOD EFFECTIVENESS
print("\n7️⃣ LOOKBACK PERIOD EFFECTIVENESS (Is 20-bar MA meaningful for each TF?)")
print("-" * 100)
for tf, df in data.items():
    # 20-bar lookback = ?
    lookback_hours = 0
    if tf == "15min":
        lookback_hours = 20 * 0.25
    elif tf == "30min":
        lookback_hours = 20 * 0.5
    elif tf == "1h":
        lookback_hours = 20 * 1
    elif tf == "2h":
        lookback_hours = 20 * 2
    elif tf == "4h":
        lookback_hours = 20 * 4
    
    print(f"{tf:8} | 20-bar MA = {lookback_hours:6.1f} hours | {'✅ Good' if 6 < lookback_hours < 96 else '⚠️  Maybe too short/long'}")

print("\n" + "=" * 100)
print("SUMMARY: What needs TF-specific adjustment?")
print("=" * 100)
