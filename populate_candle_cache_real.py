#!/usr/bin/env python3
"""
populate_candle_cache_real.py

Fetch real historical candles from KuCoin API and populate candle_cache.
Integrates with the existing ohlcv_fetch_safe module used by main.py.

This replaces synthetic candles with actual market data for accurate backtesting.

Author: Nox
Date: 2026-03-03
"""

import json
import os
import sys
from datetime import datetime

try:
    from kucoin_data import get_ohlcv
    from ohlcv_fetch_safe import safe_fetch_ohlcv_by_tf
except ImportError as e:
    print(f"❌ ERROR: Could not import KuCoin modules: {e}")
    print()
    print("Make sure you're in the smart-filter-v14-main directory with:")
    print("  - kucoin_data.py")
    print("  - ohlcv_fetch_safe.py")
    sys.exit(1)

def load_sent_signals():
    """Load SENT_SIGNALS.jsonl and extract unique symbols/TFs"""
    signals = []
    filepath = 'SENT_SIGNALS.jsonl'
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} not found")
        return []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    signals.append(json.loads(line))
        print(f"✅ Loaded {len(signals)} signals from SENT_SIGNALS.jsonl")
        return signals
    except Exception as e:
        print(f"❌ Error loading SENT_SIGNALS.jsonl: {e}")
        return []

def extract_symbol_tfs(signals):
    """Extract unique (symbol, timeframe) pairs from signals"""
    symbol_tfs = set()
    
    for sig in signals:
        symbol = sig.get('symbol')
        timeframe = sig.get('timeframe')
        
        if symbol and timeframe:
            symbol_tfs.add((symbol, timeframe))
    
    return sorted(list(symbol_tfs))

def fetch_real_candles(symbol, timeframes=['15min', '30min', '1h']):
    """
    Fetch real historical candles for a symbol using KuCoin API.
    
    Returns: dict of {timeframe: candles_list}
    """
    result = {}
    
    try:
        # Use existing safe_fetch_ohlcv_by_tf from main.py
        ohlcv_data = safe_fetch_ohlcv_by_tf(symbol, get_ohlcv)
        
        if not ohlcv_data:
            print(f"   ⚠️  {symbol}: No data fetched")
            return result
        
        for timeframe in timeframes:
            if timeframe in ohlcv_data:
                df = ohlcv_data[timeframe]
                if df is not None and len(df) > 0:
                    # Convert DataFrame to list of dicts
                    candles = []
                    for idx, row in df.iterrows():
                        candle = {
                            'timestamp': int(row['timestamp']) if 'timestamp' in row else int(idx.timestamp()) if hasattr(idx, 'timestamp') else 0,
                            'datetime': str(idx) if hasattr(idx, '__str__') else str(row.get('date', '')),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row.get('volume', 0))
                        }
                        candles.append(candle)
                    
                    result[timeframe] = candles
        
        return result
    
    except Exception as e:
        print(f"   ⚠️  {symbol}: Error fetching - {type(e).__name__}: {str(e)[:60]}")
        return result

def create_real_candle_cache():
    """Fetch real candles from KuCoin and populate cache directory"""
    
    # Load signals
    signals = load_sent_signals()
    if not signals:
        print("❌ No signals to process. Aborting.")
        return
    
    # Extract unique symbol/TF pairs
    symbol_tfs = extract_symbol_tfs(signals)
    print(f"✅ Found {len(symbol_tfs)} unique symbol/TF pairs")
    
    # Get unique symbols
    symbols = sorted(set(st[0] for st in symbol_tfs))
    print(f"   Fetching for {len(symbols)} symbols...")
    print(f"   Timeframes: 15min, 30min, 1h")
    print()
    
    # Create cache directory
    cache_dir = 'candle_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"✅ Created {cache_dir}/ directory")
    
    # Fetch candles for each symbol
    print("Fetching real candle data from KuCoin...")
    print()
    
    success_count = 0
    error_count = 0
    
    for symbol in symbols:
        print(f"  ↳ {symbol:<20}", end=' ', flush=True)
        
        # Fetch for this symbol
        ohlcv_dict = fetch_real_candles(symbol)
        
        if not ohlcv_dict:
            print("❌ No data")
            error_count += 1
            continue
        
        # Save each timeframe
        saved_count = 0
        for timeframe, candles in ohlcv_dict.items():
            if candles:
                filename = f"{symbol}_{timeframe}.json"
                filepath = os.path.join(cache_dir, filename)
                
                try:
                    with open(filepath, 'w') as f:
                        json.dump(candles, f, indent=2)
                    saved_count += 1
                except Exception as e:
                    print(f"      ERROR writing {filename}: {e}")
        
        if saved_count > 0:
            print(f"✅ ({saved_count} TF candles saved)")
            success_count += 1
        else:
            print("⚠️  No TF data saved")
            error_count += 1
    
    print()
    print("=" * 80)
    print(f"✅ Candle cache updated: {success_count} symbols with real data")
    print(f"⚠️  Symbols with issues: {error_count}")
    print()
    print(f"📁 Cache location: {cache_dir}/")
    print()
    print("🚀 Next step: Run backtest_multitf_alignment.py with real data")
    print()

if __name__ == '__main__':
    create_real_candle_cache()
