#!/usr/bin/env python3
"""
populate_candle_cache.py

Generate candle_cache from SENT_SIGNALS.jsonl by extracting symbol/TF info
and creating synthetic candle data based on signal timestamps.

This creates mock candles that allow multi-TF alignment backtesting to work.
Real candle data would be fetched from your data source instead.

Author: Nox
Date: 2026-03-03
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

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

def generate_synthetic_candles(symbol, timeframe, num_candles=20):
    """
    Generate synthetic candle data for a symbol/TF pair.
    
    In a real implementation, you would fetch actual historical candles
    from your data source (Binance API, etc).
    
    For now, generate plausible synthetic candles with:
    - Closing prices with trend
    - Reasonable OHLC
    - Volume data
    """
    
    candles = []
    base_price = 100.0  # Starting reference price
    
    now = datetime.utcnow()
    
    # Timeframe to minutes mapping
    tf_to_minutes = {
        '15min': 15,
        '30min': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    minutes = tf_to_minutes.get(timeframe, 60)
    
    # Generate candles backwards from now
    for i in range(num_candles, 0, -1):
        candle_time = now - timedelta(minutes=minutes * i)
        
        # Trend: random walk
        trend = (i % 10) - 5  # -5 to +5 trend per 10 candles
        close = base_price + (trend * 0.5) + (i % 3 - 1)
        
        # OHLC with realistic spread
        open_price = base_price + (trend * 0.3)
        high = max(open_price, close) * (1 + 0.01 * (i % 3))
        low = min(open_price, close) * (1 - 0.01 * (i % 2))
        
        # Volume (synthetic)
        volume = 1000000 + (i * 10000)
        
        candle = {
            'timestamp': int(candle_time.timestamp()),
            'datetime': candle_time.isoformat(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        }
        
        candles.append(candle)
        base_price = close  # Next candle builds on last close
    
    return candles

def create_candle_cache():
    """Create candle_cache directory and populate with symbol/TF candles"""
    
    # Load signals
    signals = load_sent_signals()
    if not signals:
        print("❌ No signals to process. Aborting.")
        return
    
    # Extract unique symbol/TF pairs
    symbol_tfs = extract_symbol_tfs(signals)
    print(f"✅ Found {len(symbol_tfs)} unique symbol/TF pairs")
    print(f"   Symbols: {set(st[0] for st in symbol_tfs)}")
    print(f"   Timeframes: {set(st[1] for st in symbol_tfs)}")
    print()
    
    # Create cache directory
    cache_dir = 'candle_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"✅ Created {cache_dir}/ directory")
    
    # Generate candles for each symbol/TF pair
    print()
    print("Generating candle data...")
    print()
    
    for symbol, timeframe in symbol_tfs:
        filename = f"{symbol}_{timeframe}.json"
        filepath = os.path.join(cache_dir, filename)
        
        # Generate synthetic candles
        candles = generate_synthetic_candles(symbol, timeframe, num_candles=20)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(candles, f, indent=2)
        
        print(f"✅ {filename:<30} ({len(candles)} candles)")
    
    print()
    print("=" * 80)
    print(f"✅ Candle cache populated: {len(symbol_tfs)} files in {cache_dir}/")
    print()
    print("📝 NOTE: Using synthetic candle data for testing")
    print("   To use real market data, update this script to fetch from:")
    print("   - Binance API")
    print("   - Your historical data store")
    print("   - CSV/database files")
    print()
    print("🚀 Next step: Run backtest_multitf_alignment.py again")
    print()

if __name__ == '__main__':
    create_candle_cache()
