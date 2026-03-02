#!/usr/bin/env python3
"""
populate_4h_candles.py

Fetch real 4h + 1d candles from KuCoin API for Phase 4A-Extended
"""

import json
import os
import sys

try:
    from kucoin_data import get_ohlcv
    from ohlcv_fetch_safe import safe_fetch_ohlcv_by_tf
except ImportError as e:
    print(f"❌ ERROR: Could not import KuCoin modules: {e}")
    sys.exit(1)

def load_symbols_from_signals():
    """Get unique symbols from SENT_SIGNALS.jsonl"""
    symbols = set()
    
    try:
        with open('SENT_SIGNALS.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    if sym := sig.get('symbol'):
                        symbols.add(sym)
    except:
        pass
    
    return sorted(list(symbols))

def fetch_4h_1d_candles():
    """Fetch 4h and 1d candles for all symbols"""
    
    symbols = load_symbols_from_signals()
    print(f"✅ Found {len(symbols)} symbols to fetch")
    print()
    
    cache_dir = 'candle_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    success = 0
    failed = 0
    
    for symbol in symbols:
        print(f"  ↳ {symbol:<20}", end=' ', flush=True)
        
        try:
            # Fetch with safe wrapper
            ohlcv_dict = safe_fetch_ohlcv_by_tf(symbol, get_ohlcv)
            
            if not ohlcv_dict:
                print("❌ No data")
                failed += 1
                continue
            
            saved = 0
            
            # Save 4h if available
            if '4h' in ohlcv_dict and ohlcv_dict['4h'] is not None:
                df = ohlcv_dict['4h']
                if len(df) > 0:
                    candles_4h = []
                    for idx, row in df.iterrows():
                        candle = {
                            'timestamp': int(row['timestamp']) if 'timestamp' in row else 0,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row.get('volume', 0))
                        }
                        candles_4h.append(candle)
                    
                    filepath = os.path.join(cache_dir, f"{symbol}_4h.json")
                    with open(filepath, 'w') as f:
                        json.dump(candles_4h, f, indent=2)
                    saved += 1
            
            # Save 1d if available
            if '1d' in ohlcv_dict and ohlcv_dict['1d'] is not None:
                df = ohlcv_dict['1d']
                if len(df) > 0:
                    candles_1d = []
                    for idx, row in df.iterrows():
                        candle = {
                            'timestamp': int(row['timestamp']) if 'timestamp' in row else 0,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row.get('volume', 0))
                        }
                        candles_1d.append(candle)
                    
                    filepath = os.path.join(cache_dir, f"{symbol}_1d.json")
                    with open(filepath, 'w') as f:
                        json.dump(candles_1d, f, indent=2)
                    saved += 1
            
            if saved > 0:
                print(f"✅ ({saved} TF candles)")
                success += 1
            else:
                print("⚠️  No 4h/1d data")
                failed += 1
        
        except Exception as e:
            print(f"❌ Error: {str(e)[:40]}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"✅ Fetched 4h+1d candles: {success} symbols")
    print(f"⚠️  Failed: {failed} symbols")
    print()
    print("Files saved to: candle_cache/")
    print()

if __name__ == '__main__':
    print("=" * 80)
    print("FETCHING 4H + 1D CANDLES FOR PHASE 4A-EXTENDED")
    print("=" * 80)
    print()
    
    fetch_4h_1d_candles()
