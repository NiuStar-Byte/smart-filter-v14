#!/usr/bin/env python3
"""
ENHANCED 1D Daily Candle Fetcher for MTF Alignment Analysis
Fetches daily candles from KuCoin public API and extracts:
- direction: LONG/SHORT (from current candle open/close)
- regime: BULL/BEAR/RANGE (from 20-day SMA comparison + multi-candle analysis)
- route: REVERSAL/TREND_CONTINUATION/NONE (from directional change detection)

Last updated: 2026-06-07 (OPTION B FIX)
"""

import requests
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List

class MTF1DFetcherEnhanced:
    """Enhanced daily candle fetcher with regime + route detection"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.base_url = "https://api.kucoin.com"
        self.symbol = symbol
        self.timeframe = "1day"
        self.timeout = 5
        self.lookback_days = 30  # Fetch 30 days for SMA + pattern detection
        
    def fetch_daily_candles(self, limit: int = 30) -> Optional[List[Dict]]:
        """
        Fetch multiple daily candles for analysis
        Returns: List of candles sorted oldest → newest
        """
        try:
            url = f"{self.base_url}/api/v1/market/candles"
            params = {
                'symbol': self.symbol,
                'type': self.timeframe,
                'limit': limit  # Get last N days
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('data') or len(data['data']) == 0:
                print(f"[MTF-1D] ⚠️ No daily candle data from KuCoin for {self.symbol}", flush=True)
                return None
            
            # KuCoin returns newest first, reverse to oldest → newest
            raw_candles = data['data'][::-1]
            
            candles = []
            for candle in raw_candles:
                if len(candle) < 5:
                    continue
                
                timestamp_ms = int(candle[0])
                open_price = float(candle[1])
                close_price = float(candle[2])
                high_price = float(candle[3])
                low_price = float(candle[4])
                
                timestamp_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
                
                direction = "LONG" if close_price > open_price else ("SHORT" if close_price < open_price else "NONE")
                
                candles.append({
                    'timestamp': timestamp_utc.isoformat(),
                    'open': open_price,
                    'close': close_price,
                    'high': high_price,
                    'low': low_price,
                    'direction': direction,
                    'symbol': self.symbol,
                    'timeframe': '1d'
                })
            
            print(f"[MTF-1D] ✅ Fetched {len(candles)} daily candles for {self.symbol}", flush=True)
            return candles
            
        except requests.exceptions.RequestException as e:
            print(f"[MTF-1D] ❌ Failed to fetch daily candles: {str(e)}", flush=True)
            return None
        except (KeyError, IndexError, ValueError) as e:
            print(f"[MTF-1D] ❌ Error parsing daily candles: {str(e)}", flush=True)
            return None
    
    def calculate_sma(self, candles: List[Dict], period: int = 20) -> Optional[float]:
        """
        Calculate Simple Moving Average of closes
        Returns: SMA value or None if insufficient data
        """
        if not candles or len(candles) < period:
            return None
        
        closes = [c['close'] for c in candles[-period:]]
        return sum(closes) / len(closes)
    
    def get_regime(self, candles: List[Dict]) -> str:
        """
        Determine daily regime: BULL/BEAR/RANGE
        
        Logic:
        - BULL: Close > 20-day SMA AND recent trend is up
        - BEAR: Close < 20-day SMA AND recent trend is down
        - RANGE: Neither (consolidating)
        """
        if not candles or len(candles) < 5:
            return 'NONE'
        
        latest_close = candles[-1]['close']
        sma_20 = self.calculate_sma(candles, 20)
        
        if sma_20 is None:
            # Fallback: just use current candle direction
            return 'BULL' if candles[-1]['direction'] == 'LONG' else 'BEAR'
        
        # Check recent trend: are last 3 closes rising or falling?
        recent_closes = [c['close'] for c in candles[-3:]]
        rising = recent_closes[-1] > recent_closes[0]
        falling = recent_closes[-1] < recent_closes[0]
        
        if latest_close > sma_20 and rising:
            return 'BULL'
        elif latest_close < sma_20 and falling:
            return 'BEAR'
        else:
            return 'RANGE'
    
    def get_route(self, candles: List[Dict]) -> str:
        """
        Detect daily route: REVERSAL/TREND_CONTINUATION/NONE
        
        Logic:
        - REVERSAL: Current candle direction differs from previous day (direction changed)
        - TREND_CONTINUATION: Current candle direction same as previous day(s) (trend continues)
        - NONE: Insufficient data or no clear direction
        """
        if not candles or len(candles) < 2:
            return 'NONE'
        
        current = candles[-1]
        previous = candles[-2]
        
        current_dir = current['direction']
        previous_dir = previous['direction']
        
        # Skip if either candle has no clear direction
        if current_dir == 'NONE' or previous_dir == 'NONE':
            return 'NONE'
        
        # Check for directional change
        if current_dir != previous_dir:
            return 'REVERSAL'
        else:
            return 'TREND_CONTINUATION'
    
    def get_daily_context(self) -> Optional[Dict]:
        """
        Comprehensive daily context: direction + regime + route
        Returns: {'direction', 'regime', 'route', 'candle_data'} or None
        """
        candles = self.fetch_daily_candles(limit=30)
        
        if not candles or len(candles) == 0:
            return None
        
        latest_candle = candles[-1]
        direction = latest_candle['direction']
        regime = self.get_regime(candles)
        route = self.get_route(candles)
        
        print(f"[MTF-1D] ✅ Daily context for {self.symbol}:", flush=True)
        print(f"         Direction: {direction} | Regime: {regime} | Route: {route}", flush=True)
        
        return {
            'symbol': self.symbol,
            'timeframe': '1d',
            'direction': direction,      # LONG/SHORT
            'regime': regime,            # BULL/BEAR/RANGE
            'route': route,              # REVERSAL/TREND_CONTINUATION/NONE
            'candle_data': {
                'timestamp': latest_candle['timestamp'],
                'open': latest_candle['open'],
                'close': latest_candle['close'],
                'high': latest_candle['high'],
                'low': latest_candle['low']
            }
        }


def get_1d_context(symbol: str = "XAUUSD") -> Optional[Dict]:
    """
    Public interface: Enhanced 1D context for MTF analysis
    Now returns direction + regime + route (no more 'NONE' values)
    
    Args:
        symbol: Trading pair symbol (e.g., 'FUEL-USDT', 'ADA-USDT')
    
    Returns: {
        'direction': 'LONG'|'SHORT'|'NONE',
        'regime': 'BULL'|'BEAR'|'RANGE'|'NONE',
        'route': 'REVERSAL'|'TREND_CONTINUATION'|'NONE',
        'candle_data': {...}
    } or None on failure
    """
    fetcher = MTF1DFetcherEnhanced(symbol)
    return fetcher.get_daily_context()


if __name__ == '__main__':
    # Test enhanced fetcher
    print("=== Testing Enhanced MTF 1D Fetcher ===\n")
    
    # Test with FUEL-USDT
    print("[TEST] Fetching enhanced 1D data for FUEL-USDT...")
    context = get_1d_context('FUEL-USDT')
    if context:
        print(json.dumps(context, indent=2, default=str))
    else:
        print("Failed to fetch 1D context for FUEL-USDT\n")
    
    # Test with ADA-USDT
    print("\n[TEST] Fetching enhanced 1D data for ADA-USDT...")
    context = get_1d_context('ADA-USDT')
    if context:
        print(json.dumps(context, indent=2, default=str))
    else:
        print("Failed to fetch 1D context for ADA-USDT\n")
    
    # Test with SOL-USDT
    print("\n[TEST] Fetching enhanced 1D data for SOL-USDT...")
    context = get_1d_context('SOL-USDT')
    if context:
        print(json.dumps(context, indent=2, default=str))
    else:
        print("Failed to fetch 1D context for SOL-USDT")
