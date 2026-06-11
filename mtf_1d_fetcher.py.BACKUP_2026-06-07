#!/usr/bin/env python3
"""
1D Daily Candle Fetcher for MTF Alignment Analysis
Fetches daily candles from KuCoin public API for any trading pair
Used to analyze 4h signals against daily trend context
Last updated: 2026-05-09
"""

import requests
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

class MTF1DFetcher:
    """Fetches daily candles from KuCoin public API for any symbol"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        # KuCoin public API endpoints (no auth needed)
        self.base_url = "https://api.kucoin.com"
        self.symbol = symbol  # Accept any symbol (FUEL-USDT, ADA-USDT, etc.)
        self.timeframe = "1day"
        self.timeout = 5
        
    def fetch_daily_candle(self) -> Optional[Dict]:
        """
        Fetch latest daily candle for XAUUSD
        Returns: {'open', 'close', 'high', 'low', 'time', 'direction', 'regime'} or None
        """
        try:
            # KuCoin klines endpoint: GET /api/v1/market/candles
            url = f"{self.base_url}/api/v1/market/candles"
            params = {
                'symbol': self.symbol,
                'type': self.timeframe  # '1day' for daily
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('data') or len(data['data']) == 0:
                print(f"[MTF-1D] ⚠️ No daily candle data from KuCoin for {self.symbol}", flush=True)
                return None
            
            # Latest candle is first in list
            candle = data['data'][0]
            
            # Parse KuCoin candle format: [time, open, close, high, low, volume, turnover]
            if len(candle) < 5:
                print(f"[MTF-1D] ⚠️ Invalid candle format from KuCoin", flush=True)
                return None
            
            timestamp_ms = int(candle[0])
            open_price = float(candle[1])
            close_price = float(candle[2])
            high_price = float(candle[3])
            low_price = float(candle[4])
            
            # Convert timestamp to UTC datetime (naive)
            timestamp_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
            
            # Calculate direction (LONG if close > open, SHORT if close < open, NONE if equal)
            direction = "LONG" if close_price > open_price else ("SHORT" if close_price < open_price else "NONE")
            
            candle_data = {
                'timestamp': timestamp_utc.isoformat(),
                'open': open_price,
                'close': close_price,
                'high': high_price,
                'low': low_price,
                'direction': direction,
                'symbol': self.symbol,
                'timeframe': '1d'
            }
            
            print(f"[MTF-1D] ✅ Fetched daily {self.symbol}: {open_price:.2f} → {close_price:.2f} ({direction})", flush=True)
            return candle_data
            
        except requests.exceptions.RequestException as e:
            print(f"[MTF-1D] ❌ Failed to fetch daily candle: {str(e)}", flush=True)
            return None
        except (KeyError, IndexError, ValueError) as e:
            print(f"[MTF-1D] ❌ Error parsing daily candle: {str(e)}", flush=True)
            return None
    
    def get_daily_regime(self) -> Optional[str]:
        """
        Get daily regime (BULL/BEAR/RANGE)
        Uses simple logic: if close > open = BULL, close < open = BEAR, else RANGE
        More sophisticated: could use daily EMA20/EMA50 if we store daily EMA history
        """
        candle = self.fetch_daily_candle()
        if not candle:
            return None
        
        # Simple direction-based regime
        if candle['direction'] == 'LONG':
            regime = 'BULL'
        elif candle['direction'] == 'SHORT':
            regime = 'BEAR'
        else:
            regime = 'RANGE'
        
        print(f"[MTF-1D] Daily regime: {regime} (based on candle direction)", flush=True)
        return regime


def get_1d_context(symbol: str = "XAUUSD") -> Optional[Dict]:
    """
    Public interface: Fetch 1D context for MTF analysis
    Args:
        symbol: Trading pair symbol (e.g., 'FUEL-USDT', 'ADA-USDT', 'XAUUSD')
    Returns: {'direction', 'regime', 'candle_data'} or None on failure
    """
    fetcher = MTF1DFetcher(symbol)
    candle = fetcher.fetch_daily_candle()
    
    if not candle:
        return None
    
    regime = fetcher.get_daily_regime()
    
    return {
        'direction': candle['direction'],
        'regime': regime,
        'candle': candle
    }


if __name__ == '__main__':
    # Test fetcher
    print("=== Testing MTF 1D Fetcher ===")
    
    # Test with FUEL-USDT
    print("\n[TEST] Fetching 1D data for FUEL-USDT...")
    context = get_1d_context('FUEL-USDT')
    if context:
        print(json.dumps(context, indent=2, default=str))
    else:
        print("Failed to fetch 1D context for FUEL-USDT")
    
    # Test with ADA-USDT
    print("\n[TEST] Fetching 1D data for ADA-USDT...")
    context = get_1d_context('ADA-USDT')
    if context:
        print(json.dumps(context, indent=2, default=str))
    else:
        print("Failed to fetch 1D context for ADA-USDT")
