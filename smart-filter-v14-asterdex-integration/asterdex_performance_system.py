#!/usr/bin/env python3
"""
ASTERDEX Performance Tracking System - Master Controller
Orchestrates the complete performance tracking pipeline:
  1. Fetches trades from Asterdex API
  2. Correlates posted entries with executed trades
  3. Calculates performance metrics
  4. Generates daily reports

Run periodically (every 5 minutes) via cron.
Completely isolated from posting logic.
"""

import sys
import time
import os
from datetime import datetime
from pathlib import Path

# CRITICAL: Load environment variables FIRST, before any imports
env_file = Path('/Users/geniustarigan/.openclaw/workspace/.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                # Handle "export KEY=VALUE" format
                line = line.replace('export ', '', 1).strip()
                key, value = line.split('=', 1)
                # Remove quotes
                value = value.strip().strip("'\"")
                os.environ[key.strip()] = value
else:
    print(f"[WARN] .env file not found at {env_file}")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from asterdex_trade_fetcher import fetch_recent_trades, get_cached_trades
from asterdex_order_matcher_v3_multitrading import match_comprehensive_multi
from asterdex_performance_analytics import analyze_performance, format_report, save_analysis


def run_performance_pipeline():
    """
    Execute complete performance tracking pipeline.
    """
    print(f"\n{'='*60}")
    print(f"ASTERDEX PERFORMANCE TRACKING SYSTEM")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: Fetch trades from Asterdex API (from Jun 7 2026 onwards)
        print("[STEP 1] Fetching trades from Asterdex API (from Jun 7 2026)...")
        print("-" * 60)
        
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'smart-filter-v14-main'))
            from symbol_config_prod import get_active_symbols
            symbols = get_active_symbols()
        except:
            symbols = ['BTC-USDT', 'ETH-USDT', 'NEAR-USDT', 'SOL-USDT', 'AVAX-USDT']
        
        # Fetch trades from Jun 7 00:00:01 GMT+7 onwards (clean tracking baseline)
        # This is the stable period after system deployment
        trades = fetch_recent_trades(symbols, start_date='2026-06-07T00:00:01+07:00')
        print(f"✅ Fetched {len(trades)} trades from history\n")
        
        # Step 2: Match ALL positions by ORDER ID (multi-trade support)
        print("[STEP 2] Matching ALL positions by ORDER ID (multi-trade v3)...")
        print("-" * 60)
        correlations = match_comprehensive_multi()
        print(f"✅ Matched positions: {correlations}\n")
        
        # Step 3: Analyze performance
        print("[STEP 3] Analyzing performance metrics...")
        print("-" * 60)
        analysis = analyze_performance()
        print(f"✅ Analysis complete\n")
        
        # Step 4: Generate and display report
        print("[STEP 4] Performance Report")
        print("-" * 60)
        if analysis:
            report = format_report(analysis)
            print(report)
            save_analysis(analysis)
        else:
            print("No trading data available yet.")
        
        print(f"\n{'='*60}")
        print(f"Complete: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_performance_pipeline()
    sys.exit(0 if success else 1)
