# main.py
# Revised: refactored single-cycle run into `run_cycle()` and a persistent outer loop,
# added safer SuperGK fetch normalization, environment-driven cycle sleep, and more defensive logging.
# NOTE: This variant FORCE-BYPASSES SuperGK for ALL signals (LONG and SHORT).

print("[INFO] main.py script started.", flush=True)
from signal_debug_log import export_signal_debug_txt, log_fired_signal
import os
import time
import pandas as pd
import random
import pytz
from datetime import datetime
import uuid as uuid_lib
import signal as signal_module
import sys

# === EARLY LOGGING CONTROL (must define before other imports/usage) ===
os.environ.setdefault("VERBOSE_LOGGING", "0")  # Default to quiet mode
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "0") == "1"
from kucoin_data import get_live_entry_price, DEFAULT_SLIPPAGE
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter, MIN_SCORE
from telegram_alert import send_telegram_alert, send_telegram_file
from tier_lookup import get_signal_tier
from symbol_classifier import classify_symbol
from kucoin_orderbook import get_order_wall_delta
from pec_engine import run_pec_check, export_pec_log
from tp_sl_retracement import calculate_tp_sl
try:
    from test_filters import run_all_filter_tests
except ImportError:
    run_all_filter_tests = None  # test_filters optional (development only)
from signal_store import get_signal_store
from pec_config import MIN_ACCEPTED_RR, SIGNALS_JSONL_PATH, validate_tp_sl_relationship, calculate_rr, enforce_rr_bounds
import math

# NEW: Enhanced tracking & safe OHLCV (PROJECT-3 SmartFilter fixes)
from signal_tracking_enhanced import get_signal_tracker
from ohlcv_fetch_safe import safe_fetch_ohlcv_by_tf, check_tf_data_available, should_skip_symbol
from signal_logger import SignalLogger
from signal_sent_tracker import get_signal_sent_tracker  # PEC: Track SENT signals only
from signals_master_writer import get_signals_master_writer  # Write to SIGNALS_MASTER.jsonl (single source of truth)
from pathlib import Path

# ===== PHASE 2 IMPORTS (Stage 2 - Direction-Aware Gatekeepers + Regime Adjustments) =====
from direction_aware_gatekeeper import DirectionAwareGatekeeper
from direction_aware_regime_adjustments import calculate_direction_aware_threshold
# ===== END PHASE 2 IMPORTS =====

# ===== PHASE 3B IMPORTS (Stage 3 - Reversal Quality Gate + Route Optimization) =====
from reversal_quality_gate import check_reversal_quality
from direction_aware_route_optimizer import calculate_route_score, get_route_recommendation
# ===== END PHASE 3B IMPORTS =====

# ===== PHASE 1 IMPORTS (Dual-Write Verification - Alert + Continue Strategy) =====
from signal_dual_write_verification import (
    initialize_dual_write_verifier,
    verify_signal_dual_write,
    get_divergence_tracker,
    send_ops_alert
)
# ===== END PHASE 1 IMPORTS =====

# ===== SYMBOL BLACKLIST & WHITELIST (Dynamic Release System) =====
try:
    from SYMBOL_BLACKLIST import BLACKLIST_SYMBOLS
    from SYMBOL_WHITELIST_OVERRIDES import is_symbol_whitelisted
    BLACKLIST_READY = True
except ImportError as e:
    print(f"[WARN] Symbol blacklist/whitelist import failed: {e}. Blacklist disabled.", flush=True)
    BLACKLIST_READY = False
    BLACKLIST_SYMBOLS = set()
    def is_symbol_whitelisted(s): return False

def apply_blacklist_filter(symbol: str) -> bool:
    """
    Check if symbol should be blocked from trading.
    
    Logic:
    1. Check whitelist first (override)
    2. Then check blacklist
    
    Args:
        symbol: Symbol string (e.g., 'EPT-USDT')
    
    Returns:
        True if BLOCK (reject signal)
        False if ALLOW (allow signal)
    """
    if not BLACKLIST_READY:
        return False  # If blacklist system failed, allow all
    
    # Check whitelist first (overrides blacklist)
    if is_symbol_whitelisted(symbol):
        print(f"[WHITELIST] {symbol} RELEASED from blacklist (override active)", flush=True)
        return False  # ALLOW this signal
    
    # Check blacklist
    if symbol in BLACKLIST_SYMBOLS:
        print(f"[BLACKLIST] {symbol} rejected (WR < 21%)", flush=True)
        return True  # BLOCK this signal
    
    return False  # ALLOW (not in either list)

# ===== END SYMBOL BLACKLIST & WHITELIST =====

# ===== PHASE 4A FUNCTION: Multi-TF Alignment (Scenario 4: 30min + 1h) =====
def check_multitf_alignment_30_1h(symbol, ohlcv_data):
    """
    PHASE 4A Scenario 4: Check if 30min trend aligns with 1h trend
    
    This acts as a consensus filter: only allow signals if higher timeframes agree
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        ohlcv_data: Dict from safe_fetch_ohlcv_by_tf with keys "15min", "30min", "1h", "2h", "4h"
    
    Returns:
        (should_allow_signal, trend_30min, trend_1h, reason_log)
    """
    try:
        # Get 30min and 1h dataframes
        df30 = ohlcv_data.get("30min")
        df1h = ohlcv_data.get("1h")
        
        if df30 is None or df1h is None or len(df30) < 1 or len(df1h) < 1:
            # Cannot check alignment without data
            return (True, "NONE", "NONE", "[PHASE4A-S4] Insufficient TF data, allowing signal")
        
        # Detect trends (close > MA20)
        trend_30 = "LONG" if len(df30) >= 20 and df30['close'].iloc[-1] > df30['close'].iloc[-20:].mean() else "SHORT"
        trend_1h = "LONG" if len(df1h) >= 20 and df1h['close'].iloc[-1] > df1h['close'].iloc[-20:].mean() else "SHORT"
        
        # Check alignment
        if trend_30 == trend_1h:
            return (True, trend_30, trend_1h, f"[PHASE4A-S4] ✅ Aligned: 30min={trend_30}, 1h={trend_1h}")
        else:
            return (False, trend_30, trend_1h, f"[PHASE4A-S4] ❌ Misaligned: 30min={trend_30}, 1h={trend_1h} (FILTERED)")
    
    except Exception as e:
        # If error during check, allow signal (fail-safe)
        return (True, "ERROR", "ERROR", f"[PHASE4A-S4] Error checking alignment: {str(e)[:50]} (allowing signal)")
# ===== END PHASE 4A =====

# === INITIALIZE SIGNAL STORAGE (EARLY & ROBUST) ===
try:
    _signal_store = get_signal_store(SIGNALS_JSONL_PATH)
    _signal_store_ready = True
    print(f"[INIT] Signal store ready: {os.path.abspath(SIGNALS_JSONL_PATH)}", flush=True)
except Exception as e:
    _signal_store_ready = False
    print(f"[ERROR] Signal store init failed: {e}. Signals will fire to Telegram only.", flush=True)

# === INITIALIZE SENT SIGNAL TRACKER (PEC) ===
try:
    # Use workspace root, not submodule directory
    sent_signals_path = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"
    _sent_signal_tracker = get_signal_sent_tracker(sent_signals_path)
    _sent_tracker_ready = True
    print(f"[INIT] Sent signal tracker ready: {os.path.abspath(sent_signals_path)}", flush=True)
except Exception as e:
    _sent_tracker_ready = False
    print(f"[ERROR] Sent signal tracker init failed: {e}. PEC will not have execution data.", flush=True)

# === INITIALIZE SIGNALS_MASTER WRITER (Single Source of Truth) ===
try:
    signals_master_path = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"
    _signals_master_writer = get_signals_master_writer(signals_master_path)
    _master_writer_ready = True
    print(f"[INIT] Signals Master writer ready: {os.path.abspath(signals_master_path)}", flush=True)
except Exception as e:
    _master_writer_ready = False
    print(f"[ERROR] Signals Master writer init failed: {e}. SIGNALS_MASTER.jsonl will not be updated.", flush=True)

# === PHASE 1: INITIALIZE DUAL-WRITE VERIFIER (Alert + Continue Strategy) ===
try:
    _dual_write_verifier = initialize_dual_write_verifier(
        master_path=os.path.abspath(signals_master_path),
        audit_path=os.path.abspath("/Users/geniustarigan/.openclaw/workspace/SIGNALS_INDEPENDENT_AUDIT.txt"),
        debug=VERBOSE_LOGGING
    )
    _divergence_tracker = get_divergence_tracker()
    _dual_write_ready = True
    print(f"[INIT] Dual-write verifier initialized (Alert+Continue strategy)", flush=True)
except Exception as e:
    _dual_write_ready = False
    _divergence_tracker = None
    print(f"[WARN] Dual-write verifier init failed: {e}. Verification disabled.", flush=True)

# === INITIALIZE SIGNAL TRACKING (EARLY & ROBUST) - NEW PROJECT-3 FIX ===
try:
    _signal_tracker = get_signal_tracker(SIGNALS_JSONL_PATH)
    _signal_tracker_ready = True
    print(f"[INIT] Signal tracker ready: {os.path.abspath(Path(SIGNALS_JSONL_PATH).parent / 'signal_status.jsonl')}", flush=True)
except Exception as e:
    _signal_tracker_ready = False
    print(f"[ERROR] Signal tracker init failed: {e}. Tracking disabled.", flush=True)

# === EXECUTOR TRIGGER (EVENT-DRIVEN PEC HYBRID MODEL) ===
def trigger_executor_on_signal(signal_uuid: str, symbol: str, timeframe: str):
    """
    Trigger PEC executor immediately when signal fires (event-driven, non-blocking)
    This is part of the HYBRID operational model (cron + event-triggered)
    """
    try:
        import subprocess
        executor_path = "/Users/geniustarigan/.openclaw/workspace/pec_executor.py"
        # Non-blocking subprocess call
        subprocess.Popen(
            ['python3', executor_path, '--signal-uuid', signal_uuid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )
        print(f"[EXECUTOR] Event-triggered backtest for {symbol} {timeframe} (UUID: {signal_uuid[:12]})", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to trigger executor: {e}", flush=True)
        # Continue anyway - cron will catch it as fallback

# --- Configuration ---
# Full liquid pairs from kucoin_orderbook.py - 90+ symbols
TOKENS = [
    # === 98 SYMBOLS (82 original + 20 new - 1 delisted - 3 invalid: 2026-03-09 validation) ===
    # WAVE 1 ADDS (2026-03-05): ATOM-USDT, AGLD-USDT, APT-USDT, INJ-USDT, NEAR-USDT, OCEAN-USDT, OP-USDT, RNDR-USDT, SEI-USDT, TAO-USDT
    # WAVE 2 ADDS (2026-03-09): BLUR-USDT, LDO-USDT, CRV-USDT, CVX-USDT, YFI-USDT, ENS-USDT, BONK-USDT (7 validated dual-listed)
    # REMOVED: ELDE-USDT (no data on KuCoin perpetuals), MKR-USDT (Binance spot only), BAL-USDT (Binance spot only), FLR-USDT (Binance spot only)
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "AVAX-USDT", 
    "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT", "SKATE-USDT", "LA-USDT", 
    "SPK-USDT", "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT", 
    "X-USDT", "RAY-USDT", "EPT-USDT", "MAGIC-USDT", "ACTSOL-USDT", 
    "FUN-USDT", "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT", 
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT", "DUCK-USDT", 
    "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT", "VOXEL-USDT", "ALU-USDT", 
    "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", "DOGE-USDT", "ALGO-USDT", "DOT-USDT", 
    "NEWT-USDT", "SAHARA-USDT", "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", 
    "ENA-USDT", "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT", 
    "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT", "GALA-USDT", 
    "PUMP-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT", "KAITO-USDT", "ARKM-USDT", 
    "ATH-USDT", "NMR-USDT", "ARB-USDT", "WLFI-USDT", "BIO-USDT", "ASTER-USDT", 
    "XPL-USDT", "AVNT-USDT", "ORDER-USDT", "XAUT-USDT",
    # WAVE 1 NEW SYMBOLS (2026-03-05 14:25 GMT+7) - 10 validated dual-listed perpetuals
    "ATOM-USDT", "AGLD-USDT", "APT-USDT", "INJ-USDT", "NEAR-USDT", "OCEAN-USDT", "OP-USDT", "RNDR-USDT", "SEI-USDT", "TAO-USDT",
    # WAVE 2 NEW SYMBOLS (2026-03-09 21:55 GMT+7) - 7 validated dual-listed perpetuals (checked Binance & KuCoin both exist)
    "BLUR-USDT", "LDO-USDT", "CRV-USDT", "CVX-USDT", "YFI-USDT", "ENS-USDT", "BONK-USDT"
]

COOLDOWN = {"15min": 120, "30min": 300, "1h": 420, "2h": 600, "4h": 900}  # Tightened: 15m=2min, 30m=5min, 1h=7min, 2h=10min, 4h=15min (more signal volume)

# === PHASE 2 GATEKEEPER CONTROL ===
# Set to False to disable DirectionAwareGatekeeper checks (data showed it reduces profitability)
# Code is preserved - set to True to re-enable if needed
ENABLE_DIRECTION_AWARE_GATEKEEPER = False  # DISABLED 2026-03-25: Data shows 4h without it has 56% WR vs 1h with it has 36% WR

last_sent = {}

# === LOGGING CONTROL ===
# Disable verbose analysis logs to reduce Railway rate limit issues (67 messages dropped)
# NOTE: VERBOSE_LOGGING is now defined at top of file (early definition)

import contextlib

@contextlib.contextmanager
def suppress_stdout():
    """Temporarily suppress stdout logging"""
    if VERBOSE_LOGGING:
        yield
    else:
        import sys
        old_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, 'w')
            yield
        finally:
            sys.stdout = old_stdout

def initialize_last_sent():
    """Load recent sent signals from SENT_SIGNALS.jsonl to restore last_sent state after daemon restart"""
    global last_sent
    import time
    now = time.time()
    
    # Use workspace root, not submodule directory
    sent_signals_path = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"
    
    try:
        if not os.path.exists(sent_signals_path):
            return
        
        # Read last 100 lines of SENT_SIGNALS.jsonl to find recent signals
        with open(sent_signals_path, "r") as f:
            lines = f.readlines()
        
        # Process recent signals (last 200 lines, ~10 mins of signals)
        for line in lines[-200:]:
            try:
                signal = json.loads(line.strip())
                symbol = signal.get("symbol")
                tf = signal.get("timeframe")
                sent_time_str = signal.get("sent_time_utc")
                
                if not (symbol and tf and sent_time_str):
                    continue
                
                # Parse sent_time_utc to get timestamp
                try:
                    sent_dt = datetime.fromisoformat(sent_time_str.replace('Z', '+00:00'))
                    if sent_dt.tzinfo is None:
                        sent_dt = sent_dt.replace(tzinfo=timezone.utc)
                    sent_timestamp = sent_dt.timestamp()
                except:
                    continue
                
                # Only consider signals sent within the last COOLDOWN window
                key = f"{symbol}_{tf}"
                max_cooldown = max(COOLDOWN.values())
                
                if now - sent_timestamp < max_cooldown:
                    # Store the latest sent time for this symbol+tf combo
                    if key not in last_sent or sent_timestamp > last_sent[key]:
                        last_sent[key] = sent_timestamp
            except:
                pass
        
        if last_sent:
            print(f"[INIT] Restored {len(last_sent)} last_sent entries from SENT_SIGNALS.jsonl", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to initialize last_sent: {e}", flush=True)

# NOTE: Updated to use 15m, 30m, 1h (removed 3m, 5m - too noisy)
# Based on Test-3 results: 15-min markets are optimal, longer timeframes reduce false signals
PEC_BARS = 5
PEC_WINDOW_MINUTES = 720
OHLCV_LIMIT = 250  # Optimized: covers EMA200 + safety buffer, reduces API calls

# EMA CONFIGURATION PER TIMEFRAME (Optimization 2026-02-27)
# Gate logic uses TF-specific EMAs instead of EMA200 for all
EMA_CONFIG = {
    "15min": 50,   # 50 × 15min = 750min = 12.5h (responsive, short-term)
    "30min": 100,  # 100 × 30min = 3000min = 50h = 2d (balanced, medium-term)
    "1h": 200,     # 200 × 60min = 12000min = 8d (standard, long-term)
    "4h": 200      # 200 × 4h = 800h = 33d (macro, long-term) - PHASE 2 ADD
}

# cycle sleep can be controlled via environment variable (seconds)
CYCLE_SLEEP = int(os.getenv("CYCLE_SLEEP", "60"))

# --- SuperGK helper: main_supergk_ok left for compatibility but main will bypass ---
def main_supergk_ok(bias, orderbook_result, density_result, analyzer_result):
    """Legacy helper retained for compatibility. Not used when global bypass is enforced in main."""
    # Keep behavior predictable if called elsewhere: call canonical function if available
    try:
        return super_gk_aligned(bias, orderbook_result, density_result)
    except Exception as e:
        print(f"[SuperGK][MAIN] Error calling super_gk_aligned(): {e}", flush=True)
        return False

def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

def get_resting_order_density(symbol, depth=100, top_n=5):
    """
    Wrapper that uses kucoin_density.get_resting_density (network fetch) and ensures
    returned densities are percentages (0..100). Defensive - returns zeros on error.
    """
    try:
        # import here to avoid circular import on module load
        from kucoin_density import get_resting_density as _kd
        res = _kd(symbol, depth=depth, levels=top_n)
        # Ensure fields exist and are numeric
        bid_density = float(res.get("bid_density", 0.0))
        ask_density = float(res.get("ask_density", 0.0))
        # Sanity clamp: density should be between 0 and 100
        bid_density = max(0.0, min(100.0, bid_density))
        ask_density = max(0.0, min(100.0, ask_density))
        # Also return totals for debugging
        return {
            "bid_density": bid_density,
            "ask_density": ask_density,
            "bid_top": float(res.get("bid_top", 0.0)),
            "ask_top": float(res.get("ask_top", 0.0)),
            "bid_total": float(res.get("bid_total", 0.0)),
            "ask_total": float(res.get("ask_total", 0.0))
        }
    except Exception as e:
        print(f"[get_resting_order_density] Error for {symbol}: {e}", flush=True)
        return {"bid_density": 0.0, "ask_density": 0.0, "bid_top": 0.0, "ask_top": 0.0, "bid_total": 0.0, "ask_total": 0.0}

def candle_color(open, close):
    return 'green' if close > open else 'red'

def create_and_store_signal(symbol, timeframe, signal_type, fired_time_utc, entry_price,
                           tp_target, sl_target, tp_pct, sl_pct, achieved_rr, fib_ratio,
                           atr_value, score, max_score, confidence, route, regime,
                           passed_gatekeepers, max_gatekeepers, passed_filters=None, 
                           failed_filters=None, passed_filter_count=0, failed_filter_count=0,
                           telegram_msg_id=''):
    """
    Create signal dict and store to JSONL.
    Uses global _signal_store initialized at module load time.
    
    Returns: signal_uuid if successful, None if failed
    """
    global _signal_store_ready
    
    if not _signal_store_ready:
        print(f"[WARN] Signal store not ready, skipping JSONL storage for {symbol} {timeframe}", flush=True)
        return None
    
    try:
        signal_uuid = str(uuid_lib.uuid4())
        
        signal_data = {
            "uuid": signal_uuid,
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_type": signal_type,
            "direction": signal_type,  # CRITICAL: direction = signal_type (LONG/SHORT) for Phase 3B RQ4 gate validation
            "fired_time_utc": fired_time_utc.isoformat() if hasattr(fired_time_utc, 'isoformat') else str(fired_time_utc),
            "entry_price": float(entry_price),
            "tp_target": float(tp_target),
            "sl_target": float(sl_target),
            "tp_pct": float(tp_pct) if tp_pct is not None else 0.0,
            "sl_pct": float(sl_pct) if sl_pct is not None else 0.0,
            "achieved_rr": float(achieved_rr) if achieved_rr is not None else 0.0,
            "fib_ratio": float(fib_ratio) if fib_ratio is not None else None,
            "atr_value": float(atr_value) if atr_value is not None else 0.0,
            "score": int(score),
            "max_score": int(max_score),
            "confidence": float(confidence),
            "route": str(route) if route else "NONE",
            "regime": str(regime) if regime else "UNKNOWN",
            "passed_gatekeepers": int(passed_gatekeepers),
            "max_gatekeepers": int(max_gatekeepers),
            "passed_filters": passed_filters if passed_filters else [],
            "failed_filters": failed_filters if failed_filters else [],
            "passed_filter_count": int(passed_filter_count),
            "failed_filter_count": int(failed_filter_count),
            "telegram_msg_id": str(telegram_msg_id) if telegram_msg_id else '',
        }
        
        # Use global store (pre-initialized)
        stored_uuid = _signal_store.append_signal(signal_data)
        
        if stored_uuid:
            print(f"[STORED] Signal captured: {symbol} {timeframe} {signal_type} UUID={stored_uuid[:8]}...", flush=True)
        else:
            print(f"[WARN] Signal store returned None for {symbol} {timeframe}", flush=True)
        
        return stored_uuid
    
    except Exception as e:
        print(f"[ERROR] create_and_store_signal failed for {symbol} {timeframe}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def early_breakout(df, lookback=3):
    if len(df) < lookback:
        return {'valid_signal': False, 'bias': None, 'price': None}

    candles = df.iloc[-lookback:]
    colors = [candle_color(row['open'], row['close']) for _, row in candles.iterrows()]
    if all(c == 'green' for c in colors):
        direction = 'green'
    elif all(c == 'red' for c in colors):
        direction = 'red'
    else:
        return {'valid_signal': False, 'bias': None, 'price': None}

    prev_high = candles['high'].iloc[0]
    prev_low = candles['low'].iloc[0]
    close = candles['close'].iloc[-1]

    if close > prev_high and direction == 'green':
        return {'valid_signal': True, 'bias': 'LONG', 'price': close}
    elif close < prev_low and direction == 'red':
        return {'valid_signal': True, 'bias': 'SHORT', 'price': close}

    return {'valid_signal': False, 'bias': None, 'price': close}

def log_orderbook_and_density(symbol):
    """
    Fetch fresh orderbook and resting density and print timestamped logs.
    Adds ts (UTC) and source=main_log so you can confirm these logs were fetched
    immediately prior to sending a signal (not a startup/import dump).
    """
    from datetime import datetime
    ts = datetime.utcnow().isoformat()
    try:
        ob = get_order_wall_delta(symbol)
        buy_wall = float(ob.get('buy_wall', 0.0))
        sell_wall = float(ob.get('sell_wall', 0.0))
        wall_delta = float(ob.get('wall_delta', 0.0))
        midprice = ob.get('midprice', 'N/A')
        print(
            f"[OrderBookDeltaLog] {symbol} | ts={ts} | source=main_log | "
            f"buy_wall={buy_wall} | sell_wall={sell_wall} | wall_delta={wall_delta} | midprice={midprice}",
            flush=True
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}", flush=True)

    try:
        dens = get_resting_order_density(symbol)
        print(
            f"[RestingOrderDensityLog] {symbol} | ts={ts} | source=main_log | "
            f"bid_density={dens.get('bid_density',0.0):.2f}% | ask_density={dens.get('ask_density',0.0):.2f}% | "
            f"bid_top={dens.get('bid_top',0.0)} | ask_top={dens.get('ask_top',0.0)} | "
            f"bid_total={dens.get('bid_total',0.0)} | ask_total={dens.get('ask_total',0.0)}",
            flush=True
        )
    except Exception as e:
        print(f"[RestingOrderDensityLog] {symbol} ERROR: {e}", flush=True)

# --- SuperGK canonical function retained for reference but not used by main (global bypass enforced below) ---
def super_gk_aligned(bias, orderbook_result, density_result,
                     wall_pct_threshold=None, density_threshold=None,
                     wall_weight=None, density_weight=None, composite_threshold=None):
    """
    Composite-first SuperGK with optional bypasses.
    """
    # Helper to read floats from env with defaults
    def _env_float(name, default):
        v = os.getenv(name)
        try:
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    default_wpct = _env_float("SUPERGK_WALL_PCT_THRESHOLD", 1.0)
    default_dpct = _env_float("SUPERGK_DENSITY_THRESHOLD", 0.5)
    global_wall_w = _env_float("SUPERGK_WALL_WEIGHT", 0.25)
    global_density_w = _env_float("SUPERGK_DENSITY_WEIGHT", 0.75)
    global_comp_th = _env_float("SUPERGK_COMPOSITE_THRESHOLD", 0.01)

    short_wall_w = _env_float("SUPERGK_SHORT_WALL_WEIGHT", os.getenv("SUPERGK_SHORT_WALL_WEIGHT") or global_wall_w)
    short_density_w = _env_float("SUPERGK_SHORT_DENSITY_WEIGHT", os.getenv("SUPERGK_SHORT_DENSITY_WEIGHT") or global_density_w)
    short_comp_th = _env_float("SUPERGK_SHORT_COMPOSITE_THRESHOLD", os.getenv("SUPERGK_SHORT_COMPOSITE_THRESHOLD") or global_comp_th)

    long_wall_w = _env_float("SUPERGK_LONG_WALL_WEIGHT", os.getenv("SUPERGK_LONG_WALL_WEIGHT") or global_wall_w)
    long_density_w = _env_float("SUPERGK_LONG_DENSITY_WEIGHT", os.getenv("SUPERGK_LONG_DENSITY_WEIGHT") or global_density_w)
    long_comp_th = _env_float("SUPERGK_LONG_COMPOSITE_THRESHOLD", os.getenv("SUPERGK_LONG_COMPOSITE_THRESHOLD") or global_comp_th)

    density_dom_factor = _env_float("SUPERGK_DENSITY_DOMINANCE_FACTOR", 1.6)

    wall_pct_threshold = default_wpct if wall_pct_threshold is None else wall_pct_threshold
    density_threshold = default_dpct if density_threshold is None else density_threshold

    if bias == "SHORT":
        wall_weight = short_wall_w if wall_weight is None else wall_weight
        density_weight = short_density_w if density_weight is None else density_weight
        composite_threshold = short_comp_th if composite_threshold is None else composite_threshold
    else:
        wall_weight = long_wall_w if wall_weight is None else wall_weight
        density_weight = long_density_w if density_weight is None else density_weight
        composite_threshold = long_comp_th if composite_threshold is None else composite_threshold

    buy_wall = float(orderbook_result.get('buy_wall', 0.0)) if orderbook_result else 0.0
    sell_wall = float(orderbook_result.get('sell_wall', 0.0)) if orderbook_result else 0.0
    wall_delta = float(orderbook_result.get('wall_delta', buy_wall - sell_wall)) if orderbook_result else 0.0

    bid_density = float(density_result.get('bid_density', 0.0)) if density_result else 0.0
    ask_density = float(density_result.get('ask_density', 0.0)) if density_result else 0.0

    total_wall = max(buy_wall + sell_wall, 1e-9)
    wall_pct = (abs(wall_delta) / total_wall) * 100.0
    wall_sign = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"

    density_diff = bid_density - ask_density
    density_pct = abs(density_diff)
    density_sign = "LONG" if density_diff > 0 else "SHORT" if density_diff < 0 else "NEUTRAL"

    orderbook_bias = wall_sign if wall_pct >= wall_pct_threshold else "NEUTRAL"
    density_bias = density_sign if density_pct >= density_threshold else "NEUTRAL"

    wall_score = (wall_pct / 100.0)
    density_score = (density_pct / 100.0)

    def sign_to_factor(sign, desired):
        if sign == desired:
            return 1.0
        elif sign == "NEUTRAL":
            return 0.0
        else:
            return -1.0

    wall_factor = sign_to_factor(orderbook_bias, bias)
    density_factor = sign_to_factor(density_bias, bias)

    composite = wall_weight * wall_factor * wall_score + density_weight * density_factor * density_score

    wall_comp = wall_weight * wall_factor * wall_score
    density_comp = density_weight * density_factor * density_score

    return composite >= composite_threshold

# --- run_cycle / main logic (UNCONDITIONAL SUPERGK BYPASS FOR ALL SIGNALS) ---
# NOTE: this module expects these symbols to be available in the import environment:
# TOKENS, OHLCV_LIMIT, COOLDOWN, last_sent, DEFAULT_SLIPPAGE
# get_ohlcv, SmartFilter, early_breakout, get_live_entry_price,
# get_order_wall_delta, get_resting_order_density, calculate_tp_sl,
# log_fired_signal, send_telegram_alert
# These are kept external so the function can be copy-pasted into your existing module.

import traceback
from datetime import datetime
import json

# === DEDUP WINDOWS (per timeframe) ===
DEDUP_WINDOWS = {
    "15min": 1200,   # 20 minutes in seconds
    "30min": 2400,   # 40 minutes in seconds
    "1h": 4800,      # 80 minutes in seconds
    "4h": 43200      # 43200 seconds = 3 bars (12 hours) in seconds (PHASE 2 ADD)
}

# === DEDUP CACHE (loaded once per cycle, not 603 times!) ===
_dedup_cache = None
_dedup_cache_time = None

def load_dedup_cache():
    """Load SENT_SIGNALS.jsonl ONCE per cycle into memory"""
    global _dedup_cache, _dedup_cache_time
    
    sent_signals_path = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"
    _dedup_cache = []
    _dedup_cache_time = datetime.now(pytz.UTC)
    
    if not os.path.exists(sent_signals_path):
        return  # File doesn't exist yet
    
    try:
        with open(sent_signals_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    _dedup_cache.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"[DEDUP] Loaded {len(_dedup_cache)} signals into cache (instead of reading file 603 times!)", flush=True)
    except Exception as e:
        print(f"[DEDUP-LOAD-ERROR] {e}", flush=True)
        _dedup_cache = []

def is_duplicate_signal(symbol, timeframe, signal_type):
    """
    Check if this signal (symbol + timeframe + signal_type) was sent recently.
    Uses in-memory cache loaded once per cycle (NOT file I/O 603 times!)
    
    Returns: True if duplicate (should skip), False if new signal (should send)
    """
    global _dedup_cache, _dedup_cache_time
    
    if _dedup_cache is None or _dedup_cache_time is None:
        return False  # Cache not initialized
    
    try:
        window_seconds = DEDUP_WINDOWS.get(timeframe, 1200)
        current_time = datetime.now(pytz.UTC)
        
        # Check in-memory cache (O(n) but n is small and we only do this once per cycle)
        for signal_record in _dedup_cache:
            # Check if same symbol + timeframe + signal_type
            if (signal_record.get('symbol') == symbol and 
                signal_record.get('timeframe') == timeframe and 
                signal_record.get('signal_type') == signal_type):
                
                # Check if within dedup window
                sent_time_str = signal_record.get('sent_time_utc')
                if sent_time_str:
                    sent_time = datetime.fromisoformat(sent_time_str.replace('Z', '+00:00'))
                    if not sent_time.tzinfo:
                        sent_time = pytz.UTC.localize(sent_time)
                    
                    time_diff_seconds = (current_time - sent_time).total_seconds()
                    
                    # If within window = duplicate
                    if time_diff_seconds < window_seconds:
                        return True  # Duplicate found
        
        # No duplicate found
        return False
    
    except Exception as e:
        print(f"[DEDUP-ERROR] {symbol} {timeframe}: {e}. Assuming NOT duplicate.", flush=True)
        return False

def run_cycle():
    """
    Single pass over all TOKENS. Returns list of valid_debug dicts collected during this cycle.
    """
    print("[INFO] Starting Smart Filter cycle (single pass)...", flush=True)
    
    # === CRITICAL FIX: Load dedup cache ONCE per cycle (not 603 times!) ===
    load_dedup_cache()
    
    logger = SignalLogger(enabled=True)  # Clean, informative logging
    valid_debugs = []
    now = time.time()
    
    # IN-MEMORY DEDUP: Track signals sent in THIS CYCLE (prevents intra-cycle duplicates)
    signals_sent_this_cycle = set()
    
    # NOTE: Deduplication now uses SENT_SIGNALS.jsonl with per-timeframe windows
    # PLUS in-memory set for same-cycle duplicates (see is_duplicate_signal function above)

    for idx, symbol in enumerate(TOKENS, start=1):
        # Skip verbose "[INFO] Checking" log
        try:
            # Fetch OHLCV independently for each TF (PROJECT-3 SmartFilter fix - decouple TF processing)
            ohlcv_data = safe_fetch_ohlcv_by_tf(symbol, get_ohlcv)
            
            # Check if symbol should be skipped (ALL TFs missing)
            should_skip, skip_reason = should_skip_symbol(ohlcv_data)
            if should_skip:
                print(f"[WARN] Skipping {symbol}: {skip_reason}", flush=True)
                continue
            
            # Extract data (may be None for individual TFs - that's OK now)
            df15 = ohlcv_data.get("15min")
            df30 = ohlcv_data.get("30min")
            df1h = ohlcv_data.get("1h")
            df2h = ohlcv_data.get("2h")  # 2h ADD (2026-03-25)
            if df2h is not None and not df2h.empty:
                print(f"[LOAD] {symbol}: df2h loaded - {len(df2h)} candles", flush=True)
            else:
                print(f"[LOAD] {symbol}: df2h is None or empty - ohlcv_data.keys()={list(ohlcv_data.keys())}", flush=True)
            df4h = ohlcv_data.get("4h")  # PHASE 2 ADD
            
            # DEBUG: Log TF4h fetch status (PHASE 2 - FIX TF4h not firing)
            if df4h is None:
                print(f"[DEBUG-TF4h] {symbol}: df4h is NONE - data not available from API", flush=True)
            else:
                print(f"[DEBUG-TF4h] {symbol}: df4h loaded - {len(df4h)} candles", flush=True)
            
            # Calculate early breakout only for TFs with data
            early_breakout_15m = early_breakout(df15, lookback=3) if df15 is not None else None
            early_breakout_30m = early_breakout(df30, lookback=3) if df30 is not None else None
            early_breakout_1h = early_breakout(df1h, lookback=3) if df1h is not None else None
            early_breakout_2h = early_breakout(df2h, lookback=3) if df2h is not None else None  # 2h ADD (2026-03-25)
            early_breakout_4h = early_breakout(df4h, lookback=3) if df4h is not None else None  # PHASE 2 ADD

            # --- 15min TF block ---
            try:
                key15 = f"{symbol}_15min"
                if df15 is None or df15.empty:
                    res15 = None
                else:
                    sf15 = SmartFilter(symbol, df15, tf="15min")
                    regime15 = sf15._market_regime()
                    res15 = sf15.analyze()

                # LOG ALL SIGNALS (even rejected) to ALL_SIGNALS.jsonl BEFORE filters_ok check
                if isinstance(res15, dict):
                    score_15min = res15.get("score")
                    try:
                        all_signals_path = "/Users/geniustarigan/.openclaw/workspace/ALL_SIGNALS.jsonl"
                        all_signal_entry = {
                            "symbol": symbol,
                            "timeframe": "15min",
                            "score": score_15min,
                            "min_score_required": MIN_SCORE,
                            "passed_filter": score_15min is not None and score_15min >= MIN_SCORE,
                            "fired_time_utc": datetime.utcnow().isoformat(),
                            "filters_ok": res15.get("filters_ok")
                        }
                        with open(all_signals_path, 'a') as f:
                            f.write(json.dumps(all_signal_entry) + '\n')
                    except Exception as e:
                        print(f"[WARN] Failed to log to ALL_SIGNALS.jsonl: {e}", flush=True)

                if isinstance(res15, dict) and res15.get("filters_ok") is True:
                    
                    # ===== SCORE VALIDATION GATE (Universal MIN_SCORE from smart_filter.py) =====
                    if score_15min is None or score_15min < MIN_SCORE:
                        print(f"[SCORE_GATE] 15min {symbol} REJECTED: score={score_15min} < MIN_SCORE={MIN_SCORE}", flush=True)
                        continue
                    # ===== END SCORE VALIDATION =====
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK (DISABLED 2026-03-25) =====
                    # Code preserved but disabled - data showed gatekeeper reduces profitability
                    # 4h without gatekeeper: 56% WR | 1h with gatekeeper: 36% WR
                    signal_type = res15.get("bias", "UNKNOWN")
                    if ENABLE_DIRECTION_AWARE_GATEKEEPER:
                        try:
                            gates_passed, gate_results = DirectionAwareGatekeeper.check_all_gates(
                                df15,
                                direction=signal_type,
                                regime=regime15,
                                debug=False
                            )
                            
                            if not gates_passed:
                                failed_gates = [k for k, v in gate_results.items() if not v]
                                print(f"[PHASE2-FIXED] 15min {symbol} {signal_type} REJECTED - "
                                      f"failed: {failed_gates}", flush=True)
                                continue  # Skip to next symbol
                            else:
                                print(f"[PHASE2-FIXED] 15min {symbol} {signal_type} ✓ ALL GATES PASS ({regime15})", flush=True)
                        except Exception as e:
                            print(f"[PHASE2-FIXED] Error checking gates for {symbol}: {e}", flush=True)
                            # Fail gracefully - still allow signal if gates can't be checked
                            pass
                    else:
                        print(f"[GATEKEEPER-DISABLED] 15min {symbol}: DirectionAwareGatekeeper bypassed (code preserved)", flush=True)
                    # ===== END PHASE 2-FIXED GATES (DISABLED) =====
                    
                    last15 = last_sent.get(key15, 0)
                    if now - last15 >= COOLDOWN["15min"]:
                        numbered_signal = f"{idx}.A"

                        # Fresh orderbook/density logs (main_log)
                        log_orderbook_and_density(symbol)
                        try:
                            orderbook_result = get_order_wall_delta(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_order_wall_delta failed for {symbol}: {e}", flush=True)
                            orderbook_result = {"buy_wall": 0, "sell_wall": 0, "wall_delta": 0, "midprice": None}
                        try:
                            density_result = get_resting_order_density(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_resting_order_density failed for {symbol}: {e}", flush=True)
                            density_result = {"bid_density": 0.0, "ask_density": 0.0, "bid_levels": 0, "ask_levels": 0, "midprice": None}

                        bias = res15.get("bias", "NEUTRAL")
                        sf15.bias = bias

                        # DISABLED: SuperGK validation (broken mechanism, blocking all signals)
                        # TODO: Review SuperGK logic and fix it properly
                        print("[SuperGK][MAIN] DISABLED (broken) - bypassing SuperGK for ALL signals", flush=True)
                        super_gk_ok = True  # Bypass until SuperGK is fixed
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok} (bypassed)", flush=True)

                        if not super_gk_ok:
                            # legacy compatibility block (should not execute with global bypass)
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            if len(valid_debugs) < 2:
                                valid_debugs.append({
                                "symbol": symbol,
                                "tf": "15min",
                                "bias": bias,
                                "filter_weights_long": getattr(sf15, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf15, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf15, 'gatekeepers', []),
                                "results_long": res15.get("results_long", {}),
                                "results_short": res15.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 15min",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res15.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_15m": early_breakout_15m
                                })
                            continue

                        # --- Prepare to send ---
                        print(f"[LOG] Sending 15min alert for {res15.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        # Live entry price preferred, fall back to analyzer price
                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res15.get("symbol"),
                                bias,
                                tf=res15.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res15.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res15.get("price", 0.0)

                        # Coerce defensively
                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        # Debug: final entry price vs analyzer price
                        print(f"[DEBUG] entry_price_final={entry_price:.8f} analyzer_price={res15.get('price')}", flush=True)

                        # Collect signal metadata
                        score = res15.get("score", 0)
                        score_max = res15.get("score_max", 0)
                        passes = res15.get("passes", 0)
                        gatekeepers_total = res15.get("gatekeepers_total", 0)
                        passed_weight = res15.get("passed_weight", 0.0)
                        total_weight = res15.get("total_weight", 0.0)
                        Route = res15.get("Route", None)
                        signal_type = bias
                        tf_val = res15.get("tf", "15min")
                        symbol_val = res15.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df15.index.get_loc(df15.index[-1])

                        # Log signal generated (passes SmartFilter)
                        logger.signal_generated(symbol_val, tf_val, signal_type, score, entry_price)

                        # Get market regime EARLY (needed for Option C: RANGE TP scaling)
                        regime = sf15._market_regime() if hasattr(sf15, "_market_regime") else None
                        
                        # ===== PHASE 3B: REVERSAL QUALITY GATE (15min block) =====
                        if Route == "REVERSAL":
                            reversal_type = res15.get("reversal_type", None)  # BULLISH or BEARISH
                            
                            quality_check = check_reversal_quality(
                                symbol=symbol_val,
                                df=df15,
                                reversal_type=reversal_type,
                                regime=regime,
                                direction=signal_type
                            )
                            
                            # Log quality gate results
                            gate_status = ", ".join([f"{k.split('_', 1)[1]}={'✓' if v else '✗'}" for k, v in quality_check["gate_results"].items()])
                            print(f"[PHASE3B-RQ] 15min {symbol_val} {signal_type}: {gate_status} → Strength={quality_check['reversal_strength_score']:.0f}%", flush=True)
                            
                            # Apply recommendation
                            if not quality_check["allowed"]:
                                Route = quality_check["recommendation"]
                                print(f"[PHASE3B-FALLBACK] 15min {symbol_val}: REVERSAL rejected ({quality_check['reason']}) → Routed to {Route}", flush=True)
                            else:
                                print(f"[PHASE3B-APPROVED] 15min {symbol_val}: REVERSAL approved ({quality_check['reason']})", flush=True)
                        
                        # === RE-APPLY PHASE 1 VETO (post-Phase 3B route changes) ===
                        if Route in ["NONE", "AMBIGUOUS"]:
                            print(f"[VETO-POST-3B] 15min {symbol_val} {signal_type}: Route={Route} (changed by Phase 3B). REJECTING signal.", flush=True)
                            continue
                        
                        # Route scoring & recommendation log
                        route_score = calculate_route_score(Route, signal_type, regime, reversal_strength_score=None)
                        route_rec = get_route_recommendation(Route, signal_type, regime)
                        print(f"[PHASE3B-SCORE] 15min {symbol_val}: {route_rec} (score: {route_score})", flush=True)
                        # ===== END PHASE 3B: 15min block =====

                        # Recompute TP/SL against the final live entry_price so TP/SL match execution price
                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df15, entry_price, signal_type, regime=regime)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            # Debug: show tp_sl summary
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        # Append debug object used for exporting debug files
                        if len(valid_debugs) < 2:
                            valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf15, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf15, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf15, 'gatekeepers', []),
                            "results_long": res15.get("results_long", {}),
                            "results_short": res15.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_15m": early_breakout_15m,
                            "tp": tp,
                            "sl": sl,
                            "fib_levels": fib_levels
                            })

                        # Log fired signal (tracking) and then send telegram
                        try:
                            log_fired_signal(
                                symbol=symbol_val,
                                tf=tf_val,
                                signal_type=signal_type,
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=score,
                                max_score=score_max,
                                passed=passes,
                                max_passed=gatekeepers_total,
                                weights=passed_weight,
                                max_weights=total_weight,
                                confidence_rate=confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            print(f"[WARN] log_fired_signal raised: {e}", flush=True)
                            traceback.print_exc()

                        # DISABLED: Direction-aware threshold was too strict
                        # Scores are 9-12, but thresholds are 13-25 → rejected ALL signals
                        # Disabled 2026-03-06 00:18 GMT+7
                        # (threshold check moved to SmartFilter min_score, which is now 3)

                        # ===== RR VALIDATION & ENFORCEMENT (PROJECT-11) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        # Step 1: Validate TP/SL relationship for direction
                        is_valid_relationship, rr_error = validate_tp_sl_relationship(entry_price, tp, sl, signal_type)
                        if not is_valid_relationship:
                            print(f"[RR_ENFORCE] 15min signal REJECTED: {symbol_val} - {rr_error}", flush=True)
                            continue
                        
                        # Step 2: Enforce RR bounds (may adjust TP/SL)
                        adjusted_tp, adjusted_sl, enforced_rr, rr_action = enforce_rr_bounds(entry_price, tp, sl, signal_type, achieved_rr_value)
                        
                        if rr_action.startswith("INVALID"):
                            print(f"[RR_ENFORCE] 15min signal REJECTED: {symbol_val} - {rr_action}", flush=True)
                            continue
                        
                        # Step 3: Log enforcement action
                        if rr_action == "VALID":
                            print(f"[RR_ENFORCE] 15min {symbol_val}: VALID RR {enforced_rr:.3f} (no adjustment)", flush=True)
                        else:
                            print(f"[RR_ENFORCE] 15min {symbol_val}: {rr_action} | Calculated RR: {achieved_rr_value} → Enforced RR: {enforced_rr} | TP: {tp} → {adjusted_tp} | SL: {sl} → {adjusted_sl}", flush=True)
                            tp = adjusted_tp
                            sl = adjusted_sl
                        
                        # RR valid and enforced: Store signal + Fire
                        print(f"[RR_ENFORCE] 15min signal ACCEPTED: {symbol_val} - RR {enforced_rr:.3f} ({rr_action})", flush=True)
                        
                        # BLACKLIST CHECK: Reject blacklisted symbols
                        if apply_blacklist_filter(symbol_val):
                            continue
                        
                        # Store signal to JSONL
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
                        # Extract passed/failed filters from SmartFilter results (PROJECT-5B instrumentation)
                        passed_filters = []
                        failed_filters = []
                        passed_filter_count = 0
                        failed_filter_count = 0
                        
                        if signal_type == 'LONG':
                            results = res15.get('results_long', {})
                        else:  # SHORT
                            results = res15.get('results_short', {})
                        
                        passed_filters = results.get('passed_filters', [])
                        failed_filters = results.get('failed_filters', [])
                        passed_filter_count = results.get('passed_filter_count', 0)
                        failed_filter_count = results.get('failed_filter_count', 0)
                        
                        # Generate telegram_msg_id upfront (will be used for Telegram + stored in signal)
                        telegram_msg_id = str(uuid_lib.uuid4())[:12]
                        
                        signal_uuid = create_and_store_signal(
                            symbol=symbol_val,
                            timeframe=tf_val,
                            signal_type=signal_type,
                            fired_time_utc=fired_time_utc,
                            entry_price=entry_price,
                            tp_target=tp,
                            sl_target=sl,
                            tp_pct=tp_pct_val,
                            sl_pct=sl_pct_val,
                            achieved_rr=achieved_rr_value,
                            fib_ratio=None,  # ATR-based 2:1 RR (no Fibonacci)
                            atr_value=tp_sl.get('atr_value') if isinstance(tp_sl, dict) else None,
                            score=score,
                            max_score=score_max,
                            confidence=confidence,
                            route=Route,
                            regime=regime,
                            passed_gatekeepers=passes,
                            max_gatekeepers=gatekeepers_total,
                            passed_filters=passed_filters,
                            failed_filters=failed_filters,
                            passed_filter_count=passed_filter_count,
                            failed_filter_count=failed_filter_count,
                            telegram_msg_id=telegram_msg_id  # ← NEW: Pass telegram_msg_id upfront
                        )
                        
                        if not signal_uuid:
                            print(f"[WARN] Signal {symbol_val} (15min) REJECTED - duplicate within 120s. NOT sending Telegram alert.", flush=True)
                            continue

                        # IN-MEMORY DEDUP: Check if sent in THIS CYCLE (prevents rapid duplicates)
                        cycle_key = f"{symbol_val}|15min|{signal_type}"
                        if cycle_key in signals_sent_this_cycle:
                            print(f"[DEDUP-CYCLE] 15min {symbol_val} {signal_type}: Already sent in THIS CYCLE. SKIPPING.", flush=True)
                            continue
                        
                        # DISABLED: PHASE 4A: Check 30min+1h alignment (Scenario 4)
                        # This was rejecting ALL signals - disabled 2026-03-06 00:10 GMT+7
                        # alignment_allowed, trend_30, trend_1h, alignment_reason = check_multitf_alignment_30_1h(symbol_val, ohlcv_data)
                        # if not alignment_allowed:
                        #     continue
                        alignment_allowed = True  # FIX: Allow all signals to proceed
                        
                        # CRITICAL: Deduplication check (prevent duplicate within DEDUP_WINDOWS)
                        if is_duplicate_signal(symbol_val, "15min", signal_type):
                            continue
                        
                        # Mark as sent in this cycle
                        signals_sent_this_cycle.add(cycle_key)
                        
                        # Define symbol_group early (for tier matching + write_signal)
                        symbol_group = classify_symbol(symbol_val)
                        
                        # Send trade alert to Telegram
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                # Get signal tier (dynamic - will be Tier-X initially, populates over time)
                                # symbol_group already defined above
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime, symbol_group)
                                print(f"[TIER] 15min {symbol_val} ({symbol_group}): {signal_tier}", flush=True)
                                
                                print(f"[DEBUG] 15min: Calling send_telegram_alert for {symbol_val} (tf={tf_val}, Entry={entry_price})", flush=True)
                                sent_ok = send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    passed=passes,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    score_max=score_max,
                                    gatekeepers_total=gatekeepers_total,
                                    total_weight=total_weight,
                                    reversal_side=res15.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_15m=early_breakout_15m,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=None,
                                    achieved_rr=achieved_rr_value,
                                    tier=signal_tier
                                )
                                print(f"[DEBUG] 15min: send_telegram_alert returned {sent_ok} for {symbol_val}", flush=True)
                                if sent_ok:
                                    last_sent[key15] = now
                                    # signals_sent_this_cycle.add(signal_key)  # Dedup now uses SENT_SIGNALS.jsonl
                                    logger.signal_sent(symbol_val, "15min", signal_uuid[:12])
                                    
                                    # Log to SENT_SIGNALS for PEC tracking
                                    try:
                                        _sent_signal_tracker.log_sent_signal(
                                            signal_uuid=signal_uuid,
                                            symbol=symbol_val,
                                            timeframe="15min",
                                            signal_type=signal_type,
                                            entry_price=entry_price,
                                            tp_target=tp,
                                            sl_target=sl,
                                            tp_pct=tp_pct_val,
                                            sl_pct=sl_pct_val,
                                            achieved_rr=achieved_rr_value,
                                            score=score,
                                            max_score=score_max,
                                            confidence=confidence,
                                            route=Route,
                                            regime=regime,
                                            telegram_msg_id=signal_uuid[:12],
                                            fired_time_utc=fired_time_utc.isoformat(),
                                            passed_filters=passed_filters,
                                            failed_filters=failed_filters,
                                            passed_filter_count=passed_filter_count,
                                            failed_filter_count=failed_filter_count
                                        )
                                        # Also write to SIGNALS_MASTER.jsonl (single source of truth)
                                        if _master_writer_ready:
                                            _signals_master_writer.write_signal({
                                                'signal_uuid': signal_uuid,
                                                'symbol': symbol_val,
                                                'timeframe': '15min',
                                                'signal_type': signal_type,
                                                'entry_price': entry_price,
                                                'tp_target': tp,
                                                'sl_target': sl,
                                                'tp_pct': tp_pct_val,
                                                'sl_pct': sl_pct_val,
                                                'achieved_rr': achieved_rr_value,
                                                'score': score,
                                                'max_score': score_max,
                                                'confidence': confidence,
                                                'route': Route,
                                                'regime': regime,
                                                'telegram_msg_id': signal_uuid[:12],
                                                'fired_time_utc': fired_time_utc.isoformat(),
                                                'sent_time_utc': datetime.utcnow().isoformat(),
                                                'passed_filters': passed_filters,
                                                'failed_filters': failed_filters,
                                                'passed_filter_count': passed_filter_count,
                                                'failed_filter_count': failed_filter_count,
                                                'status': 'OPEN',
                                                'signal_origin': 'NEW_LIVE',
                                                'weighted_score': confidence * score / 100 if score_max else 0,
                                                'tier': signal_tier,

                                                'symbol_group': symbol_group,

                                                'confidence_level': 'HIGH' if confidence >= 75 else 'MID' if confidence >= 50 else 'LOW'

                                                })
                                            
                                            # PHASE 1: Dual-Write Verification (Alert + Continue strategy)
                                            dual_write_ok = True
                                            if _dual_write_ready:
                                                try:
                                                    verify_result = verify_signal_dual_write(
                                                        signal_uuid=signal_uuid,
                                                        signal_data={'symbol': symbol_val, 'timeframe': '15min', 'entry_price': entry_price},
                                                        raise_on_failure=False  # Alert+Continue (don't halt)
                                                    )
                                                    if verify_result:
                                                        print(f"[DUAL-WRITE] ✅ Verified {signal_uuid[:12]} to both files | FIRE: {symbol_val} 15min", flush=True)
                                                    else:
                                                        dual_write_ok = False
                                                        _divergence_tracker.record_failure(signal_uuid)
                                                        alert_msg = f"Dual-write failed: {signal_uuid[:12]} {symbol_val} 15min | MASTER:✓ AUDIT:✗"
                                                        print(f"[DUAL-WRITE] ❌ {alert_msg}", flush=True)
                                                        send_ops_alert(alert_msg, "CRITICAL")
                                                except Exception as e:
                                                    print(f"[DUAL-WRITE] Warning: Verification error (continuing): {e}", flush=True)
                                                    dual_write_ok = False
                                            
                                            # NEW: Trigger executor on signal fire (hybrid model - event-driven)
                                            trigger_executor_on_signal(signal_uuid, symbol_val, '15min')
                                    except Exception as e:
                                        print(f"[ERROR] Failed to log PEC signal for {symbol_val}: {e}", flush=True)
                                else:
                                    logger.signal_rejected(symbol_val, "15min", "Telegram send failed")
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val}: {e}", flush=True)

                else:
                    pass  # Silent - not a signal
            except Exception as e:
                print(f"[ERROR] Exception in processing 15min for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 30min TF block (mirror of 15min with identical safe flow) ---
            try:
                key30 = f"{symbol}_30min"
                if df30 is None or df30.empty:
                    res30 = None
                else:
                    sf30 = SmartFilter(symbol, df30, tf="30min")  # FIX: 2026-03-06 - use default min_score from smart_filter.py (now 10)
                    regime30 = sf30._market_regime()
                    res30 = sf30.analyze()

                # LOG ALL SIGNALS (even rejected) to ALL_SIGNALS.jsonl BEFORE filters_ok check
                if isinstance(res30, dict):
                    score_30min = res30.get("score")
                    try:
                        all_signals_path = "/Users/geniustarigan/.openclaw/workspace/ALL_SIGNALS.jsonl"
                        all_signal_entry = {
                            "symbol": symbol,
                            "timeframe": "30min",
                            "score": score_30min,
                            "min_score_required": MIN_SCORE,
                            "passed_filter": score_30min is not None and score_30min >= MIN_SCORE,
                            "fired_time_utc": datetime.utcnow().isoformat(),
                            "filters_ok": res30.get("filters_ok")
                        }
                        with open(all_signals_path, 'a') as f:
                            f.write(json.dumps(all_signal_entry) + '\n')
                    except Exception as e:
                        print(f"[WARN] Failed to log to ALL_SIGNALS.jsonl: {e}", flush=True)

                if isinstance(res30, dict) and res30.get("filters_ok") is True:
                    
                    # ===== SCORE VALIDATION GATE (Universal MIN_SCORE from smart_filter.py) =====
                    if score_30min is None or score_30min < MIN_SCORE:
                        print(f"[SCORE_GATE] 30min {symbol} REJECTED: score={score_30min} < MIN_SCORE={MIN_SCORE}", flush=True)
                        continue
                    # ===== END SCORE VALIDATION =====
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK - 30min (DISABLED 2026-03-25) =====
                    signal_type = res30.get("bias", "UNKNOWN")
                    if ENABLE_DIRECTION_AWARE_GATEKEEPER:
                        try:
                            gates_passed, gate_results = DirectionAwareGatekeeper.check_all_gates(
                                df30,
                                direction=signal_type,
                                regime=regime30,
                                debug=False
                            )
                            
                            if not gates_passed:
                                failed_gates = [k for k, v in gate_results.items() if not v]
                                print(f"[PHASE2-FIXED] 30min {symbol} {signal_type} REJECTED - "
                                      f"failed: {failed_gates}", flush=True)
                                continue  # Skip to next symbol
                            else:
                                print(f"[PHASE2-FIXED] 30min {symbol} {signal_type} ✓ ALL GATES PASS ({regime30})", flush=True)
                        except Exception as e:
                            print(f"[PHASE2-FIXED] Error checking gates for {symbol}: {e}", flush=True)
                            pass
                    else:
                        print(f"[GATEKEEPER-DISABLED] 30min {symbol}: DirectionAwareGatekeeper bypassed (code preserved)", flush=True)
                    # ===== END PHASE 2-FIXED GATES (DISABLED) =====
                    
                    last30 = last_sent.get(key30, 0)
                    if now - last30 >= COOLDOWN["30min"]:
                        numbered_signal = f"{idx}.B"

                        log_orderbook_and_density(symbol)
                        try:
                            orderbook_result = get_order_wall_delta(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_order_wall_delta failed for {symbol}: {e}", flush=True)
                            orderbook_result = {"buy_wall": 0, "sell_wall": 0, "wall_delta": 0, "midprice": None}
                        try:
                            density_result = get_resting_order_density(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_resting_order_density failed for {symbol}: {e}", flush=True)
                            density_result = {"bid_density": 0.0, "ask_density": 0.0, "bid_levels": 0, "ask_levels": 0, "midprice": None}

                        bias = res30.get("bias", "NEUTRAL")
                        sf30.bias = bias

                        # DISABLED: SuperGK validation (broken mechanism, blocking all signals)
                        # TODO: Review SuperGK logic and fix it properly
                        print("[SuperGK][MAIN] DISABLED (broken) - bypassing SuperGK for ALL signals", flush=True)
                        super_gk_ok = True  # Bypass until SuperGK is fixed
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok} (bypassed)", flush=True)

                        if not super_gk_ok:
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            if len(valid_debugs) < 2:
                                valid_debugs.append({
                                "symbol": symbol,
                                "tf": "30min",
                                "bias": bias,
                                "filter_weights_long": getattr(sf30, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf30, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf30, 'gatekeepers', []),
                                "results_long": res30.get("results_long", {}),
                                "results_short": res30.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 30min",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res30.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_30m": early_breakout_30m
                                })
                            continue

                        print(f"[LOG] Sending 30min alert for {res30.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res30.get("symbol"),
                                bias,
                                tf=res30.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res30.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res30.get("price", 0.0)

                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        print(f"[DEBUG] entry_price_final={entry_price:.8f} analyzer_price={res30.get('price')}", flush=True)

                        score = res30.get("score", 0)
                        score_max = res30.get("score_max", 0)
                        passes = res30.get("passes", 0)
                        gatekeepers_total = res30.get("gatekeepers_total", 0)
                        passed_weight = res30.get("passed_weight", 0.0)
                        total_weight = res30.get("total_weight", 0.0)
                        Route = res30.get("Route", None)
                        signal_type = bias
                        tf_val = res30.get("tf", "30min")
                        symbol_val = res30.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df30.index.get_loc(df30.index[-1])

                        # Log signal generated (passes SmartFilter)
                        logger.signal_generated(symbol_val, tf_val, signal_type, score, entry_price)

                        # Get market regime EARLY (needed for Option C: RANGE TP scaling)
                        regime = sf30._market_regime() if hasattr(sf30, "_market_regime") else None
                        
                        # ===== PHASE 3B: REVERSAL QUALITY GATE (30min block) =====
                        if Route == "REVERSAL":
                            reversal_type = res30.get("reversal_type", None)  # BULLISH or BEARISH
                            
                            quality_check = check_reversal_quality(
                                symbol=symbol_val,
                                df=df30,
                                reversal_type=reversal_type,
                                regime=regime,
                                direction=signal_type
                            )
                            
                            # Log quality gate results
                            gate_status = ", ".join([f"{k.split('_', 1)[1]}={'✓' if v else '✗'}" for k, v in quality_check["gate_results"].items()])
                            print(f"[PHASE3B-RQ] 30min {symbol_val} {signal_type}: {gate_status} → Strength={quality_check['reversal_strength_score']:.0f}%", flush=True)
                            
                            # Apply recommendation
                            if not quality_check["allowed"]:
                                Route = quality_check["recommendation"]
                                print(f"[PHASE3B-FALLBACK] 30min {symbol_val}: REVERSAL rejected ({quality_check['reason']}) → Routed to {Route}", flush=True)
                            else:
                                print(f"[PHASE3B-APPROVED] 30min {symbol_val}: REVERSAL approved ({quality_check['reason']})", flush=True)
                        
                        # === RE-APPLY PHASE 1 VETO (post-Phase 3B route changes) ===
                        if Route in ["NONE", "AMBIGUOUS"]:
                            print(f"[VETO-POST-3B] 30min {symbol_val} {signal_type}: Route={Route} (changed by Phase 3B). REJECTING signal.", flush=True)
                            continue
                        
                        # Route scoring & recommendation log
                        route_score = calculate_route_score(Route, signal_type, regime, reversal_strength_score=None)
                        route_rec = get_route_recommendation(Route, signal_type, regime)
                        print(f"[PHASE3B-SCORE] 30min {symbol_val}: {route_rec} (score: {route_score})", flush=True)
                        # ===== END PHASE 3B: 30min block =====

                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df30, entry_price, signal_type, regime=regime)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        if len(valid_debugs) < 2:
                            valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf30, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf30, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf30, 'gatekeepers', []),
                            "results_long": res30.get("results_long", {}),
                            "results_short": res30.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_30m": early_breakout_30m,
                            "tp": tp,
                            "sl": sl,
                            "fib_levels": fib_levels
                            })

                        try:
                            log_fired_signal(
                                symbol=symbol_val,
                                tf=tf_val,
                                signal_type=signal_type,
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=score,
                                max_score=score_max,
                                passed=passes,
                                max_passed=gatekeepers_total,
                                weights=passed_weight,
                                max_weights=total_weight,
                                confidence_rate=confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            print(f"[WARN] log_fired_signal raised: {e}", flush=True)
                            traceback.print_exc()

                        # DISABLED: Direction-aware threshold was too strict (30min)
                        # Disabled 2026-03-06 00:18 GMT+7
                        # ===== END PHASE 2-FIXED THRESHOLD =====

                        # ===== RR VALIDATION & ENFORCEMENT (PROJECT-11) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        # Step 1: Validate TP/SL relationship for direction
                        is_valid_relationship, rr_error = validate_tp_sl_relationship(entry_price, tp, sl, signal_type)
                        if not is_valid_relationship:
                            print(f"[RR_ENFORCE] 30min signal REJECTED: {symbol_val} - {rr_error}", flush=True)
                            continue
                        
                        # Step 2: Enforce RR bounds (may adjust TP/SL)
                        adjusted_tp, adjusted_sl, enforced_rr, rr_action = enforce_rr_bounds(entry_price, tp, sl, signal_type, achieved_rr_value)
                        
                        if rr_action.startswith("INVALID"):
                            print(f"[RR_ENFORCE] 30min signal REJECTED: {symbol_val} - {rr_action}", flush=True)
                            continue
                        
                        # Step 3: Log enforcement action
                        if rr_action == "VALID":
                            print(f"[RR_ENFORCE] 30min {symbol_val}: VALID RR {enforced_rr:.3f} (no adjustment)", flush=True)
                        else:
                            print(f"[RR_ENFORCE] 30min {symbol_val}: {rr_action} | Calculated RR: {achieved_rr_value} → Enforced RR: {enforced_rr} | TP: {tp} → {adjusted_tp} | SL: {sl} → {adjusted_sl}", flush=True)
                            tp = adjusted_tp
                            sl = adjusted_sl
                        
                        # RR valid and enforced: Store signal + Fire
                        print(f"[RR_ENFORCE] 30min signal ACCEPTED: {symbol_val} - RR {enforced_rr:.3f} ({rr_action})", flush=True)
                        
                        # BLACKLIST CHECK: Reject blacklisted symbols
                        if apply_blacklist_filter(symbol_val):
                            continue
                        
                        # Store signal to JSONL
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
                        # Extract passed/failed filters from SmartFilter results (PROJECT-5B instrumentation)
                        passed_filters = []
                        failed_filters = []
                        passed_filter_count = 0
                        failed_filter_count = 0
                        
                        if signal_type == 'LONG':
                            results = res30.get('results_long', {})
                        else:  # SHORT
                            results = res30.get('results_short', {})
                        
                        passed_filters = results.get('passed_filters', [])
                        failed_filters = results.get('failed_filters', [])
                        passed_filter_count = results.get('passed_filter_count', 0)
                        failed_filter_count = results.get('failed_filter_count', 0)
                        
                        # Generate telegram_msg_id upfront (will be used for Telegram + stored in signal)
                        telegram_msg_id = str(uuid_lib.uuid4())[:12]
                        
                        signal_uuid = create_and_store_signal(
                            symbol=symbol_val,
                            timeframe=tf_val,
                            signal_type=signal_type,
                            fired_time_utc=fired_time_utc,
                            entry_price=entry_price,
                            tp_target=tp,
                            sl_target=sl,
                            tp_pct=tp_pct_val,
                            sl_pct=sl_pct_val,
                            achieved_rr=achieved_rr_value,
                            fib_ratio=None,  # ATR-based 2:1 RR (no Fibonacci)
                            atr_value=tp_sl.get('atr_value') if isinstance(tp_sl, dict) else None,
                            score=score,
                            max_score=score_max,
                            confidence=confidence,
                            route=Route,
                            regime=regime,
                            passed_gatekeepers=passes,
                            max_gatekeepers=gatekeepers_total,
                            passed_filters=passed_filters,
                            failed_filters=failed_filters,
                            passed_filter_count=passed_filter_count,
                            failed_filter_count=failed_filter_count,
                            telegram_msg_id=telegram_msg_id  # ← NEW: Pass telegram_msg_id upfront
                        )
                        
                        if not signal_uuid:
                            print(f"[WARN] Signal {symbol_val} (30min) REJECTED - duplicate within 120s. NOT sending Telegram alert.", flush=True)
                            continue

                        # PHASE 4A: Check 30min+1h alignment (Scenario 4)
                        alignment_allowed, trend_30, trend_1h, alignment_reason = check_multitf_alignment_30_1h(symbol_val, ohlcv_data)
                        print(f"[PHASE4A-S4] 30min {symbol_val}: {alignment_reason}", flush=True)
                        
                        if not alignment_allowed:
                            print(f"[PHASE4A-S4-FILTERED] 30min {symbol_val} {signal_type}: Rejected by 30min+1h filter", flush=True)
                            continue
                        
                        # IN-MEMORY DEDUP: Check if sent in THIS CYCLE (prevents rapid duplicates)
                        cycle_key = f"{symbol_val}|30min|{signal_type}"
                        if cycle_key in signals_sent_this_cycle:
                            print(f"[DEDUP-CYCLE] 30min {symbol_val} {signal_type}: Already sent in THIS CYCLE. SKIPPING.", flush=True)
                            continue
                        
                        # CRITICAL: Deduplication check (prevent duplicate within DEDUP_WINDOWS)
                        if is_duplicate_signal(symbol_val, "30min", signal_type):
                            print(f"[DEDUP-WINDOW] 30min {symbol_val} {signal_type}: Duplicate within dedup window. SKIPPING.", flush=True)
                            continue
                        
                        # Mark as sent in this cycle
                        signals_sent_this_cycle.add(cycle_key)
                        
                        # Define symbol_group early (for tier matching + write_signal)
                        symbol_group = classify_symbol(symbol_val)
                        
                        # TEMP DEBUG: Verify gate code is reached
                        print(f"[GATE-ENTRY] 30min {symbol_val} signal reached gate block", flush=True)
                        
                        # EMA CONFIRMATION GATE (TF-Specific): Align short-term signal to long-term trend
                        # Optimization 2026-02-27: Use EMA100 for 30min (better short-term alignment than EMA200)
                        try:
                            ema_period_30m = EMA_CONFIG.get("30min", 100)  # Default EMA100 for 30min
                            ema_col_name = f"ema{ema_period_30m}"
                            
                            close_price = sf30.df['close'].iat[-1] if sf30.df is not None and not sf30.df.empty else None
                            ema_val = sf30.df[ema_col_name].iat[-1] if sf30.df is not None and ema_col_name in sf30.df.columns and not sf30.df.empty else None
                            
                            # DIAGNOSTIC: Always log gate status
                            has_ema_col = ema_col_name in sf30.df.columns if sf30.df is not None else False
                            print(f"[EMA{ema_period_30m}-DEBUG] 30min {symbol_val}: has_col={has_ema_col}, close={close_price}, ema{ema_period_30m}={ema_val}", flush=True)
                            
                            if close_price is not None and ema_val is not None:
                                if signal_type == "LONG" and close_price < ema_val:
                                    print(f"[EMA{ema_period_30m}-GATE] 30min {symbol_val}: LONG rejected (close ${close_price:.6f} < EMA{ema_period_30m} ${ema_val:.6f})", flush=True)
                                    continue
                                elif signal_type == "SHORT" and close_price > ema_val:
                                    print(f"[EMA{ema_period_30m}-GATE] 30min {symbol_val}: SHORT rejected (close ${close_price:.6f} > EMA{ema_period_30m} ${ema_val:.6f})", flush=True)
                                    continue
                                else:
                                    signal_alignment = f"LONG above EMA{ema_period_30m}" if signal_type == "LONG" else f"SHORT below EMA{ema_period_30m}"
                                    print(f"[EMA{ema_period_30m}-PASS] 30min {symbol_val}: {signal_alignment} ✓", flush=True)
                            else:
                                print(f"[EMA{ema_period_30m}-SKIP] 30min {symbol_val}: Skipping gate (values missing)", flush=True)
                        except Exception as e:
                            print(f"[EMA{ema_period_30m}-WARN] 30min {symbol_val}: Error checking gate: {e}", flush=True)
                        
                        # Send trade alert to Telegram
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                # Get signal tier (dynamic - will be Tier-X initially, populates over time)
                                # symbol_group already defined above
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime, symbol_group)
                                print(f"[TIER] 30min {symbol_val} ({symbol_group}): {signal_tier}", flush=True)
                                
                                print(f"[DEBUG] 30min: Calling send_telegram_alert for {symbol_val} (tf={tf_val}, Entry={entry_price})", flush=True)
                                sent_ok = send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    passed=passes,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    score_max=score_max,
                                    gatekeepers_total=gatekeepers_total,
                                    total_weight=total_weight,
                                    reversal_side=res30.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_15m=early_breakout_30m,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=None,
                                    achieved_rr=achieved_rr_value,
                                    tier=signal_tier
                                )
                                print(f"[DEBUG] 30min: send_telegram_alert returned {sent_ok} for {symbol_val}", flush=True)
                                if sent_ok:
                                    last_sent[key30] = now
                                    # signals_sent_this_cycle.add(signal_key)  # Dedup now uses SENT_SIGNALS.jsonl
                                    logger.signal_sent(symbol_val, "30min", signal_uuid[:12])
                                    
                                    # Log to SENT_SIGNALS for PEC tracking
                                    try:
                                        _sent_signal_tracker.log_sent_signal(
                                            signal_uuid=signal_uuid,
                                            symbol=symbol_val,
                                            timeframe="30min",
                                            signal_type=signal_type,
                                            entry_price=entry_price,
                                            tp_target=tp,
                                            sl_target=sl,
                                            tp_pct=tp_pct_val,
                                            sl_pct=sl_pct_val,
                                            achieved_rr=achieved_rr_value,
                                            score=score,
                                            max_score=score_max,
                                            confidence=confidence,
                                            route=Route,
                                            regime=regime,
                                            telegram_msg_id=signal_uuid[:12],
                                            fired_time_utc=fired_time_utc.isoformat(),
                                            passed_filters=passed_filters,
                                            failed_filters=failed_filters,
                                            passed_filter_count=passed_filter_count,
                                            failed_filter_count=failed_filter_count
                                        )
                                        # Also write to SIGNALS_MASTER.jsonl (single source of truth)
                                        if _master_writer_ready:
                                            _signals_master_writer.write_signal({
                                                'signal_uuid': signal_uuid,
                                                'symbol': symbol_val,
                                                'timeframe': '30min',
                                                'signal_type': signal_type,
                                                'entry_price': entry_price,
                                                'tp_target': tp,
                                                'sl_target': sl,
                                                'tp_pct': tp_pct_val,
                                                'sl_pct': sl_pct_val,
                                                'achieved_rr': achieved_rr_value,
                                                'score': score,
                                                'max_score': score_max,
                                                'confidence': confidence,
                                                'route': Route,
                                                'regime': regime,
                                                'telegram_msg_id': signal_uuid[:12],
                                                'fired_time_utc': fired_time_utc.isoformat(),
                                                'sent_time_utc': datetime.utcnow().isoformat(),
                                                'passed_filters': passed_filters,
                                                'failed_filters': failed_filters,
                                                'passed_filter_count': passed_filter_count,
                                                'failed_filter_count': failed_filter_count,
                                                'status': 'OPEN',
                                                'signal_origin': 'NEW_LIVE',
                                                'weighted_score': confidence * score / 100 if score_max else 0,
                                                'tier': signal_tier,

                                                'symbol_group': symbol_group,

                                                'confidence_level': 'HIGH' if confidence >= 75 else 'MID' if confidence >= 50 else 'LOW'

                                                })
                                            
                                            # PHASE 1: Dual-Write Verification (Alert + Continue strategy)
                                            dual_write_ok = True
                                            if _dual_write_ready:
                                                try:
                                                    verify_result = verify_signal_dual_write(
                                                        signal_uuid=signal_uuid,
                                                        signal_data={'symbol': symbol_val, 'timeframe': '30min', 'entry_price': entry_price},
                                                        raise_on_failure=False  # Alert+Continue (don't halt)
                                                    )
                                                    if verify_result:
                                                        print(f"[DUAL-WRITE] ✅ Verified {signal_uuid[:12]} to both files | FIRE: {symbol_val} 30min", flush=True)
                                                    else:
                                                        dual_write_ok = False
                                                        _divergence_tracker.record_failure(signal_uuid)
                                                        alert_msg = f"Dual-write failed: {signal_uuid[:12]} {symbol_val} 30min | MASTER:✓ AUDIT:✗"
                                                        print(f"[DUAL-WRITE] ❌ {alert_msg}", flush=True)
                                                        send_ops_alert(alert_msg, "CRITICAL")
                                                except Exception as e:
                                                    print(f"[DUAL-WRITE] Warning: Verification error (continuing): {e}", flush=True)
                                                    dual_write_ok = False
                                            
                                            # NEW: Trigger executor on signal fire (hybrid model - event-driven)
                                            trigger_executor_on_signal(signal_uuid, symbol_val, '30min')
                                    except Exception as e:
                                        print(f"[ERROR] Failed to log PEC signal for {symbol_val}: {e}", flush=True)
                                else:
                                    logger.signal_rejected(symbol_val, "30min", "Telegram send failed")
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (30min): {e}", flush=True)
                                traceback.print_exc()
                else:
                    pass  # Silent - not a signal
            except Exception as e:
                print(f"[ERROR] Exception in processing 30min for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 1h TF block (mirror of 15min with identical safe flow) ---
            try:
                key1h = f"{symbol}_1h"
                if df1h is None or df1h.empty:
                    res1h = None
                else:
                    sf1h = SmartFilter(symbol, df1h, tf="1h")
                    regime1h = sf1h._market_regime()
                    res1h = sf1h.analyze()

                # LOG ALL SIGNALS (even rejected) to ALL_SIGNALS.jsonl BEFORE filters_ok check
                if isinstance(res1h, dict):
                    score_1h = res1h.get("score")
                    try:
                        all_signals_path = "/Users/geniustarigan/.openclaw/workspace/ALL_SIGNALS.jsonl"
                        all_signal_entry = {
                            "symbol": symbol,
                            "timeframe": "1h",
                            "score": score_1h,
                            "min_score_required": MIN_SCORE,
                            "passed_filter": score_1h is not None and score_1h >= MIN_SCORE,
                            "fired_time_utc": datetime.utcnow().isoformat(),
                            "filters_ok": res1h.get("filters_ok")
                        }
                        with open(all_signals_path, 'a') as f:
                            f.write(json.dumps(all_signal_entry) + '\n')
                    except Exception as e:
                        print(f"[WARN] Failed to log to ALL_SIGNALS.jsonl: {e}", flush=True)

                if isinstance(res1h, dict) and res1h.get("filters_ok") is True:
                    
                    # ===== SCORE VALIDATION GATE (Universal MIN_SCORE from smart_filter.py) =====
                    if score_1h is None or score_1h < MIN_SCORE:
                        print(f"[SCORE_GATE] 1h {symbol} REJECTED: score={score_1h} < MIN_SCORE={MIN_SCORE}", flush=True)
                        continue
                    # ===== END SCORE VALIDATION =====
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK - 1h (DISABLED 2026-03-25) =====
                    signal_type = res1h.get("bias", "UNKNOWN")
                    if ENABLE_DIRECTION_AWARE_GATEKEEPER:
                        try:
                            gates_passed, gate_results = DirectionAwareGatekeeper.check_all_gates(
                                df1h,
                                direction=signal_type,
                                regime=regime1h,
                                debug=False
                            )
                            
                            if not gates_passed:
                                failed_gates = [k for k, v in gate_results.items() if not v]
                                print(f"[PHASE2-FIXED] 1h {symbol} {signal_type} REJECTED - "
                                      f"failed: {failed_gates}", flush=True)
                                continue  # Skip to next symbol
                            else:
                                print(f"[PHASE2-FIXED] 1h {symbol} {signal_type} ✓ ALL GATES PASS ({regime1h})", flush=True)
                        except Exception as e:
                            print(f"[PHASE2-FIXED] Error checking gates for {symbol}: {e}", flush=True)
                            pass
                    else:
                        print(f"[GATEKEEPER-DISABLED] 1h {symbol}: DirectionAwareGatekeeper bypassed (code preserved)", flush=True)
                    # ===== END PHASE 2-FIXED GATES (DISABLED) =====
                    
                    last1h = last_sent.get(key1h, 0)
                    if now - last1h >= COOLDOWN["1h"]:
                        numbered_signal = f"{idx}.C"

                        log_orderbook_and_density(symbol)
                        try:
                            orderbook_result = get_order_wall_delta(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_order_wall_delta failed for {symbol}: {e}", flush=True)
                            orderbook_result = {"buy_wall": 0, "sell_wall": 0, "wall_delta": 0, "midprice": None}
                        try:
                            density_result = get_resting_order_density(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_resting_order_density failed for {symbol}: {e}", flush=True)
                            density_result = {"bid_density": 0.0, "ask_density": 0.0, "bid_levels": 0, "ask_levels": 0, "midprice": None}

                        bias = res1h.get("bias", "NEUTRAL")
                        sf1h.bias = bias

                        # DISABLED: SuperGK validation (broken mechanism, blocking all signals)
                        # TODO: Review SuperGK logic and fix it properly
                        print("[SuperGK][MAIN] DISABLED (broken) - bypassing SuperGK for ALL signals", flush=True)
                        super_gk_ok = True  # Bypass until SuperGK is fixed
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok} (bypassed)", flush=True)

                        if not super_gk_ok:
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            if len(valid_debugs) < 2:
                                valid_debugs.append({
                                "symbol": symbol,
                                "tf": "1h",
                                "bias": bias,
                                "filter_weights_long": getattr(sf1h, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf1h, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf1h, 'gatekeepers', []),
                                "results_long": res1h.get("results_long", {}),
                                "results_short": res1h.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 1h",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res1h.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_1h": early_breakout_1h
                                })
                            continue

                        print(f"[LOG] Sending 1h alert for {res1h.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res1h.get("symbol"),
                                bias,
                                tf=res1h.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res1h.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res1h.get("price", 0.0)

                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        print(f"[DEBUG] entry_price_final={entry_price:.8f} analyzer_price={res1h.get('price')}", flush=True)

                        score = res1h.get("score", 0)
                        score_max = res1h.get("score_max", 0)
                        passes = res1h.get("passes", 0)
                        gatekeepers_total = res1h.get("gatekeepers_total", 0)
                        passed_weight = res1h.get("passed_weight", 0.0)
                        total_weight = res1h.get("total_weight", 0.0)
                        Route = res1h.get("Route", None)
                        signal_type = bias
                        tf_val = res1h.get("tf", "1h")
                        symbol_val = res1h.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df1h.index.get_loc(df1h.index[-1])

                        # Log signal generated (passes SmartFilter)
                        logger.signal_generated(symbol_val, tf_val, signal_type, score, entry_price)

                        # Get market regime EARLY (needed for Option C: RANGE TP scaling)
                        regime = sf1h._market_regime() if hasattr(sf1h, "_market_regime") else None
                        
                        # ===== PHASE 3B: REVERSAL QUALITY GATE (1h block) =====
                        if Route == "REVERSAL":
                            reversal_type = res1h.get("reversal_type", None)  # BULLISH or BEARISH
                            
                            quality_check = check_reversal_quality(
                                symbol=symbol_val,
                                df=df1h,
                                reversal_type=reversal_type,
                                regime=regime,
                                direction=signal_type
                            )
                            
                            # Log quality gate results
                            gate_status = ", ".join([f"{k.split('_', 1)[1]}={'✓' if v else '✗'}" for k, v in quality_check["gate_results"].items()])
                            print(f"[PHASE3B-RQ] 1h {symbol_val} {signal_type}: {gate_status} → Strength={quality_check['reversal_strength_score']:.0f}%", flush=True)
                            
                            # Apply recommendation
                            if not quality_check["allowed"]:
                                Route = quality_check["recommendation"]
                                print(f"[PHASE3B-FALLBACK] 1h {symbol_val}: REVERSAL rejected ({quality_check['reason']}) → Routed to {Route}", flush=True)
                            else:
                                print(f"[PHASE3B-APPROVED] 1h {symbol_val}: REVERSAL approved ({quality_check['reason']})", flush=True)
                        
                        # === RE-APPLY PHASE 1 VETO (post-Phase 3B route changes) ===
                        if Route in ["NONE", "AMBIGUOUS"]:
                            print(f"[VETO-POST-3B] 1h {symbol_val} {signal_type}: Route={Route} (changed by Phase 3B). REJECTING signal.", flush=True)
                            continue
                        
                        # Route scoring & recommendation log
                        route_score = calculate_route_score(Route, signal_type, regime, reversal_strength_score=None)
                        route_rec = get_route_recommendation(Route, signal_type, regime)
                        print(f"[PHASE3B-SCORE] 1h {symbol_val}: {route_rec} (score: {route_score})", flush=True)
                        # ===== END PHASE 3B: 1h block =====

                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df1h, entry_price, signal_type, regime=regime)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        if len(valid_debugs) < 2:
                            valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf1h, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf1h, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf1h, 'gatekeepers', []),
                            "results_long": res1h.get("results_long", {}),
                            "results_short": res1h.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_1h": early_breakout_1h,
                            "tp": tp,
                            "sl": sl,
                            "fib_levels": fib_levels
                            })

                        try:
                            log_fired_signal(
                                symbol=symbol_val,
                                tf=tf_val,
                                signal_type=signal_type,
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=score,
                                max_score=score_max,
                                passed=passes,
                                max_passed=gatekeepers_total,
                                weights=passed_weight,
                                max_weights=total_weight,
                                confidence_rate=confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            print(f"[WARN] log_fired_signal raised: {e}", flush=True)
                            traceback.print_exc()

                        # DISABLED: Direction-aware threshold was too strict (1h)
                        # Disabled 2026-03-06 00:18 GMT+7

                        # ===== RR VALIDATION & ENFORCEMENT (PROJECT-11) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        # Step 1: Validate TP/SL relationship for direction
                        is_valid_relationship, rr_error = validate_tp_sl_relationship(entry_price, tp, sl, signal_type)
                        if not is_valid_relationship:
                            print(f"[RR_ENFORCE] 1h signal REJECTED: {symbol_val} - {rr_error}", flush=True)
                            continue
                        
                        # Step 2: Enforce RR bounds (may adjust TP/SL)
                        adjusted_tp, adjusted_sl, enforced_rr, rr_action = enforce_rr_bounds(entry_price, tp, sl, signal_type, achieved_rr_value)
                        
                        if rr_action.startswith("INVALID"):
                            print(f"[RR_ENFORCE] 1h signal REJECTED: {symbol_val} - {rr_action}", flush=True)
                            continue
                        
                        # Step 3: Log enforcement action
                        if rr_action == "VALID":
                            print(f"[RR_ENFORCE] 1h {symbol_val}: VALID RR {enforced_rr:.3f} (no adjustment)", flush=True)
                        else:
                            print(f"[RR_ENFORCE] 1h {symbol_val}: {rr_action} | Calculated RR: {achieved_rr_value} → Enforced RR: {enforced_rr} | TP: {tp} → {adjusted_tp} | SL: {sl} → {adjusted_sl}", flush=True)
                            tp = adjusted_tp
                            sl = adjusted_sl
                        
                        # RR valid and enforced: Store signal + Fire
                        print(f"[RR_ENFORCE] 1h signal ACCEPTED: {symbol_val} - RR {enforced_rr:.3f} ({rr_action})", flush=True)
                        
                        # BLACKLIST CHECK: Reject blacklisted symbols
                        if apply_blacklist_filter(symbol_val):
                            continue
                        
                        # Store signal to JSONL
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
                        # Extract passed/failed filters from SmartFilter results (PROJECT-5B instrumentation)
                        passed_filters = []
                        failed_filters = []
                        passed_filter_count = 0
                        failed_filter_count = 0
                        
                        if signal_type == 'LONG':
                            results = res1h.get('results_long', {})
                        else:  # SHORT
                            results = res1h.get('results_short', {})
                        
                        passed_filters = results.get('passed_filters', [])
                        failed_filters = results.get('failed_filters', [])
                        passed_filter_count = results.get('passed_filter_count', 0)
                        failed_filter_count = results.get('failed_filter_count', 0)
                        
                        # Generate telegram_msg_id upfront (will be used for Telegram + stored in signal)
                        telegram_msg_id = str(uuid_lib.uuid4())[:12]
                        
                        signal_uuid = create_and_store_signal(
                            symbol=symbol_val,
                            timeframe=tf_val,
                            signal_type=signal_type,
                            fired_time_utc=fired_time_utc,
                            entry_price=entry_price,
                            tp_target=tp,
                            sl_target=sl,
                            tp_pct=tp_pct_val,
                            sl_pct=sl_pct_val,
                            achieved_rr=achieved_rr_value,
                            fib_ratio=None,  # ATR-based 2:1 RR (no Fibonacci)
                            atr_value=tp_sl.get('atr_value') if isinstance(tp_sl, dict) else None,
                            score=score,
                            max_score=score_max,
                            confidence=confidence,
                            route=Route,
                            regime=regime,
                            passed_gatekeepers=passes,
                            max_gatekeepers=gatekeepers_total,
                            passed_filters=passed_filters,
                            failed_filters=failed_filters,
                            passed_filter_count=passed_filter_count,
                            failed_filter_count=failed_filter_count,
                            telegram_msg_id=telegram_msg_id  # ← NEW: Pass telegram_msg_id upfront
                        )
                        
                        if not signal_uuid:
                            print(f"[WARN] Signal {symbol_val} (1h) REJECTED - duplicate within 120s. NOT sending Telegram alert.", flush=True)
                            continue

                        # PHASE 4A: Check 30min+1h alignment (Scenario 4)
                        # Note: For 1h signals, we check if 30min aligns with 1h (we're already at 1h, so both TFs must agree)
                        alignment_allowed, trend_30, trend_1h, alignment_reason = check_multitf_alignment_30_1h(symbol_val, ohlcv_data)
                        print(f"[PHASE4A-S4] 1h {symbol_val}: {alignment_reason}", flush=True)
                        
                        if not alignment_allowed:
                            print(f"[PHASE4A-S4-FILTERED] 1h {symbol_val} {signal_type}: Rejected by 30min+1h filter", flush=True)
                            continue

                        # IN-MEMORY DEDUP: Check if sent in THIS CYCLE (prevents rapid duplicates)
                        cycle_key = f"{symbol_val}|1h|{signal_type}"
                        if cycle_key in signals_sent_this_cycle:
                            print(f"[DEDUP-CYCLE] 1h {symbol_val} {signal_type}: Already sent in THIS CYCLE. SKIPPING.", flush=True)
                            continue
                        
                        # CRITICAL: Deduplication check (prevent duplicate within DEDUP_WINDOWS)
                        if is_duplicate_signal(symbol_val, "1h", signal_type):
                            continue
                        
                        # Mark as sent in this cycle
                        signals_sent_this_cycle.add(cycle_key)
                        
                        # Define symbol_group early (for tier matching + write_signal)
                        symbol_group = classify_symbol(symbol_val)
                        
                        # Send trade alert to Telegram
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                # Get signal tier (dynamic - will be Tier-X initially, populates over time)
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime, symbol_group)
                                print(f"[TIER] 1h {symbol_val} ({symbol_group}): {signal_tier}", flush=True)
                                
                                print(f"[DEBUG] 1h: Calling send_telegram_alert for {symbol_val} (tf={tf_val}, Entry={entry_price})", flush=True)
                                sent_ok = send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    passed=passes,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    score_max=score_max,
                                    gatekeepers_total=gatekeepers_total,
                                    total_weight=total_weight,
                                    reversal_side=res1h.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_15m=early_breakout_1h,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=None,
                                    achieved_rr=achieved_rr_value,
                                    tier=signal_tier
                                )
                                print(f"[DEBUG] 1h: send_telegram_alert returned {sent_ok} for {symbol_val}", flush=True)
                                if sent_ok:
                                    last_sent[key1h] = now
                                    # signals_sent_this_cycle.add(signal_key)  # Dedup now uses SENT_SIGNALS.jsonl
                                    logger.signal_sent(symbol_val, "1h", signal_uuid[:12])
                                    
                                    # Log to SENT_SIGNALS for PEC tracking
                                    try:
                                        _sent_signal_tracker.log_sent_signal(
                                            signal_uuid=signal_uuid,
                                            symbol=symbol_val,
                                            timeframe="1h",
                                            signal_type=signal_type,
                                            entry_price=entry_price,
                                            tp_target=tp,
                                            sl_target=sl,
                                            tp_pct=tp_pct_val,
                                            sl_pct=sl_pct_val,
                                            achieved_rr=achieved_rr_value,
                                            score=score,
                                            max_score=score_max,
                                            confidence=confidence,
                                            route=Route,
                                            regime=regime,
                                            telegram_msg_id=signal_uuid[:12],
                                            fired_time_utc=fired_time_utc.isoformat(),
                                            passed_filters=passed_filters,
                                            failed_filters=failed_filters,
                                            passed_filter_count=passed_filter_count,
                                            failed_filter_count=failed_filter_count
                                        )
                                        # Also write to SIGNALS_MASTER.jsonl (single source of truth)
                                        if _master_writer_ready:
                                            _signals_master_writer.write_signal({
                                                'signal_uuid': signal_uuid,
                                                'symbol': symbol_val,
                                                'timeframe': '1h',
                                                'signal_type': signal_type,
                                                'entry_price': entry_price,
                                                'tp_target': tp,
                                                'sl_target': sl,
                                                'tp_pct': tp_pct_val,
                                                'sl_pct': sl_pct_val,
                                                'achieved_rr': achieved_rr_value,
                                                'score': score,
                                                'max_score': score_max,
                                                'confidence': confidence,
                                                'route': Route,
                                                'regime': regime,
                                                'telegram_msg_id': signal_uuid[:12],
                                                'fired_time_utc': fired_time_utc.isoformat(),
                                                'sent_time_utc': datetime.utcnow().isoformat(),
                                                'passed_filters': passed_filters,
                                                'failed_filters': failed_filters,
                                                'passed_filter_count': passed_filter_count,
                                                'failed_filter_count': failed_filter_count,
                                                'status': 'OPEN',
                                                'signal_origin': 'NEW_LIVE',
                                                'weighted_score': confidence * score / 100 if score_max else 0,
                                                'tier': signal_tier,

                                                'symbol_group': symbol_group,

                                                'confidence_level': 'HIGH' if confidence >= 75 else 'MID' if confidence >= 50 else 'LOW'

                                                })
                                            
                                            # PHASE 1: Dual-Write Verification (Alert + Continue strategy)
                                            dual_write_ok = True
                                            if _dual_write_ready:
                                                try:
                                                    verify_result = verify_signal_dual_write(
                                                        signal_uuid=signal_uuid,
                                                        signal_data={'symbol': symbol_val, 'timeframe': '1h', 'entry_price': entry_price},
                                                        raise_on_failure=False  # Alert+Continue (don't halt)
                                                    )
                                                    if verify_result:
                                                        print(f"[DUAL-WRITE] ✅ Verified {signal_uuid[:12]} to both files | FIRE: {symbol_val} 1h", flush=True)
                                                    else:
                                                        dual_write_ok = False
                                                        _divergence_tracker.record_failure(signal_uuid)
                                                        alert_msg = f"Dual-write failed: {signal_uuid[:12]} {symbol_val} 1h | MASTER:✓ AUDIT:✗"
                                                        print(f"[DUAL-WRITE] ❌ {alert_msg}", flush=True)
                                                        send_ops_alert(alert_msg, "CRITICAL")
                                                except Exception as e:
                                                    print(f"[DUAL-WRITE] Warning: Verification error (continuing): {e}", flush=True)
                                                    dual_write_ok = False
                                            
                                            # NEW: Trigger executor on signal fire (hybrid model - event-driven)
                                            trigger_executor_on_signal(signal_uuid, symbol_val, '1h')
                                    except Exception as e:
                                        print(f"[ERROR] Failed to log PEC signal for {symbol_val}: {e}", flush=True)
                                else:
                                    logger.signal_rejected(symbol_val, "1h", "Telegram send failed")
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (1h): {e}", flush=True)
                                traceback.print_exc()
                else:
                    pass  # Silent - not a signal
            except Exception as e:
                print(f"[ERROR] Exception in processing 1h for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 2h TF block (NEW - mirror of 1h with identical safe flow) ---
            try:
                key2h = f"{symbol}_2h"
                if df2h is None or df2h.empty:
                    res2h = None
                    print(f"[DEBUG-2h] {symbol}: df2h is None/empty - SKIPPING 2h processing", flush=True)
                else:
                    sf2h = SmartFilter(symbol, df2h, tf="2h")
                    regime2h = sf2h._market_regime()
                    res2h = sf2h.analyze()
                    print(f"[DEBUG-2h] {symbol}: analyze() returned - filters_ok={res2h.get('filters_ok') if isinstance(res2h, dict) else 'NOT_DICT'}", flush=True)

                if isinstance(res2h, dict):
                    score_2h = res2h.get("score")
                    filters_ok_2h = res2h.get("filters_ok")
                    bias_2h = res2h.get("bias", "UNKNOWN")
                    passes_2h = res2h.get("passes", 0)
                    max_gatekeepers_2h = res2h.get("gatekeepers_total", 0)
                    score_max_2h = res2h.get("score_max", 0)
                    
                    # ===== 4TH FACTOR: DYNAMIC MIN_SCORE FOR 2h =====
                    # 2h has higher volatility (3.2x) and fewer filters passing due to structural constraints
                    # Instead of requiring MIN_SCORE=10 (designed for 15min), use relaxed threshold for 2h
                    # Target: Allow 2h signals at score=8-9/19 (67% filter confidence vs 83% for 15min)
                    MIN_SCORE_2h = 8  # Relaxed from global 10 to account for TF volatility differences
                    
                    print(f"[DEBUG-2h] {symbol}: score={score_2h}/{score_max_2h}, passes={passes_2h}/{max_gatekeepers_2h}, bias={bias_2h}, filters_ok={filters_ok_2h}", flush=True)
                    if filters_ok_2h is not True:
                        print(f"[DEBUG-2h] {symbol}: REJECTED - filters_ok={filters_ok_2h} (score={score_2h}, MIN_SCORE={MIN_SCORE} from smart_filter.py)", flush=True)
                    
                    # ===== APPLY DYNAMIC MIN_SCORE FOR 2h =====
                    # If global MIN_SCORE check failed but TF-specific check passes, override
                    if not filters_ok_2h and score_2h is not None and score_2h >= MIN_SCORE_2h:
                        filters_ok_2h = True  # Override: allow signal with relaxed 2h threshold
                        print(f"[MIN-SCORE-OVERRIDE] 2h {symbol}: score={score_2h} >= MIN_SCORE_2h={MIN_SCORE_2h} → filters_ok=True (overridden from global MIN_SCORE={MIN_SCORE})", flush=True)
                    try:
                        all_signals_path = "/Users/geniustarigan/.openclaw/workspace/ALL_SIGNALS.jsonl"
                        # Use TF-specific MIN_SCORE for 2h
                        min_score_to_use = MIN_SCORE_2h if score_2h is not None else MIN_SCORE
                        all_signal_entry = {
                            "symbol": symbol,
                            "timeframe": "2h",
                            "score": score_2h,
                            "min_score_required": min_score_to_use,
                            "min_score_global": MIN_SCORE,
                            "passed_filter": score_2h is not None and score_2h >= min_score_to_use,
                            "fired_time_utc": datetime.utcnow().isoformat(),
                            "filters_ok": filters_ok_2h
                        }
                        with open(all_signals_path, 'a') as f:
                            f.write(json.dumps(all_signal_entry) + '\n')
                    except Exception as e:
                        print(f"[WARN] Failed to log to ALL_SIGNALS.jsonl: {e}", flush=True)

                if isinstance(res2h, dict) and res2h.get("filters_ok") is True:
                    # SmartFilter already enforces MIN_SCORE in filters_ok calculation, so no need to check again
                    # Just proceed with signal processing
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK - 2h (DISABLED 2026-03-25) =====
                    signal_type = res2h.get("bias", "UNKNOWN")
                    if ENABLE_DIRECTION_AWARE_GATEKEEPER:
                        try:
                            gates_passed, gate_results = DirectionAwareGatekeeper.check_all_gates(
                                df2h,
                                direction=signal_type,
                                regime=regime2h,
                                debug=False
                            )
                            
                            if not gates_passed:
                                failed_gates = [k for k, v in gate_results.items() if not v]
                                print(f"[PHASE2-FIXED] 2h {symbol} {signal_type} REJECTED - failed: {failed_gates}", flush=True)
                                continue
                            else:
                                print(f"[PHASE2-FIXED] 2h {symbol} {signal_type} ✓ ALL GATES PASS ({regime2h})", flush=True)
                        except Exception as e:
                            print(f"[PHASE2-FIXED] Error checking gates for {symbol}: {e}", flush=True)
                            pass
                    else:
                        print(f"[GATEKEEPER-DISABLED] 2h {symbol}: DirectionAwareGatekeeper bypassed (code preserved)", flush=True)
                    # ===== END PHASE 2-FIXED GATES (DISABLED) =====
                    
                    last2h = last_sent.get(key2h, 0)
                    if now - last2h >= COOLDOWN["2h"]:
                        numbered_signal = f"{idx}.D"

                        log_orderbook_and_density(symbol)
                        try:
                            orderbook_result = get_order_wall_delta(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_order_wall_delta failed for {symbol}: {e}", flush=True)
                            orderbook_result = {"buy_wall": 0, "sell_wall": 0, "wall_delta": 0, "midprice": None}
                        try:
                            density_result = get_resting_order_density(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_resting_order_density failed for {symbol}: {e}", flush=True)
                            density_result = {"bid_density": 0.0, "ask_density": 0.0, "bid_levels": 0, "ask_levels": 0, "midprice": None}

                        bias = res2h.get("bias", "NEUTRAL")
                        sf2h.bias = bias

                        print("[SuperGK][MAIN] DISABLED (broken) - bypassing SuperGK for ALL signals", flush=True)
                        super_gk_ok = True
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok} (bypassed)", flush=True)

                        if not super_gk_ok:
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias} — NO SIGNAL SENT", flush=True)
                            continue

                        # Define symbol_group early (for tier matching + write_signal)
                        symbol_group = classify_symbol(res2h.get('symbol'))
                        
                        print(f"[LOG] Sending 2h alert for {res2h.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res2h.get("symbol"),
                                bias,
                                tf=res2h.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res2h.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res2h.get("price", 0.0)

                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        print(f"[DEBUG] 2h entry_price_final={entry_price:.8f} analyzer_price={res2h.get('price')}", flush=True)

                        score = res2h.get("score", 0)
                        score_max = res2h.get("score_max", 0)
                        passes = res2h.get("passes", 0)
                        gatekeepers_total = res2h.get("gatekeepers_total", 0)
                        passed_weight = res2h.get("passed_weight", 0.0)
                        total_weight = res2h.get("total_weight", 0.0)
                        Route = res2h.get("Route", None)
                        signal_type = bias
                        tf_val = res2h.get("tf", "2h")
                        symbol_val = res2h.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df2h.index.get_loc(df2h.index[-1])

                        logger.signal_generated(symbol_val, tf_val, signal_type, score, entry_price)

                        regime = sf2h._market_regime() if hasattr(sf2h, "_market_regime") else None
                        
                        if Route == "REVERSAL":
                            reversal_type = res2h.get("reversal_type", None)
                            
                            quality_check = check_reversal_quality(
                                symbol=symbol_val,
                                df=df2h,
                                reversal_type=reversal_type,
                                regime=regime,
                                direction=signal_type
                            )
                            
                            gate_status = ", ".join([f"{k.split('_', 1)[1]}={'✓' if v else '✗'}" for k, v in quality_check["gate_results"].items()])
                            print(f"[PHASE3B-RQ] 2h {symbol_val} {signal_type}: {gate_status} → Strength={quality_check['reversal_strength_score']:.0f}%", flush=True)
                            
                            if not quality_check["allowed"]:
                                Route = quality_check["recommendation"]
                                print(f"[PHASE3B-FALLBACK] 2h {symbol_val}: REVERSAL rejected ({quality_check['reason']}) → Routed to {Route}", flush=True)
                            else:
                                print(f"[PHASE3B-APPROVED] 2h {symbol_val}: REVERSAL approved ({quality_check['reason']})", flush=True)
                        
                        if Route in ["NONE", "AMBIGUOUS"]:
                            print(f"[VETO-POST-3B] 2h {symbol_val} {signal_type}: Route={Route} (changed by Phase 3B). REJECTING signal.", flush=True)
                            continue
                        
                        route_score = calculate_route_score(Route, signal_type, regime, reversal_strength_score=None)
                        route_rec = get_route_recommendation(Route, signal_type, regime)
                        print(f"[PHASE3B-SCORE] 2h {symbol_val}: {route_rec} (score: {route_score})", flush=True)

                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df2h, entry_price, signal_type, regime=regime)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        try:
                            log_fired_signal(
                                symbol=symbol_val,
                                tf=tf_val,
                                signal_type=signal_type,
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=score,
                                max_score=score_max,
                                passed=passes,
                                max_passed=gatekeepers_total,
                                weights=passed_weight,
                                max_weights=total_weight,
                                confidence_rate=confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            print(f"[WARN] log_fired_signal raised: {e}", flush=True)
                            traceback.print_exc()

                        # ===== RR VALIDATION & ENFORCEMENT (PROJECT-11) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        # Step 1: Validate TP/SL relationship for direction
                        is_valid_relationship, rr_error = validate_tp_sl_relationship(entry_price, tp, sl, signal_type)
                        if not is_valid_relationship:
                            print(f"[RR_ENFORCE] 2h signal REJECTED: {symbol_val} - {rr_error}", flush=True)
                            continue
                        
                        # Step 2: Enforce RR bounds (may adjust TP/SL)
                        adjusted_tp, adjusted_sl, enforced_rr, rr_action = enforce_rr_bounds(entry_price, tp, sl, signal_type, achieved_rr_value)
                        
                        if rr_action.startswith("INVALID"):
                            print(f"[RR_ENFORCE] 2h signal REJECTED: {symbol_val} - {rr_action}", flush=True)
                            continue
                        
                        # Step 3: Log enforcement action
                        if rr_action == "VALID":
                            print(f"[RR_ENFORCE] 2h {symbol_val}: VALID RR {enforced_rr:.3f} (no adjustment)", flush=True)
                        else:
                            print(f"[RR_ENFORCE] 2h {symbol_val}: {rr_action} | Calculated RR: {achieved_rr_value} → Enforced RR: {enforced_rr} | TP: {tp} → {adjusted_tp} | SL: {sl} → {adjusted_sl}", flush=True)
                            tp = adjusted_tp
                            sl = adjusted_sl
                        
                        # RR valid and enforced: Store signal + Fire
                        print(f"[RR_ENFORCE] 2h signal ACCEPTED: {symbol_val} - RR {enforced_rr:.3f} ({rr_action})", flush=True)
                        
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
                        if signal_type == 'LONG':
                            results = res2h.get('results_long', {})
                        else:
                            results = res2h.get('results_short', {})
                        
                        passed_filters = results.get('passed_filters', [])
                        failed_filters = results.get('failed_filters', [])
                        passed_filter_count = results.get('passed_filter_count', 0)
                        failed_filter_count = results.get('failed_filter_count', 0)
                        
                        telegram_msg_id = str(uuid_lib.uuid4())[:12]
                        
                        signal_uuid = create_and_store_signal(
                            symbol=symbol_val,
                            timeframe=tf_val,
                            signal_type=signal_type,
                            fired_time_utc=fired_time_utc,
                            entry_price=entry_price,
                            tp_target=tp,
                            sl_target=sl,
                            tp_pct=tp_pct_val,
                            sl_pct=sl_pct_val,
                            achieved_rr=achieved_rr_value,
                            fib_ratio=None,
                            atr_value=tp_sl.get('atr_value') if isinstance(tp_sl, dict) else None,
                            score=score,
                            max_score=score_max,
                            confidence=confidence,
                            route=Route,
                            regime=regime,
                            passed_gatekeepers=passes,
                            max_gatekeepers=gatekeepers_total,
                            passed_filters=passed_filters,
                            failed_filters=failed_filters,
                            passed_filter_count=passed_filter_count,
                            failed_filter_count=failed_filter_count,
                            telegram_msg_id=telegram_msg_id
                        )
                        
                        if not signal_uuid:
                            print(f"[WARN] Signal {symbol_val} (2h) REJECTED - duplicate within 120s. NOT sending Telegram alert.", flush=True)
                            pass
                        
                        # Send trade alert to Telegram
                        if signal_uuid and os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                symbol_group = classify_symbol(symbol_val)
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime, symbol_group)
                                print(f"[TIER] 2h {symbol_val} ({symbol_group}): {signal_tier}", flush=True)
                                
                                print(f"[DEBUG] 2h: Calling send_telegram_alert for {symbol_val} (tf={tf_val}, Entry={entry_price})", flush=True)
                                sent_ok = send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    passed=passes,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    score_max=score_max,
                                    gatekeepers_total=gatekeepers_total,
                                    total_weight=total_weight,
                                    reversal_side=res2h.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_15m=early_breakout_2h,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=None,
                                    achieved_rr=achieved_rr_value,
                                    tier=signal_tier
                                )
                                print(f"[DEBUG] 2h: send_telegram_alert returned {sent_ok} for {symbol_val}", flush=True)
                                if sent_ok:
                                    last_sent[key2h] = now
                                    logger.signal_sent(symbol_val, "2h", signal_uuid[:12])
                                    
                                    try:
                                        _sent_signal_tracker.log_sent_signal(
                                            signal_uuid=signal_uuid,
                                            symbol=symbol_val,
                                            timeframe="2h",
                                            signal_type=signal_type,
                                            entry_price=entry_price,
                                            tp_target=tp,
                                            sl_target=sl,
                                            tp_pct=tp_pct_val,
                                            sl_pct=sl_pct_val,
                                            achieved_rr=achieved_rr_value,
                                            score=score,
                                            max_score=score_max,
                                            confidence=confidence,
                                            route=Route,
                                            regime=regime,
                                            telegram_msg_id=signal_uuid[:12],
                                            fired_time_utc=fired_time_utc.isoformat(),
                                            passed_filters=passed_filters,
                                            failed_filters=failed_filters,
                                            passed_filter_count=passed_filter_count,
                                            failed_filter_count=failed_filter_count
                                        )
                                        if _master_writer_ready:
                                            _signals_master_writer.write_signal({
                                                'signal_uuid': signal_uuid,
                                                'symbol': symbol_val,
                                                'timeframe': '2h',
                                                'signal_type': signal_type,
                                                'entry_price': entry_price,
                                                'tp_target': tp,
                                                'sl_target': sl,
                                                'tp_pct': tp_pct_val,
                                                'sl_pct': sl_pct_val,
                                                'achieved_rr': achieved_rr_value,
                                                'score': score,
                                                'max_score': score_max,
                                                'confidence': confidence,
                                                'route': Route,
                                                'regime': regime,
                                                'telegram_msg_id': signal_uuid[:12],
                                                'fired_time_utc': fired_time_utc.isoformat(),
                                                'sent_time_utc': datetime.utcnow().isoformat(),
                                                'passed_filters': passed_filters,
                                                'failed_filters': failed_filters,
                                                'passed_filter_count': passed_filter_count,
                                                'failed_filter_count': failed_filter_count,
                                                'status': 'OPEN',
                                                'signal_origin': 'NEW_LIVE',
                                                'weighted_score': confidence * score / 100 if score_max else 0,
                                                'tier': signal_tier,

                                                'symbol_group': symbol_group,

                                                'confidence_level': 'HIGH' if confidence >= 75 else 'MID' if confidence >= 50 else 'LOW'

                                                })
                                            
                                            if _dual_write_ready:
                                                try:
                                                    verify_result = verify_signal_dual_write(
                                                        signal_uuid=signal_uuid,
                                                        signal_data={'symbol': symbol_val, 'timeframe': '2h', 'entry_price': entry_price},
                                                        raise_on_failure=False
                                                    )
                                                    if verify_result:
                                                        print(f"[DUAL-WRITE] ✅ Verified {signal_uuid[:12]} to both files | FIRE: {symbol_val} 2h", flush=True)
                                                    else:
                                                        _divergence_tracker.record_failure(signal_uuid)
                                                        alert_msg = f"Dual-write failed: {signal_uuid[:12]} {symbol_val} 2h | MASTER:✓ AUDIT:✗"
                                                        print(f"[DUAL-WRITE] ❌ {alert_msg}", flush=True)
                                                        send_ops_alert(alert_msg, "CRITICAL")
                                                except Exception as e:
                                                    print(f"[DUAL-WRITE] Warning: Verification error (continuing): {e}", flush=True)
                                            
                                            trigger_executor_on_signal(signal_uuid, symbol_val, '2h')
                                    except Exception as e:
                                        print(f"[ERROR] Failed to log PEC signal for {symbol_val}: {e}", flush=True)
                                else:
                                    logger.signal_rejected(symbol_val, "2h", "Telegram send failed")
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (2h): {e}", flush=True)
                                traceback.print_exc()
                else:
                    pass
            except Exception as e:
                print(f"[ERROR] Exception in processing 2h for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 4h TF block (PHASE 2 ADD - mirror of 15min/30min/1h with identical safe flow) ---
            # --- 4h TF block (SIMPLIFIED - mirror of 1h exactly, no extra gates) ---
            try:
                key4h = f"{symbol}_4h"
                if df4h is None or df4h.empty:
                    res4h = None
                else:
                    sf4h = SmartFilter(symbol, df4h, tf="4h")
                    res4h = sf4h.analyze()

                # ===== 4TH FACTOR: DYNAMIC MIN_SCORE FOR 4h =====
                # 4h has highest volatility (4.4x) and fewest filters passing
                # Instead of requiring MIN_SCORE=10, use MIN_SCORE_4h=8 for 4h
                # This allows 4h signals at score=8-9/19 (same as 2h)
                if isinstance(res4h, dict):
                    score_4h = res4h.get("score")
                    filters_ok_4h = res4h.get("filters_ok")
                    MIN_SCORE_4h = 8  # Relaxed from global 10 for high-volatility TF
                    
                    # If global MIN_SCORE check failed but TF-specific check passes, override
                    if not filters_ok_4h and score_4h is not None and score_4h >= MIN_SCORE_4h:
                        filters_ok_4h = True  # Override: allow signal with relaxed 4h threshold
                        res4h["filters_ok"] = True
                        print(f"[MIN-SCORE-OVERRIDE] 4h {symbol}: score={score_4h} >= MIN_SCORE_4h={MIN_SCORE_4h} → filters_ok=True (overridden from global MIN_SCORE={MIN_SCORE})", flush=True)

                if isinstance(res4h, dict) and res4h.get("filters_ok") is True:
                    bias = res4h.get("bias", "UNKNOWN")
                    score = res4h.get("score", 0)
                    
                    if score is None or score < MIN_SCORE:
                        pass
                    else:
                        last4h = last_sent.get(key4h, 0)
                        
                        if now - last4h >= COOLDOWN["4h"]:
                            numbered_signal = f"{idx}.E"
                            
                            try:
                                entry_price_raw = get_live_entry_price(
                                    res4h.get("symbol"),
                                    bias,
                                    tf=res4h.get("tf"),
                                    slippage=DEFAULT_SLIPPAGE
                                ) or res4h.get("price", 0.0)
                            except Exception as e:
                                print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                                entry_price_raw = res4h.get("price", 0.0)

                            try:
                                entry_price = float(entry_price_raw)
                            except Exception:
                                try:
                                    entry_price = float(str(entry_price_raw))
                                    print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                                except Exception:
                                    print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                    entry_price = 0.0

                            print(f"[DEBUG] 4h entry_price_final={entry_price:.8f} analyzer_price={res4h.get('price')}", flush=True)

                            score_max = res4h.get("score_max", 0)
                            passes = res4h.get("passes", 0)
                            gatekeepers_total = res4h.get("gatekeepers_total", 0)
                            passed_weight = res4h.get("passed_weight", 0.0)
                            total_weight = res4h.get("total_weight", 0.0)
                            Route = res4h.get("Route", None)
                            signal_type = bias
                            tf_val = res4h.get("tf", "4h")
                            symbol_val = res4h.get("symbol", symbol)
                            try:
                                confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                            except Exception:
                                confidence = 0.0

                            logger.signal_generated(symbol_val, tf_val, signal_type, score, entry_price)

                            # Get market regime from SmartFilter (extracted early for TP/SL scaling)
                            regime = sf4h._market_regime() if hasattr(sf4h, "_market_regime") else None
                            
                            tp_sl = calculate_tp_sl(df4h, entry_price, signal_type, regime=regime)
                            tp = tp_sl.get("tp") if isinstance(tp_sl, dict) else 0
                            sl = tp_sl.get("sl") if isinstance(tp_sl, dict) else 0
                            tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                            sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                            achieved_rr_value = tp_sl.get("achieved_rr") if isinstance(tp_sl, dict) else 0

                            try:
                                entry_idx = df4h.index.get_loc(df4h.index[-1])
                            except Exception:
                                entry_idx = len(df4h) - 1

                            if signal_type == 'LONG':
                                results = res4h.get('results_long', {})
                            else:
                                results = res4h.get('results_short', {})
                            
                            passed_filters = results.get('passed_filters', [])
                            failed_filters = results.get('failed_filters', [])
                            passed_filter_count = results.get('passed_filter_count', 0)
                            failed_filter_count = results.get('failed_filter_count', 0)

                            numbered_signal = f"{idx}.E"
                            fired_time_utc = datetime.utcnow()
                            telegram_msg_id = str(uuid_lib.uuid4())[:12]
                            
                            # Define symbol_group early (for tier matching + write_signal)
                            symbol_group = classify_symbol(symbol_val)
                            
                            signal_uuid = create_and_store_signal(
                                symbol=symbol_val,
                                timeframe=tf_val,
                                signal_type=signal_type,
                                fired_time_utc=fired_time_utc,
                                entry_price=entry_price,
                                tp_target=tp,
                                sl_target=sl,
                                tp_pct=tp_pct_val,
                                sl_pct=sl_pct_val,
                                achieved_rr=achieved_rr_value,
                                fib_ratio=None,
                                atr_value=tp_sl.get('atr_value') if isinstance(tp_sl, dict) else None,
                                score=score,
                                max_score=score_max,
                                confidence=confidence,
                                route=Route,
                                regime=regime,
                                passed_gatekeepers=passes,
                                max_gatekeepers=gatekeepers_total,
                                passed_filters=passed_filters,
                                failed_filters=failed_filters,
                                passed_filter_count=passed_filter_count,
                                failed_filter_count=failed_filter_count,
                                telegram_msg_id=telegram_msg_id
                            )
                            
                            if not signal_uuid:
                                print(f"[WARN] Signal {symbol_val} (4h) REJECTED - duplicate within cooldown. NOT sending Telegram alert.", flush=True)
                                pass

                            # Send trade alert to Telegram
                            if signal_uuid and os.getenv("DRY_RUN", "false").lower() != "true":
                                try:
                                    # symbol_group already defined above
                                    signal_tier = get_signal_tier(tf_val, signal_type, Route, regime, symbol_group)
                                    print(f"[TIER] 4h {symbol_val} ({symbol_group}): {signal_tier}", flush=True)
                                    
                                    print(f"[DEBUG] 4h: Calling send_telegram_alert for {symbol_val} (tf={tf_val}, Entry={entry_price})", flush=True)
                                    sent_ok = send_telegram_alert(
                                        numbered_signal=numbered_signal,
                                        symbol=symbol_val,
                                        signal_type=signal_type,
                                        Route=Route,
                                        price=entry_price,
                                        tf=tf_val,
                                        score=score,
                                        passed=passes,
                                        confidence=confidence,
                                        weighted=passed_weight,
                                        score_max=score_max,
                                        gatekeepers_total=gatekeepers_total,
                                        total_weight=total_weight,
                                        reversal_side=res4h.get("reversal_side"),
                                        regime=regime,
                                        early_breakout_15m=early_breakout_4h,
                                        tp=tp,
                                        sl=sl,
                                        tp_sl=tp_sl,
                                        chosen_ratio=None,
                                        achieved_rr=achieved_rr_value,
                                        tier=signal_tier
                                    )
                                    print(f"[DEBUG] 4h: send_telegram_alert returned {sent_ok} for {symbol_val}", flush=True)
                                    if sent_ok:
                                        last_sent[key4h] = now
                                        logger.signal_sent(symbol_val, "4h", signal_uuid[:12])
                                        
                                        try:
                                            _sent_signal_tracker.log_sent_signal(
                                                signal_uuid=signal_uuid,
                                                symbol=symbol_val,
                                                timeframe="4h",
                                                signal_type=signal_type,
                                                entry_price=entry_price,
                                                tp_target=tp,
                                                sl_target=sl,
                                                tp_pct=tp_pct_val,
                                                sl_pct=sl_pct_val,
                                                achieved_rr=achieved_rr_value,
                                                score=score,
                                                max_score=score_max,
                                                confidence=confidence,
                                                route=Route,
                                                regime=regime,
                                                telegram_msg_id=signal_uuid[:12],
                                                fired_time_utc=fired_time_utc.isoformat(),
                                                passed_filters=passed_filters,
                                                failed_filters=failed_filters,
                                                passed_filter_count=passed_filter_count,
                                                failed_filter_count=failed_filter_count
                                            )
                                            
                                            if _master_writer_ready:
                                                _signals_master_writer.write_signal({
                                                    'signal_uuid': signal_uuid,
                                                    'symbol': signal_val,
                                                    'timeframe': '4h',
                                                    'signal_type': signal_type,
                                                    'entry_price': entry_price,
                                                    'tp_target': tp,
                                                    'sl_target': sl,
                                                    'tp_pct': tp_pct_val,
                                                    'sl_pct': sl_pct_val,
                                                    'achieved_rr': achieved_rr_value,
                                                    'score': score,
                                                    'max_score': score_max,
                                                    'confidence': confidence,
                                                    'route': Route,
                                                    'regime': regime,
                                                    'telegram_msg_id': signal_uuid[:12],
                                                    'fired_time_utc': fired_time_utc.isoformat(),
                                                    'sent_time_utc': datetime.utcnow().isoformat(),
                                                    'passed_filters': passed_filters,
                                                    'failed_filters': failed_filters,
                                                    'passed_filter_count': passed_filter_count,
                                                    'failed_filter_count': failed_filter_count,
                                                    'status': 'OPEN',
                                                    'signal_origin': 'NEW_LIVE',
                                                    'weighted_score': confidence * score / 100 if score_max else 0,
                                                    'tier': signal_tier,

                                                    'symbol_group': symbol_group,

                                                    'confidence_level': 'HIGH' if confidence >= 75 else 'MID' if confidence >= 50 else 'LOW'

                                                    })
                                                
                                                if _dual_write_ready:
                                                    try:
                                                        verify_result = verify_signal_dual_write(
                                                            signal_uuid=signal_uuid,
                                                            signal_data={'symbol': symbol_val, 'timeframe': '4h', 'entry_price': entry_price},
                                                            raise_on_failure=False
                                                        )
                                                        if verify_result:
                                                            print(f"[DUAL-WRITE] ✅ Verified {signal_uuid[:12]} 4h | FIRE: {symbol_val}", flush=True)
                                                        else:
                                                            _divergence_tracker.record_failure(signal_uuid)
                                                            alert_msg = f"Dual-write failed: {signal_uuid[:12]} {symbol_val} 4h"
                                                            print(f"[DUAL-WRITE] ❌ {alert_msg}", flush=True)
                                                            send_ops_alert(alert_msg, "CRITICAL")
                                                    except Exception as e:
                                                        print(f"[DUAL-WRITE] Warning: Verification error (continuing): {e}", flush=True)
                                                
                                                trigger_executor_on_signal(signal_uuid, symbol_val, '4h')
                                        except Exception as e:
                                            print(f"[ERROR] Failed to log PEC signal for {symbol_val}: {e}", flush=True)
                                    else:
                                        logger.signal_rejected(symbol_val, "4h", "Telegram send failed")
                                except Exception as e:
                                    print(f"[ERROR] Exception during Telegram send for {symbol_val} (4h): {e}", flush=True)
                                    traceback.print_exc()
            except Exception as e:
                print(f"[ERROR] Exception in processing 4h for {symbol}: {e}", flush=True)
                traceback.print_exc()
            except Exception as e:
                print(f"[ERROR] Exception in processing 4h for {symbol}: {e}", flush=True)
                traceback.print_exc()

        except Exception as e:
            print(f"[FATAL] Exception processing {symbol}: {e}", flush=True)
            traceback.print_exc()

    # End of per-symbol loop
    
    logger.cycle_summary()
    
    # Print SENT_SIGNALS summary for PEC tracking
    if _sent_tracker_ready:
        stats = _sent_signal_tracker.get_summary_stats()
        if stats:
            print("\n" + "="*70, flush=True)
            print("PEC SIGNAL TRACKING SUMMARY (for Position Entry Confidence):", flush=True)
            print(f"  Total Sent:      {stats.get('total_sent', 0)}", flush=True)
            print(f"  Open (Running):  {stats.get('open', 0)}", flush=True)
            print(f"  TP Hit:          {stats.get('tp_hit', 0)}", flush=True)
            print(f"  SL Hit:          {stats.get('sl_hit', 0)}", flush=True)
            print(f"  Timeout:         {stats.get('timeout', 0)}", flush=True)
            print(f"  Closed Trades:   {stats.get('closed', 0)}", flush=True)
            print(f"  Total P&L:       ${stats.get('total_pnl_usd', 0)}", flush=True)
            print(f"  Win Rate:        {stats.get('win_rate_pct', 0)}% ({stats.get('winning_trades', 0)}W/{stats.get('losing_trades', 0)}L)", flush=True)
            print("="*70 + "\n", flush=True)
    
    return valid_debugs

def run():
    """
    Continuous loop that calls run_cycle() then sleeps.
    Enforces CYCLE_SLEEP interval and prevents overlapping cycles.
    """
    print("[INFO] Starting Smart Filter engine (LIVE MODE)...\n", flush=True)
    
    # Initialize last_sent from recent SENT_SIGNALS.jsonl to avoid resending after restart
    initialize_last_sent()
    
    last_cycle_end = 0
    while True:
        try:
            cycle_start = time.time()
            print(f"[CYCLE] START at {datetime.utcnow().isoformat()} (wall: {datetime.now().isoformat()})", flush=True)
            
            valid_debugs = run_cycle()

            # --- Send 2 random debug files from fired signals (PROVEN METHOD) ---
            try:
                if valid_debugs:
                    print(f"[FIRED] About to send {min(len(valid_debugs), 2)} debug files to Telegram.", flush=True)
                    num = min(len(valid_debugs), 2)
                    for debug_info in random.sample(valid_debugs, num):
                        try:
                            print("[FIRED] LONG filter weights:", debug_info["filter_weights_long"], flush=True)
                            print("[FIRED] SHORT filter weights:", debug_info["filter_weights_short"], flush=True)
                            export_signal_debug_txt(
                                symbol=debug_info["symbol"],
                                tf=debug_info["tf"],
                                bias=debug_info["bias"],
                                filter_weights_long=debug_info["filter_weights_long"],
                                filter_weights_short=debug_info["filter_weights_short"],
                                gatekeepers=debug_info["gatekeepers"],
                                results_long=debug_info.get("results_long", {}),
                                results_short=debug_info.get("results_short", {}),
                                orderbook_result=debug_info.get("orderbook_result"),
                                density_result=debug_info.get("density_result")
                            )
                            send_telegram_file(
                                "signal_debug_temp.txt",
                                caption=debug_info["caption"]
                            )
                        except Exception as e:
                            print(f"[ERROR] Exception in Telegram debug send: {e}", flush=True)
                else:
                    print("[FIRED] valid_debugs is empty — no debug files to send to Telegram.", flush=True)
            except Exception as e:
                print(f"[FATAL] Exception in debug sending block: {e}", flush=True)

            if valid_debugs:
                print(f"[FIRED] Processed {len(valid_debugs)} valid signals this cycle", flush=True)
            else:
                print("[FIRED] No valid signals processed this cycle", flush=True)

            cycle_end = time.time()
            cycle_duration = cycle_end - cycle_start
            print(f"[CYCLE] END at {datetime.utcnow().isoformat()} (duration: {cycle_duration:.2f}s)", flush=True)
            
            # ENFORCE CYCLE_SLEEP interval
            time_to_sleep = CYCLE_SLEEP - cycle_duration
            if time_to_sleep > 0:
                print(f"[INFO] ✅ Cycle complete ({cycle_duration:.2f}s). Sleeping {time_to_sleep:.2f}s (total interval: {CYCLE_SLEEP}s)...\n", flush=True)
                time.sleep(time_to_sleep)
            else:
                print(f"[WARN] Cycle took {cycle_duration:.2f}s (longer than CYCLE_SLEEP={CYCLE_SLEEP}s). No sleep. Next cycle starts immediately.\n", flush=True)
            
            last_cycle_end = time.time()

        except Exception as e:
            print(f"[FATAL] Exception in main loop: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("[INFO] Sleeping 10 seconds before retrying main loop...\n", flush=True)
            time.sleep(10)

if __name__ == "__main__":
    print(">>> ENTERED main.py - LIVE SIGNAL MODE", flush=True)
    print(">>> For PEC backtesting, run: python3 run_pec_backtest.py", flush=True)
    print(">>> For historical backtest, run: python3 backtest_real.py", flush=True)
    run()
