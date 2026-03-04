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
from kucoin_data import get_live_entry_price, DEFAULT_SLIPPAGE
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert, send_telegram_file
from tier_lookup import get_signal_tier
from kucoin_orderbook import get_order_wall_delta
from pec_engine import run_pec_check, export_pec_log
from tp_sl_retracement import calculate_tp_sl
try:
    from test_filters import run_all_filter_tests
except ImportError:
    run_all_filter_tests = None  # test_filters optional (development only)
from signal_store import get_signal_store
from pec_config import MIN_ACCEPTED_RR, SIGNALS_JSONL_PATH
import math

# NEW: Enhanced tracking & safe OHLCV (PROJECT-3 SmartFilter fixes)
from signal_tracking_enhanced import get_signal_tracker
from ohlcv_fetch_safe import safe_fetch_ohlcv_by_tf, check_tf_data_available, should_skip_symbol
from signal_logger import SignalLogger
from signal_sent_tracker import get_signal_sent_tracker  # PEC: Track SENT signals only
from pathlib import Path

# ===== PHASE 2 IMPORTS (Stage 2 - Direction-Aware Gatekeepers + Regime Adjustments) =====
from direction_aware_gatekeeper import DirectionAwareGatekeeper
from direction_aware_regime_adjustments import calculate_direction_aware_threshold
# ===== END PHASE 2 IMPORTS =====

# ===== PHASE 3B IMPORTS (Stage 3 - Reversal Quality Gate + Route Optimization) =====
from reversal_quality_gate import check_reversal_quality
from direction_aware_route_optimizer import calculate_route_score, get_route_recommendation
# ===== END PHASE 3B IMPORTS =====

# ===== PHASE 4A FUNCTION: Multi-TF Alignment (Scenario 4: 30min + 1h) =====
def check_multitf_alignment_30_1h(symbol, ohlcv_data):
    """
    PHASE 4A Scenario 4: Check if 30min trend aligns with 1h trend
    
    This acts as a consensus filter: only allow signals if higher timeframes agree
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        ohlcv_data: Dict from safe_fetch_ohlcv_by_tf with keys "15min", "30min", "1h"
    
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
    _sent_signal_tracker = get_signal_sent_tracker("SENT_SIGNALS.jsonl")
    _sent_tracker_ready = True
    print(f"[INIT] Sent signal tracker ready: {os.path.abspath('SENT_SIGNALS.jsonl')}", flush=True)
except Exception as e:
    _sent_tracker_ready = False
    print(f"[ERROR] Sent signal tracker init failed: {e}. PEC will not have execution data.", flush=True)

# === INITIALIZE SIGNAL TRACKING (EARLY & ROBUST) - NEW PROJECT-3 FIX ===
try:
    _signal_tracker = get_signal_tracker(SIGNALS_JSONL_PATH)
    _signal_tracker_ready = True
    print(f"[INIT] Signal tracker ready: {os.path.abspath(Path(SIGNALS_JSONL_PATH).parent / 'signal_status.jsonl')}", flush=True)
except Exception as e:
    _signal_tracker_ready = False
    print(f"[ERROR] Signal tracker init failed: {e}. Tracking disabled.", flush=True)

# --- Configuration ---
# Full liquid pairs from kucoin_orderbook.py - 90+ symbols
TOKENS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "AVAX-USDT", 
    "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT", "SKATE-USDT", "LA-USDT", 
    "SPK-USDT", "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT", 
    "X-USDT", "RAY-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", 
    "FUN-USDT", "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT", 
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT", "DUCK-USDT", 
    "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT", "VOXEL-USDT", "ALU-USDT", 
    "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", "DOGE-USDT", "ALGO-USDT", "DOT-USDT", 
    "NEWT-USDT", "SAHARA-USDT", "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", 
    "ENA-USDT", "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT", 
    "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT", "GALA-USDT", 
    "PUMP-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT", "KAITO-USDT", "ARKM-USDT", 
    "ATH-USDT", "NMR-USDT", "ARB-USDT", "WLFI-USDT", "BIO-USDT", "ASTER-USDT", 
    "XPL-USDT", "AVNT-USDT", "ORDER-USDT", "XAUT-USDT"
]

COOLDOWN = {"15min": 120, "30min": 240, "1h": 600}
last_sent = {}

# === LOGGING CONTROL ===
# Disable verbose analysis logs to reduce Railway rate limit issues (67 messages dropped)
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "0") == "1"

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
    
    try:
        if not os.path.exists("SENT_SIGNALS.jsonl"):
            return
        
        # Read last 100 lines of SENT_SIGNALS.jsonl to find recent signals
        with open("SENT_SIGNALS.jsonl", "r") as f:
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
    "1h": 200      # 200 × 60min = 12000min = 8d (standard, long-term)
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
                           passed_gatekeepers, max_gatekeepers):
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
    "1h": 4800       # 80 minutes in seconds
}

def is_duplicate_signal(symbol, timeframe, signal_type):
    """
    Check if this signal (symbol + timeframe + signal_type) was sent recently
    within the dedup window for that timeframe.
    
    Returns: True if duplicate (should skip), False if new signal (should send)
    """
    try:
        # Check if SENT_SIGNALS.jsonl exists
        sent_signals_path = "SENT_SIGNALS.jsonl"
        if not os.path.exists(sent_signals_path):
            return False  # No file = no duplicates possible
        
        # Get dedup window for this timeframe
        window_seconds = DEDUP_WINDOWS.get(timeframe, 1200)
        current_time = datetime.now(pytz.UTC)
        
        # Read SENT_SIGNALS.jsonl line by line
        with open(sent_signals_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    signal_record = json.loads(line)
                    
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
                                print(f"[DEDUP] {symbol} {timeframe} {signal_type}: Found duplicate from {time_diff_seconds:.0f}s ago. SKIPPING.", flush=True)
                                return True
                
                except json.JSONDecodeError:
                    continue
        
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
            
            # Calculate early breakout only for TFs with data
            early_breakout_15m = early_breakout(df15, lookback=3) if df15 is not None else None
            early_breakout_30m = early_breakout(df30, lookback=3) if df30 is not None else None
            early_breakout_1h = early_breakout(df1h, lookback=3) if df1h is not None else None

            # --- 15min TF block ---
            try:
                key15 = f"{symbol}_15min"
                if df15 is None or df15.empty:
                    res15 = None
                else:
                    sf15 = SmartFilter(symbol, df15, tf="15min")
                    regime15 = sf15._market_regime()
                    res15 = sf15.analyze()

                if isinstance(res15, dict) and res15.get("filters_ok") is True:
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK =====
                    signal_type = res15.get("bias", "UNKNOWN")
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
                    # ===== END PHASE 2-FIXED GATES =====
                    
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

                        # ===== PHASE 2-FIXED: DIRECTION-AWARE THRESHOLD CHECK (15min) =====
                        try:
                            min_threshold, threshold_reason = calculate_direction_aware_threshold(
                                Route,
                                regime,
                                signal_type,
                                debug=False
                            )
                            
                            print(f"[PHASE2-FIXED-THRESHOLD] 15min {symbol}: score={score:.1f} | "
                                  f"threshold={min_threshold} | {threshold_reason}", flush=True)
                            
                            if score < min_threshold:
                                print(f"[PHASE2-FIXED-REJECT] 15min {symbol} {signal_type}: "
                                      f"{score:.1f} < {min_threshold} (below {Route} threshold)", flush=True)
                                continue  # Skip signal
                        
                        except Exception as e:
                            print(f"[SCORE-ADJUST] Error in 15min: {e}", flush=True)
                            # Continue with original score if error
                            pass
                        # ===== END REGIME-AWARE ADJUSTMENT =====

                        # ===== RR FILTERING (Enhanced PEC) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        if achieved_rr_value is None or achieved_rr_value < MIN_ACCEPTED_RR:
                            # Filter out: RR too low
                            print(f"[RR_FILTER] 15min signal REJECTED: {symbol_val} - RR {achieved_rr_value} < MIN {MIN_ACCEPTED_RR}", flush=True)
                            continue
                        
                        # RR acceptable: Store signal + Fire
                        print(f"[RR_FILTER] 15min signal ACCEPTED: {symbol_val} - RR {achieved_rr_value} >= MIN {MIN_ACCEPTED_RR}", flush=True)
                        
                        # Store signal to JSONL
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
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
                            max_gatekeepers=gatekeepers_total
                        )
                        
                        if not signal_uuid:
                            print(f"[WARN] Signal {symbol_val} (15min) REJECTED - duplicate within 120s. NOT sending Telegram alert.", flush=True)
                            continue

                        # IN-MEMORY DEDUP: Check if sent in THIS CYCLE (prevents rapid duplicates)
                        cycle_key = f"{symbol_val}|15min|{signal_type}"
                        if cycle_key in signals_sent_this_cycle:
                            print(f"[DEDUP-CYCLE] 15min {symbol_val} {signal_type}: Already sent in THIS CYCLE. SKIPPING.", flush=True)
                            continue
                        
                        # PHASE 4A: Check 30min+1h alignment (Scenario 4)
                        alignment_allowed, trend_30, trend_1h, alignment_reason = check_multitf_alignment_30_1h(symbol_val, ohlcv_data)
                        print(f"[PHASE4A-S4] 15min {symbol_val}: {alignment_reason}", flush=True)
                        
                        if not alignment_allowed:
                            print(f"[PHASE4A-S4-FILTERED] 15min {symbol_val} {signal_type}: Rejected by 30min+1h filter", flush=True)
                            continue
                        
                        # CRITICAL: Deduplication check (prevent duplicate within DEDUP_WINDOWS)
                        if is_duplicate_signal(symbol_val, "15min", signal_type):
                            continue
                        
                        # Mark as sent in this cycle
                        signals_sent_this_cycle.add(cycle_key)
                        
                        # Send trade alert to Telegram
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                # Get signal tier (dynamic - will be Tier-X initially, populates over time)
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime)
                                print(f"[TIER] 15min {symbol_val}: {signal_tier}", flush=True)
                                
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
                                            fired_time_utc=fired_time_utc.isoformat()
                                        )
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
                    sf30 = SmartFilter(symbol, df30, min_score=14, tf="30min")  # FIX: 30min scores 14/19 (1 point lower than 15min)
                    regime30 = sf30._market_regime()
                    res30 = sf30.analyze()

                if isinstance(res30, dict) and res30.get("filters_ok") is True:
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK - 30min =====
                    signal_type = res30.get("bias", "UNKNOWN")
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
                    # ===== END PHASE 2-FIXED GATES =====
                    
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

                        # ===== PHASE 2-FIXED: DIRECTION-AWARE THRESHOLD CHECK - 30min =====
                        try:
                            min_threshold, threshold_reason = calculate_direction_aware_threshold(
                                Route,
                                regime,
                                signal_type,
                                debug=False
                            )
                            
                            print(f"[PHASE2-FIXED-THRESHOLD] 30min {symbol}: score={score:.1f} | "
                                  f"threshold={min_threshold} | {threshold_reason}", flush=True)
                            
                            if score < min_threshold:
                                print(f"[PHASE2-FIXED-REJECT] 30min {symbol} {signal_type}: "
                                      f"{score:.1f} < {min_threshold} (below {Route} threshold)", flush=True)
                                continue  # Skip signal
                        
                        except Exception as e:
                            print(f"[PHASE2-FIXED] Error in 30min threshold: {e}", flush=True)
                            pass
                        # ===== END PHASE 2-FIXED THRESHOLD =====

                        # ===== RR FILTERING (Enhanced PEC) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        if achieved_rr_value is None or achieved_rr_value < MIN_ACCEPTED_RR:
                            # Filter out: RR too low
                            print(f"[RR_FILTER] 30min signal REJECTED: {symbol_val} - RR {achieved_rr_value} < MIN {MIN_ACCEPTED_RR}", flush=True)
                            continue
                        
                        # RR acceptable: Store signal + Fire
                        print(f"[RR_FILTER] 30min signal ACCEPTED: {symbol_val} - RR {achieved_rr_value} >= MIN {MIN_ACCEPTED_RR}", flush=True)
                        
                        # Store signal to JSONL
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
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
                            max_gatekeepers=gatekeepers_total
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
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime)
                                print(f"[TIER] 30min {symbol_val}: {signal_tier}", flush=True)
                                
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
                                            fired_time_utc=fired_time_utc.isoformat()
                                        )
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

                if isinstance(res1h, dict) and res1h.get("filters_ok") is True:
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK - 1h =====
                    signal_type = res1h.get("bias", "UNKNOWN")
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
                    # ===== END PHASE 2-FIXED GATES =====
                    
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

                        # ===== PHASE 2-FIXED: DIRECTION-AWARE THRESHOLD CHECK - 1h =====
                        try:
                            min_threshold, threshold_reason = calculate_direction_aware_threshold(
                                Route,
                                regime,
                                signal_type,
                                debug=False
                            )
                            
                            print(f"[PHASE2-FIXED-THRESHOLD] 1h {symbol}: score={score:.1f} | "
                                  f"threshold={min_threshold} | {threshold_reason}", flush=True)
                            
                            if score < min_threshold:
                                print(f"[PHASE2-FIXED-REJECT] 1h {symbol} {signal_type}: "
                                      f"{score:.1f} < {min_threshold} (below {Route} threshold)", flush=True)
                                continue  # Skip signal
                        
                        except Exception as e:
                            print(f"[PHASE2-FIXED] Error in 1h threshold: {e}", flush=True)
                            pass
                        # ===== END PHASE 2-FIXED THRESHOLD =====

                        # ===== RR FILTERING (Enhanced PEC) ===== (regime already calculated earlier)
                        achieved_rr_value = tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None
                        
                        if achieved_rr_value is None or achieved_rr_value < MIN_ACCEPTED_RR:
                            # Filter out: RR too low
                            print(f"[RR_FILTER] 1h signal REJECTED: {symbol_val} - RR {achieved_rr_value} < MIN {MIN_ACCEPTED_RR}", flush=True)
                            continue
                        
                        # RR acceptable: Store signal + Fire
                        print(f"[RR_FILTER] 1h signal ACCEPTED: {symbol_val} - RR {achieved_rr_value} >= MIN {MIN_ACCEPTED_RR}", flush=True)
                        
                        # Store signal to JSONL
                        tp_pct_val = ((tp - entry_price) / entry_price * 100) if tp and entry_price else 0
                        sl_pct_val = ((sl - entry_price) / entry_price * 100) if sl and entry_price else 0
                        
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
                            max_gatekeepers=gatekeepers_total
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
                        
                        # Send trade alert to Telegram
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                # Get signal tier (dynamic - will be Tier-X initially, populates over time)
                                signal_tier = get_signal_tier(tf_val, signal_type, Route, regime)
                                print(f"[TIER] 1h {symbol_val}: {signal_tier}", flush=True)
                                
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
                                            fired_time_utc=fired_time_utc.isoformat()
                                        )
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
