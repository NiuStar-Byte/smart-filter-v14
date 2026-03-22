# 🔧 FIX PLAN: Signal Closure Tracking (Priority: CRITICAL)

## Problem Summary
- **818 OPEN signals** accumulated Mar 1-18, never closed
- **TP/SL targets** identical (0.18/0.18) → impossible to close
- **No auto-closure mechanism** → system waits for PEC to manually update
- **Result:** Can't calculate fair WR, can't optimize filters

---

## Step 1: Diagnose TP/SL Calculation Bug (30 min)

### Check tp_sl_retracement.py
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
grep -n "def calculate_tp_sl" tp_sl_retracement.py
```

### What to look for:
1. **Entry price handling** - Is it being read correctly?
2. **TP calculation** - Should be entry + (ATR × multiplier) for LONG
3. **SL calculation** - Should be entry - (ATR × multiplier) for LONG
4. **Regime handling** - Different multipliers for BULL/BEAR/RANGE?
5. **Return dict** - tp and sl fields populated correctly?

### Common bugs to check:
```python
# Bug 1: tp = sl = entry_price (both zero)
tp = entry_price + 0.0
sl = entry_price + 0.0

# Bug 2: Regime not applied
if regime == "BULL":
    atr_mult = 2.0  # Not being used
# Still returns original broken tp/sl

# Bug 3: Return value wrong structure
return {'tp': tp, 'sl': sl, ...}  # Wrong key names?

# Bug 4: Entry price type error
entry_price = "0.18"  # String instead of float
tp = entry_price + atr * 2  # Type error, falls back to 0
```

---

## Step 2: Fix TP/SL Calculation (1-2 hours)

### Once bug found, apply fix:

**Option A: Fix tp_sl_retracement.py** (preferred)
```python
def calculate_tp_sl(df, entry_price, signal_type, regime=None):
    """
    Calculate TP and SL based on ATR and regime.
    
    Args:
        entry_price: float, must be > 0
        signal_type: 'LONG' or 'SHORT'
        regime: 'BULL', 'BEAR', 'RANGE'
    
    Returns:
        dict with keys: tp, sl, achieved_rr, chosen_ratio, fib_levels
    """
    try:
        # 1. VALIDATE entry_price
        entry_price = float(entry_price)
        if entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")
        
        # 2. CALCULATE ATR
        atr = compute_atr(df, period=14).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01  # Fallback: 1% of price
        
        # 3. SET MULTIPLIERS BY REGIME
        if regime == "BULL":
            tp_mult, sl_mult = 2.5, 1.0  # Favor TP (trend following)
        elif regime == "BEAR":
            tp_mult, sl_mult = 2.5, 1.0
        else:  # RANGE
            tp_mult, sl_mult = 1.5, 1.5  # Equal risk/reward
        
        # 4. CALCULATE TP/SL
        if signal_type == "LONG":
            tp = entry_price + (atr * tp_mult)
            sl = entry_price - (atr * sl_mult)
        else:  # SHORT
            tp = entry_price - (atr * tp_mult)
            sl = entry_price + (atr * sl_mult)
        
        # 5. VALIDATE RESULT
        if signal_type == "LONG":
            assert tp > entry_price, f"LONG TP must be > entry: {tp} vs {entry_price}"
            assert sl < entry_price, f"LONG SL must be < entry: {sl} vs {entry_price}"
        else:
            assert tp < entry_price, f"SHORT TP must be < entry: {tp} vs {entry_price}"
            assert sl > entry_price, f"SHORT SL must be > entry: {sl} vs {entry_price}"
        
        # 6. CALCULATE RR
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        return {
            'tp': float(tp),
            'sl': float(sl),
            'achieved_rr': float(rr),
            'chosen_ratio': f"{tp_mult}:{sl_mult}",
            'fib_levels': calculate_fib_levels(entry_price, tp, sl)
        }
    
    except Exception as e:
        print(f"[ERROR] calculate_tp_sl failed: {e}")
        return None  # Don't fire signal if TP/SL calc fails
```

**Option B: Bypass TP/SL Calculation** (temporary workaround)
- If time constrained, generate synthetic TP/SL from ATR
- Better than identical 0.18/0.18

---

## Step 3: Implement Signal Closure Tracking (2-3 hours)

### Create new file: `signal_closure_tracker.py`
```python
#!/usr/bin/env python3
"""
Automatic signal closure tracking.
Polls OPEN signals, checks exchange order status, updates SENT_SIGNALS.jsonl.
"""

import json
import time
from datetime import datetime, timedelta
from kucoin_data import get_live_price, get_order_status

class SignalClosureTracker:
    """Monitor OPEN signals and auto-update when they hit TP/SL."""
    
    def __init__(self, signals_file):
        self.signals_file = signals_file
        self.timeout_hours = 6  # Close OPEN signals after 6 hours
    
    def check_and_update_signals(self):
        """Check all OPEN signals, update closure status."""
        try:
            signals = []
            with open(self.signals_file, 'r') as f:
                for line in f:
                    if line.strip():
                        signals.append(json.loads(line))
            
            now = datetime.utcnow()
            updated = 0
            
            for sig in signals:
                if sig['status'] != 'OPEN':
                    continue  # Skip already-closed
                
                # Check if signal timed out
                fired_time = datetime.fromisoformat(sig['fired_time_utc'].replace('Z', '+00:00'))
                age_hours = (now - fired_time).total_seconds() / 3600
                
                if age_hours > self.timeout_hours:
                    # Signal timeout: mark as TIMEOUT_WIN (conservative)
                    sig['status'] = 'TIMEOUT_WIN'
                    sig['closed_at'] = now.isoformat()
                    sig['actual_exit_price'] = sig['entry_price']  # Break-even
                    sig['pnl_usd'] = 0.0
                    sig['pnl_pct'] = 0.0
                    print(f"[TIMEOUT] {sig['symbol']} {sig['timeframe']}: Marked TIMEOUT_WIN after {age_hours:.1f}h")
                    updated += 1
                    continue
                
                # Check if hit TP or SL in live market
                try:
                    current_price = get_live_price(sig['symbol'])
                    tp = sig['tp_target']
                    sl = sig['sl_target']
                    entry = sig['entry_price']
                    
                    if sig['signal_type'] == 'LONG':
                        if current_price >= tp:
                            sig['status'] = 'TP'
                            sig['actual_exit_price'] = tp
                        elif current_price <= sl:
                            sig['status'] = 'SL'
                            sig['actual_exit_price'] = sl
                    else:  # SHORT
                        if current_price <= tp:
                            sig['status'] = 'TP'
                            sig['actual_exit_price'] = tp
                        elif current_price >= sl:
                            sig['status'] = 'SL'
                            sig['actual_exit_price'] = sl
                    
                    if sig['status'] != 'OPEN':
                        sig['closed_at'] = now.isoformat()
                        pnl_pct = ((sig['actual_exit_price'] - entry) / entry * 100)
                        sig['pnl_pct'] = pnl_pct
                        sig['pnl_usd'] = pnl_pct * entry / 100  # Approximate
                        print(f"[CLOSED] {sig['symbol']}: {sig['status']} at {sig['actual_exit_price']} ({pnl_pct:.2f}%)")
                        updated += 1
                
                except Exception as e:
                    print(f"[WARN] Could not check {sig['symbol']}: {e}")
            
            # Write updated signals back
            if updated > 0:
                with open(self.signals_file, 'w') as f:
                    for sig in signals:
                        f.write(json.dumps(sig) + '\n')
                print(f"[UPDATE] Closed {updated} signals")
            
            return updated
        
        except Exception as e:
            print(f"[ERROR] check_and_update_signals: {e}")
            return 0
    
    def run_daemon(self, interval_seconds=1800):
        """Run continuous closure checker (every 30 min)."""
        while True:
            try:
                self.check_and_update_signals()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] Daemon error: {e}")
                time.sleep(60)  # Retry after 1 min

# Usage in main.py:
# from signal_closure_tracker import SignalClosureTracker
# tracker = SignalClosureTracker('/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl')
# # Call periodically or run as daemon
# tracker.check_and_update_signals()
```

### Integrate into main.py
```python
# At startup (after signal store init)
from signal_closure_tracker import SignalClosureTracker

try:
    _closure_tracker = SignalClosureTracker(
        '/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl'
    )
    print("[INIT] Signal closure tracker initialized")
except Exception as e:
    print(f"[WARN] Signal closure tracker init failed: {e}")

# In run_cycle(), call periodically
def run_cycle():
    global _closure_tracker
    
    # ... existing cycle code ...
    
    # Update OPEN signals (every cycle, ~every hour)
    if _closure_tracker:
        try:
            _closure_tracker.check_and_update_signals()
        except Exception as e:
            print(f"[WARN] Closure update failed: {e}")
```

---

## Step 4: Force-Close Old Signals for Fair Comparison (30 min)

### Force-close all 818 stuck signals
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 << 'SCRIPT'
import json
from datetime import datetime, timedelta

# Mark all OPEN signals older than 6h as TIMEOUT_WIN
signals = []
now = datetime.utcnow()
timeout_threshold = timedelta(hours=6)

with open('SENT_SIGNALS.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            sig = json.loads(line)
            if sig['status'] == 'OPEN':
                fired = datetime.fromisoformat(sig['fired_time_utc'].replace('Z', '+00:00'))
                age = now - fired
                
                if age > timeout_threshold:
                    sig['status'] = 'TIMEOUT_WIN'
                    sig['closed_at'] = now.isoformat()
                    sig['actual_exit_price'] = sig['entry_price']
                    sig['pnl_pct'] = 0.0
                    sig['pnl_usd'] = 0.0
                    print(f"Force-closed: {sig['symbol']} ({age.total_seconds()/3600:.1f}h old)")
            signals.append(sig)

# Write back
with open('SENT_SIGNALS.jsonl', 'w') as f:
    for sig in signals:
        f.write(json.dumps(sig) + '\n')

print(f"\nTotal signals processed: {len(signals)}")
print(f"Force-closed (TIMEOUT): {sum(1 for s in signals if s['status'] == 'TIMEOUT_WIN')}")

# Recalculate WR
closed = [s for s in signals if s['status'] in ['TP', 'SL', 'TIMEOUT_WIN']]
wins = sum(1 for s in closed if s['status'] == 'TP')
print(f"\nUpdated WR: {wins}/{len(closed)} = {100*wins/len(closed):.2f}%")
SCRIPT
```

---

## Step 5: Test & Validate (1-2 hours)

### Verify TP/SL fix
```bash
# Check if any signals still have TP ≈ SL
python3 << 'SCRIPT'
import json
bad_count = 0
with open('SENT_SIGNALS.jsonl', 'r') as f:
    for line in f:
        sig = json.loads(line)
        if abs(sig['tp_target'] - sig['sl_target']) < 0.001:
            bad_count += 1

print(f"Signals with TP ≈ SL: {bad_count} (should be 0)")
SCRIPT
```

### Verify closure tracking
```bash
# Check that signals are actually closing
ps aux | grep signal_closure_tracker  # Should be running
tail -20 main_daemon.log | grep -E "TIMEOUT|CLOSED"  # Should see recent closures
```

---

## Step 6: Clean Restart & Fair Comparison (30 min)

### Archive old signals (optional but recommended)
```bash
cp SENT_SIGNALS.jsonl SENT_SIGNALS_BACKUP_MAR18.jsonl
cp SIGNALS_MASTER.jsonl SIGNALS_MASTER_BACKUP_MAR18.jsonl
```

### Recalculate WR on fixed data
```bash
python3 << 'SCRIPT'
import json

# Read all closed signals
with open('SENT_SIGNALS.jsonl', 'r') as f:
    signals = [json.loads(line) for line in f if line.strip()]

# Split by era
foundation = [s for s in signals if datetime.fromisoformat(s['fired_time_utc'].replace('Z', '+00:00')) < datetime(2026, 3, 16)]
new = [s for s in signals if datetime.fromisoformat(s['fired_time_utc'].replace('Z', '+00:00')) >= datetime(2026, 3, 16)]

# Closed only
foundation_closed = [s for s in foundation if s['status'] in ['TP', 'SL', 'TIMEOUT_WIN']]
new_closed = [s for s in new if s['status'] in ['TP', 'SL', 'TIMEOUT_WIN']]

# Calculate WR
def calc_wr(signals):
    if not signals: return 0
    wins = sum(1 for s in signals if s['status'] == 'TP')
    return 100 * wins / len(signals)

print(f"Foundation WR: {calc_wr(foundation_closed):.2f}% ({sum(1 for s in foundation_closed if s['status'] == 'TP')}/{len(foundation_closed)})")
print(f"New WR: {calc_wr(new_closed):.2f}% ({sum(1 for s in new_closed if s['status'] == 'TP')}/{len(new_closed)})")
SCRIPT
```

---

## Timeline

| Step | Time | Owner | Notes |
|------|------|-------|-------|
| 1. Diagnose TP/SL | 30 min | You | Check tp_sl_retracement.py |
| 2. Fix TP/SL | 1-2 hr | You | Apply patch, test |
| 3. Add closure tracker | 2-3 hr | You | Implement daemon |
| 4. Force-close old | 30 min | You | Mark 818 signals TIMEOUT |
| 5. Test | 1-2 hr | You | Verify signals closing |
| **TOTAL** | **6-8 hr** | - | Full fix + validation |

---

## Success Criteria

✅ No signals with TP ≈ SL  
✅ All OPEN signals older than 6h marked TIMEOUT_WIN  
✅ New signals auto-close within 6h of firing  
✅ WR calculated fairly (Foundation vs New on closed-only data)  
✅ Phase 2 gates can be properly evaluated

---

## Alternative: Quick Workaround (1 hour)

If time is critical:
1. Force-close all 818 stuck signals as TIMEOUT_WIN
2. Don't fix TP/SL yet, just mark the old ones closed
3. Restart daemon fresh with empty SENT_SIGNALS.jsonl
4. Run for 24-48h on clean slate
5. Then fix TP/SL for next iteration

This gives you fair apples-to-apples comparison immediately.
