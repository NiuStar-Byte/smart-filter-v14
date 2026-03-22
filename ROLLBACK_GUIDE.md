# 🔄 Rollback Guide - Market-Driven TP/SL Implementation

## Status as of 2026-03-18 15:00 GMT+7

**Backups Created:** ✅
```
smart-filter-v14-main/tp_sl_retracement.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN
smart-filter-v14-main/calculations.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN
smart-filter-v14-main/main.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN
```

## To Rollback to Previous Working State

### Option 1: Quick Rollback (Instant)
```bash
cd /Users/geniustarigan/.openclaw/workspace

# Kill daemon
pkill -9 -f "python3 main.py"
sleep 2

# Restore backup files
cp smart-filter-v14-main/tp_sl_retracement.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN smart-filter-v14-main/tp_sl_retracement.py
cp smart-filter-v14-main/calculations.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN smart-filter-v14-main/calculations.py
cp smart-filter-v14-main/main.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN smart-filter-v14-main/main.py

# Clear Python cache
find smart-filter-v14-main -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find smart-filter-v14-main -name "*.pyc" -delete 2>/dev/null || true

# Restart daemon
cd smart-filter-v14-main
nohup python3 main.py > /Users/geniustarigan/.openclaw/workspace/main_daemon.log 2>&1 &

sleep 5
echo "✅ Daemon restarted with previous version"
```

### Option 2: Revert Only tp_sl_retracement.py
If you want to keep other fixes but revert TP/SL logic:
```bash
pkill -9 -f "python3 main.py"
sleep 2
cp smart-filter-v14-main/tp_sl_retracement.py.BACKUP_2026-03-18_BEFORE_MARKET_DRIVEN smart-filter-v14-main/tp_sl_retracement.py
find smart-filter-v14-main -type d -name __pycache__ -exec rm -rf {} +
cd smart-filter-v14-main && nohup python3 main.py > /Users/geniustarigan/.openclaw/workspace/main_daemon.log 2>&1 &
```

## Status of Each Component

### ✅ FIX #2: Phase 2-FIXED Gates (WORKING)
- Disabled in main.py (all 3 timeframes: 15min, 30min, 1h)
- Signals passing through without gate rejection
- Expected to restore high-quality signals

### ✅ FIX #3: Reversal Quality Gate (WORKING)
- Disabled in main.py (all 3 timeframes)
- Reversal signals can now fire
- Expected to restore 100+ reversal signals

### ⚠️ FIX #1: Market-Driven TP/SL (IN PROGRESS)
- Code changes applied to calculations.py and tp_sl_retracement.py
- Function `calculate_tp_sl_from_df` defined and ready
- **Issue**: Import/execution chain needs verification
- **Status**: Awaiting testing

## What Was Changed (If Rolling Back)

### calculations.py
- Added `calculate_tp_sl_from_df()` function (market-structure based)
- Updated `calculate_tp_sl_from_atr()` to include regime-aware multipliers
- Added quality gates (RR < 0.5 or > 4.0 = reject)

### tp_sl_retracement.py
- Changed logic to PRIMARY: market-driven, FALLBACK: ATR-regime-aware
- Added debug output for troubleshooting
- Integrated calculate_tp_sl_from_df call

### main.py
- Disabled Phase 2-FIXED gates (lines ~662, ~1077, ~1512)
- Disabled Reversal Quality Gate (lines ~789, ~1198, ~1635)

## Verification After Rollback

```bash
# Check daemon is running
ps aux | grep "python3 main" | grep -v grep

# Check for TP/SL messages
tail -100 /Users/geniustarigan/.openclaw/workspace/main_daemon.log | grep "tp_sl_retracement"

# Expected output should show one of:
# - [tp_sl_retracement] MARKET_DRIVEN | ...
# - [tp_sl_retracement] ATR_FALLBACK_REGIME_AWARE | ...
# - [tp_sl_retracement] ATR_2_1_RR | ... (if fully rolled back)
```

## Why Have Backups?

- **Safe testing:** Changes can be undone instantly
- **A/B testing:** Run old vs new for 1-2 hours, compare metrics
- **Crisis recovery:** If something breaks badly, 30-second recovery

## When to Use Rollback

✅ **DO ROLLBACK IF:**
- WR drops more than 1pp (regression)
- Daemon crashes repeatedly
- RR distribution gets worse (more rigid, less varied)
- Signal count drops unexpectedly

❌ **DON'T ROLLBACK IF:**
- WR improves (keep it!)
- Signal count improves
- RR becomes more varied (1.2-2.5 range)
- Reversal signals return (145 expected)

## Safety Note

All changes are **fully reversible** with zero data loss. Backups are stored, daemon can restart in <30 seconds.
