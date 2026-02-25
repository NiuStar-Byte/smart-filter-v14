# PEC Execution Tracking - Complete Guide

## Overview

**PEC Execution Tracker** monitors SENT_SIGNALS.jsonl and updates each signal's execution status:

1. Signal fires to Telegram → Status: **OPEN**
2. Market hits TP → Status: **TP_HIT** (with P&L recorded)
3. Market hits SL → Status: **SL_HIT** (with P&L recorded)
4. Timeout (>24h) → Status: **TIMEOUT** (with P&L at market price)

---

## Quick Start

### 1. Get Tracker Instance

```python
from pec_execution_tracker import get_pec_execution_tracker

tracker = get_pec_execution_tracker("SENT_SIGNALS.jsonl")
```

### 2. Get Open Signals

```python
open_signals = tracker.get_open_signals()

for signal in open_signals:
    print(f"{signal['symbol']} {signal['timeframe']}")
    print(f"  Entry: {signal['entry_price']}")
    print(f"  TP: {signal['tp_target']}")
    print(f"  SL: {signal['sl_target']}")
```

### 3. Monitor for TP/SL

```python
# When market reaches TP
tracker.update_signal_if_tp_hit(signal_uuid, current_price)

# When market reaches SL
tracker.update_signal_if_sl_hit(signal_uuid, current_price)

# When timeout expires
tracker.mark_as_timeout(signal_uuid, current_market_price)
```

### 4. Get Statistics

```python
stats = tracker.get_execution_stats()

print(f"Win Rate: {stats['win_rate_pct']}%")
print(f"Total P&L: ${stats['total_pnl_usd']}")
print(f"Closed: {stats['closed']} trades")
```

---

## API Reference

### Class: PECExecutionTracker

#### Methods

**`__init__(sent_signals_path, timeout_hours=24)`**
- Initialize tracker
- Args:
  - `sent_signals_path`: Path to SENT_SIGNALS.jsonl
  - `timeout_hours`: Hours before signal expires (default: 24)

**`update_signal_if_tp_hit(signal_uuid, current_price) → bool`**
- Check if signal hit TP and update status
- Returns: True if TP was hit, False otherwise
- Automatically calculates P&L

**`update_signal_if_sl_hit(signal_uuid, current_price) → bool`**
- Check if signal hit SL and update status
- Returns: True if SL was hit, False otherwise
- Automatically calculates P&L

**`mark_as_timeout(signal_uuid, exit_price) → bool`**
- Mark signal as expired without hitting TP/SL
- `exit_price`: Current market price when timeout triggered
- Calculates P&L at timeout price

**`get_open_signals() → List[Dict]`**
- Returns all signals with status='OPEN'
- Ready for TP/SL monitoring

**`get_execution_stats() → Dict`**
- Returns:
  ```python
  {
    "total_sent": 127,
    "open": 12,
    "tp_hit": 78,
    "sl_hit": 31,
    "timeout": 6,
    "closed": 115,
    "total_pnl_usd": 1245.67,
    "winning_trades": 78,
    "losing_trades": 31,
    "win_rate_pct": 71.6
  }
  ```

**`get_signals_by_status(status) → List[Dict]`**
- Get signals filtered by status
- Status: "OPEN", "TP_HIT", "SL_HIT", "TIMEOUT"

**`print_execution_summary()`**
- Print formatted execution statistics to console

---

## Integration with PEC Backtest

### Example: Monitor Signals Every 5 Minutes

```python
from pec_execution_tracker import get_pec_execution_tracker
from kucoin_data import get_live_entry_price
import time

tracker = get_pec_execution_tracker("SENT_SIGNALS.jsonl")

while True:
    open_signals = tracker.get_open_signals()
    
    for signal in open_signals:
        symbol = signal['symbol']
        uuid = signal['uuid']
        
        try:
            # Get current market price
            current_price = get_live_entry_price(symbol, "BUY")
            
            # Check if TP or SL hit
            if tracker.update_signal_if_tp_hit(uuid, current_price):
                print(f"✅ {symbol} TP HIT!")
            
            elif tracker.update_signal_if_sl_hit(uuid, current_price):
                print(f"❌ {symbol} SL HIT!")
        
        except Exception as e:
            print(f"Error monitoring {symbol}: {e}")
    
    # Print summary every cycle
    tracker.print_execution_summary()
    
    # Wait 5 minutes
    time.sleep(300)
```

---

## Output Examples

### TP Hit Signal

```
[PEC-TP] BTC-USDT 15min TP HIT @ 66520.52 | P&L: $909.37 (1.39%)
```

### SL Hit Signal

```
[PEC-SL] ETH-USDT 30min SL HIT @ 1850.00 | LOSS: -$45.23 (-2.38%)
```

### Timeout Signal

```
[PEC-TIMEOUT] SOL-USDT 1h TIMEOUT @ 142.50 | PROFIT: $12.34 (0.87%)
```

### Summary

```
======================================================================
PEC EXECUTION SUMMARY:
  Total Sent:      127
  Open (Running):  12
  TP Hit:          78
  SL Hit:          31
  Timeout:         6
  Closed Trades:   115
  Total P&L:       $1,245.67
  Win Rate:        71.6% (78W/31L)
======================================================================
```

---

## Signal Record Structure

Each signal in SENT_SIGNALS.jsonl has:

```json
{
  "uuid": "408f02fe-0fa4-486a-acb9-95e7dee83859",
  "symbol": "BTC-USDT",
  "timeframe": "15min",
  "signal_type": "LONG",
  
  "entry_price": 65611.15,
  "tp_target": 66520.52,
  "sl_target": 65156.46,
  
  "tp_pct": 1.39,
  "sl_pct": -0.69,
  "achieved_rr": 2.0,
  
  "score": 15,
  "max_score": 19,
  "confidence": 82.8,
  "route": "TREND CONTINUATION",
  "regime": "BULL",
  
  "telegram_msg_id": "408f02fe-0fa",
  "fired_time_utc": "2026-02-25T05:33:31.646175",
  "sent_time_utc": "2026-02-25T05:33:32.984125",
  
  "status": "OPEN",
  "closed_at": null,
  "actual_exit_price": null,
  "pnl_usd": null,
  "pnl_pct": null
}
```

When TP/SL/Timeout hits:
- `status` → "TP_HIT" / "SL_HIT" / "TIMEOUT"
- `closed_at` → Timestamp when closed
- `actual_exit_price` → Price where TP/SL hit or timeout price
- `pnl_usd` → Dollar profit/loss
- `pnl_pct` → Percentage profit/loss

---

## Key Features

✅ **Automatic P&L Calculation** - Computes profit/loss on TP/SL hits  
✅ **Side-Aware** - Handles LONG and SHORT signals differently  
✅ **Timeout Management** - Closes signals after 24h (configurable)  
✅ **Statistics** - Win rate, ROI, closed trades count  
✅ **Clean Logging** - [PEC-TP], [PEC-SL], [PEC-TIMEOUT] tags  

---

## Next Steps

1. **Run PEC Monitor** - Start background process monitoring SENT_SIGNALS
2. **Track Executions** - Update signals as they hit TP/SL
3. **Calculate Statistics** - Generate win rate and P&L reports
4. **Optimize** - Use metrics to improve SmartFilter configuration
