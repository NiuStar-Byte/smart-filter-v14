# PROJECT-5: PEC - AUDIT CLOSURE REMARKS (Architectural Correction)

**Decision Date:** 2026-03-21 01:08 GMT+7  
**Status:** ✅ Architectural correction locked  
**Impact:** Critical - fixes immutability gap in SIGNALS_INDEPENDENT_AUDIT.txt

---

## 🚨 **The Problem Found**

Original locked model had a critical flaw:

```
AUDIT (append-only):
├─ Signal fires → Append with status="OPEN"
└─ Signal closes → ??? (Never recorded in AUDIT)

Result: AUDIT is INCOMPLETE
├─ Missing closure information
├─ Can't rebuild MASTER from AUDIT (would lose all closure info)
└─ AUDIT not truly "source of truth"
```

**User asked:** "If independent.txt is always appended, do you put remarks on independent.txt once signals closed?"

**Answer:** YES. We must. CLOSURE REMARKS must be appended to AUDIT.

---

## ✅ **SOLUTION: CLOSURE REMARKS**

AUDIT is truly append-only by adding separate **CLOSURE remark** lines:

```
Signal Event 1: FIRED (Daemon appends at signal fire)
{
  "type": "FIRED",
  "signal_uuid": "abc123",
  "symbol": "BTC-USDT",
  "timeframe": "1h",
  "direction": "LONG",
  "entry_price": 45000.00,
  "tp_price": 46000.00,
  "sl_price": 44000.00,
  "fired_time_utc": "2026-03-21T14:00:00",
  "signal_origin": "NEW_LIVE",
  "status": "OPEN",
  ... (all daemon-created fields)
}

Signal Event 2: CLOSURE (Executor appends when signal closes)
{
  "type": "CLOSURE",
  "signal_uuid": "abc123",
  "status": "TP_HIT",
  "actual_exit_price": 46000.00,
  "pnl_usd": 1000.00,
  "pnl_pct": 2.2,
  "closed_at": "2026-03-21T14:45:00"
}

Signal Event 3: FIRED (Another signal fires)
{
  "type": "FIRED",
  "signal_uuid": "def456",
  ... (all fields)
}

Signal Event 4: CLOSURE (Another signal closes)
{
  "type": "CLOSURE",
  "signal_uuid": "def456",
  "status": "SL_HIT",
  ... (closure info)
}
```

---

## 🔄 **Updated Data Flow**

### **Daemon Flow**
```
Signal fires:
├─ Analyze (PROJECT-3: SmartFilter generates signal)
├─ Create signal object with all fields
├─ Append to SIGNALS_INDEPENDENT_AUDIT.txt
│  └─ Line type: {"type": "FIRED", uuid, symbol, entry_price, tp_price, sl_price, fired_time_utc, ...}
├─ Append to SIGNALS_MASTER.jsonl
│  └─ Same signal object with status="OPEN"
└─ Send Telegram alert
```

### **Executor Flow**
```
Backtest phase:
├─ Read SIGNALS_INDEPENDENT_AUDIT.txt
│  └─ Find all FIRED lines to process
├─ For each OPEN signal:
│  ├─ Get immutable facts from AUDIT FIRED line
│  ├─ Backtest against OHLCV
│  ├─ Calculate exit price & P&L
│  └─ Determine outcome (TP_HIT / SL_HIT / TIMEOUT)
│
Update phase:
├─ Update SIGNALS_MASTER.jsonl
│  └─ Modify existing signal: status, actual_exit_price, pnl_usd, pnl_pct, closed_at
├─ Append to SIGNALS_INDEPENDENT_AUDIT.txt
│  └─ New line type: {"type": "CLOSURE", signal_uuid, status, actual_exit_price, pnl_usd, ...}
└─ Both updates must succeed or fail together (atomic)
```

### **Reporter Flow**
```
Read phase:
├─ Read SIGNALS_INDEPENDENT_AUDIT.txt
│  ├─ Parse all FIRED lines (signal facts)
│  └─ Parse all CLOSURE remarks (closure info)
├─ Read SIGNALS_MASTER.jsonl
│  └─ Get current status (for validation)
│
Merge phase:
├─ For each signal UUID:
│  ├─ Get FIRED info from AUDIT (entry, tp, sl, fired_time, signal_origin)
│  ├─ Find matching CLOSURE remark from AUDIT (status, exit_price, pnl)
│  ├─ Validate against MASTER (should match)
│  └─ Use merged data for metrics
│
Calculate phase:
├─ Group by signal_origin (FOUNDATION / NEW_IMMUTABLE / NEW_LIVE)
├─ Calculate WR, P&L, duration per group
└─ Output metrics
```

### **Recovery Flow (If MASTER Corrupts)**
```
Disaster: SIGNALS_MASTER.jsonl corrupted/lost

Recovery:
├─ Read SIGNALS_INDEPENDENT_AUDIT.txt (backup source)
├─ Parse FIRED lines → original signal data
├─ Parse CLOSURE remarks → closure data
├─ Reconstruct signal object: FIRED + matching CLOSURE
├─ Write clean SIGNALS_MASTER.jsonl
└─ ✅ System recovered, no data loss
```

---

## 📋 **FIELD REQUIREMENTS**

### **What goes in AUDIT FIRED lines?**
All daemon-created fields:
```
signal_uuid          (unique identifier)
symbol              (BTC-USDT)
timeframe           (1h, 15min, 30min)
direction           (LONG / SHORT)
entry_price         (45000.00)
tp_price            (46000.00)
sl_price            (44000.00)
tp_pct              (2.2)
sl_pct              (-2.0)
rr                  (1.8)
score               (15)
max_score           (19)
confidence          (78.5)
route               (TREND_CONTINUATION)
regime              (BULL)
fired_time_utc      (2026-03-21T14:00:00)
fired_time_jakarta  (2026-03-21T21:00:00+07:00)
fired_date_jakarta  (2026-03-21)
sent_time_utc       (2026-03-21T14:00:00.123)
signal_origin       (FOUNDATION / NEW_IMMUTABLE / NEW_LIVE)
status              (OPEN at fire)
weighted_score      (calculated by daemon)
consensus           (calculated by daemon)
passed_min_score_gate (boolean)
tier                (optional metadata)
```

### **What goes in AUDIT CLOSURE remarks?**
Only executor-calculated fields:
```
type                (literal: "CLOSURE")
signal_uuid         (matches FIRED line)
status              (TP_HIT / SL_HIT / TIMEOUT)
actual_exit_price   (46000.00)
pnl_usd             (1000.00)
pnl_pct             (2.2)
closed_at           (2026-03-21T14:45:00)
```

### **What's in MASTER?**
Complete merged data:
```
FIRED fields (from AUDIT FIRED line)
+
CLOSURE fields (from AUDIT CLOSURE remark)
+
All combined in one signal object with updated status/exit/pnl
```

---

## 🔐 **Immutability Maintained**

With CLOSURE REMARKS:

✅ **AUDIT is APPEND-ONLY**
- FIRED lines never modified
- CLOSURE remarks only appended
- No changes to existing lines

✅ **AUDIT is COMPLETE**
- Has both fire and closure info
- All data needed to recover MASTER
- No missing information

✅ **AUDIT is IMMUTABLE PROOF**
- FIRED line timestamped at fire
- CLOSURE remark timestamped at close
- Chronological ordering proves causation
- Can't retroactively change closure info

---

## ⚙️ **Implementation Details**

### **Executor Closure Remark Creation**
```python
# After backtest determines exit
closure_remark = {
    "type": "CLOSURE",
    "signal_uuid": signal_uuid,
    "status": "TP_HIT",  # or SL_HIT, TIMEOUT
    "actual_exit_price": 46000.00,
    "pnl_usd": (actual_exit_price - entry_price) * quantity,
    "pnl_pct": (pnl_usd / entry_price) * 100,
    "closed_at": datetime.utcnow().isoformat()
}

# Append to AUDIT
with open('SIGNALS_INDEPENDENT_AUDIT.txt', 'a') as f:
    f.write(json.dumps(closure_remark) + '\n')
```

### **Reporter Merge Logic**
```python
# Load all signals from AUDIT
fired_signals = {}
closure_remarks = {}

with open('SIGNALS_INDEPENDENT_AUDIT.txt') as f:
    for line in f:
        event = json.loads(line)
        if event['type'] == 'FIRED':
            fired_signals[event['signal_uuid']] = event
        elif event['type'] == 'CLOSURE':
            closure_remarks[event['signal_uuid']] = event

# Reconstruct complete signals
for uuid in fired_signals:
    signal = fired_signals[uuid]
    if uuid in closure_remarks:
        # Merge closure info
        signal.update(closure_remarks[uuid])
```

---

## ✅ **Answer to User's Question**

**"Do you put remarks on independent.txt once signals closed?"**

✅ **YES, absolutely.**

CLOSURE REMARKS must be appended to SIGNALS_INDEPENDENT_AUDIT.txt:
- Keeps AUDIT complete (fire + closure)
- Keeps AUDIT append-only (never modifies existing)
- Enables full MASTER recovery from AUDIT
- Maintains immutability guarantee

---

## 📌 **Updated File Format**

Each line in SIGNALS_INDEPENDENT_AUDIT.txt is ONE of:
- Type: "FIRED" (when signal fires)
- Type: "CLOSURE" (when signal closes)
- Nothing else should be appended

This makes AUDIT:
- ✅ Append-only (new lines added, never modify existing)
- ✅ Complete (has all information needed)
- ✅ Immutable (can't change past events)
- ✅ Recoverable (MASTER rebuild source)

**This is how AUDIT becomes the true source of truth.**
