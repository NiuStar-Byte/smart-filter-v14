# PEC SYSTEM - CRITICAL ANALYSIS & REQUIREMENTS

**Date:** 2026-02-22 17:00 GMT+7  
**Status:** 🔴 CURRENT IMPLEMENTATION HAS CRITICAL GAPS  
**Author:** Nox

---

## ⚠️ CRITICAL FINDINGS

### **FINDING 1: Signal Tracking is Broken**

**Current Approach:**
- Signals logged to `logs.txt` via text log lines
- Example: `[FIRED] Logged: uuid, symbol, tf, signal_type, entry_time, entry_price, SCORE: X, ...`
- **Problem:** Text parsing is fragile and incomplete
- **Real Issue:** logs.txt is manually copy-pasted from Railway (incomplete)

**Result:**  
❌ PEC is reading from an INCOMPLETE data source  
❌ Only a few manual signals in logs.txt  
❌ No structured storage mechanism  
❌ Cannot reliably backtest actual Telegram signals

---

### **FINDING 2: Signal Data is Incomplete**

**Data in Current logs.txt:**
- UUID ✅
- Symbol ✅
- Timeframe ✅
- Signal Type (LONG/SHORT) ✅
- Entry Time ✅
- Entry Price ✅
- Score, Confidence ✅

**Data MISSING but Available in Telegram:**
- TP Target ❌ (sent to Telegram but not tracked)
- SL Target ❌ (sent to Telegram but not tracked)
- TP/SL Source ❌ (retracement or ratio-based?)
- Filter Details ❌ (which filters passed?)
- Route ❌ (explicit trade direction)
- Regime ❌ (market regime when signal fired)

**Main.py SENDS all this to Telegram, but it's NOT STORED for PEC!**

---

### **FINDING 3: No Structured Signal Store**

**What's needed:**
```
signals_fired.jsonl
├─ Each line = 1 complete signal (JSON)
├─ All Telegram data included
├─ Easy to query (timestamp, symbol, etc.)
└─ Append-only (permanent record)

Example:
{
  "uuid": "23dee4d1...",
  "symbol": "BTC-USDT",
  "timeframe": "15min",
  "signal_type": "LONG",
  "fired_time_utc": "2026-02-22T15:30:00Z",
  "entry_price": 42500.00,
  "tp_target": 43140.00,
  "sl_target": 42075.00,
  "tp_pct": 1.5,
  "sl_pct": -1.0,
  "score": 18,
  "max_score": 20,
  "confidence": 90.0,
  "route": "LONG_CONFIRMED",
  "regime": "UPTREND",
  "telegram_msg_id": "12345"
}
```

**What exists now:**  
❌ No structured store  
❌ Only fragile text logs  
❌ Data loss (manual copy-paste)

---

## 📋 PROPER PEC REQUIREMENTS

### **Requirement 1: Signal Capture**

**Source:** When SmartFilter fires signal + sends to Telegram

**Must Capture:**
1. ✅ UUID (unique signal ID)
2. ✅ Symbol, Timeframe, Direction
3. ✅ Entry Price (actual bid/ask at signal time)
4. ✅ Fired Time (UTC timestamp)
5. ✅ **TP Target** (from calculate_tp_sl)
6. ✅ **SL Target** (from calculate_tp_sl)
7. ✅ **TP/SL Pct** (1.5% / -1.0%)
8. ✅ Score & Confidence
9. ✅ Filter/Gatekeeper Status
10. ✅ Telegram Message ID (for audit trail)

**Storage:** Append to `signals_fired.jsonl` (permanent record)

---

### **Requirement 2: Signal Retrieval for PEC**

**Use Case:** "Get all signals fired between date1 and date2"

```python
# Pseudo-code
signals = load_signals_jsonl(
    start_date="2026-02-22",
    end_date="2026-02-25",
    symbols=["BTC-USDT", "ETH-USDT"],  # optional filter
    timeframes=["15min", "30min"],      # optional filter
)
# Returns: List[Dict] with all complete signal data
```

---

### **Requirement 3: PEC Validation Logic**

**Input:** 1 signal from signals_fired.jsonl

**Process:**
```
1. Extract: symbol, timeframe, entry_price, tp_target, sl_target, fired_time
2. Fetch: OHLCV data for symbol/timeframe (250 bars)
3. Match: Find bar closest to fired_time
4. Scan: Next 20 bars for TP/SL hits
5. Calculate:
   - Exit Price = TP if hit | SL if hit | Close at bar 20 if timeout
   - P&L = (Exit - Entry) × Qty - Fees
   - Status = WIN (TP) | LOSS (SL) | BREAK (Timeout)
6. Output: Result dict with all validation data
```

**TP/SL Targets:** Come from signal (not hardcoded in PEC)

---

### **Requirement 4: Batch Reporting**

**Input:** 50-300 signals

**Process:**
1. Load all signals in batch
2. Run PEC validation on each
3. Aggregate results:
   - Win/loss count
   - P&L totals
   - False positive rate
   - Filter effectiveness
   - By-timeframe breakdown
4. Export: Excel + Markdown report

---

## 🔧 ENHANCED PEC MECHANISM (Design)

### **New File: `signal_store.py`**

```python
# Signal capturing & storage
class SignalStore:
    
    def __init__(self, jsonl_path="signals_fired.jsonl"):
        self.path = jsonl_path
    
    def append_signal(self, signal_dict):
        """Append a complete signal to JSONL"""
        # Validate all required fields
        # Append to jsonl_path
        # Return signal_uuid
    
    def load_signals(self, start_date, end_date, filters=None):
        """Load signals within date range"""
        # Read jsonl file
        # Filter by date + optional filters
        # Return List[Dict]
    
    def get_signal_by_uuid(self, uuid):
        """Retrieve specific signal"""
        # Query by uuid
        # Return Dict
```

### **Modified File: `main.py`**

**Before sending to Telegram**, also store to JSONL:

```python
# After calculate_tp_sl (line ~390)

# Capture complete signal
signal_data = {
    "uuid": str(uuid.uuid4()),
    "symbol": symbol_val,
    "timeframe": tf_val,
    "signal_type": signal_type,
    "fired_time_utc": fired_time_utc.isoformat(),
    "entry_price": entry_price,
    "tp_target": tp,
    "sl_target": sl,
    "tp_pct": 1.5,  # from calculate_tp_sl
    "sl_pct": -1.0,
    "score": score,
    "max_score": score_max,
    "confidence": confidence,
    "route": Route,
    "regime": regime,
}

# Store to JSONL
signal_store.append_signal(signal_data)

# THEN send to Telegram
send_telegram_alert(..., uuid=signal_data["uuid"], ...)
```

### **Modified File: `pec_backtest.py`**

**Instead of parsing logs.txt:**

```python
from signal_store import SignalStore

def run_pec_backtest(start_date, end_date, batch_num):
    store = SignalStore()
    
    # Load ACTUAL signals from JSONL
    signals = store.load_signals(start_date, end_date)
    
    if not signals:
        print("No signals in date range")
        return
    
    # Run PEC on each
    results = []
    for signal in signals:
        result = validate_signal_pec(signal)  # Uses TP/SL from signal
        results.append(result)
    
    # Report
    generate_report(results, batch_num)
```

---

## 📊 DATA FLOW (Corrected)

```
Signal Fired
    ↓
SmartFilter + calculate_tp_sl
    ├─ Entry Price ✅
    ├─ TP Target (1.5% or dynamic) ✅
    ├─ SL Target (-1.0% or dynamic) ✅
    └─ Score, Confidence ✅
    ↓
[NEW] Store to signals_fired.jsonl ✅
    ├─ uuid, symbol, timeframe
    ├─ fired_time, entry_price
    ├─ tp_target, sl_target
    ├─ tp_pct, sl_pct
    ├─ score, confidence
    └─ route, regime
    ↓
Send to Telegram (with uuid reference)
    ├─ Human reads signal
    ├─ Makes trading decision
    └─ (Optional) Records actual execution
    ↓
Later: Run PEC Backtest
    ├─ Load signals from signals_fired.jsonl
    ├─ Validate each (using TP/SL from signal)
    ├─ Generate report
    └─ Compare actual execution vs PEC
```

---

## ✅ WHAT NEEDS TO BE BUILT

1. **`signal_store.py`** - JSONL-based signal storage
   - append_signal(signal_dict)
   - load_signals(start_date, end_date, filters)
   - get_signal_by_uuid(uuid)

2. **Modified `main.py`**
   - After calculate_tp_sl, store signal to JSONL
   - Include uuid in Telegram message

3. **Modified `pec_backtest.py`**
   - Use SignalStore instead of text log parsing
   - Load from signals_fired.jsonl
   - Use TP/SL from signal (not hardcoded)

4. **Modified `pec_engine.py`**
   - Accept TP/SL from signal data
   - No hardcoded 1.5%/-1.0% (use from signal)

5. **Documentation**
   - Signal data schema
   - How to query signals
   - How PEC uses them

---

## 🎯 ACCEPTANCE CRITERIA

**Before running any PEC backtest:**

- [ ] signals_fired.jsonl exists and is being appended to
- [ ] Each signal has: uuid, symbol, tf, entry_price, tp_target, sl_target, tp_pct, sl_pct
- [ ] Telegram signals match JSONL signals
- [ ] PEC loads from signals_fired.jsonl (not logs.txt)
- [ ] PEC uses TP/SL from signal data
- [ ] Can query signals by date range
- [ ] Can run PEC on any signal (not just recent ones)

---

## ⏭️ NEXT STEPS

**For Nox:**
1. Build signal_store.py (Signal capture & storage)
2. Modify main.py to store signals when fired
3. Modify pec_backtest.py to use SignalStore
4. Test: Verify signals are captured correctly
5. Run PEC on actual signals (not old test data)

**For Jetro:**
- Confirm the signal schema (is TP/SL always 1.5%/-1.0% or dynamic?)
- Confirm exit logic (5 bars or 20 bars max hold?)
- Specify where signals should be stored (current directory or elsewhere?)

---

## 📝 SUMMARY

**Current Problem:** PEC mechanism is incomplete - no proper signal storage, relying on fragile text logs

**Solution:** Build proper JSONL-based signal store with complete data capture

**Impact:** Can reliably backtest ALL actual signals fired to Telegram, not just old manual logs

**Timeline:** 2-3 hours to build + test properly

