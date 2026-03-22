# 📐 PROJECT-5: PEC SYSTEM ARCHITECTURE

**Scope:** Complete architecture, data flow, failure modes, and integration with PROJECT-1,2,3,4  
**Status:** Currently Broken ❌ — Misaligned components causing recurring issues  
**Document:** Definitive system design (user requested)

---

## 🏗️ SYSTEM CONTEXT (All Projects)

### **PROJECT-1: NEXUS (Polymarket Copy Trading)**
- **Input:** Polymarket leader positions (real-time)
- **Logic:** Copy trades at 5.0× ratio, scale to market (buy when leader buys)
- **Output:** Copy trades to Polymarket (actual capital deployed)
- **Status:** ✅ LIVE (collecting data daily)

### **PROJECT-2: NEXAU (Gold/Macro Trading)**
- **Input:** XAU-USD price + macro indicators
- **Logic:** Entry/exit rules based on gold trends
- **Output:** Trade signals to manual execution
- **Status:** In development

### **PROJECT-3: SMARTFILTER (Crypto Signals)**
- **Input:** 92 crypto symbols × 3 timeframes = 276 analyses per cycle
- **Logic:** 5+ filters (Support/Resistance, Volatility, MACD, Volume, ATR, etc.)
- **Output:** Trade signals with scores (1-19 range)
- **Firing Rate:** ~150-250 signals/hour (very high frequency)
- **Status:** ✅ LIVE (daemon running continuously)

### **PROJECT-4: ASTERDEX SPOT BOT**
- **Input:** Signals from PROJECT-3 + wallet balance
- **Logic:** Auto-execute 1% of balance at +3% TP, -2% SL
- **Output:** Spot trades on Asterdex (live capital deployed)
- **Status:** ✅ CODE READY (not yet activated)

### **PROJECT-5: PEC (Position Entry Closure & Backtest)**
- **Input:** Signals from PROJECT-3 (2,540+ signals historical, ~1,600 new daily)
- **Logic:** Backtest signals against historical OHLCV, mark as TP_HIT/SL_HIT/TIMEOUT
- **Output:** Structured trade results with P&L, WR, and metrics
- **Status:** ❌ BROKEN (components misaligned)
- **Problem:** Executor runs when? Reporter when? How do they sync?

---

## ⚙️ PEC ARCHITECTURE (CURRENT BROKEN STATE)

### **Current (Broken) Design**

```
[SMARTFILTER DAEMON]
      ↓ (fires signal)
[SENT_SIGNALS.jsonl] ← Status: OPEN
      ↓
[PEC_EXECUTOR.py] ← When does this run? (Not clear!)
      ↓ (should mark: TP_HIT/SL_HIT/TIMEOUT)
[SIGNALS_MASTER.jsonl] ← Updated with exit data? (Inconsistent)
      ↓
[PEC_REPORTER.py] ← When does this run? (Hourly? When?)
      ↓
[Reports] ← Showing NEW signals as ZERO (Why?)
      ↓
[USER] ← "Why is everything broken?" ❌
```

### **Problems (Documented)**

| Component | Issue | Symptom |
|-----------|-------|---------|
| **Executor** | Runs sporadically, timing unknown | Signals stay OPEN indefinitely |
| **Reporter** | Reads stale data, filters don't work | NEW signals show as ZERO |
| **Sync** | Master file & executor updates not atomic | Foundation/NEW mismatch |
| **Timing** | UTC vs GMT+7 confusion | Data cross-contamination |
| **Files** | Multiple sources of truth (SENT_SIGNALS, MASTER, AUDIT) | Which one to trust? |

---

## ✅ PEC ARCHITECTURE (CORRECTED DESIGN)

### **Proposed Clean Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                   SMARTFILTER DAEMON                         │
│  (PROJECT-3: Fires signals continuously, 24/7)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓ (fires signal with entry/TP/SL)
┌─────────────────────────────────────────────────────────────┐
│              SIGNALS_MASTER.jsonl (ATOMIC)                   │
│  Single source of truth:                                     │
│  - Lines 1-2,540: FOUNDATION (baseline, immutable)           │
│  - Lines 2,541+: NEW (append-only, firing today)             │
│  - Status field: OPEN → TP_HIT/SL_HIT/TIMEOUT (updated by E) │
│  - fired_time_utc (naive, assumed UTC)                       │
│  - actual_exit_price (filled by executor)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
       ┌─────────────┴─────────────┐
       ↓                           ↓
   [EXECUTOR HOUR]           [REPORTER HOUR]
   (Every :00 GMT+7)         (Every :00 GMT+7)
   
   PEC_EXECUTOR                 PEC_REPORTER
   1. Read MASTER               1. Read MASTER
   2. Find OPEN signals         2. Split by fired_time
   3. Fetch OHLCV (KuCoin)      3. Calc stats
   4. Walk bars → mark close    4. Generate report
   5. Write exit_price          5. Save to file
   6. Write status (TP/SL/TO)   
   7. Commit atomically
```

### **Critical Rules**

1. **SIGNALS_MASTER.jsonl is the SINGLE SOURCE OF TRUTH**
   - Daemon APPENDS new signals only
   - Executor UPDATES status/exit_price fields only
   - Reporter READS snapshot only
   - Never delete, never reorder

2. **Executor and Reporter MUST use same file, same timestamp**
   - Both read MASTER at :00:00 GMT+7
   - Executor closes signals → updates MASTER
   - Reporter reads updated MASTER → generates report
   - NO stale data, NO cross-file confusion

3. **Timestamps are NAIVE (assumed UTC everywhere)**
   - fired_time_utc: "2026-03-20T16:57:35" (no TZ info)
   - All comparisons as naive datetimes
   - Convert to GMT+7 ONLY for display/user output
   - Never mix offset-aware + naive in same comparison

4. **Immutability boundaries are HARD**
   - FOUNDATION (2,540 signals): Locked, never changes
   - NEW (today's signals): Append-only until end of day
   - No updates to old data, ever
   - If executor needs to modify historical → FAIL & ALERT

---

## 🔄 HOURLY CYCLE (Synchronized)

### **Time: :00:00 GMT+7 (every hour)**

```
00:00:00 ┌─────────────────────────────────────┐
         │ [EXECUTOR START]                    │
         │ 1. Load SIGNALS_MASTER.jsonl        │
         │ 2. Find all OPEN signals            │
         │ 3. For each OPEN signal:            │
         │    a) Fetch symbol OHLCV (KuCoin)   │
         │    b) Walk candles from fire time   │
         │    c) Check if TP/SL/TIMEOUT hit    │
         │    d) Store exit_price              │
         │ 4. Update status in MASTER          │
         │ 5. Commit atomically                │
         └────────┬─────────────────────────────┘
                  │
                  ↓ (after exec finishes)
         ┌────────────────────────────────┐
         │ [REPORTER START]               │
         │ 1. Load SIGNALS_MASTER.jsonl   │
         │ 2. Read exec updates           │
         │ 3. Split: FOUNDATION + NEW     │
         │ 4. Calc WR/P&L for each        │
         │ 5. Save report snapshot        │
         │ 6. Send metrics to user        │
         └────────────────────────────────┘

00:05:00 Both complete, ready for next hour
```

### **Constraints**
- Executor timeout: 120s (KuCoin API fetch + walk all OPEN signals)
- Reporter timeout: 60s (calculation only, no API)
- If executor takes >120s: Skip report that hour, alert user
- If report fails: Retry once, log error, continue next hour

---

## 📊 DATA MODEL (Clean)

### **SIGNALS_MASTER.jsonl Structure**

```json
{
  "uuid": "b00b91d5-48dd-40a2-877d-f6b6249e76e2",
  "fired_time_utc": "2026-03-20T16:57:35.822737",
  "symbol": "BTC-USDT",
  "timeframe": "1h",
  "signal_type": "LONG",
  "route": "TREND_CONTINUATION",
  "regime": "BULL",
  
  "entry_price": 45000.50,
  "tp_target": 46500.00,
  "sl_target": 44000.00,
  
  "score": 15,
  "confidence": 78.5,
  "achieved_rr": 1.8,
  
  "status": "TP_HIT",
  "actual_exit_price": 46500.00,
  "exit_time_utc": "2026-03-20T18:30:45.123456",
  "exit_reason": "TP_HIT",
  
  "pnl_usd": 150.00,
  "pnl_pct": 2.0,
  
  "signal_origin": "NEW"
}
```

### **Key Fields**

| Field | Set By | Updated By | Immutable After |
|-------|--------|-----------|-----------------|
| uuid | Daemon | - | Creation |
| fired_time_utc | Daemon | - | Creation |
| symbol, timeframe, route, regime | Daemon | - | Creation |
| entry_price, tp_target, sl_target | Daemon | - | Creation |
| score, confidence, achieved_rr | Daemon | - | Creation |
| status | - | Executor | After executor runs |
| actual_exit_price | - | Executor | After executor runs |
| exit_time_utc | - | Executor | After executor runs |
| pnl_usd, pnl_pct | - | Executor | After executor runs |
| signal_origin | Daemon | - | Creation |

---

## 🔧 EXECUTOR LOGIC (Detailed)

### **Pseudocode: PEC_EXECUTOR.py**

```python
def pec_executor_hourly():
    """
    Process all OPEN signals in MASTER via OHLCV backtest
    """
    
    # 1. LOAD
    master = load_jsonl("SIGNALS_MASTER.jsonl")
    open_signals = [s for s in master if s['status'] == 'OPEN']
    
    if not open_signals:
        print("[EXECUTOR] No OPEN signals. Idle this hour.")
        return
    
    # 2. BATCH BY SYMBOL (avoid API rate limits)
    by_symbol = group_by_symbol(open_signals)
    
    for symbol in by_symbol:
        signals_for_symbol = by_symbol[symbol]
        
        # 3. FETCH OHLCV (once per symbol per hour)
        ohlcv = kucoin_fetch_ohlcv(symbol, "1m", lookback="24h")
        
        for signal in signals_for_symbol:
            # 4. WALK CANDLES FROM FIRE TIME
            fire_time = parse_naive_datetime(signal['fired_time_utc'])
            entry = signal['entry_price']
            tp = signal['tp_target']
            sl = signal['sl_target']
            direction = signal['signal_type']  # LONG or SHORT
            
            # Start from candle after fire_time
            start_idx = find_candle_index(ohlcv, fire_time)
            exit_price = None
            exit_time = None
            exit_reason = None
            
            # 5. WALK FORWARD THROUGH CANDLES
            for i in range(start_idx, len(ohlcv)):
                candle = ohlcv[i]  # [time, open, high, low, close, volume]
                high = candle['high']
                low = candle['low']
                close = candle['close']
                candle_time = candle['time']
                
                # Check TP HIT
                if direction == 'LONG':
                    if high >= tp:
                        exit_price = tp
                        exit_time = candle_time
                        exit_reason = 'TP_HIT'
                        break
                else:  # SHORT
                    if low <= tp:
                        exit_price = tp
                        exit_time = candle_time
                        exit_reason = 'TP_HIT'
                        break
                
                # Check SL HIT
                if direction == 'LONG':
                    if low <= sl:
                        exit_price = sl
                        exit_time = candle_time
                        exit_reason = 'SL_HIT'
                        break
                else:  # SHORT
                    if high >= sl:
                        exit_price = sl
                        exit_time = candle_time
                        exit_reason = 'SL_HIT'
                        break
            
            # 6. IF NOT HIT: TIMEOUT AT CLOSE OF TODAY
            if exit_price is None:
                exit_price = close  # Use today's close
                exit_time = candle_time
                exit_reason = 'TIMEOUT'
            
            # 7. CALC P&L
            if direction == 'LONG':
                pnl = (exit_price - entry)
            else:  # SHORT
                pnl = (entry - exit_price)
            
            pnl_pct = (pnl / entry) * 100
            
            # 8. UPDATE SIGNAL
            signal['status'] = exit_reason
            signal['actual_exit_price'] = exit_price
            signal['exit_time_utc'] = exit_time
            signal['pnl_usd'] = pnl
            signal['pnl_pct'] = pnl_pct
    
    # 9. WRITE BACK (ATOMIC)
    write_jsonl("SIGNALS_MASTER.jsonl", master, atomic=True)
    print(f"[EXECUTOR] Updated {len(open_signals)} signals. Master file committed.")
```

### **Why This Works**

- ✅ **Single pass:** Each symbol fetched once, all signals processed
- ✅ **Chronological:** Walks candles forward from fire time (realistic)
- ✅ **Fair:** Uses actual OHLCV, not assumptions
- ✅ **Atomic:** Single write at end (no partial updates)
- ✅ **Idempotent:** Running twice gives same results (safe to retry)

---

## 📈 REPORTER LOGIC (Detailed)

### **Pseudocode: PEC_REPORTER.py**

```python
def pec_reporter_hourly():
    """
    Generate metrics snapshot for FOUNDATION + NEW signals
    """
    
    # 1. LOAD & SPLIT
    master = load_jsonl("SIGNALS_MASTER.jsonl")
    
    cutoff = datetime(2026, 3, 19, 18, 3, 17, 605610)  # Naive UTC
    
    foundation = []
    new_signals = []
    
    for s in master:
        fired = datetime.fromisoformat(s['fired_time_utc'])  # Naive
        if fired <= cutoff:
            foundation.append(s)
        else:
            new_signals.append(s)
    
    # 2. CALC STATS (for each group)
    foundation_stats = calc_stats(foundation)
    new_stats = calc_stats(new_signals)
    
    # 3. GENERATE REPORT
    report = f"""
    ╔══════════════════════════════════════╗
    ║  PEC HOURLY REPORT - {now_gmt7()}    ║
    ╚══════════════════════════════════════╝
    
    📊 FOUNDATION (Baseline, Immutable)
    ────────────────────────────────────────
    Total: {foundation_stats['total']}
    Closed: {foundation_stats['closed']} ({foundation_stats['closed_pct']:.1f}%)
    WR: {foundation_stats['wr']:.1f}%
    P&L: ${foundation_stats['pnl']:+,.2f}
    
    📊 NEW (Signals fired after {cutoff})
    ────────────────────────────────────────
    Total: {new_stats['total']}
    Closed: {new_stats['closed']} ({new_stats['closed_pct']:.1f}%)
    Open: {new_stats['open']}
    WR: {new_stats['wr']:.1f}%
    P&L: ${new_stats['pnl']:+,.2f}
    
    ✅ Report generated successfully
    """
    
    # 4. SAVE SNAPSHOT
    timestamp = now_gmt7().strftime('%Y-%m-%d_%H-00')
    write(f"pec_hourly_reports/{timestamp}-report.txt", report)
    
    print(report)
```

### **Key Points**

- ✅ **Same file, same timestamp:** Both executor & reporter read MASTER at :00
- ✅ **No stale data:** Uses fresh executor output
- ✅ **Immutable baseline:** FOUNDATION never changes
- ✅ **Progressive tracking:** NEW signals gradually close over time

---

## 🚨 FAILURE MODES (Root Causes of Current Issues)

### **Failure Mode #1: "Executor not running"**
- **Symptom:** NEW signals stay OPEN indefinitely
- **Cause:** Executor doesn't run on schedule, OR crashes silently
- **Fix:** 
  - Add `pec_executor.py` to cron job (currently missing!)
  - Log executor start/end to file
  - Alert user if executor fails

### **Failure Mode #2: "NEW signals show as ZERO"**
- **Symptom:** Reporter SECTION 2 empty despite 1,600 new signals
- **Cause:** Datetime filtering broken (offset-naive vs offset-aware) OR executor never ran
- **Current Fix:** Already applied (datetime comparison fixed)
- **Deeper Fix:** Executor MUST run first to populate exit data

### **Failure Mode #3: "Foundation/NEW mismatch"**
- **Symptom:** FOUNDATION shows 2,540 signals but MASTER has 4,879
- **Cause:** Multiple files (SENT_SIGNALS vs MASTER vs AUDIT) with different data
- **Fix:** Single file SIGNALS_MASTER.jsonl is source of truth
- **Rule:** Ignore SENT_SIGNALS after migration

### **Failure Mode #4: "UTC vs GMT+7 confusion"**
- **Symptom:** Signals fire at time X but show in time Y
- **Cause:** Inconsistent timezone handling across daemon/executor/reporter
- **Fix:** 
  - ALL timestamps stored as naive (UTC assumed)
  - Conversion to GMT+7 ONLY for user display
  - Never mix timezones in comparisons

### **Failure Mode #5: "Back and forth fixes"**
- **Symptom:** Fix A breaks B, fix B breaks C
- **Cause:** Patching individual components without clear architecture
- **Fix:** This document. Clean architecture, clear contracts.

---

## 🎯 IMPLEMENTATION CHECKLIST (To Fix PEC)

- [ ] **Step 1:** Verify SIGNALS_MASTER.jsonl is single source of truth
- [ ] **Step 2:** Update cron job to run: Executor → Reporter (in sequence)
- [ ] **Step 3:** Add timeout handling (if executor >120s, skip report)
- [ ] **Step 4:** Add logging to executor (what signals processed, how many closed)
- [ ] **Step 5:** Add idempotency check to executor (don't double-update)
- [ ] **Step 6:** Verify NEW signals gradually close over subsequent hours
- [ ] **Step 7:** Monitor for 24h: Check that NEW section populates
- [ ] **Step 8:** Once stable, integrate with PROJECT-4 (feed results to bot)

---

## 📌 COMPARISON: What I Did vs What Should Have Been Done

### **What I Did (Patching)**
```
NEW signals show zero? → Fix reporter filtering
Executor not running? → Add executor to cron
Timezone mismatch? → Update datetime logic
Still broken? → Try again...
(repeat 5 times ❌)
```

### **What Should Be Done (Architecture First)**
```
1. Define single source of truth (SIGNALS_MASTER.jsonl)
2. Define hourly cycle (executor :00 → reporter :00)
3. Define data contracts (fields, formats, timezones)
4. Define failure handling (timeouts, retries, alerts)
5. Implement with clear contracts
6. Monitor + iterate
(success ✅)
```

---

## 🏁 CONCLUSION

**PEC is currently broken because:**
1. Multiple files (SENT_SIGNALS, MASTER, AUDIT) with conflicting data
2. Executor and reporter run on different schedules (or not at all)
3. Timezone mixing (UTC vs GMT+7 in comparisons)
4. No clear contracts or data flow

**To fix it:**
1. Use SIGNALS_MASTER.jsonl as single source of truth
2. Synchronize executor + reporter to run together hourly
3. Use naive datetimes consistently
4. Add logging to diagnose failures

**Once fixed, PEC integrates with:**
- PROJECT-3 (SmartFilter) as signal generator
- PROJECT-4 (Asterdex Bot) as trade executor
- PROJECT-1,2 as sibling systems

---

**Document Created:** 2026-03-21 00:06 GMT+7  
**Status:** Ready for implementation
