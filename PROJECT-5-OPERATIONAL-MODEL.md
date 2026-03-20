# PROJECT-5: PEC - Operational Model (How to Keep Executor & Reporter Running)

**Question:** How to make executor and reporter always ON?  
**Asked:** 2026-03-21 01:14 GMT+7  
**Status:** Discussing options, awaiting user decision

---

## 🎯 **What Does "Always ON" Mean?**

Three interpretations:

1. **Continuously monitoring** - Executor checks for OPEN signals constantly, backtests whenever possible
2. **Always responsive** - When daemon fires signal, executor backtests immediately (no waiting for cron)
3. **Always available** - Reporter generates metrics whenever requested (on-demand)

---

## 🔧 **OPTION 1: Cron (Current Model)**

**How it works:**
```
Every hour at :00 GMT+7:
├─ Cron triggers executor
│  ├─ Read AUDIT (all OPEN signals)
│  ├─ Backtest against OHLCV
│  ├─ Append CLOSURE remarks to AUDIT
│  └─ Update SIGNALS_MASTER.jsonl
├─ Then cron triggers reporter
│  ├─ Read AUDIT + MASTER
│  ├─ Calculate metrics
│  └─ Generate report
└─ Both finish, wait 1 hour for next cycle
```

**Pros:**
- ✅ Simple to implement (one cron job)
- ✅ Predictable timing (exact hour)
- ✅ Low resource usage (runs once/hour)
- ✅ Easy to debug (scheduled, repeatable)
- ✅ Current model (works)

**Cons:**
- ❌ 1-hour delay between signal fire and backtest
- ❌ If executor slow, reporter delayed
- ❌ If executor crashes, manual restart needed

**Call from main.py?** 
NO - Executor runs separately, doesn't need main.py (daemon)

---

## 🎯 **OPTION 2: Event-Driven (Trigger on Signal Fire)**

**How it works:**
```
Daemon fires signal:
├─ Append FIRED line to AUDIT + MASTER
├─ Send Telegram
└─ IMMEDIATELY trigger executor
   ├─ Find newly-fired signal
   ├─ Backtest if OHLCV available
   ├─ Append CLOSURE remark if closed
   └─ Reporter auto-generates latest report

Result: Backtest happens in seconds, not 1 hour
```

**Implementation:**
```python
# In main.py (daemon):
def fire_signal(signal):
    # ... existing code ...
    write_to_audit(signal)
    write_to_master(signal)
    send_telegram(signal)
    
    # NEW: Trigger executor immediately
    trigger_executor(signal['signal_uuid'])
```

**Pros:**
- ✅ Real-time backtest (seconds, not 1 hour)
- ✅ Responsive (no wait for cron)
- ✅ Can integrate into main.py easily
- ✅ Better user experience

**Cons:**
- ❌ More complex (need signal communication)
- ❌ Race condition risk (daemon + executor both touching files)
- ❌ Higher CPU usage (backtest on every signal)
- ❌ OHLCV may not be available immediately after fire

**Call from main.py?**
YES - Can call executor function directly or spawn subprocess

---

## 🚀 **OPTION 3: Long-Running Executor Daemon**

**How it works:**
```
Start two separate processes:

Process 1: DAEMON (main.py - PROJECT-3)
├─ Continuously analyze and fire signals
├─ Append to AUDIT + MASTER
└─ Send Telegram alerts

Process 2: EXECUTOR_DAEMON (separate, always running)
├─ Continuously monitor for OPEN signals
├─ Check SIGNALS_MASTER.jsonl every 5 seconds
├─ If OPEN signals exist:
│  ├─ Backtest (fetch OHLCV, walk candles)
│  ├─ Append CLOSURE remarks to AUDIT
│  └─ Update SIGNALS_MASTER.jsonl
└─ Repeat forever

Process 3: REPORTER_DAEMON (optional, separate, always running)
├─ Generate metrics every 15-60 seconds
├─ Read AUDIT + MASTER
└─ Write to report file (on-demand served)
```

**Pros:**
- ✅ Truly "always ON" (never sleeps)
- ✅ Real-time execution (instant backtest)
- ✅ Independent from daemon
- ✅ Can run on separate machine if needed

**Cons:**
- ❌ More complex (manage multiple processes)
- ❌ Higher resource usage (always polling)
- ❌ Rate limiting issues (constant OHLCV fetches)
- ❌ Need process monitoring (auto-restart if crash)

**Call from main.py?**
OPTIONAL - Can be separate standalone script, but could share same process

---

## 🔌 **OPTION 4: Hybrid (Cron + Event-Triggered)**

**How it works:**
```
Best of both worlds:

Primary: Event-triggered (when signal fires)
├─ Main.py triggers executor immediately
└─ Fast backtest for signals that close quickly

Fallback: Hourly cron check
├─ Catch any signals executor missed
├─ Handle edge cases
└─ Guaranteed catch-up every hour

Result: Real-time for most, guaranteed 1-hour max for stragglers
```

**Implementation:**
```python
# In main.py:
def fire_signal(signal):
    # ... write to AUDIT/MASTER ...
    try:
        executor.backtest_signal(signal['signal_uuid'])
    except Exception as e:
        logger.error(f"Executor failed: {e}")
        # Continue, cron will catch it

# Plus: Keep hourly cron as backup
```

---

## 📊 **Comparison Table**

| Aspect | Cron | Event-Driven | Daemon | Hybrid |
|--------|------|--------------|--------|--------|
| **Responsiveness** | 1 hour delay | Seconds | Instant | Seconds |
| **Complexity** | Simple | Medium | Complex | Medium |
| **Resource Usage** | Low | Medium | High | Medium |
| **Integration** | Separate | With main.py | Standalone | Both |
| **Reliability** | Good | Medium | Excellent | Excellent |
| **Crash Recovery** | Manual | Manual | Auto-restart | Manual + Cron |
| **Cost** | Low | Low-Medium | Medium | Medium |

---

## 🎯 **RECOMMENDATION FOR PEC**

Given PEC requirements:
- Backtest needs OHLCV data (fetching takes seconds)
- Can't backtest signal that just fired (no historical bars yet)
- 1-hour latency is acceptable (signal fired hours ago)

**Recommended: OPTION 4 (Hybrid)**

```
primary: Event-triggered for responsiveness
├─ When daemon fires signal, trigger executor
├─ Backtest runs in background (non-blocking)
└─ Completes in 5-30 seconds (depends on OHLCV fetch)

Fallback: Hourly cron for catch-all
├─ Guaranteed executor run every hour
├─ Catches any missed signals
└─ Safety net if event-trigger fails
```

**Implementation flow:**
```python
# main.py (daemon)
def fire_signal(signal):
    append_to_audit(signal)
    append_to_master(signal)
    send_telegram(signal)
    
    # Trigger executor in background (non-blocking)
    import subprocess
    subprocess.Popen(['python3', 'pec_executor.py', '--signal-uuid', signal['signal_uuid']])

# pec_executor.py (can be called directly or via cron)
def main(signal_uuid=None):
    if signal_uuid:
        # Event-triggered: backtest single signal
        backtest_signal(signal_uuid)
    else:
        # Cron-triggered: backtest all OPEN
        backtest_all_open()

# Cron job (hourly fallback)
# 0 * * * * cd /workspace && python3 pec_executor.py
```

---

## ❓ **YOUR DECISION**

Which operational model do you want?

1. **Keep Cron Only** (Simple, current model)
2. **Event-Driven** (Real-time, integrated with main.py)
3. **Long-Running Daemon** (True "always ON")
4. **Hybrid** (Best of both - recommended)
5. **Something else?**

Let me know and I can detail the implementation.
