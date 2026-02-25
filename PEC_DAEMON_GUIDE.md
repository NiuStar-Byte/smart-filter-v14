# PEC Daemon Guide - Auto-Update Every 5 Minutes

## What It Does

✅ Runs **silently in background** every 5 minutes  
✅ Scans SENT_SIGNALS.jsonl for OPEN signals  
✅ Checks current prices via public KuCoin API  
✅ Updates signal status: OPEN → TP_HIT / SL_HIT / TIMEOUT  
✅ **Announces ONLY when signals hit targets** (no spam)  

## Quick Commands

```bash
# Start daemon (runs in background)
./pec_control.sh start

# Check status (show if running, last 10 lines of log)
./pec_control.sh status

# Watch live updates
./pec_control.sh watch

# Restart daemon
./pec_control.sh restart

# Stop daemon
./pec_control.sh stop
```

## What You'll See

**Silent operation** (every 5 min):
- Checks 34 open signals
- Updates SENT_SIGNALS.jsonl
- Logs nothing (working)

**When signals hit targets**, you see:

```
================================================================================
[PEC UPDATE] 2026-02-25 13:34:58
================================================================================

✅ TP HIT (4 signals):
   NMR-USDT     30min  | P&L: $    6.70 (+1.14%)
   NMR-USDT     30min  | P&L: $    7.76 (+1.33%)
   NMR-USDT     30min  | P&L: $    7.61 (+1.30%)
   NMR-USDT     30min  | P&L: $    8.12 (+1.39%)

❌ SL HIT (4 signals):
   TUT-USDT     15min  | P&L: $   -0.00 (-0.61%)
   TUT-USDT     15min  | P&L: $   -0.00 (-0.51%)
   TUT-USDT     15min  | P&L: $   -0.00 (-0.61%)
   TUT-USDT     15min  | P&L: $   -0.00 (-0.61%)

📊 PEC STATS:
   Total: 42 | Open: 34 | TP: 4 | SL: 4 | Timeout: 0
   Win Rate: 100.0% | Total P&L: $+30.19
================================================================================
```

## File Structure

```
smart-filter-v14-main/
├── pec_daemon.py          ← Main loop (5-min updates)
├── pec_executor.py        ← Logic (check TP/SL, calculate P&L)
├── pec_control.sh         ← Control script (start/stop/status)
├── pec_runner.sh          ← Runner for cron (fallback)
├── SENT_SIGNALS.jsonl     ← Telegram-sent signals + execution data
└── pec_daemon.log         ← Daemon output log
```

## How It Tracks Signals

**SENT_SIGNALS.jsonl** (one signal per line):
```json
{
  "uuid": "abc123...",
  "symbol": "BTC-USDT",
  "timeframe": "15min",
  "signal_type": "LONG",
  "entry_price": 65611.15,
  "tp_target": 66200,
  "sl_target": 65000,
  "fired_time_utc": "2026-02-25T13:26:45Z",
  "status": "OPEN",
  "closed_at": null,
  "actual_exit_price": null,
  "pnl_usd": null,
  "pnl_pct": null
}
```

When TP hits, updates to:
```json
{
  "status": "TP_HIT",
  "closed_at": "2026-02-25T13:27:30Z",
  "actual_exit_price": 66250,
  "pnl_usd": 639.13,
  "pnl_pct": +0.98
}
```

## Credit Cost

**$0** — Uses **public KuCoin API** (no authentication required)
- Every 5 minutes: $0
- Running 24/7: $0
- 288 updates/day: $0

## Stats Available

After running for a while, check:
```bash
python3 -c "from pec_executor import PECExecutor; e = PECExecutor(); print(e.get_stats())"
```

Output:
```python
{
  'total': 42,
  'open': 34,
  'tp_hit': 4,
  'sl_hit': 4,
  'timeout': 0,
  'total_pnl_usd': 30.19,
  'win_rate_pct': 50.0
}
```

## Troubleshooting

**Daemon not starting?**
```bash
# Check if already running
ps aux | grep pec_daemon

# Kill any stuck processes
pkill -9 -f pec_daemon.py

# Try starting again
./pec_control.sh start
```

**Check for errors:**
```bash
tail -50 pec_daemon.log | grep ERROR
```

**Watch in real-time:**
```bash
./pec_control.sh watch
```

## Integration with SmartFilter

1. ✅ SmartFilter generates signals → SENT_SIGNALS.jsonl
2. ✅ PEC daemon monitors every 5 min
3. ✅ Updates execution status (TP/SL/TIMEOUT)
4. ✅ Calculates P&L automatically
5. ✅ Announces changes only when they happen

**Result:** Zero manual tracking, automatic P&L accounting.
