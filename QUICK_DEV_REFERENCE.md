# SmartFilter Local Dev - Quick Reference

## 🚀 Start/Restart in 2 Commands

```bash
cd ~/path/to/smart-filter-v14
./restart.sh
```

Done. Changes take effect immediately.

---

## 📊 Monitor in Real-Time

### Watch Signals Only (Clean)
```bash
tail -f smartfilter_run.log | grep "^\[SIGNAL\]\|^\[✅ SENT\]\|^\[⛔"
```

### Watch Everything
```bash
tail -f smartfilter_run.log
```

### Watch Cycle Summary
```bash
tail -f smartfilter_run.log | grep "CYCLE SUMMARY" -A 5
```

---

## ✏️ Efficient Workflow

**Terminal 1 (Watch Logs):**
```bash
tail -f smartfilter_run.log | grep "^\[SIGNAL\]\|SENT"
```

**Terminal 2 (Make Changes & Restart):**
```bash
nano main.py              # Edit code
./restart.sh              # Restart (2 seconds)
# See changes in Terminal 1 immediately!
```

---

## 🐛 Common Commands

**Check if running:**
```bash
ps aux | grep "python.*main.py" | grep -v grep
```

**Kill process:**
```bash
pkill -f "python.*main.py"
```

**View last 100 lines:**
```bash
tail -100 smartfilter_run.log
```

**Search logs:**
```bash
grep "BTC-USDT" smartfilter_run.log
grep "ERROR" smartfilter_run.log
```

---

## 📝 Test Cycle (Full)

```bash
./restart.sh
sleep 120  # Wait 2 minutes for signals
tail -50 smartfilter_run.log | grep "^\[SIGNAL\]"
```

---

## 🔧 Key Files to Edit

| File | Purpose |
|------|---------|
| `main.py` | Signal processing, logging |
| `smart_filter.py` | Filter logic, scoring |
| `signal_logger.py` | Output formatting |
| `tp_sl_retracement.py` | TP/SL calculation |
| `kucoin_data.py` | API data fetching |

---

## ⚡ When You Fix Something

1. Find the bug (read logs)
2. Edit the file
3. `./restart.sh`
4. Watch `tail -f smartfilter_run.log`
5. Verify fix works in next cycle (~1-2 min)

**Total time: ~3-5 minutes from diagnosis to verification**

---

## 🎯 Debug Mode

To see verbose filter logs (slow, spammy):
```bash
export DEBUG_FILTERS=true
./restart.sh
```

Turn off:
```bash
export DEBUG_FILTERS=false
./restart.sh
```

---

## 📦 Push to GitHub (When Done)

After testing locally:
```bash
git add main.py smart_filter.py signal_logger.py  # Your changes
git commit -m "FIX: description of what you fixed"
git push origin main
```

---

## 💡 Pro Tips

✅ Use 2 terminals: one for logs, one for coding  
✅ Restart is fast (<2 sec) — test frequently  
✅ Keep `tail -f` running — catch bugs immediately  
✅ Git push only when tested & working locally  
✅ Edit one file at a time — easier to debug  

---

**Questions?** Check `RUN_LOCAL_SETUP.md` for detailed setup.
