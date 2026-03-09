# 📋 CHECKPOINT TEMPLATE (Always Include These Sections)

## REQUIRED: Signal File Sync Comparison

**ALWAYS include in every checkpoint:**

```markdown
### Signal File Sync (MASTER vs AUDIT)
- **SIGNALS_MASTER.jsonl:** [X] lines (daemon primary source)
- **SIGNALS_INDEPENDENT_AUDIT.txt:** [Y] lines (audit trail)
- **Divergence:** [Z] signals (difference)
- **Lag Type:** ⚠️ [description of lag/divergence]
  - If AUDIT > MASTER: "AUDIT catching up from earlier"
  - If MASTER > AUDIT: "MASTER ahead, normal flow"
  - If difference = 0: "✅ Perfectly synced"
- **Status:** ✅ Acceptable / ⚠️ Monitor / 🔴 Critical
- **Trend:** Converging / Stable / Diverging
- **Action:** [If needed, document what to do]
```

---

## Comparison Command (Copy/Paste)

```bash
# Quick sync check
echo "MASTER: $(wc -l < SIGNALS_MASTER.jsonl) lines"
echo "AUDIT: $(wc -l < SIGNALS_INDEPENDENT_AUDIT.txt) lines"
echo "Extra in AUDIT: $(comm -13 <(jq -r '.signal_uuid' SIGNALS_MASTER.jsonl | sort) <(jq -r '.signal_uuid' SIGNALS_INDEPENDENT_AUDIT.txt | sort) | wc -l)"
```

---

## Interpretation Guide

| Scenario | Divergence | Status | Action |
|----------|-----------|--------|--------|
| AUDIT = MASTER | 0 | ✅ Perfect | None needed |
| AUDIT > MASTER (1-10) | +1 to +10 | ✅ Normal | Monitor, will converge |
| AUDIT > MASTER (11+) | +11+ | ⚠️ Watch | Check if MASTER is stuck |
| MASTER > AUDIT | Any | 🔴 Critical | AUDIT file corruption risk |
| Duplicate UUIDs | Any | 🔴 Critical | Dedup logic failure |

---

## What This Tracks

**SIGNALS_MASTER.jsonl:**
- Primary daemon output
- Source of truth for signal generation
- Written immediately after Telegram send

**SIGNALS_INDEPENDENT_AUDIT.txt:**
- Backup audit trail
- Verification checkpoint
- May lag slightly (async update)

**Acceptable Lag:** 1-8 signals (normal accumulation delay)  
**Warning Threshold:** >15 signals divergence (investigate)  
**Critical:** MASTER diverging from AUDIT (data loss risk)

---

## When to Include

✅ Every checkpoint (daily/weekly)  
✅ After major changes (filter update, symbol add, etc.)  
✅ If divergence is >0 (always explain why)  
✅ When troubleshooting signal issues

---

## Template Placement

Put this section in the **Signal Generation / Monitoring** area:

```markdown
## 📊 CURRENT OPERATIONAL STATUS

### Daemon
- PID: [X]
- ...

### Signal Generation
- **Daily Snapshot:** [X] signals
- **Closed:** [Y]
- **Open:** [Z]
- **Hour Rate:** [~A signals/hour]

### Signal File Sync (MASTER vs AUDIT)  ← ADD HERE
- **SIGNALS_MASTER.jsonl:** [X] lines
- **SIGNALS_INDEPENDENT_AUDIT.txt:** [Y] lines
- **Divergence:** [Z] signals
- **Status:** ✅ Acceptable
```

---

## Example (Good)
```
SIGNALS_MASTER.jsonl: 2,346 lines
SIGNALS_INDEPENDENT_AUDIT.txt: 2,354 lines
Divergence: 8 signals (AUDIT ahead)
Status: ✅ Acceptable (normal lag)
```

## Example (Bad - Needs Investigation)
```
SIGNALS_MASTER.jsonl: 2,346 lines
SIGNALS_INDEPENDENT_AUDIT.txt: 2,346 lines
Divergence: 0 signals
Status: ⚠️ Monitor (may indicate MASTER stopped writing?)
```

---

**Maintained by:** Nox  
**Last Updated:** 2026-03-09 23:03 GMT+7  
**Frequency:** Every checkpoint from now on
