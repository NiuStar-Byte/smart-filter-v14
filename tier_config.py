"""
Tier Configuration - Dynamic Thresholds (Production Version B - AGREED)

Manual control over tier criteria. Tags appear on Telegram only when combos meet thresholds.

Active Since: 2026-02-28 11:45 GMT+7
"""

# === VERSION B (AGREED - PRODUCTION) ===
TIER_THRESHOLDS = {
    "min_trades": 25,         # Minimum closed trades to qualify for tier evaluation
    "tier1_wr": 0.60,         # 60%+ win rate for Tier-1
    "tier1_pnl": 5.0,         # $5.00+ avg P&L per trade for Tier-1
    "tier2_wr_min": 0.40,     # 40%+ win rate for Tier-2
    "tier2_pnl": 2.0,         # $2.00+ avg P&L per trade for Tier-2
    "tier3_min_pnl": 0.00,    # Tier-3: positive avg P&L, below Tier-2 threshold
}


def get_tier_thresholds():
    """Get current tier thresholds"""
    return TIER_THRESHOLDS.copy()
