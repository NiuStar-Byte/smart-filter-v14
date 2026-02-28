"""
Tier Configuration - Dynamic Thresholds
Manual control over tier criteria

VERSION A (LOOSE) - TEMPORARY FOR TELEGRAM PROOF
Min Trades: 5 (super low for fast population)
Tier-1: WR≥30% + P&L≥$0.50
Tier-2: WR≥20% + P&L≥$0.20
Tier-3: Anything positive

After demo complete, switch to VERSION B (AGREED)
"""

# === VERSION A (LOOSE - TEMPORARY FOR DEMO) ===
# TIER_THRESHOLDS = {
#     "min_trades": 5,              # Super low - populates fast
#     "tier1_wr": 0.30,             # 30% WR (demo only, normally 60%)
#     "tier1_pnl": 0.50,            # $0.50 (demo only, normally $5)
#     "tier2_wr_min": 0.20,         # 20%+ WR (demo only, normally 40%)
#     "tier2_pnl": 0.20,            # $0.20 (demo only, normally $2)
#     "tier3_min_pnl": 0.00,        # Anything positive = Tier-3
# }

# === VERSION B (AGREED - PRODUCTION) ===
# Active as of 2026-02-28 11:45 GMT+7
TIER_THRESHOLDS = {
    "min_trades": 25,
    "tier1_wr": 0.60,
    "tier1_pnl": 5.0,
    "tier2_wr_min": 0.40,
    "tier2_pnl": 2.0,
    "tier3_min_pnl": 0.00,
}

# === VERSION C (FINAL - STRICT) ===
# For future use (Day 5-7):
# TIER_THRESHOLDS = {
#     "min_trades": 100,
#     "tier1_wr": 0.65,
#     "tier1_pnl": 8.0,
#     "tier2_wr_min": 0.40,
#     "tier2_pnl": 3.0,
#     "tier3_min_pnl": 0.00,
# }


def get_tier_thresholds():
    """Get current tier thresholds"""
    return TIER_THRESHOLDS.copy()
