"""
Tier Configuration - Dynamic Thresholds (Version C - CONSENSUS CASCADE + OPTION B)

Consensus Cascade: Evaluates 5D → 4D → 3D → 2D, assigns tier at finest qualified level
Option B: Graduated criteria with trade volume gates

Active Since: 2026-03-01 22:31 GMT+7
"""

# === VERSION C (CONSENSUS CASCADE + OPTION B) ===
TIER_THRESHOLDS = {
    # Tier-1: Elite (60% WR, $5.50+ avg, 60+ trades)
    "tier1_wr": 0.60,
    "tier1_pnl": 5.50,
    "tier1_min_trades": 60,
    
    # Tier-2: Good (50% WR, $3.50+ avg, 50+ trades)
    "tier2_wr": 0.50,
    "tier2_pnl": 3.50,
    "tier2_min_trades": 50,
    
    # Tier-3: Acceptable (40% WR, $2.00+ avg, 40+ trades)
    "tier3_wr": 0.40,
    "tier3_pnl": 2.00,
    "tier3_min_trades": 40,
    
    # Tier-X: Below minimum (hidden from Telegram)
    "tierx_threshold": 0.40,  # WR below this or negative P&L
}


def get_tier_thresholds():
    """Get current tier thresholds"""
    return TIER_THRESHOLDS.copy()
