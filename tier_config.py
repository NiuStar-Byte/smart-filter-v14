"""
Tier Configuration - Dynamic Thresholds (Version D - OPTIMIZED FOR BEAR MARKET)

Optimized thresholds based on actual BEAR market performance data:
- Tier-1: 57.7% WR confirmed achievable (78 trades, +$678.09)
- Tier-2: 50% WR target (realistic midpoint)
- Tier-3: 45% WR target (above baseline 35.55%)

Updated: 2026-04-04 23:42 GMT+7
"""

# === VERSION D (BEAR MARKET OPTIMIZED) ===
TIER_THRESHOLDS = {
    # Tier-1: Elite (55%+ WR, $4.50+ avg, 55+ trades) - ACHIEVABLE target
    "tier1_wr": 0.55,
    "tier1_pnl": 4.50,
    "tier1_min_trades": 55,
    
    # Tier-2: Good (50%+ WR, $3.50+ avg, 50+ trades) - REALISTIC midpoint
    "tier2_wr": 0.50,
    "tier2_pnl": 3.50,
    "tier2_min_trades": 50,
    
    # Tier-3: Acceptable (45%+ WR, $2.50+ avg, 45+ trades) - ABOVE BASELINE
    "tier3_wr": 0.45,
    "tier3_pnl": 2.50,
    "tier3_min_trades": 45,
    
    # Tier-X: Below minimum (hidden from Telegram)
    "tierx_threshold": 0.45,  # WR below this or negative P&L
}


def get_tier_thresholds():
    """Get current tier thresholds"""
    return TIER_THRESHOLDS.copy()
