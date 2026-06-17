"""
TODAY'S LOCKED ALLOWABLE COMBOS - June 17, 2026
Generated: 2026-06-17 07:03:05 GMT+7
Source Report: PEC_POST_DEPLOYMENT_TRACKER_v2_2026-06-16_22-56-41.txt

12 COMBOS - LOCKED FOR THE DAY
Tier-1: 4 combos (6D only)
Tier-2: 4 combos (6D, 5D)
Tier-3: 4 combos (6D, 5D, 4D)

main.py should read from this file ONLY via get_locked_combos()
"""

# DATETIME VALIDATION: Ensures main.py is using TODAY's combos only
GENERATED_DATETIME = "2026-06-17 07:03:05 GMT+7"
GENERATED_DATE = "2026-06-17"  # YYYY-MM-DD for date comparison
LOCK_EXPIRES_DATE = "2026-06-18"  # Expires at midnight GMT+7

LOCKED_COMBOS = [
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_2h_LONG_TREND CONTINUATION_BEAR_LOW_ALTS_HIGH", "tier": "Tier-1", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_1h_LONG_TREND CONTINUATION_BULL_LOW_ALTS_HIGH", "tier": "Tier-1", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_SHORT_TREND CONTINUATION_BEAR_LOW_ALTS_HIGH", "tier": "Tier-1", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_30min_LONG_TREND CONTINUATION_BULL_LOW_ALTS_HIGH", "tier": "Tier-1", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_30min_LONG_TREND CONTINUATION_BULL_LOW_ALTS", "tier": "Tier-2", "dimension": "5D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_2h_LONG_TREND CONTINUATION_BULL_LOW_ALTS_HIGH", "tier": "Tier-2", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_2h_LONG_TREND CONTINUATION_BULL_LOW_ALTS", "tier": "Tier-2", "dimension": "5D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_LONG_TREND CONTINUATION_BULL_LOW_ALTS_HIGH", "tier": "Tier-2", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_4h_LONG_TREND CONTINUATION_BULL_LOW_ALTS", "tier": "Tier-3", "dimension": "5D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_30min_SHORT_TREND CONTINUATION_BEAR_LOW_ALTS_HIGH", "tier": "Tier-3", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_30min_SHORT_TREND CONTINUATION_BEAR_LOW_ALTS", "tier": "Tier-3", "dimension": "5D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_LONG_REVERSAL_BEAR_LOW_ALTS_HIGH", "tier": "Tier-3", "dimension": "6D"},
]

def get_locked_combos():
    """Return the locked combos for today"""
    return LOCKED_COMBOS
