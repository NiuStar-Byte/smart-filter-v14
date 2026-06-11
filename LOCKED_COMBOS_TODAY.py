"""
TODAY'S LOCKED ALLOWABLE COMBOS - June 12, 2026
Generated: 2026-06-12 00:14:26 GMT+7
Source Report: PEC_POST_DEPLOYMENT_TRACKER_v2_2026-06-11_23-08-27.txt

10 COMBOS - LOCKED FOR THE DAY
Tier-1: 2 combos (6D only)
Tier-2: 4 combos (6D, 5D)
Tier-3: 4 combos (6D, 5D, 4D)

main.py should read from this file ONLY via get_locked_combos()
"""

# DATETIME VALIDATION: Ensures main.py is using TODAY's combos only
GENERATED_DATETIME = "2026-06-12 00:14:26 GMT+7"
GENERATED_DATE = "2026-06-12"  # YYYY-MM-DD for date comparison
LOCK_EXPIRES_DATE = "2026-06-13"  # Expires at midnight GMT+7

LOCKED_COMBOS = [
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_LONG_TREND CONTINUATION_BULL_MID_ALTS_HIGH", "tier": "Tier-1", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_2h_SHORT_TREND CONTINUATION_BEAR_TOP_ALTS_MID", "tier": "Tier-1", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_15min_LONG_TREND CONTINUATION_RANGE_MID_ALTS_LOW", "tier": "Tier-2", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_LONG_TREND CONTINUATION_RANGE_MAIN_BLOCKCHAIN_MID", "tier": "Tier-2", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_LONG_TREND CONTINUATION_BULL_MAIN_BLOCKCHAIN_MID", "tier": "Tier-2", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_LONG_REVERSAL_BEAR_LOW_ALTS_LOW", "tier": "Tier-2", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_SHORT_REVERSAL_RANGE_MAIN_BLOCKCHAIN_LOW", "tier": "Tier-3", "dimension": "6D"},
    {"combo": "TF_DIR_ROUTE_REGIME_4h_SHORT_AMBIGUOUS_BEAR", "tier": "Tier-3", "dimension": "4D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_15min_LONG_TREND CONTINUATION_RANGE_MID_ALTS", "tier": "Tier-3", "dimension": "5D"},
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_4h_SHORT_TREND CONTINUATION_BULL_MAIN_BLOCKCHAIN_MID", "tier": "Tier-3", "dimension": "6D"},
]

def get_locked_combos():
    """Return the locked combos for today"""
    return LOCKED_COMBOS
