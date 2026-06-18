"""
TODAY'S LOCKED ALLOWABLE COMBOS - June 18, 2026
Generated: 2026-06-18 19:19:50 GMT+7
Source Report: PEC_POST_DEPLOYMENT_TRACKER_v2_2026-06-17_23-44-14.txt

1 COMBOS - LOCKED FOR THE DAY
Tier-1: 1 combos (6D only)
Tier-2: 0 combos (6D, 5D)
Tier-3: 0 combos (6D, 5D, 4D)

main.py should read from this file ONLY via get_locked_combos()
"""

# DATETIME VALIDATION: Ensures main.py is using TODAY's combos only
GENERATED_DATETIME = "2026-06-18 19:19:50 GMT+7"
GENERATED_DATE = "2026-06-18"  # YYYY-MM-DD for date comparison
LOCK_EXPIRES_DATE = "2026-06-19"  # Expires at midnight GMT+7

LOCKED_COMBOS = [
    {"combo": "TF_DIR_ROUTE_REGIME_SG_CONF_30min_SHORT_TREND CONTINUATION_BEAR_LOW_ALTS_HIGH", "tier": "Tier-1", "dimension": "6D"},
]

def get_locked_combos():
    """Return the locked combos for today"""
    return LOCKED_COMBOS
