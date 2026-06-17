# mtf_alignment_config.py
# Multi-Timeframe (MTF) Alignment Configuration
# All parameters here - changeable without code modification
# Last updated: 2026-04-07

import os

# ============================================
# FEATURE TOGGLE
# ============================================
MTF_ALIGNMENT_ENABLED = True  # Set to False to disable feature entirely

# ============================================
# ALIGNMENT SCORING WEIGHTS
# ============================================
# How to weight direction vs regime vs route match
# Sum should equal 1.0 (100%)
ALIGNMENT_WEIGHTS = {
    'direction': 0.40,      # LONG/SHORT match importance
    'regime': 0.35,         # BULL/BEAR match importance
    'route': 0.25           # TREND/REVERSAL match importance
}

# ============================================
# CONFIDENCE ADJUSTMENT MULTIPLIERS (V2 ENHANCED)
# ============================================
# Applied to base_confidence based on alignment_score band
# Format: alignment_score >= threshold → multiply confidence by factor
# V2: Uses (strong, weak, conflict, neutral) - NO 'partial'
CONFIDENCE_ADJUSTMENTS = {
    'strong': 1.25,         # 75+ alignment → boost confidence by 25% (1.25x)
    'weak': 0.90,           # 50-74 alignment → reduce confidence by 10% (0.90x)
    'conflict': 0.60,       # 1-49 alignment → reduce confidence by 40% (0.60x)
    'neutral': 1.0          # 0 or no data → no adjustment (1.0x)
}

# ============================================
# ALIGNMENT THRESHOLD BANDS (V2 ENHANCED)
# ============================================
# Define which band an alignment_score falls into
# Score 0-100, bands are: conflict, weak, strong, neutral
# V2: NO 'partial' - 'weak' (50-74) replaces old 'partial' (50-79)
ALIGNMENT_THRESHOLDS = {
    'strong': 75,           # 75-100: Strong alignment (full confirmation)
    'weak': 50,             # 50-74: Weak alignment (partial match)
    # 1-49: Conflict (opposing signals)
    # 0: Neutral (no data)
}

# ============================================
# CONFLICT HARD GATE (2026-04-11 IMPROVEMENT)
# ============================================
# Reject signals with conflicting TF alignment (counter-trend risk)
# NEW: Implemented to remove worst-performing signals
CONFLICT_HARD_GATE_ENABLED = True      # Enable/disable hard gate for conflict signals
CONFLICT_REJECTION_THRESHOLD = 30      # alignment_score < 30 = REJECT (conflict band)
CONFLICT_GATE_LABEL = "MTF Conflict Gate"
CONFLICT_GATE_REASON = "Counter-trend signal rejected: Lower TFs show opposing direction/regime"

# ============================================
# ALIGNMENT BAND LABELS & ICONS (V2 ENHANCED)
# ============================================
# V2 ICONS: 💪=strong, 📉=weak, ⚔️=conflict, ➖=neutral
ALIGNMENT_LABELS = {
    'strong': {
        'label': 'Strong Alignment',
        'icon': '💪',  # V2 enhanced icon (was 🔗)
        'color': '✅',
        'description': 'All TFs aligned (direction + regime + route)'
    },
    'weak': {
        'label': 'Weak Alignment',
        'icon': '📉',  # V2 enhanced icon (was ❔)
        'color': '⚠️',
        'description': 'Partial TF alignment (some match, some differ)'
    },
    'conflict': {
        'label': 'Counter-Trend Risk',
        'icon': '⚔️',  # V2 enhanced icon (was ⛔)
        'color': '❌',
        'description': 'Conflicting TF signals (counter-trend entry risk)'
    },
    'neutral': {
        'label': 'No Alignment Data',
        'icon': '➖',  # V2 enhanced icon (new)
        'color': '⚪',
        'description': 'Insufficient data for MTF alignment check'
    }
}

# ============================================
# TIMEFRAME CASCADE RULES (V2 ENHANCED)
# ============================================
# Which TFs to check for each TF that generates a signal
# Format: 'current_tf': ['tf_to_check_1', 'tf_to_check_2']
# V2: Added 1d check for 4h signals (daily MTF confirmation)
TF_CHECK_CASCADE = {
    '15min': ['30min', '1h'],       # 15min signal checks 30min + 1h
    '30min': ['1h', '2h'],          # 30min signal checks 1h + 2h
    '1h': ['2h', '4h'],             # 1h signal checks 2h + 4h
    '2h': ['4h'],                   # 2h signal checks 4h
    '4h': ['1d', '1w']              # 4h checks 1d (daily) + 1w (weekly) for stronger signal confirmation
}

# ============================================
# LOOKBACK WINDOW
# ============================================
# For each checked TF, which candle to use for alignment
# 'current': Use latest (possibly incomplete) candle
# 'completed': Wait for completed candle (but introduces latency)
LOOKBACK_MODE = 'current'  # Use latest candle (real-time approach)

# ============================================
# MATCH SCORING
# ============================================
# How to score each dimension match (0.0 = no match, 0.5 = partial, 1.0 = match)
MATCH_SCORES = {
    'full_match': 1.0,      # Direction/regime/route exactly match
    'partial_match': 0.5,   # One TF waiting/neutral, other has signal
    'no_match': 0.0,        # Opposite directions/regimes
    'unknown': 0.25         # One TF has no data available
}

# ============================================
# MONTHLY CALIBRATION TARGETS (V2 ENHANCED)
# ============================================
# Used for verifying that adjustment multipliers are working correctly
# V2: Only strong, weak, conflict, neutral (no partial)
# Will be checked during MTF_alignment_comparison_tracker runs
CALIBRATION_TARGETS = {
    'strong': {
        'min_wr': 0.65,     # Expect strong alignment signals to have 65%+ WR (improved target)
        'adjustment': 1.25
    },
    'weak': {
        'min_wr': 0.50,     # Expect weak alignment signals to have 50%+ WR (was partial 55%)
        'adjustment': 0.90
    },
    'conflict': {
        'max_wr': 0.40,     # Expect conflict signals to have ≤40% WR (tighter gate)
        'adjustment': 0.60
    },
    'neutral': {
        'min_wr': 0.45,     # Expect neutral (no data) signals to have 45%+ WR baseline
        'adjustment': 1.0
    }
}

# ============================================
# DATA STORAGE
# ============================================
# Use absolute path to workspace root for consistency across all deployments (ROOT + BACKUP)
_WORKSPACE_ROOT = '/Users/geniustarigan/.openclaw/workspace'
MTF_ALIGNMENT_RESULTS_FILE = os.path.join(_WORKSPACE_ROOT, 'MTF_ALIGNMENT_RESULTS.jsonl')
MTF_ALIGNMENT_DAILY_REPORT_FILE = os.path.join(_WORKSPACE_ROOT, 'MTF_ALIGNMENT_DAILY_TRACKER.md')

# ============================================
# LOGGING
# ============================================
MTF_ALIGNMENT_LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_DETAIL = True  # If True, log detailed TF match info; if False, summary only

print("[CONFIG] MTF Alignment config loaded")
print(f"[CONFIG] MTF_ALIGNMENT_ENABLED = {MTF_ALIGNMENT_ENABLED}")
print(f"[CONFIG] Weights: Direction={ALIGNMENT_WEIGHTS['direction']}, Regime={ALIGNMENT_WEIGHTS['regime']}, Route={ALIGNMENT_WEIGHTS['route']}")
