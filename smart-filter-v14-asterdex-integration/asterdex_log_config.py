"""
Lightweight logging configuration for Asterdex Entry Poster
- Minimal verbose output
- Only logs important events (posts, errors)
- Automatic log rotation (daily)
- Simple, compact format
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# SIMPLE LOG FORMAT - Only essential info
# ============================================================================
SIMPLE_FORMAT = "[%(asctime)s] %(message)s"
SIMPLE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# FILTER - Only log filter matches and errors (clean output)
# ============================================================================
class ImportantEventsFilter(logging.Filter):
    """Only log successful matches and errors, skip all rejections"""
    
    def filter(self, record):
        msg = record.getMessage()
        
        # SKIP: Only skip filter rejections and dedup
        skip_keywords = [
            "NO FILTER MATCH",   # Skip filter rejections (too noisy)
            "Already processed", # Skip dedup rejections (rare now with 12h window)
        ]
        
        for keyword in skip_keywords:
            if keyword in msg:
                return False
        
        # ONLY SHOW: Filter matches, steps, errors, posts, and rate limiter info
        show_keywords = [
            "FILTER MATCH:",     # Show ✅ TIER FILTER MATCH and ✅ MTF COMBO FILTER MATCH
            "Step",              # Show Step 1/5, Step 2/5, etc. (detailed posting process)
            "ERROR",             # Show all errors
            "FAILED",            # Show failures
            "COMPLETE",          # Show completed posts
            "POSTED",            # Show posted orders
            "SUCCESS",           # Show successful operations
            "REJECTED:",         # Show rate limiter rejections (important!)
            "Cooldown:",         # Show cooldown info
            "Web3 account",      # Show initialization
        ]
        
        # LOG: If contains show keywords OR is ERROR level
        for keyword in show_keywords:
            if keyword in msg:
                return True
        
        # Always log ERROR level
        if record.levelno >= logging.ERROR:
            return True
        
        # SKIP: Everything else (debug, etc.)
        return False

# ============================================================================
# SETUP LOGGER
# ============================================================================
def setup_asterdex_logger(name="AsterdexEntryPoster"):
    """
    Setup lightweight logger with rotation and filtering.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
    
    # Remove existing handlers
    logger.handlers = []
    
    # ========================================================================
    # HANDLER 1: Important Events Log (Simple, Compact)
    # ========================================================================
    important_log_file = LOG_DIR / "asterdex_important.log"
    important_handler = RotatingFileHandler(
        important_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB per file
        backupCount=5,               # Keep 5 old files
    )
    important_handler.setLevel(logging.DEBUG)
    important_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT, SIMPLE_DATE_FORMAT))
    important_handler.addFilter(ImportantEventsFilter())
    logger.addHandler(important_handler)
    
    # ========================================================================
    # HANDLER 2: Console Output (Real-time, filtered)
    # ========================================================================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Show all messages including warnings
    console_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT, SIMPLE_DATE_FORMAT))
    console_handler.addFilter(ImportantEventsFilter())
    logger.addHandler(console_handler)
    
    # ========================================================================
    # HANDLER 3: Error Log (Errors only - for debugging)
    # ========================================================================
    error_log_file = LOG_DIR / "asterdex_errors.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB per file
        backupCount=3,               # Keep 3 old files
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        SIMPLE_DATE_FORMAT
    ))
    logger.addHandler(error_handler)
    
    logger.info(f"✅ Asterdex logger initialized (lightweight mode)")
    logger.info(f"  ├─ Important events: {important_log_file}")
    logger.info(f"  ├─ Errors only: {error_log_file}")
    logger.info(f"  └─ Format: [timestamp] message (no file:line)")
    
    return logger

# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    logger = setup_asterdex_logger()
    
    print("Testing lightweight logging...\n")
    
    logger.debug("[ENTRY_POSTER] ⛔ REJECTED: SOL-USDT - Tier Tier-X (should NOT appear)")
    logger.info("[ENTRY_POSTER] ✅ POSTED: SOL-USDT 10x leverage (should appear)")
    logger.warning("[ENTRY_POSTER] ⚠️ Rate limited, retrying... (should appear)")
    logger.error("[ENTRY_POSTER] ❌ Entry failed: invalid symbol (should appear)")
    
    print("\n✅ Test complete. Check asterdex_important.log and asterdex_errors.log")
