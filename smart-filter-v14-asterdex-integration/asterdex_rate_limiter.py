"""
asterdex_rate_limiter.py - Cooldown Manager

Prevents duplicate entries for the same (symbol, timeframe) pair within the timeframe duration.

Example:
- BTC-USDT 2h signal enters at 10:00
- Cannot enter BTC-USDT 2h again until 12:00 (after 2 hours)
- BTC-USDT 4h is independent (can enter anytime)
- SOL-USDT 2h is independent (can enter anytime)

Storage: cooldown_state.json (persistent across restarts)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """Tracks (symbol, timeframe) → cooldown expiration time"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Dict[str, str] = {}  # "BTC-USDT_4h" → "2026-05-29T17:50:00Z"
        self._load_state()

    def _load_state(self):
        """Load cooldown state from persistent file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logger.info(f"[RATE_LIMITER] Loaded {len(self.state)} cooldown entries")
            except Exception as e:
                logger.error(f"[RATE_LIMITER] Failed to load state: {e}")
                self.state = {}
        else:
            self.state = {}

    def _save_state(self):
        """Persist cooldown state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"[RATE_LIMITER] Failed to save state: {e}")

    def _get_key(self, symbol: str, timeframe: str) -> str:
        """Generate cooldown key"""
        return f"{symbol}_{timeframe}"

    def _parse_iso_time(self, iso_str: str) -> datetime:
        """Parse ISO 8601 timestamp"""
        # Handle both with and without 'Z' suffix
        iso_str = iso_str.rstrip('Z')
        return datetime.fromisoformat(iso_str)

    def _to_iso_time(self, dt: datetime) -> str:
        """Convert datetime to ISO 8601 string"""
        return dt.isoformat() + 'Z'

    def can_enter(self, symbol: str, timeframe: str) -> Tuple[bool, str]:
        """
        Check if (symbol, timeframe) can enter now.

        Returns:
            (can_enter: bool, reason: str)
        """
        key = self._get_key(symbol, timeframe)

        # Not in cooldown → can enter
        if key not in self.state:
            return True, "First entry"

        expiration_str = self.state[key]
        expiration_time = self._parse_iso_time(expiration_str)
        now = datetime.utcnow()

        if now >= expiration_time:
            # Cooldown expired → can enter again
            del self.state[key]
            self._save_state()
            return True, f"Cooldown expired at {expiration_str}"
        else:
            # Still in cooldown
            remaining = expiration_time - now
            return False, f"Cooldown active until {expiration_str} ({remaining.seconds}s remaining)"

    def start_cooldown(self, symbol: str, timeframe: str, entry_time: datetime = None):
        """
        Start cooldown for (symbol, timeframe).

        The cooldown lasts for the duration of the timeframe.
        Example: 2h signal entered at 10:00 → expires at 12:00
        """
        if entry_time is None:
            entry_time = datetime.utcnow()

        # Parse timeframe to duration
        tf_minutes = self._parse_timeframe_to_minutes(timeframe)
        expiration_time = entry_time + timedelta(minutes=tf_minutes)

        key = self._get_key(symbol, timeframe)
        self.state[key] = self._to_iso_time(expiration_time)
        self._save_state()

        logger.info(
            f"[RATE_LIMITER] Cooldown started for {key}: "
            f"expires at {self._to_iso_time(expiration_time)}"
        )

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            "15min": 15,
            "30min": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
        }
        minutes = mapping.get(timeframe)
        if minutes is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        return minutes

    def get_status(self) -> Dict:
        """Get current cooldown status"""
        now = datetime.utcnow()
        status = {}

        for key, expiration_str in self.state.items():
            expiration_time = self._parse_iso_time(expiration_str)
            remaining_seconds = (expiration_time - now).total_seconds()

            if remaining_seconds > 0:
                status[key] = {
                    "expires_at": expiration_str,
                    "remaining_sec": int(remaining_seconds),
                }
            else:
                # Expired, will be cleaned up on next can_enter() check
                status[key] = {"status": "EXPIRED"}

        return status

    def clear_all(self):
        """Clear all cooldown entries (use with caution)"""
        self.state.clear()
        self._save_state()
        logger.warning("[RATE_LIMITER] All cooldown entries cleared")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile

    # Test
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "cooldown_state.json"
        limiter = RateLimiter(state_file)

        # Test 1: First entry allowed
        can_enter, reason = limiter.can_enter("BTC-USDT", "4h")
        print(f"✅ Test 1 - First entry: {can_enter} ({reason})")

        # Test 2: Start cooldown
        limiter.start_cooldown("BTC-USDT", "4h")
        can_enter, reason = limiter.can_enter("BTC-USDT", "4h")
        print(f"✅ Test 2 - During cooldown: {can_enter} (expected: False)")

        # Test 3: Different timeframe allowed
        can_enter, reason = limiter.can_enter("BTC-USDT", "2h")
        print(f"✅ Test 3 - Different TF: {can_enter} ({reason})")

        # Test 4: Status
        status = limiter.get_status()
        print(f"✅ Test 4 - Status: {json.dumps(status, indent=2)}")
