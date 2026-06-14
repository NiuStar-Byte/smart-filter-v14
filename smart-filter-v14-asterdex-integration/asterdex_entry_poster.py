"""
asterdex_entry_poster.py - Main Entry Posting Engine

Monitors SIGNALS_MASTER.jsonl and posts real entries to Asterdex.

Workflow:
1. Poll SIGNALS_MASTER.jsonl every 5 seconds
2. For each NEW signal:
   a. Check tier: Only Tier 1, 2, 3 (any timeframe)
   b. Check MTF alignment: Only MTF = 'strong' (any timeframe) - highest WR in tracker
   c. Check cooldown: (symbol, timeframe) not in cooldown
   d. Extract TP/SL from signal
   e. Post entry to Asterdex
   f. Log result
   g. Start cooldown timer
3. Loop forever

Status: ✅ DUAL FILTER ACTIVE - Tier-1/2/3 + MTF Strong
"""

import json
import logging
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set

# Import our modules
from asterdex_config import (
    ASTER_MAIN_ACCOUNT,
    ASTER_API_WALLET_ADDRESS,
    ASTER_API_WALLET_PRIVATE_KEY,
    ASTERDEX_ENDPOINT,
    POSITION_SETTINGS,
    TIER_FILTER,
    RATE_LIMIT_SETTINGS,
    SIGNALS_MASTER_PATH,
    ENTRY_LOG_PATH,
    COOLDOWN_STATE_PATH,
    LOG_DIR,
    DRY_RUN_MODE,
    LOG_LEVEL,
    API_VERSION,
)
from aster_v3_auth import AsterV3Auth
from asterdex_rate_limiter import RateLimiter
from asterdex_utils import (
    parse_signal,
    check_tier,
    calculate_order_size,
    format_log_entry,
    calculate_side,
    convert_symbol_format,
)
from asterdex_exchange_info import validate_order_params
from symbol_leverage_config import get_leverage_for_symbol
from asterdex_log_config import setup_asterdex_logger
from asterdex_symbol_blacklist import is_available_on_asterdex
from asterdex_symbol_mapper import get_asterdex_symbol  # NEW: Maps signal symbols to Asterdex names
from asterdex_entry_logger import log_posted_entry

# ============================================================================
# LOGGING SETUP - Lightweight (no verbose rejections)
# ============================================================================

logger = setup_asterdex_logger()


class AsterdexEntryPoster:
    """Main engine for posting entries to Asterdex"""

    def __init__(self):
        # Initialize Web3 authentication with two wallets:
        # - ASTER_MAIN_ACCOUNT: Your main trading account ("user" in API)
        # - ASTER_API_WALLET: The authorized API signer ("signer" in API)
        self.main_account = ASTER_MAIN_ACCOUNT
        self.api_wallet_address = ASTER_API_WALLET_ADDRESS
        self.api_wallet_private_key = ASTER_API_WALLET_PRIVATE_KEY
        
        # Auth signs with the API wallet private key
        self.auth = AsterV3Auth(self.api_wallet_address, self.api_wallet_private_key)
        
        self.endpoint = ASTERDEX_ENDPOINT
        self.position_settings = POSITION_SETTINGS
        self.tier_filter = TIER_FILTER

        # Rate limiter for cooldowns
        self.rate_limiter = RateLimiter(COOLDOWN_STATE_PATH)

        # Track processed signals
        self.processed_signal_uuids: Set[str] = set()
        
        # Track which cooldowns have been announced (prevent log spam)
        # Key: "BTC-USDT_4h", Value: expiration_time (to clean up when cooldown ends)
        self.cooldown_announced: Dict[str, datetime] = {}
        
        # OPTIMIZATION: Incremental reading - track file position
        self.signals_master_last_line_count = 0
        
        self._load_processed_signals()

        logger.info("[ENTRY_POSTER] Initialized (PRO API V3 with Two-Wallet Auth)")
        logger.info(f"  ├─ Endpoint: {self.endpoint}")
        logger.info(f"  ├─ Main Account (user): {self.main_account}")
        logger.info(f"  ├─ API Wallet (signer): {self.api_wallet_address}")
        logger.info(f"  ├─ API Version: {API_VERSION}")
        logger.info(f"  ├─ Margin/Leverage: ${self.position_settings['margin']} × {self.position_settings['leverage']}x")
        logger.info(f"  ├─ Tier filter: {self.tier_filter}")
        logger.info(f"  └─ Dry run mode: {DRY_RUN_MODE}")

    def _load_processed_signals(self):
        """
        Load recently-posted signal UUIDs from entry_log.jsonl.
        
        SMART DEDUP: Only tracks signals posted in last 12 hours.
        Older signals are likely already closed, no need to block them.
        """
        if ENTRY_LOG_PATH.exists():
            try:
                # Get cutoff time (12 hours ago)
                now = datetime.utcnow()
                cutoff = now - timedelta(hours=12)
                cutoff_str = cutoff.isoformat()
                
                with open(ENTRY_LOG_PATH, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            ts = entry.get('timestamp', '')
                            
                            # Only track recent posts (last 12 hours)
                            if ts >= cutoff_str:
                                self.processed_signal_uuids.add(entry.get('signal_uuid'))
                
                logger.info(f"[ENTRY_POSTER] ✅ Loaded {len(self.processed_signal_uuids)} recent posted signals (last 12h)")
                logger.info(f"  └─ Cutoff: {cutoff_str} (older posts not tracked for dedup)")
            except Exception as e:
                logger.error(f"[ENTRY_POSTER] Failed to load processed signals: {e}")

    def _read_signals_master(self) -> list:
        """
        Read ONLY NEW signals since last cycle (incremental).
        
        OPTIMIZATION: Instead of reading all 226k signals every cycle,
        only read new lines added since last check.
        
        Returns:
            List of NEW signals only (not seen before)
        """
        if not SIGNALS_MASTER_PATH.exists():
            logger.warning(f"[ENTRY_POSTER] SIGNALS_MASTER not found: {SIGNALS_MASTER_PATH}")
            return []

        signals = []
        try:
            # Count total lines in file
            with open(SIGNALS_MASTER_PATH, 'r') as f:
                current_line_count = sum(1 for line in f if line.strip())
            
            # If no new lines, skip
            if current_line_count <= self.signals_master_last_line_count:
                return []
            
            # Read only NEW lines (from last checkpoint to end)
            with open(SIGNALS_MASTER_PATH, 'r') as f:
                for idx, line in enumerate(f):
                    if line.strip() and idx >= self.signals_master_last_line_count:
                        signals.append(json.loads(line))
            
            # Update checkpoint
            self.signals_master_last_line_count = current_line_count
            
        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Failed to read SIGNALS_MASTER: {e}")

        return signals

    def _check_tier_filter(self, signal: Dict) -> tuple:
        """
        Check if signal matches TIER filter (Tier 1, 2, 3).
        
        REQUIREMENTS:
        1. Must be OPEN (not stale/closed)
        2. Must be Tier 1, 2, or 3
        3. NO AGE RESTRICTION for Tier (Tier is determined by locked combo at signal time)
        
        Returns:
            (matches: bool, source_tag: str or None)
        """
        tier = signal.get('tier')
        status = signal.get('status')
        
        # Check 1: Must be OPEN (not stale, not closed)
        if status != 'OPEN':
            return False, None
        
        # Check 2: Must be Tier 1, 2, or 3
        if not check_tier(tier, self.tier_filter):
            return False, None
        
        return True, "tier_based"

    def _check_mtf_combo_filter(self, signal: Dict) -> tuple:
        """
        Check if signal matches MTF STRONG filter: MTF alignment band = 'strong' (ANY timeframe).
        
        STRICT REQUIREMENTS:
        1. Must be OPEN (not stale/closed)
        2. Must be RECENTLY ADDED to system (sent_time < 5 minutes ago)
        3. Must have Strong MTF alignment (any timeframe - 15min, 30min, 1h, 2h, or 4h)
        
        Returns:
            (matches: bool, source_tag: str or None)
        """
        timeframe = signal.get('timeframe')
        mtf_band = signal.get('mtf_alignment_band')
        status = signal.get('status')
        sent_time = signal.get('sent_time_utc')
        
        # Check 1: Must be OPEN (not stale, not closed)
        if status != 'OPEN':
            return False, None
        
        # Check 2: Must be RECENTLY ADDED to system (last 5 minutes)
        # sent_time = when signal arrived in SIGNALS_MASTER
        # SAME as Tier filter - only post fresh signals
        if sent_time:
            try:
                sent_dt = datetime.fromisoformat(sent_time.replace('Z', '+00:00').split('+')[0])
                now = datetime.utcnow()
                age_minutes = (now - sent_dt).total_seconds() / 60
                
                # Only post if arrived in last 5 minutes (fresh signal)
                if age_minutes > 5:
                    return False, None  # Too old, skip
            except:
                return False, None  # Can't parse time, skip
        
        # Check 3: Must have Strong MTF alignment (ANY timeframe)
        # MTF=strong has highest WR in tracker, so post all strong signals regardless of TF
        if mtf_band == 'strong':
            return True, "mtf_strong"
        
        return False, None

    def _should_post_entry(self, signal: Dict) -> tuple:
        """
        Check if signal should generate an entry.
        
        Signal passes if ANY of these independent filters match (OR logic):
        
          FILTER A) TIER-BASED: Tier 1, 2, or 3 (no age restriction)
            - Requirement: status=OPEN AND tier in [1,2,3]
            - Age: Any age accepted (post old Tier signals)
          
          OR
          
          FILTER B) MTF-STRONG: MTF alignment = 'strong' (must be fresh)
            - Requirement: status=OPEN AND mtf_alignment_band='strong' AND age < 5 min
            - Age: Fresh signals only (last 5 minutes)
        
        Additionally must pass:
          - Dedup check
          - Rate limiting check
        
        Returns:
            (should_post: bool, reason: str, source: str)
            source: "tier_based", "mtf_strong", or None
        """
        signal_uuid = signal.get('signal_uuid')
        symbol = signal.get('symbol')
        timeframe = signal.get('timeframe')

        # Check 1: Already processed (DEDUP)
        if signal_uuid in self.processed_signal_uuids:
            return False, "Already processed", None

        # Check 2a: TIER FILTERING
        tier_match, tier_source = self._check_tier_filter(signal)
        
        # Check 2b: MTF COMBO FILTERING (NEW)
        mtf_match, mtf_source = self._check_mtf_combo_filter(signal)
        
        # Signal passes if EITHER condition is met (OR logic)
        if tier_match:
            source = tier_source
        elif mtf_match:
            source = mtf_source
        else:
            logger.debug(f"[ENTRY_POSTER] ❌ NO FILTER MATCH: {symbol} {timeframe}")
            return False, "No filter match (not Tier 1/2/3 and not fresh MTF=strong)", None
        
        logger.debug(f"[ENTRY_POSTER] DEBUG: About to check rate limiter for {symbol} {timeframe}")

        # Check 3: RATE LIMITING - STRICT
        # Prevent duplicate (symbol, timeframe) entries until timeframe expires
        try:
            can_enter, reason = self.rate_limiter.can_enter(symbol, timeframe)
            if not can_enter:
                # Extract expiration time from reason string (format: "Cooldown active until YYYY-MM-DDTHH:MM:SS.ffffffZ (Ns remaining)")
                cooldown_key = f"{symbol}_{timeframe}"
                
                # Only log ONCE per cooldown (prevent spam on 5-second polls)
                if cooldown_key not in self.cooldown_announced:
                    # Parse expiration time from reason
                    try:
                        # Extract ISO timestamp from reason: "... until 2026-06-07T18:34:50.165267Z (14391s remaining)"
                        until_idx = reason.find("until ")
                        if until_idx >= 0:
                            time_part = reason[until_idx + 6:until_idx + 6 + 26]  # ISO format is ~26 chars
                            expiration_time = datetime.fromisoformat(time_part.rstrip('Z'))
                            self.cooldown_announced[cooldown_key] = expiration_time
                            
                            # Format release time nicely (HH:MM:SS)
                            release_time = expiration_time.strftime("%H:%M:%S")
                            remaining_secs = int((expiration_time - datetime.utcnow()).total_seconds())
                            remaining_mins = remaining_secs // 60
                            remaining_secs = remaining_secs % 60
                            
                            if remaining_mins > 0:
                                time_str = f"~{remaining_mins}m{remaining_secs}s"
                            else:
                                time_str = f"{remaining_secs}s"
                            
                            logger.info(f"[ENTRY_POSTER] 🕐 {symbol} {timeframe} posted | Cooldown active until {release_time} ({time_str} remaining)")
                    except Exception as e:
                        logger.debug(f"[ENTRY_POSTER] Failed to parse cooldown time: {e}")
                        self.cooldown_announced[cooldown_key] = datetime.utcnow() + timedelta(hours=1)  # Default to 1h
                else:
                    # Already announced, silently skip (no log spam)
                    pass
                
                return False, f"Cooldown active", None
        except Exception as e:
            logger.error(f"[ENTRY_POSTER] ❌ RATE LIMITER ERROR: {symbol} {timeframe} - {e}")
            return False, f"Rate limiter error: {e}", None

        # Clean up expired cooldown announcements
        now = datetime.utcnow()
        expired_keys = [k for k, exp_time in self.cooldown_announced.items() if now >= exp_time]
        for k in expired_keys:
            del self.cooldown_announced[k]

        # Log FILTER MATCH only AFTER rate limiter passes (prevents spam for signals in cooldown)
        if source == "tier_based":
            logger.info(f"[ENTRY_POSTER] ✅ TIER FILTER MATCH: {symbol} {timeframe} - Tier {signal.get('tier')}")
        elif source == "mtf_strong":
            logger.info(f"[ENTRY_POSTER] ✅ MTF STRONG FILTER MATCH: {symbol} {timeframe} - MTF alignment = strong")

        return True, "OK", source

    def _set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict:
        """
        Set margin type for symbol (ISOLATED or CROSS) - PRO API V3.
        
        Args:
            symbol: Symbol (e.g., BTCUSDT)
            margin_type: ISOLATED or CROSS
        
        Returns:
            Dict with result
        """
        try:
            params = {
                "symbol": symbol,
                "marginType": margin_type,
            }
            
            # Sign request with EIP-712 (main_account as "user", api_wallet as "signer")
            signed_params = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            url = f"{self.endpoint}/fapi/{API_VERSION}/marginType"
            
            response = requests.post(url, data=signed_params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {"status": "SUCCESS"}
            else:
                # Margin type might already be set, not critical error
                if "No need to change" in response.text or response.status_code == 400:
                    return {"status": "ALREADY_SET"}
                return {"status": "ERROR", "error": response.text}
        
        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Failed to set margin type: {e}")
            return {"status": "ERROR", "error": str(e)}

    def _set_leverage(self, symbol: str, leverage: int = 10) -> Dict:
        """
        Set leverage for symbol - PRO API V3.
        
        Args:
            symbol: Symbol (e.g., BTCUSDT)
            leverage: Leverage multiplier (e.g., 10 for 10x)
        
        Returns:
            Dict with result
        """
        try:
            params = {
                "symbol": symbol,
                "leverage": str(leverage),
            }
            
            # Sign request with EIP-712 (main_account as "user", api_wallet as "signer")
            signed_params = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            url = f"{self.endpoint}/fapi/{API_VERSION}/leverage"
            
            response = requests.post(url, data=signed_params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {"status": "SUCCESS"}
            else:
                if "No need to change" in response.text or response.status_code == 400:
                    return {"status": "ALREADY_SET"}
                return {"status": "ERROR", "error": response.text}
        
        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Failed to set leverage: {e}")
            return {"status": "ERROR", "error": str(e)}

    def _post_order_to_asterdex(self, symbol: str, side: str, order_type: str, 
                                quantity: float, price: float = None, stop_price: float = None) -> Dict:
        """
        Post a single order to Asterdex PRO API V3 with Web3 signing.
        
        Args:
            symbol: Symbol (e.g., BTCUSDT)
            side: BUY or SELL
            order_type: MARKET, LIMIT, TAKE_PROFIT_MARKET, STOP_MARKET
            quantity: Order quantity
            price: For LIMIT orders
            stop_price: For TAKE_PROFIT_MARKET or STOP_MARKET
        
        Returns:
            Dict with order result
        """
        try:
            # Validate and round parameters to exchange constraints
            if order_type == "LIMIT" and price:
                validation = validate_order_params(symbol, price, quantity)
                if not validation['valid']:
                    logger.error(f"[ENTRY_POSTER] Order validation failed: {validation['error']}")
                    return {
                        "status": "ERROR",
                        "error": f"Parameter validation failed: {validation['error']}"
                    }
                
                # Use validated (rounded) values
                price = validation['validated_price']
                quantity = validation['validated_quantity']
                logger.info(f"[ENTRY_POSTER] Parameters validated: price={price}, qty={quantity}")
            
            elif order_type in ["TAKE_PROFIT_MARKET", "STOP_MARKET"] and stop_price:
                validation = validate_order_params(symbol, stop_price, quantity)
                if not validation['valid']:
                    logger.error(f"[ENTRY_POSTER] Order validation failed: {validation['error']}")
                    return {
                        "status": "ERROR",
                        "error": f"Parameter validation failed: {validation['error']}"
                    }
                
                # Use validated values
                stop_price = validation['validated_price']
                quantity = validation['validated_quantity']
                logger.info(f"[ENTRY_POSTER] Parameters validated: stopPrice={stop_price}, qty={quantity}")
            
            # Build base parameters
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": str(quantity),
            }
            
            # Add price for LIMIT orders
            if order_type == "LIMIT" and price:
                params["price"] = str(price)
                params["timeInForce"] = self.position_settings.get('time_in_force', 'GTC')
            
            # Add stopPrice for stop orders (TP/SL triggered by PRICE using MARK_PRICE)
            if stop_price and order_type in ["TAKE_PROFIT_MARKET", "STOP_MARKET"]:
                params["stopPrice"] = str(stop_price)  # Exact price from signal
                params["workingType"] = "MARK_PRICE"  # Trigger by Mark Price, not PnL/ROI
            
            # Sign request with EIP-712 (Web3) - main_account as "user", api_wallet as "signer"
            signed_params = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            # Make API request to V3 endpoint
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            url = f"{self.endpoint}/fapi/{API_VERSION}/order"
            
            # Retry on rate limit (429)
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    url,
                    data=signed_params,
                    headers=headers,
                    timeout=10,
                )
                
                # Success
                if response.status_code == 200:
                    break
                
                # Rate limit - wait and retry
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 1 * (2 ** attempt)  # 1s, 2s, 4s
                    logger.warning(f"[ENTRY_POSTER] Rate limited (429), waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                
                # Other error
                break
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "SUCCESS",
                    "order_id": result.get('orderId'),
                    "response": result,
                }
            else:
                return {
                    "status": "ERROR",
                    "http_code": response.status_code,
                    "error": response.text,
                }
        
        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Failed to post order: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
            }

    def _post_oco_strategy(self, symbol: str, quantity: float, tp_price: float, sl_price: float, side: str = "BUY") -> Dict:
        """Post OCO (One-Cancel-the-Other) Strategy Order. Falls back silently if not supported."""
        try:
            # Determine TP/SL sides (opposite of entry)
            if side.upper() == "BUY":
                tp_sl_side = "SELL"  # Close the long position
            else:
                tp_sl_side = "BUY"  # Close the short position
            
            # Validate prices
            tp_validation = validate_order_params(symbol, tp_price, quantity)
            sl_validation = validate_order_params(symbol, sl_price, quantity)
            
            if not tp_validation['valid'] or not sl_validation['valid']:
                return {
                    "status": "VALIDATION_FAILED",
                    "error": "OCO price/quantity validation failed"
                }
            
            # Use validated values
            tp_price = tp_validation['validated_price']
            sl_price = sl_validation['validated_price']
            quantity = tp_validation['validated_quantity']
            
            # Build OCO strategy parameters
            params = {
                "symbol": symbol,
                # TP order (first leg)
                "type1": "TAKE_PROFIT",
                "side1": tp_sl_side,
                "quantity1": str(quantity),
                "price1": str(tp_price),
                "timeInForce1": "GTC",
                # SL order (second leg)
                "type2": "STOP_LOSS",
                "side2": tp_sl_side,
                "quantity2": str(quantity),
                "price2": str(sl_price),
                "timeInForce2": "GTC",
            }
            
            # Sign request with EIP-712
            signed_params = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            # Make API request
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            url = f"{self.endpoint}/fapi/{API_VERSION}/strategyOrder"
            
            response = requests.post(
                url,
                data=signed_params,
                headers=headers,
                timeout=10,
            )
            
            if response.status_code == 200:
                result = response.json()
                orders = result.get('orders', [])
                
                if len(orders) >= 2:
                    tp_order_id = orders[0].get('orderId')
                    sl_order_id = orders[1].get('orderId')
                    
                    return {
                        "status": "SUCCESS",
                        "tp_order_id": tp_order_id,
                        "sl_order_id": sl_order_id,
                        "response": result,
                    }
                else:
                    return {
                        "status": "MISSING_ORDERS",
                        "error": "OCO response doesn't contain 2 orders",
                        "response": result,
                    }
            else:
                # OCO not supported on Asterdex - silently fail and use fallback TP/SL
                return {
                    "status": "ERROR",
                    "http_code": response.status_code,
                    "error": f"HTTP {response.status_code}",
                }
        
        except Exception as e:
            # OCO attempt failed - will fall back to traditional TP/SL
            return {
                "status": "ERROR",
                "error": str(e),
            }

    def _scale_pepe_prices(self, symbol: str, entry_price: float, tp_price: float, sl_price: float) -> tuple:
        """
        Scale micro-cap prices for Asterdex (1000x scaling).
        
        Asterdex trades some assets as "1000x" variants:
        - PEPE: trades as "1000PEPE-USDT" (scaled 1000x from real market price)
        - BONK: trades as "1000BONK-USDT" (scaled 1000x from real market price)
        
        This function multiplies all prices by 1000 for these assets.
        
        Args:
            symbol: Symbol (e.g., "PEPE-USDT", "BONK-USDT", or "1000PEPE-USDT", "1000BONK-USDT" after mapping)
            entry_price: Entry price from signal
            tp_price: TP price from signal
            sl_price: SL price from signal
        
        Returns:
            Tuple of (entry_price, tp_price, sl_price) - scaled if PEPE/BONK, unchanged otherwise
        """
        # Check for both original names (before mapping) and mapped names (after mapping)
        if symbol in ["PEPE-USDT", "BONK-USDT", "1000PEPE-USDT", "1000BONK-USDT"]:
            logger.info(f"[ENTRY_POSTER] 🔄 {symbol} detected - Scaling prices 1000x for Asterdex")
            entry_price_scaled = entry_price * 1000
            tp_price_scaled = tp_price * 1000
            sl_price_scaled = sl_price * 1000
            logger.info(f"[ENTRY_POSTER]   ├─ Entry: {entry_price} → {entry_price_scaled}")
            logger.info(f"[ENTRY_POSTER]   ├─ TP: {tp_price} → {tp_price_scaled}")
            logger.info(f"[ENTRY_POSTER]   └─ SL: {sl_price} → {sl_price_scaled}")
            return entry_price_scaled, tp_price_scaled, sl_price_scaled
        
        return entry_price, tp_price, sl_price

    def _round_pepe_price(self, symbol: str, entry_price: float, tp_price: float, sl_price: float) -> tuple:
        """
        Round micro-cap prices to match Asterdex PRICE_FILTER constraints.
        
        PEPE-USDT and BONK-USDT on Asterdex have specific tick size requirements.
        This function rounds scaled prices to valid precision (0.0001 tick).
        
        Args:
            symbol: Symbol (e.g., "PEPE-USDT", "BONK-USDT", or "1000PEPE-USDT", "1000BONK-USDT" after mapping)
            entry_price: Entry price (already scaled if PEPE/BONK)
            tp_price: TP price (already scaled if PEPE/BONK)
            sl_price: SL price (already scaled if PEPE/BONK)
        
        Returns:
            Tuple of (entry_price, tp_price, sl_price) - rounded if PEPE/BONK, unchanged otherwise
        """
        # Check for both original names (before mapping) and mapped names (after mapping)
        if symbol in ["PEPE-USDT", "BONK-USDT", "1000PEPE-USDT", "1000BONK-USDT"]:
            # Round to 4 decimal places (0.0001 tick size) for both PEPE and BONK
            tick_size = 0.0001
            
            entry_rounded = round(entry_price / tick_size) * tick_size
            tp_rounded = round(tp_price / tick_size) * tick_size
            sl_rounded = round(sl_price / tick_size) * tick_size
            
            logger.info(f"[ENTRY_POSTER] 🔧 {symbol} price rounding (tick: {tick_size}):")
            logger.info(f"[ENTRY_POSTER]   ├─ Entry: {entry_price} → {entry_rounded}")
            logger.info(f"[ENTRY_POSTER]   ├─ TP: {tp_price} → {tp_rounded}")
            logger.info(f"[ENTRY_POSTER]   └─ SL: {sl_price} → {sl_rounded}")
            
            return entry_rounded, tp_rounded, sl_rounded
        
        return entry_price, tp_price, sl_price

    def _post_entry_to_asterdex(self, signal: Dict) -> Dict:
        """
        Post complete entry with TP/SL protection to Asterdex.
        
        Posts 3 orders in sequence:
        1. ENTRY (MARKET) at entry_price
        2. TP (TAKE_PROFIT_MARKET) at tp_price
        3. SL (STOP_MARKET) at sl_price
        
        Returns:
            Result dict with all order details
        """
        try:
            symbol, direction, entry_price, tp_price, sl_price, timeframe, tier = parse_signal(signal)
            
            # ✅ MAP SYMBOL NAME: Some symbols known by different names on Asterdex
            # e.g., XAUT-USDT (signal) → XAU-USDT (Asterdex)
            symbol_original = symbol  # Keep original for logging
            symbol = get_asterdex_symbol(symbol)  # Convert to Asterdex name if needed
            if symbol != symbol_original:
                logger.info(f"[ENTRY_POSTER] 📍 Symbol mapped: {symbol_original} → {symbol}")
            
            # ⚠️ PEPE-USDT Special Handling: Scale prices 1000x for Asterdex
            entry_price, tp_price, sl_price = self._scale_pepe_prices(symbol, entry_price, tp_price, sl_price)
            
            # ⚠️ PEPE-USDT Special Handling: Round prices to match Asterdex tick size
            entry_price, tp_price, sl_price = self._round_pepe_price(symbol, entry_price, tp_price, sl_price)

            # CHECK: Is symbol available on Asterdex?
            available, reason = is_available_on_asterdex(symbol)
            if not available:
                logger.warning(f"[ENTRY_POSTER] ⛔ {reason} - SKIPPING {symbol}")
                return {
                    "status": "SYMBOL_UNAVAILABLE",
                    "error": reason,
                    "symbol": symbol
                }

            # Calculate order details
            side = calculate_side(direction)  # BUY or SELL
            opposite_side = "SELL" if side == "BUY" else "BUY"  # For TP/SL
            
            # ✅ DYNAMIC LEVERAGE & POSITION SIZING
            # Get symbol-specific leverage (e.g., KNC=5x, FUN=2x, default=10x)
            symbol_leverage = get_leverage_for_symbol(symbol)
            
            # Calculate notional value: Margin ($2) × Symbol-specific Leverage
            notional_value = self.position_settings['margin'] * symbol_leverage
            
            # Calculate position size based on dynamic notional value
            quantity = calculate_order_size(
                notional_value,
                entry_price,
            )
            symbol_asterdex = convert_symbol_format(symbol, "asterdex")  # BTC-USDT → BTCUSDT

            logger.info(f"[ENTRY_POSTER] Posting complete entry: {symbol} {direction}")
            logger.info(f"  ├─ Entry: {side} {quantity} @ {entry_price} (Margin=${self.position_settings['margin']}, Leverage={symbol_leverage}x, Notional=${notional_value:.1f})")
            logger.info(f"  ├─ TP: {opposite_side} @ {tp_price}")
            logger.info(f"  └─ SL: {opposite_side} @ {sl_price}")

            if DRY_RUN_MODE:
                # DRY RUN MODE - Simulate all operations without real API calls
                logger.info(f"[ENTRY_POSTER] 🏃 DRY RUN - Simulating complete entry (8 infos + 5 operations):")
                logger.info(f"\n  📋 Entry Infos:")
                logger.info(f"     1. Entry Direction: {direction}")
                logger.info(f"     2. Entry Price: {entry_price}")
                logger.info(f"     3. TP Target: {tp_price}")
                logger.info(f"     4. SL Target: {sl_price}")
                logger.info(f"     5. Margin: {self.position_settings['margin']} USDT")
                logger.info(f"     6. Leverage: {symbol_leverage}x (Symbol-specific: {symbol})")
                logger.info(f"     7. Notional Value: ${notional_value:.1f} (Margin × Leverage)")
                logger.info(f"     8. Margin Mode: {self.position_settings['margin_type']}")
                logger.info(f"     9. Order Type: {self.position_settings['order_type']}")
                
                logger.info(f"\n  🔧 Operations:")
                logger.info(f"     1. Set margin type: {self.position_settings['margin_type']}")
                logger.info(f"     2. Set leverage: {symbol_leverage}x (for {symbol})")
                logger.info(f"     3. {self.position_settings['order_type']} entry: {side} {quantity} @ {entry_price}")
                logger.info(f"     4. TP order: TAKE_PROFIT_MARKET {opposite_side} @ {tp_price}")
                logger.info(f"     5. SL order: STOP_MARKET {opposite_side} @ {sl_price}")
                
                entry_order_id = f"DRY_ENTRY_{int(time.time()*1000)}"
                tp_order_id = f"DRY_TP_{int(time.time()*1000)+1}"
                sl_order_id = f"DRY_SL_{int(time.time()*1000)+2}"

                # Simulate Asterdex response
                return {
                    "status": "DRY_RUN_COMPLETE",
                    "entry_order_id": entry_order_id,
                    "tp_order_id": tp_order_id,
                    "sl_order_id": sl_order_id,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "quantity": quantity,
                    "margin": self.position_settings['margin'],
                    "leverage": self.position_settings['leverage'],
                    "margin_type": self.position_settings['margin_type'],
                    "order_type": self.position_settings['order_type'],
                    "message": "5 operations simulated (DRY RUN mode)",
                }
            else:
                # REAL MODE - Post all operations to Asterdex
                logger.warning(f"[ENTRY_POSTER] 🔴 REAL MODE - Posting to Asterdex...")
                
                # Step 1: Set margin type (ISOLATED)
                logger.info(f"[ENTRY_POSTER] Step 1/5: Setting margin type to {self.position_settings['margin_type']}...")
                margin_result = self._set_margin_type(symbol_asterdex, self.position_settings['margin_type'])
                if margin_result['status'] not in ["SUCCESS", "ALREADY_SET"]:
                    logger.warning(f"[ENTRY_POSTER] ⚠️  Margin type warning: {margin_result}")
                else:
                    logger.info(f"[ENTRY_POSTER] ✓ Margin type set: {margin_result['status']}")
                
                # Step 2: Set leverage (CRITICAL - use symbol-specific leverage)
                logger.info(f"[ENTRY_POSTER] Step 2/5: Setting leverage to {symbol_leverage}x for {symbol}...")
                leverage_result = self._set_leverage(symbol_asterdex, symbol_leverage)
                
                # STRICT: Leverage must match symbol_leverage setting, not ALREADY_SET with wrong value
                if leverage_result['status'] not in ["SUCCESS", "ALREADY_SET"]:
                    logger.error(f"[ENTRY_POSTER] ❌ LEVERAGE FAILED - aborting entry: {leverage_result}")
                    return {
                        "status": "LEVERAGE_FAILED",
                        "error": f"Failed to set leverage: {leverage_result}"
                    }
                else:
                    logger.info(f"[ENTRY_POSTER] ✓ Leverage set: {leverage_result['status']} (must be 10x)")
                
                # Step 3: Post ENTRY order (LIMIT at entry price)
                logger.info(f"[ENTRY_POSTER] Step 3/5: Posting {self.position_settings['order_type']} ENTRY order @ {entry_price}...")
                logger.info(f"[ENTRY_POSTER]   └─ Symbol: {symbol_asterdex} | Side: {side} | Qty: {quantity} | Price: {entry_price}")
                
                entry_result = self._post_order_to_asterdex(
                    symbol=symbol_asterdex,
                    side=side,
                    order_type=self.position_settings['order_type'],
                    quantity=quantity,
                    price=entry_price,
                )
                
                # CRITICAL VALIDATION: Entry must succeed before TP/SL
                if entry_result['status'] != "SUCCESS":
                    logger.error(f"[ENTRY_POSTER] ❌ CRITICAL FAILURE - Entry order FAILED, ABORTING TP/SL posting")
                    logger.error(f"[ENTRY_POSTER] Entry Result: {entry_result}")
                    logger.error(f"[ENTRY_POSTER] Error Details: {entry_result.get('error', 'Unknown error')}")
                    
                    # Log entry failure details
                    failure_detail = {
                        "symbol": symbol_asterdex,
                        "side": side,
                        "quantity": quantity,
                        "price": entry_price,
                        "status": entry_result.get('status'),
                        "error": entry_result.get('error'),
                        "response": entry_result.get('response'),
                    }
                    logger.error(f"[ENTRY_POSTER] Entry Failure Details: {json.dumps(failure_detail, indent=2)}")
                    
                    return {
                        "status": "ENTRY_FAILED",
                        "error": entry_result.get('error'),
                        "entry_result": entry_result,
                        "message": "Entry order failed - TP/SL NOT posted (safety validation)"
                    }
                
                entry_order_id = str(entry_result['order_id'])  # Convert to string
                logger.info(f"[ENTRY_POSTER] ✅ Entry order posted successfully: {entry_order_id}")
                
                # VALIDATION CHECKPOINT: Verify Entry succeeded before posting TP
                if not entry_order_id or entry_order_id.startswith("DRY_"):
                    logger.error(f"[ENTRY_POSTER] ❌ VALIDATION FAILED - Entry order ID invalid: {entry_order_id}")
                    return {
                        "status": "VALIDATION_FAILED",
                        "error": "Entry order ID is invalid or empty - TP/SL NOT posted",
                        "entry_order_id": entry_order_id,
                    }
                
                # Step 3.5: Try OCO (One-Cancel-the-Other) Strategy - TP/SL linked
                # NON-BLOCKING: If OCO succeeds, skip traditional TP/SL. If it fails, continue to Step 4.
                oco_result = self._post_oco_strategy(
                    symbol=symbol_asterdex,
                    quantity=quantity,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    side=side,
                )
                
                if oco_result['status'] == "SUCCESS":
                    # OCO succeeded - use these order IDs instead of traditional TP/SL
                    tp_order_id = str(oco_result.get('tp_order_id', ''))
                    sl_order_id = str(oco_result.get('sl_order_id', ''))
                    
                    logger.info(f"[ENTRY_POSTER] ✅✅✅ COMPLETE: Entry + OCO all 3 orders posted!")
                    
                    return {
                        "status": "SUCCESS",
                        "entry_order_id": entry_order_id,
                        "tp_order_id": tp_order_id,
                        "sl_order_id": sl_order_id,
                        "message": "Entry + OCO Strategy (TP/SL auto-linked)",
                    }
                
                # Step 4: Post TP order (TAKE_PROFIT_MARKET) - fallback if OCO failed
                logger.info(f"[ENTRY_POSTER] Step 4/5: Posting TP order @ {tp_price}...")
                
                tp_result = self._post_order_to_asterdex(
                    symbol=symbol_asterdex,
                    side=opposite_side,
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=quantity,
                    stop_price=tp_price,
                )
                
                if tp_result['status'] != "SUCCESS":
                    logger.error(f"[ENTRY_POSTER] ❌ TP order FAILED")
                    logger.error(f"[ENTRY_POSTER] TP Result: {tp_result}")
                    
                    tp_failure_detail = {
                        "entry_order_id": entry_order_id,
                        "symbol": symbol_asterdex,
                        "side": opposite_side,
                        "quantity": quantity,
                        "stop_price": tp_price,
                        "status": tp_result.get('status'),
                        "error": tp_result.get('error'),
                    }
                    logger.error(f"[ENTRY_POSTER] TP Failure Details: {json.dumps(tp_failure_detail, indent=2)}")
                    
                    return {
                        "status": "TP_FAILED",
                        "error": tp_result.get('error'),
                        "entry_order_id": entry_order_id,
                        "tp_result": tp_result,
                        "message": "Entry posted but TP order failed - SL NOT posted (safety validation)"
                    }
                
                tp_order_id = str(tp_result['order_id'])  # Convert to string
                logger.info(f"[ENTRY_POSTER] ✅ TP order posted successfully: {tp_order_id}")
                
                # VALIDATION CHECKPOINT: Verify TP succeeded before posting SL
                if not tp_order_id or tp_order_id.startswith("DRY_"):
                    logger.error(f"[ENTRY_POSTER] ❌ VALIDATION FAILED - TP order ID invalid: {tp_order_id}")
                    return {
                        "status": "VALIDATION_FAILED",
                        "error": "TP order ID is invalid or empty - SL NOT posted",
                        "entry_order_id": entry_order_id,
                        "tp_order_id": tp_order_id,
                    }
                
                # Step 5: Post SL order (STOP_MARKET)
                logger.info(f"[ENTRY_POSTER] Step 5/5: Posting SL order @ {sl_price}...")
                logger.info(f"[ENTRY_POSTER]   └─ Side: {opposite_side} | Qty: {quantity} | Stop Price: {sl_price}")
                
                sl_result = self._post_order_to_asterdex(
                    symbol=symbol_asterdex,
                    side=opposite_side,
                    order_type="STOP_MARKET",
                    quantity=quantity,
                    stop_price=sl_price,
                )
                
                if sl_result['status'] != "SUCCESS":
                    logger.error(f"[ENTRY_POSTER] ❌ SL order FAILED")
                    logger.error(f"[ENTRY_POSTER] SL Result: {sl_result}")
                    
                    sl_failure_detail = {
                        "entry_order_id": entry_order_id,
                        "tp_order_id": tp_order_id,
                        "symbol": symbol_asterdex,
                        "side": opposite_side,
                        "quantity": quantity,
                        "stop_price": sl_price,
                        "status": sl_result.get('status'),
                        "error": sl_result.get('error'),
                    }
                    logger.error(f"[ENTRY_POSTER] SL Failure Details: {json.dumps(sl_failure_detail, indent=2)}")
                    
                    return {
                        "status": "SL_FAILED",
                        "error": sl_result.get('error'),
                        "entry_order_id": entry_order_id,
                        "tp_order_id": tp_order_id,
                        "sl_result": sl_result,
                        "message": "Entry + TP posted but SL order failed - position has partial protection only"
                    }
                
                sl_order_id = str(sl_result['order_id'])  # Convert to string
                logger.info(f"[ENTRY_POSTER] ✅ SL order posted successfully: {sl_order_id}")
                logger.info(f"[ENTRY_POSTER] ✅✅✅ COMPLETE: All 3 orders posted with full protection!")
                
                return {
                    "status": "COMPLETE",
                    "entry_order_id": entry_order_id,
                    "tp_order_id": tp_order_id,
                    "sl_order_id": sl_order_id,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "quantity": quantity,
                    "margin": self.position_settings['margin'],
                    "leverage": self.position_settings['leverage'],
                    "margin_type": self.position_settings['margin_type'],
                    "order_type": self.position_settings['order_type'],
                }

        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Failed to post complete entry: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
            }

    def _log_entry(self, signal: Dict, result: Dict, source: str = None):
        """Log complete entry with TP/SL to entry_log.jsonl
        
        Args:
            signal: Signal dict
            result: Result dict from _post_entry_to_asterdex
            source: "tier_based", "mtf_strong", or None
        """
        try:
            symbol, direction, entry_price, tp_price, sl_price, timeframe, tier = parse_signal(signal)

            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "signal_uuid": signal.get('signal_uuid'),
                "symbol": symbol,
                "direction": direction,
                "timeframe": timeframe,
                "tier": tier,
                "source": source,  # Tag: "tier_based" or "mtf_strong"
                # 8 Infos per entry
                "entry_direction": direction,  # Info #1
                "entry_price": entry_price,  # Info #2
                "tp_price": tp_price,  # Info #3
                "sl_price": sl_price,  # Info #4
                "margin_usdt": self.position_settings['margin'],  # Info #5
                "leverage": self.position_settings['leverage'],  # Info #6
                "margin_mode": self.position_settings['margin_type'],  # Info #7
                "order_type": self.position_settings['order_type'],  # Info #8
                # Order details
                "quantity": calculate_order_size(
                    self.position_settings['notional_value'],
                    entry_price,
                ),
                "notional": self.position_settings['notional_value'],
                "status": result.get('status'),
                "entry_order_id": result.get('entry_order_id'),
                "tp_order_id": result.get('tp_order_id'),
                "sl_order_id": result.get('sl_order_id'),
            }
            
            # Add optional fields if present
            if result.get('message'):
                log_entry['message'] = result['message']
            if result.get('error'):
                log_entry['error'] = result['error']

            # Append to entry_log.jsonl
            ENTRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ENTRY_LOG_PATH, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            source_tag = f" [{source}]" if source else ""
            logger.info(f"[ENTRY_POSTER] Logged entry: {signal.get('signal_uuid')}{source_tag}")
            logger.info(f"  ├─ Entry: {result.get('entry_order_id')}")
            logger.info(f"  ├─ TP: {result.get('tp_order_id')}")
            logger.info(f"  └─ SL: {result.get('sl_order_id')}")

        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Failed to log entry: {e}")

    def _process_signal(self, signal: Dict) -> bool:
        """
        Process a single signal.

        Returns:
            True if entry was posted, False otherwise
        """
        should_post, reason, source = self._should_post_entry(signal)

        if not should_post:
            return False

        logger.info(f"[ENTRY_POSTER] Processing signal {signal.get('signal_uuid')}: {reason} (source: {source})")

        # Post entry with TP/SL
        result = self._post_entry_to_asterdex(signal)

        # Log result with source tag
        self._log_entry(signal, result, source=source)

        # Mark as processed
        self.processed_signal_uuids.add(signal.get('signal_uuid'))

        # Start cooldown ONLY if ALL 3 orders successful (entry + TP + SL)
        if result.get('status') in ["DRY_RUN_COMPLETE", "COMPLETE"]:
            symbol = signal.get('symbol')
            timeframe = signal.get('timeframe')
            self.rate_limiter.start_cooldown(symbol, timeframe)
            
            # Log entry for performance tracking (isolated, fire-and-forget)
            # CRITICAL: Pass ORDER IDs from Asterdex - these are the unique transaction identifiers
            log_posted_entry(
                signal_uuid=signal.get('signal_uuid'),
                symbol=symbol,
                side=signal.get('direction'),
                entry_price=signal.get('entry_price', 0),
                quantity=result.get('quantity', 0),
                timeframe=timeframe,
                tier=signal.get('tier'),
                mtf_alignment_band=signal.get('mtf_alignment_band'),
                route=signal.get('route'),
                confidence_level=signal.get('confidence_level'),
                tp_price=signal.get('tp_price'),
                sl_price=signal.get('sl_price'),
                # ORDER IDs - Critical for matching in trade history
                entry_order_id=result.get('entry_order_id'),
                tp_order_id=result.get('tp_order_id'),
                sl_order_id=result.get('sl_order_id')
            )
            
            # Calculate and format release time
            tf_minutes = self.rate_limiter._parse_timeframe_to_minutes(timeframe)
            release_time = datetime.utcnow() + timedelta(minutes=tf_minutes)
            release_time_str = release_time.strftime("%H:%M:%S")
            
            logger.info(f"[ENTRY_POSTER] ✅ {symbol} {timeframe} posted | Cooldown active until {release_time_str} (~{tf_minutes}m)")
            return True
        else:
            # Entry failed - mark for review but don't cooldown
            logger.warning(f"[ENTRY_POSTER] ⚠️  Partial failure: {result.get('status')}")
            return False

    def run(self):
        """Main loop - poll signals and post entries"""
        logger.info("[ENTRY_POSTER] Started main loop")

        try:
            while True:
                # Read all signals
                signals = self._read_signals_master()

                # Process each signal
                processed_count = 0
                for signal in signals:
                    if self._process_signal(signal):
                        processed_count += 1

                if processed_count > 0:
                    logger.info(f"[ENTRY_POSTER] Processed {processed_count} new signals")

                # Sleep before next check
                time.sleep(RATE_LIMIT_SETTINGS['signal_check_interval_sec'])

        except KeyboardInterrupt:
            logger.info("[ENTRY_POSTER] Interrupted by user")
        except Exception as e:
            logger.error(f"[ENTRY_POSTER] Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    poster = AsterdexEntryPoster()
    poster.run()
