"""
asterdex_utils.py - Helper functions for Asterdex integration

Functions:
- sign_request() → HMAC-SHA256 signature for API calls
- parse_signal() → Extract symbol, direction, TP, SL from signal dict
- check_tier() → Validate Tier 1/2/3
- calculate_order_size() → Convert $10 notional to contracts
- format_log_entry() → Standardized logging
"""

import hashlib
import hmac
import time
import json
import logging
from typing import Dict, Tuple, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


def sign_request(
    api_secret: str,
    query_string: str,
) -> str:
    """
    Create HMAC-SHA256 signature for Asterdex API requests.

    Args:
        api_secret: Your Asterdex API secret
        query_string: Query parameters (e.g., "symbol=BTCUSDT&side=BUY&...")

    Returns:
        Hex-encoded signature
    """
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature


def parse_signal(signal_dict: Dict) -> Tuple[str, str, float, float, float, str, int]:
    """
    Extract trading parameters from signal dictionary.

    Args:
        signal_dict: Signal from SIGNALS_MASTER.jsonl

    Returns:
        (symbol, direction, entry_price, tp_price, sl_price, timeframe, tier)
    """
    try:
        symbol = signal_dict.get('symbol', '').upper()
        
        # Direction: try 'direction' first, fall back to 'signal_type'
        direction = signal_dict.get('direction')
        if not direction:
            signal_type = signal_dict.get('signal_type', '').upper()
            if signal_type in ['LONG', 'SHORT']:
                direction = signal_type
        direction = (direction or '').upper()
        
        entry_price = float(signal_dict.get('entry_price', 0))
        
        # TP Price: try 'tp_price' first, fall back to 'tp_target'
        tp_price = signal_dict.get('tp_price')
        if tp_price is None:
            tp_price = signal_dict.get('tp_target')
        tp_price = float(tp_price or 0)
        
        # SL Price: try 'sl_price' first, fall back to 'sl_target'
        sl_price = signal_dict.get('sl_price')
        if sl_price is None:
            sl_price = signal_dict.get('sl_target')
        sl_price = float(sl_price or 0)
        
        timeframe = signal_dict.get('timeframe', '')
        tier = signal_dict.get('tier')

        # Validate required fields
        if not all([symbol, direction, entry_price, tp_price, sl_price, timeframe]):
            missing = []
            if not symbol: missing.append('symbol')
            if not direction: missing.append('direction')
            if not entry_price: missing.append('entry_price')
            if not tp_price: missing.append('tp_price')
            if not sl_price: missing.append('sl_price')
            if not timeframe: missing.append('timeframe')
            raise ValueError(f"Missing fields: {missing}")

        return symbol, direction, entry_price, tp_price, sl_price, timeframe, tier

    except Exception as e:
        logger.error(f"[UTILS] Failed to parse signal: {e}")
        raise


def check_tier(tier: Optional[int], allowed_tiers: list = [1, 2, 3]) -> bool:
    """
    Check if signal tier is in the allowed list.

    Args:
        tier: Signal tier (1, 2, 3, "Tier-1"/"Tier-2"/"Tier-3", or None for unassigned)
        allowed_tiers: List of acceptable tiers (integers)

    Returns:
        True if tier is allowed, False otherwise
    """
    # Handle string format "Tier-2" → extract 2
    if isinstance(tier, str):
        if tier == "Tier-X" or tier.lower() == "unassigned":
            tier_num = None
        else:
            # Extract number from "Tier-2" format
            try:
                tier_num = int(tier.split("-")[-1])
            except (ValueError, IndexError):
                tier_num = None
    else:
        tier_num = tier
    
    return tier_num in allowed_tiers


def convert_symbol_format(symbol: str, to_format: str = "asterdex") -> str:
    """
    Convert between symbol formats.

    Args:
        symbol: Symbol string (e.g., "BTC-USDT" or "BTCUSDT")
        to_format: Target format ("asterdex" = BTCUSDT, "binance" = BTC-USDT)

    Returns:
        Converted symbol
    """
    if to_format == "asterdex":
        return symbol.replace("-", "")
    elif to_format == "binance":
        # Assume symbol is like BTCUSDT, need to insert hyphen before USDT
        if symbol.endswith("USDT"):
            return symbol[:-4] + "-USDT"
        return symbol
    else:
        return symbol


def calculate_order_size(
    notional_value: float,
    entry_price: float,
    precision: int = 8,
) -> float:
    """
    Calculate order quantity from notional value.

    Args:
        notional_value: Dollar amount (e.g., $10)
        entry_price: Entry price per unit
        precision: Decimal precision for quantity

    Returns:
        Order quantity
    """
    if entry_price <= 0:
        raise ValueError(f"Invalid entry price: {entry_price}")

    quantity = notional_value / entry_price
    # Round to precision
    quantity = float(f"{quantity:.{precision}f}")

    return quantity


def calculate_side(direction: str) -> str:
    """
    Convert direction to Asterdex side.

    Args:
        direction: "LONG" or "SHORT"

    Returns:
        "BUY" (for LONG) or "SELL" (for SHORT)
    """
    direction = direction.upper()
    if direction == "LONG":
        return "BUY"
    elif direction == "SHORT":
        return "SELL"
    else:
        raise ValueError(f"Invalid direction: {direction}")


def format_log_entry(
    timestamp: str,
    signal_uuid: str,
    symbol: str,
    direction: str,
    tier: int,
    timeframe: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    quantity: float,
    notional: float,
    asterdex_order_id: str = None,
    status: str = "PENDING",
    fill_price: float = None,
    fill_time: str = None,
) -> Dict:
    """
    Format entry into standardized log format.

    Returns:
        Dictionary ready for JSON serialization to entry_log.jsonl
    """
    entry = {
        "timestamp": timestamp,
        "signal_uuid": signal_uuid,
        "symbol": symbol,
        "direction": direction,
        "timeframe": timeframe,
        "tier": tier,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "quantity": quantity,
        "notional": notional,
        "status": status,
    }

    if asterdex_order_id:
        entry["asterdex_order_id"] = asterdex_order_id

    if fill_price:
        entry["fill_price"] = fill_price

    if fill_time:
        entry["fill_time"] = fill_time

    return entry


def get_unix_timestamp_ms() -> int:
    """Get current Unix timestamp in milliseconds"""
    return int(time.time() * 1000)


def build_query_string(params: Dict) -> str:
    """
    Build URL query string from dict.

    Args:
        params: Dictionary of parameters

    Returns:
        Query string (e.g., "symbol=BTCUSDT&side=BUY&...")
    """
    items = []
    for key, value in sorted(params.items()):
        items.append(f"{key}={value}")
    return "&".join(items)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test parse_signal
    signal = {
        "symbol": "BTC-USDT",
        "direction": "LONG",
        "entry_price": 67500,
        "tp_price": 69000,
        "sl_price": 66000,
        "timeframe": "4h",
        "tier": 2,
    }
    symbol, direction, ep, tp, sl, tf, tier = parse_signal(signal)
    print(f"✅ Parse signal: {symbol} {direction} @ {ep} (TP:{tp}, SL:{sl})")

    # Test tier check
    print(f"✅ Tier check: {check_tier(2, [1,2,3])}")
    print(f"✅ Tier check (reject X): {check_tier(None, [1,2,3])}")

    # Test symbol conversion
    print(f"✅ Symbol convert: {convert_symbol_format('BTC-USDT', 'asterdex')}")

    # Test order size
    size = calculate_order_size(10, 67500)
    print(f"✅ Order size: ${10} / $67500 = {size:.8f} contracts")

    # Test side conversion
    print(f"✅ Direction to side: LONG → {calculate_side('LONG')}, SHORT → {calculate_side('SHORT')}")

    # Test log entry
    log_entry = format_log_entry(
        timestamp="2026-05-29T13:50:00Z",
        signal_uuid="abc-123",
        symbol="BTC-USDT",
        direction="LONG",
        tier=2,
        timeframe="4h",
        entry_price=67500,
        tp_price=69000,
        sl_price=66000,
        quantity=0.000148,
        notional=10.0,
        status="PENDING",
    )
    print(f"✅ Log entry: {json.dumps(log_entry, indent=2)}")
