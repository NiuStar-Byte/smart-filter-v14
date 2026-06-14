"""
asterdex_exchange_info.py - Exchange Info Cache & Parameter Validation

Fetches symbol precision rules from Asterdex exchangeInfo endpoint.
Validates and rounds prices/quantities to meet exchange constraints.

Based on: https://github.com/asterdex/api-docs/blob/master/V3(Recommended)/EN/aster-finance-futures-api-v3.md
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, Optional
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)

EXCHANGE_INFO_CACHE_PATH = Path(__file__).parent / "exchange_info_cache.json"
ASTERDEX_ENDPOINT = "https://fapi.asterdex.com"


class ExchangeInfoCache:
    """Cached exchange info with symbol filter validation"""
    
    def __init__(self):
        self.symbols: Dict = {}
        self.loaded = False
        self._load_or_fetch()
    
    def _load_or_fetch(self):
        """Load from cache or fetch fresh from API"""
        if EXCHANGE_INFO_CACHE_PATH.exists():
            try:
                with open(EXCHANGE_INFO_CACHE_PATH, 'r') as f:
                    data = json.load(f)
                    self.symbols = {s['symbol']: s for s in data.get('symbols', [])}
                    self.loaded = True
                    logger.info(f"✅ Loaded {len(self.symbols)} symbols from cache")
                    return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, fetching fresh...")
        
        # Load fallback defaults (known symbols with conservative constraints)
        self._load_fallback_symbols()
        
        # Try to fetch from API to refresh cache (but don't fail if we can't)
        try:
            response = requests.get(f"{ASTERDEX_ENDPOINT}/fapi/v3/exchangeInfo", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.symbols = {s['symbol']: s for s in data.get('symbols', [])}
                
                # Cache for future use
                with open(EXCHANGE_INFO_CACHE_PATH, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.loaded = True
                logger.info(f"✅ Fetched {len(self.symbols)} symbols from API, cached")
                return
            else:
                logger.warning(f"Exchange info fetch returned {response.status_code}, using fallback")
        except Exception as e:
            logger.warning(f"Exchange info fetch failed ({e}), using fallback symbols")
        
        # If we got here, we're using fallback
        self.loaded = True  # Loaded from fallback
    
    def _load_fallback_symbols(self):
        """Load fallback symbol definitions for common trading pairs"""
        # Conservative estimates based on typical Asterdex constraints
        # 2-decimal tickSize for most, 1-decimal for higher prices
        self.symbols = {
            'BTCUSDT': {
                'symbol': 'BTCUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.00001', 'maxQty': '10000', 'stepSize': '0.00001'}
                ]
            },
            'ETHUSDT': {
                'symbol': 'ETHUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.0001', 'maxQty': '10000', 'stepSize': '0.0001'}
                ]
            },
            'SOLUSDT': {
                'symbol': 'SOLUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '10000', 'stepSize': '0.001'}
                ]
            },
            'BNBUSDT': {
                'symbol': 'BNBUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '10000', 'stepSize': '0.001'}
                ]
            },
            'AVAXUSDT': {
                'symbol': 'AVAXUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '10000', 'stepSize': '0.001'}
                ]
            },
            'LINKUSDT': {
                'symbol': 'LINKUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '10000', 'stepSize': '0.001'}
                ]
            },
            'BONKUSDT': {
                'symbol': 'BONKUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.000001', 'maxPrice': '1', 'tickSize': '0.000001'},
                    {'filterType': 'LOT_SIZE', 'minQty': '1', 'maxQty': '1000000000', 'stepSize': '1'}
                ]
            },
            'SPKUSDT': {
                'symbol': 'SPKUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.00001', 'maxPrice': '1000', 'tickSize': '0.00001'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.1', 'maxQty': '100000', 'stepSize': '0.1'}
                ]
            },
            'CROSSUSDT': {
                'symbol': 'CROSSUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.00001', 'maxPrice': '1000', 'tickSize': '0.00001'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.1', 'maxQty': '100000', 'stepSize': '0.1'}
                ]
            }
        }
        logger.info(f"✅ Loaded {len(self.symbols)} fallback symbols (common pairs)")
    
    def get_filters(self, symbol: str) -> Optional[Dict]:
        """Get PRICE_FILTER and LOT_SIZE for symbol"""
        sym_data = self.symbols.get(symbol)
        if not sym_data:
            return None
        
        filters = {}
        for f in sym_data.get('filters', []):
            if f['filterType'] in ['PRICE_FILTER', 'LOT_SIZE']:
                filters[f['filterType']] = f
        
        return filters if filters else None
    
    def validate_price(self, symbol: str, price: float) -> Optional[float]:
        """
        Validate and round price to PRICE_FILTER constraints.
        
        Returns:
            Rounded price, or None if invalid
        """
        filters = self.get_filters(symbol)
        if not filters or 'PRICE_FILTER' not in filters:
            logger.warning(f"No PRICE_FILTER for {symbol}, skipping validation")
            return None
        
        pf = filters['PRICE_FILTER']
        min_price = float(pf['minPrice'])
        max_price = float(pf['maxPrice'])
        tick_size = float(pf['tickSize'])
        
        # Check bounds
        if price < min_price or price > max_price:
            logger.error(f"Price {price} out of bounds [{min_price}, {max_price}] for {symbol}")
            return None
        
        # Round to tick size
        if tick_size == 0:
            return price  # No constraint
        
        # Round DOWN to nearest tick (conservative)
        decimal_price = Decimal(str(price))
        decimal_tick = Decimal(str(tick_size))
        rounded = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_tick
        
        return float(rounded)
    
    def validate_quantity(self, symbol: str, quantity: float) -> Optional[float]:
        """
        Validate and round quantity to LOT_SIZE constraints.
        
        Returns:
            Rounded quantity, or None if invalid
        """
        filters = self.get_filters(symbol)
        if not filters or 'LOT_SIZE' not in filters:
            logger.warning(f"No LOT_SIZE for {symbol}, skipping validation")
            return None
        
        ls = filters['LOT_SIZE']
        min_qty = float(ls['minQty'])
        max_qty = float(ls['maxQty'])
        step_size = float(ls['stepSize'])
        
        # Check bounds
        if quantity < min_qty or quantity > max_qty:
            logger.error(f"Quantity {quantity} out of bounds [{min_qty}, {max_qty}] for {symbol}")
            return None
        
        # Round to step size
        if step_size == 0:
            return quantity  # No constraint
        
        # Round DOWN to nearest step (conservative)
        decimal_qty = Decimal(str(quantity))
        decimal_step = Decimal(str(step_size))
        rounded = (decimal_qty / decimal_step).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_step
        
        return float(rounded)


# Global cache instance
_exchange_info = None


def get_exchange_info() -> ExchangeInfoCache:
    """Get or create global exchange info cache"""
    global _exchange_info
    if _exchange_info is None:
        _exchange_info = ExchangeInfoCache()
    return _exchange_info


def validate_order_params(symbol: str, price: float, quantity: float) -> Dict:
    """
    Validate and round order parameters to exchange constraints.
    
    Returns:
        Dict with validated_price, validated_quantity, valid (bool), error (str)
    """
    info = get_exchange_info()
    
    if not info.loaded:
        return {
            "valid": False,
            "error": "Exchange info not loaded",
            "validated_price": None,
            "validated_quantity": None
        }
    
    # Validate price
    validated_price = info.validate_price(symbol, price)
    if validated_price is None:
        return {
            "valid": False,
            "error": f"Invalid price {price} for {symbol}",
            "validated_price": None,
            "validated_quantity": None
        }
    
    # Validate quantity
    validated_qty = info.validate_quantity(symbol, quantity)
    if validated_qty is None:
        return {
            "valid": False,
            "error": f"Invalid quantity {quantity} for {symbol}",
            "validated_price": validated_price,
            "validated_quantity": None
        }
    
    # Success
    return {
        "valid": True,
        "error": None,
        "validated_price": validated_price,
        "validated_quantity": validated_qty
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    result = validate_order_params("BTCUSDT", 66660.0732, 0.00015001)
    print(f"✅ Result: {result}")
    
    # Try other symbols
    for sym, price, qty in [
        ("ETHUSDT", 1995.28272, 0.00501182),
        ("SOLUSDT", 83.30661, 0.12003849),
    ]:
        result = validate_order_params(sym, price, qty)
        print(f"  {sym}: valid={result['valid']}, price={result['validated_price']}, qty={result['validated_quantity']}")
