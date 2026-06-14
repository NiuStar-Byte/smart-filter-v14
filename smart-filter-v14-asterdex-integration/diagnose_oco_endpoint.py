"""
diagnose_oco_endpoint.py - Test both possible OCO endpoints

Tests:
1. POST /fapi/v3/strategyOrder (what we tried first - got 404)
2. POST /fapi/v3/updateStrategyOrder (user's suggestion from API docs)
"""

import sys
import time
import requests
import json

sys.path.insert(0, '.')

from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
from aster_v3_auth import AsterV3Auth

print("\n" + "=" * 80)
print("OCO ENDPOINT DIAGNOSTICS - Testing Both Possible Endpoints")
print("=" * 80)

auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)

# Test parameters
params = {
    'symbol': 'BTCUSDT',
    'type1': 'TAKE_PROFIT',
    'side1': 'SELL',
    'quantity1': '0.001',
    'price1': '65500',
    'timeInForce1': 'GTC',
    'type2': 'STOP_LOSS',
    'side2': 'SELL',
    'quantity2': '0.001',
    'price2': '64500',
    'timeInForce2': 'GTC',
}

endpoints = [
    "https://fapi.asterdex.com/fapi/v3/strategyOrder",      # Original (404)
    "https://fapi.asterdex.com/fapi/v3/updateStrategyOrder", # From API docs
]

for i, url in enumerate(endpoints, 1):
    print(f"\n{'─' * 80}")
    print(f"TEST {i}: {url}")
    print(f"{'─' * 80}")
    
    # Sign
    signed = auth.sign_request_v3(params, main_account=ASTER_MAIN_ACCOUNT)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, data=signed, headers=headers, timeout=10)
        
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ SUCCESS!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Failed")
            try:
                print(f"Response: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"Response: {response.text[:200]}")
                
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {str(e)[:200]}")

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print("""
If updateStrategyOrder returns 200:
  ✅ That's the correct endpoint for creating OCO orders
  → Update asterdex_entry_poster.py to use /fapi/v3/updateStrategyOrder

If both return 404:
  → OCO not supported on Asterdex (use fallback TP/SL)
  
If neither works but we get a different error:
  → Need to adjust request parameters or auth
""")
