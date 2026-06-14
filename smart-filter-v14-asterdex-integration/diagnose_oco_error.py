"""
diagnose_oco_error.py - Diagnose why OCO is failing

This script tests the OCO endpoint directly to see what error we get.
"""

import sys
import time
import requests
import json

sys.path.insert(0, '.')

from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
from aster_v3_auth import AsterV3Auth

print("\n" + "=" * 70)
print("OCO ENDPOINT DIAGNOSTICS")
print("=" * 70)

auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)

# Try OCO request
print("\n[TEST] Sending OCO Strategy Order request...")

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

print(f"\nRequest Parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

# Sign
signed = auth.sign_request_v3(params, main_account=ASTER_MAIN_ACCOUNT)

headers = {'Content-Type': 'application/x-www-form-urlencoded'}
url = "https://fapi.asterdex.com/fapi/v3/strategyOrder"

print(f"\nEndpoint: POST {url}")

# Send
response = requests.post(url, data=signed, headers=headers, timeout=10)

print(f"\nResponse Status: {response.status_code}")
print(f"Response Body:")
print(json.dumps(response.json(), indent=2))

# Analyze
if response.status_code != 200:
    error_data = response.json()
    print(f"\n⚠️  ERROR DETAILS:")
    print(f"  Code: {error_data.get('code')}")
    print(f"  Message: {error_data.get('msg')}")
    print(f"\n💡 DIAGNOSIS:")
    
    if error_data.get('code') == -1000:
        print("  → Invalid symbol or endpoint not found")
        print("  → Try: /fapi/v3/batchOrders with 2 separate orders?")
    elif "order type" in str(error_data.get('msg')).lower():
        print("  → Order type not valid for strategy orders")
        print("  → Try: Different order types or different endpoint")
    elif "unknown" in str(error_data.get('msg')).lower():
        print("  → Endpoint /fapi/v3/strategyOrder might not exist")
        print("  → Try: Check API documentation for correct endpoint")
    else:
        print(f"  → {error_data.get('msg')}")
else:
    print("\n✅ OCO request succeeded!")
    print(f"Orders: {response.json().get('orders')}")

print("\n" + "=" * 70)
