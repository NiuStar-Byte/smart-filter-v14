"""
diagnose_oco_params.py - Test updateStrategyOrder with different parameter formats

The endpoint exists but signature check fails. Try:
1. Different parameter names
2. Different request structure
"""

import sys
import time
import requests
import json

sys.path.insert(0, '.')

from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
from aster_v3_auth import AsterV3Auth

print("\n" + "=" * 80)
print("OCO Parameter Format Testing")
print("=" * 80)

auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)

# Try different parameter formats
test_params = [
    {
        "name": "Format 1: type1/type2 with details",
        "params": {
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
    },
    {
        "name": "Format 2: Simple (orders as array?)",
        "params": {
            'symbol': 'BTCUSDT',
            'orders': json.dumps([
                {
                    'type': 'TAKE_PROFIT',
                    'side': 'SELL',
                    'quantity': '0.001',
                    'price': '65500',
                    'timeInForce': 'GTC'
                },
                {
                    'type': 'STOP_LOSS',
                    'side': 'SELL',
                    'quantity': '0.001',
                    'price': '64500',
                    'timeInForce': 'GTC'
                }
            ])
        }
    },
    {
        "name": "Format 3: orderDetails as string",
        "params": {
            'symbol': 'BTCUSDT',
            'orderDetails': '[{"type":"TAKE_PROFIT","side":"SELL","quantity":"0.001","price":"65500","timeInForce":"GTC"},{"type":"STOP_LOSS","side":"SELL","quantity":"0.001","price":"64500","timeInForce":"GTC"}]'
        }
    }
]

url = "https://fapi.asterdex.com/fapi/v3/updateStrategyOrder"
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

for test in test_params:
    print(f"\n{'─' * 80}")
    print(f"Testing: {test['name']}")
    print(f"{'─' * 80}")
    
    # Sign
    signed = auth.sign_request_v3(test['params'], main_account=ASTER_MAIN_ACCOUNT)
    
    try:
        response = requests.post(url, data=signed, headers=headers, timeout=10)
        
        print(f"HTTP Status: {response.status_code}")
        
        try:
            result = response.json()
            if response.status_code == 200:
                print(f"✅ SUCCESS!")
            else:
                print(f"Code: {result.get('code')}")
                print(f"Message: {result.get('msg')}")
        except:
            print(f"Response: {response.text[:300]}")
                
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {str(e)[:200]}")

print(f"\n{'=' * 80}")
