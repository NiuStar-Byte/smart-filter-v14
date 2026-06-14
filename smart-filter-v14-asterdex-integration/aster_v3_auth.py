"""
aster_v3_auth.py - EIP-712 Web3 Authentication for Asterdex PRO API V3

Implements Web3-native request signing using EIP-712 standard.
"""

import time
import logging
import urllib.parse
from typing import Dict
from eth_account import Account
from eth_account.messages import encode_typed_data
import json

logger = logging.getLogger(__name__)


class AsterV3Auth:
    """EIP-712 authentication for Asterdex PRO API V3"""

    def __init__(self, wallet_address: str, private_key: str):
        """
        Initialize Web3 authentication.
        
        Args:
            wallet_address: Wallet address (0x...)
            private_key: Private key (0x...)
        """
        self.wallet_address = wallet_address.lower()
        self.private_key = private_key
        
        # Initialize account from private key
        try:
            self.account = Account.from_key(private_key)
            if self.account.address.lower() != self.wallet_address:
                raise ValueError(
                    f"Private key doesn't match wallet address!\n"
                    f"Expected: {self.wallet_address}\n"
                    f"Got: {self.account.address.lower()}"
                )
            logger.info(f"✅ Web3 account initialized: {self.wallet_address}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Web3 account: {e}")
            raise

    def _get_nonce(self) -> int:
        """
        Get current nonce (microseconds since epoch).
        
        Returns:
            Nonce in microseconds
        """
        # Use current time in microseconds
        # Can be monotonic: now_seconds * 1e6 + sequence
        nonce = int(time.time() * 1_000_000)
        return nonce

    def _build_param_string(self, params: Dict) -> str:
        """
        Build parameter string for signing using urllib.parse.urlencode (matches official Asterdex docs).
        
        This properly URL-encodes all special characters and builds the exact format
        that Asterdex signature verification expects.
        
        Args:
            params: Parameters dict
        
        Returns:
            URL-encoded parameter string (e.g., "symbol=BTCUSDT&nonce=123456&signer=0x...")
        """
        # Convert all values to strings
        string_params = {k: str(v) for k, v in params.items()}
        
        # Use urllib.parse.urlencode - this is what official Asterdex example uses
        # It properly handles URL encoding and parameter ordering
        param_string = urllib.parse.urlencode(string_params)
        
        return param_string

    def _eip712_encode(self, param_string: str) -> str:
        """
        Sign using EIP-712 (matches official Asterdex PRO API V3 documentation).
        
        This uses the exact same format as the official Asterdex example:
        - EIP712Domain with chainId=1666, name="AsterSignTransaction"
        - Message containing the URL-encoded parameter string
        - eth_account's encode_typed_data for proper EIP-712 encoding
        
        Args:
            param_string: URL-encoded parameter string (e.g., "symbol=BTCUSDT&nonce=123&signer=0x...")
        
        Returns:
            Hex signature (e.g., "0x...")
        """
        try:
            # Domain data (matches official Asterdex example)
            domain_data = {
                "name": "AsterSignTransaction",
                "version": "1",
                "chainId": 1666,
                "verifyingContract": "0x0000000000000000000000000000000000000000"
            }
            
            # Message types (EIP-712 format)
            message_types = {
                "Message": [
                    {"name": "msg", "type": "string"}
                ]
            }
            
            # Message data (contains the URL-encoded parameter string)
            message_data = {
                "msg": param_string  # The URL-encoded param string (e.g., "symbol=BTCUSDT&nonce=...")
            }
            
            # Encode using eth_account's encode_typed_data (available in eth_account >= 0.5.x)
            message = encode_typed_data(
                domain_data=domain_data,
                message_types=message_types,
                message_data=message_data
            )
            
            # Sign the encoded message
            signed_message = Account.sign_message(message, private_key=self.private_key)
            
            # Return signature as hex (includes 0x prefix)
            signature_hex = signed_message.signature.hex()
            
            logger.debug(f"Signature generated: {signature_hex[:20]}...")
            return signature_hex
            
        except Exception as e:
            logger.error(f"❌ EIP-712 signing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def sign_request(self, params: Dict, main_account: str = None) -> Dict:
        """
        Sign a request with nonce and EIP-712 signature (matches official Asterdex PRO API V3).
        
        Adds: nonce (microseconds), user (main_account), signer (API wallet), signature
        
        Args:
            params: Request parameters (symbol, side, type, quantity, price, etc.)
            main_account: Main trading account ("user" in API). If None, uses wallet_address.
        
        Returns:
            Params dict with nonce, user, signer, and signature added (ready for POST)
        """
        # Step 1: Add nonce (microseconds since epoch)
        nonce = self._get_nonce()
        params["nonce"] = str(nonce)  # Convert to string for URL encoding
        
        # Step 2: Add user and signer (both required for signature)
        params["user"] = main_account if main_account else self.wallet_address
        params["signer"] = self.wallet_address  # Always the API wallet (signer)
        
        # Step 3: Build parameter string for signing (URL-encoded, like official docs)
        param_string = self._build_param_string(params)
        
        logger.debug(f"Parameters for signing:")
        logger.debug(f"  Nonce: {params['nonce']}")
        logger.debug(f"  User: {params['user']}")
        logger.debug(f"  Signer: {params['signer']}")
        logger.debug(f"  Param string (first 100 chars): {param_string[:100]}...")
        
        # Step 4: Sign with EIP-712
        signature = self._eip712_encode(param_string)
        
        # Step 5: Add signature to params
        params["signature"] = signature
        
        logger.debug(f"  Signature: {signature[:20]}...")
        
        return params

    def sign_request_v3(self, base_params: Dict, main_account: str = None) -> Dict:
        """
        Sign request following PRO API V3 format (exactly matches official Asterdex docs).
        
        The signature includes: business params + nonce + user + signer
        
        Args:
            base_params: Base parameters (symbol, side, type, quantity, price, etc.)
            main_account: Main trading account ("user" in API). If None, uses wallet_address.
        
        Returns:
            Signed params ready for API call with nonce, user, signer, signature
        """
        # Call sign_request which adds nonce, user, signer, and signature
        # This matches the official Asterdex PRO API V3 implementation
        signed_params = self.sign_request(base_params, main_account=main_account)
        
        return signed_params


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Test with environment variables
    wallet_address = os.environ.get("ASTER_WALLET_ADDRESS")
    private_key = os.environ.get("ASTER_WALLET_PRIVATE_KEY")
    
    if not wallet_address or not private_key:
        print("❌ Environment variables not set")
        print("Set: export ASTER_WALLET_ADDRESS='0x...'")
        print("Set: export ASTER_WALLET_PRIVATE_KEY='0x...'")
        exit(1)
    
    # Initialize auth
    auth = AsterV3Auth(wallet_address, private_key)
    
    # Test signing
    test_params = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "LIMIT",
        "quantity": "0.001",
        "price": "67500",
    }
    
    signed = auth.sign_request_v3(test_params)
    print(f"✅ Test signature successful!")
    print(f"   Nonce: {signed.get('nonce')}")
    print(f"   Signature: {signed.get('signature')[:20]}...")
    print(f"   User: {signed.get('user')}")
    print(f"   Signer: {signed.get('signer')}")
