"""
Signal Dual-Write Verification Module

Verifies that signals are written to BOTH SIGNALS_MASTER.jsonl and 
SIGNALS_INDEPENDENT_AUDIT.txt successfully before confirming the signal fire.

This is Phase 1 of the dual-write prevention strategy:
- Verify both writes succeed
- Fail-safe: halt on any write failure
- Detailed logging for troubleshooting
"""

import json
import time
import os
from datetime import datetime

class DualWriteVerifier:
    """Verifies signals are written to both files"""
    
    def __init__(
        self,
        master_path: str,
        audit_path: str,
        timeout_sec: float = 10.0,
        debug: bool = False
    ):
        self.master_path = master_path
        self.audit_path = audit_path
        self.timeout_sec = timeout_sec
        self.debug = debug
        self.failed_writes = []
    
    def log(self, level: str, message: str):
        """Log dual-write events"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        prefix = f"[{timestamp}] [{level:8s}]"
        print(f"{prefix} {message}", flush=True)
    
    def verify_write(self, signal_uuid: str, signal_data: dict) -> bool:
        """
        Verify that signal was written to BOTH files
        
        Args:
            signal_uuid: The signal UUID to verify
            signal_data: The signal data that was written
            
        Returns:
            True if both writes confirmed, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < self.timeout_sec:
            try:
                # Check MASTER file
                master_found = self._check_file(self.master_path, signal_uuid, signal_data)
                
                # Check AUDIT file
                audit_found = self._check_file(self.audit_path, signal_uuid, signal_data)
                
                if master_found and audit_found:
                    if self.debug:
                        self.log("DEBUG", f"Dual-write verified for {signal_uuid}")
                    return True
                
                # If either not found, retry after brief pause
                time.sleep(0.1)
                
            except Exception as e:
                self.log("ERROR", f"Error during verification: {e}")
                return False
        
        # Timeout reached
        self.log("WARN", f"Dual-write verification timeout for {signal_uuid}")
        self.failed_writes.append(signal_uuid)
        return False
    
    def _check_file(self, file_path: str, signal_uuid: str, signal_data: dict) -> bool:
        """Check if signal exists in file with matching key fields"""
        try:
            if not os.path.exists(file_path):
                return False
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line.strip())
                        if sig.get('signal_uuid') == signal_uuid:
                            # Verify key fields match
                            if self._fields_match(sig, signal_data):
                                return True
                    except json.JSONDecodeError:
                        continue
            
            return False
        
        except Exception as e:
            self.log("ERROR", f"Error checking file {file_path}: {e}")
            return False
    
    def _fields_match(self, file_sig: dict, original_sig: dict) -> bool:
        """Verify critical fields match between stored and original"""
        critical_fields = [
            'signal_uuid', 'symbol', 'direction', 'timeframe',
            'entry_price', 'tp_price', 'sl_price', 'fired_time_utc'
        ]
        
        for field in critical_fields:
            if file_sig.get(field) != original_sig.get(field):
                return False
        
        return True
    
    def get_status(self) -> dict:
        """Get current verification status"""
        return {
            'failed_writes': len(self.failed_writes),
            'failed_uuids': self.failed_writes,
            'master_path': self.master_path,
            'audit_path': self.audit_path
        }


# Global verifier instance
_dual_write_verifier = None


def initialize_dual_write_verifier(
    master_path: str,
    audit_path: str,
    debug: bool = False
) -> DualWriteVerifier:
    """Initialize the global dual-write verifier"""
    global _dual_write_verifier
    _dual_write_verifier = DualWriteVerifier(
        master_path=master_path,
        audit_path=audit_path,
        debug=debug
    )
    return _dual_write_verifier


def verify_signal_dual_write(
    signal_uuid: str,
    signal_data: dict,
    raise_on_failure: bool = True
) -> bool:
    """
    Verify signal was written to both files
    
    Args:
        signal_uuid: UUID to verify
        signal_data: Signal data that was written
        raise_on_failure: Raise exception if verification fails
        
    Returns:
        True if both files verified
        
    Raises:
        RuntimeError if raise_on_failure=True and verification fails
    """
    global _dual_write_verifier
    
    if _dual_write_verifier is None:
        raise RuntimeError("Dual-write verifier not initialized")
    
    result = _dual_write_verifier.verify_write(signal_uuid, signal_data)
    
    if not result and raise_on_failure:
        raise RuntimeError(
            f"Dual-write verification failed for signal {signal_uuid}. "
            f"Signal may not be in both MASTER and AUDIT files."
        )
    
    return result


def get_dual_write_status() -> dict:
    """Get current dual-write verification status"""
    global _dual_write_verifier
    
    if _dual_write_verifier is None:
        return {'status': 'not_initialized'}
    
    return _dual_write_verifier.get_status()
