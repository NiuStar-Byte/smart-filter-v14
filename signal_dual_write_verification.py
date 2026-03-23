"""
Signal Dual-Write Verification Module

Verifies that signals are written to BOTH SIGNALS_MASTER.jsonl and 
SIGNALS_INDEPENDENT_AUDIT.txt successfully before confirming the signal fire.

This is Phase 1 of the dual-write prevention strategy:
- Verify both writes succeed
- Alert on failure (don't halt)
- Continue signal generation (trading never stops)
- Auto-recovery in background (Phases 2-3)
- Detailed logging for troubleshooting
"""

import json
import time
import os
from datetime import datetime
from collections import defaultdict


class DivergenceTracker:
    """Track dual-write failures for monitoring and auto-recovery"""
    
    def __init__(self):
        self.failures = []  # List of failed signal UUIDs
        self.failure_times = []  # Timestamps of failures
        self.failure_counts_by_hour = defaultdict(int)
    
    def record_failure(self, signal_uuid: str):
        """Record a dual-write failure"""
        timestamp = datetime.utcnow()
        self.failures.append(signal_uuid)
        self.failure_times.append(timestamp)
        
        # Track hourly stats
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        self.failure_counts_by_hour[hour_key] += 1
    
    def get_gap_size(self) -> int:
        """Current estimated divergence gap"""
        return len(self.failures)
    
    def get_failure_rate(self, window: int = 100) -> float:
        """% of signals with failures (last N signals)"""
        recent = self.failures[-window:]
        return len(recent) / window if recent else 0
    
    def get_failure_rate_percent(self, window: int = 100) -> float:
        """Failure rate as percentage"""
        return self.get_failure_rate(window) * 100
    
    def get_hourly_stats(self) -> dict:
        """Get failure counts by hour"""
        return dict(self.failure_counts_by_hour)
    
    def get_status(self) -> dict:
        """Get full status"""
        return {
            'total_failures': len(self.failures),
            'gap_size': self.get_gap_size(),
            'failure_rate_percent': self.get_failure_rate_percent(),
            'hourly_stats': self.get_hourly_stats(),
            'last_failure': self.failure_times[-1] if self.failure_times else None
        }
    
    def clear_old_failures(self, older_than_hours: int = 24):
        """Clear old failures (for long-running processes)"""
        cutoff_time = datetime.utcnow() - __import__('datetime').timedelta(hours=older_than_hours)
        
        # Keep only recent failures
        recent_failures = []
        recent_times = []
        
        for uuid, timestamp in zip(self.failures, self.failure_times):
            if timestamp > cutoff_time:
                recent_failures.append(uuid)
                recent_times.append(timestamp)
        
        self.failures = recent_failures
        self.failure_times = recent_times
        
        return len(self.failures)


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
    
    def verify_write(self, signal_uuid: str, signal_data: dict, raise_on_failure: bool = False) -> bool:
        """
        Verify that signal was written to BOTH files
        
        STRATEGY: Alert + Continue (default behavior)
        - Returns False if either write missing
        - raise_on_failure=False: Log alert, return False (RECOMMENDED)
        - raise_on_failure=True: Raise exception (not recommended, halts daemon)
        
        Args:
            signal_uuid: The signal UUID to verify
            signal_data: The signal data that was written
            raise_on_failure: If True, raise exception on failure (causes halt)
                             If False, log alert and return False (signals continue)
            
        Returns:
            True if both writes confirmed, False otherwise (when raise_on_failure=False)
            
        Raises:
            RuntimeError if raise_on_failure=True and verification fails
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
                if raise_on_failure:
                    raise RuntimeError(f"Verification error: {e}")
                return False
        
        # Timeout reached - verification failed
        self.log("WARN", f"Dual-write verification timeout for {signal_uuid}")
        self.failed_writes.append(signal_uuid)
        
        if raise_on_failure:
            raise RuntimeError(f"Dual-write verification timeout for {signal_uuid}")
        
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
    raise_on_failure: bool = False  # ✅ DEFAULT: Alert + Continue (don't halt)
) -> bool:
    """
    Verify signal was written to both files
    
    STRATEGY: Alert + Continue (default)
    - raise_on_failure=False (default): Log alert, return False (signals continue)
    - raise_on_failure=True: Raise exception (causes halt, not recommended)
    
    Args:
        signal_uuid: UUID to verify
        signal_data: Signal data that was written
        raise_on_failure: If True, halt on failure. If False, alert and continue (RECOMMENDED)
        
    Returns:
        True if both files verified, False if either missing
        
    Raises:
        RuntimeError if raise_on_failure=True and verification fails
    """
    global _dual_write_verifier
    
    if _dual_write_verifier is None:
        raise RuntimeError("Dual-write verifier not initialized")
    
    result = _dual_write_verifier.verify_write(signal_uuid, signal_data, raise_on_failure=raise_on_failure)
    
    if not result and raise_on_failure:
        raise RuntimeError(
            f"Dual-write verification failed for signal {signal_uuid}. "
            f"Signal may not be in both MASTER and AUDIT files."
        )
    
    return result


# Global divergence tracker (used by main.py to record failures)
_divergence_tracker = None


def initialize_divergence_tracker() -> DivergenceTracker:
    """Initialize the global divergence tracker"""
    global _divergence_tracker
    _divergence_tracker = DivergenceTracker()
    return _divergence_tracker


def get_divergence_tracker() -> DivergenceTracker:
    """Get the global divergence tracker"""
    global _divergence_tracker
    
    if _divergence_tracker is None:
        initialize_divergence_tracker()
    
    return _divergence_tracker


def get_dual_write_status() -> dict:
    """Get current dual-write verification status"""
    global _dual_write_verifier
    global _divergence_tracker
    
    status = {}
    
    if _dual_write_verifier is not None:
        status['verifier'] = _dual_write_verifier.get_status()
    else:
        status['verifier'] = {'status': 'not_initialized'}
    
    if _divergence_tracker is not None:
        status['divergence_tracker'] = _divergence_tracker.get_status()
    else:
        status['divergence_tracker'] = {'status': 'not_initialized'}
    
    return status


def send_ops_alert(message: str, severity: str = "WARNING"):
    """
    Send alert to operations team
    
    In production, this would integrate with:
    - Email (ops@example.com)
    - Slack (#alerts)
    - PagerDuty
    - etc.
    
    For now, just logs to file
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Log to file (always)
    alert_log_path = "/tmp/dual_write_alerts.log"
    try:
        with open(alert_log_path, 'a') as f:
            f.write(f"[{timestamp}] [{severity:8s}] {message}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to alert log: {e}", flush=True)
    
    # For critical alerts, also print to stdout
    if severity == "CRITICAL":
        print(f"[ALERT] 🚨 {severity}: {message}", flush=True)
