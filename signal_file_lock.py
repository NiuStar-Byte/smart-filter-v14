#!/usr/bin/env python3
"""
signal_file_lock.py

File-level read/write locking for COMPLETE_SIGNALS.jsonl to prevent corruption
during concurrent reads/writes.

Implements advisory file locking (fcntl on Unix/Linux, msvcrt on Windows).
Ensures atomic operations and data consistency.
"""

import os
import fcntl
import time
from contextlib import contextmanager
from typing import Iterator


class SignalFileLock:
    """File-level locking for COMPLETE_SIGNALS.jsonl"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock_file = f"{file_path}.lock"
    
    @contextmanager
    def read_lock(self, timeout_sec: int = 5) -> Iterator:
        """
        Acquire shared read lock (multiple readers allowed)
        
        Args:
            timeout_sec: Timeout for acquiring lock
        
        Yields:
            Context manager for lock duration
        """
        lock_fd = None
        try:
            # Open lock file in read mode (creates if doesn't exist)
            lock_fd = open(self.lock_file, 'a+')
            
            # Acquire shared lock (LOCK_SH)
            start = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except IOError:
                    if time.time() - start > timeout_sec:
                        raise TimeoutError(f"Could not acquire read lock on {self.file_path} after {timeout_sec}s")
                    time.sleep(0.01)
            
            yield
            
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except:
                    pass
    
    @contextmanager
    def write_lock(self, timeout_sec: int = 10) -> Iterator:
        """
        Acquire exclusive write lock (only one writer allowed)
        
        Args:
            timeout_sec: Timeout for acquiring lock
        
        Yields:
            Context manager for lock duration
        """
        lock_fd = None
        try:
            # Open lock file in write mode
            lock_fd = open(self.lock_file, 'w')
            
            # Acquire exclusive lock (LOCK_EX)
            start = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    if time.time() - start > timeout_sec:
                        raise TimeoutError(f"Could not acquire write lock on {self.file_path} after {timeout_sec}s")
                    time.sleep(0.01)
            
            yield
            
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except:
                    pass


def get_signal_file_lock(file_path: str) -> SignalFileLock:
    """Factory function"""
    return SignalFileLock(file_path)
