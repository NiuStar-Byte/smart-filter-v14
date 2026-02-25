"""
PEC Execution Tracker - Monitor SENT signals and update with execution results

Monitors SENT_SIGNALS.jsonl for open signals and updates them when:
- TP is hit (take profit)
- SL is hit (stop loss)  
- Timeout (signal expires without hitting TP/SL)

Returns execution statistics for PEC backtesting validation.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from signal_sent_tracker import get_signal_sent_tracker

class PECExecutionTracker:
    """Monitor and track signal executions"""
    
    def __init__(self, 
                 sent_signals_path: str = "SENT_SIGNALS.jsonl",
                 timeout_hours: int = 24):
        """
        Initialize tracker
        
        Args:
            sent_signals_path: Path to SENT_SIGNALS.jsonl
            timeout_hours: How long before signal times out
        """
        self.sent_signals_path = sent_signals_path
        self.timeout_hours = timeout_hours
        self.tracker = get_signal_sent_tracker(sent_signals_path)
    
    def update_signal_if_tp_hit(self, 
                                signal_uuid: str,
                                current_price: float) -> bool:
        """
        Check if signal hit TP, update if so
        
        Returns: True if TP was hit and updated, False otherwise
        """
        try:
            signal = self._get_signal(signal_uuid)
            if not signal or signal.get('status') != 'OPEN':
                return False
            
            tp_target = float(signal.get('tp_target', 0))
            entry_price = float(signal.get('entry_price', 0))
            
            if signal.get('signal_type') == 'LONG':
                # For LONG: TP hit if current_price >= tp_target
                if current_price >= tp_target:
                    pnl_usd = (current_price - entry_price) * 1  # Assuming 1 share
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    self.tracker.update_signal_execution(
                        signal_uuid=signal_uuid,
                        status='TP_HIT',
                        exit_price=tp_target,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct
                    )
                    print(f"[PEC-TP] {signal.get('symbol')} {signal.get('timeframe')} TP HIT @ {tp_target} | P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)", flush=True)
                    return True
            
            elif signal.get('signal_type') == 'SHORT':
                # For SHORT: TP hit if current_price <= tp_target
                if current_price <= tp_target:
                    pnl_usd = (entry_price - current_price) * 1  # Short profit reversed
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    self.tracker.update_signal_execution(
                        signal_uuid=signal_uuid,
                        status='TP_HIT',
                        exit_price=tp_target,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct
                    )
                    print(f"[PEC-TP] {signal.get('symbol')} {signal.get('timeframe')} TP HIT @ {tp_target} | P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)", flush=True)
                    return True
        
        except Exception as e:
            print(f"[ERROR] Failed to check TP for {signal_uuid}: {e}", flush=True)
        
        return False
    
    def update_signal_if_sl_hit(self,
                                signal_uuid: str,
                                current_price: float) -> bool:
        """
        Check if signal hit SL, update if so
        
        Returns: True if SL was hit and updated, False otherwise
        """
        try:
            signal = self._get_signal(signal_uuid)
            if not signal or signal.get('status') != 'OPEN':
                return False
            
            sl_target = float(signal.get('sl_target', 0))
            entry_price = float(signal.get('entry_price', 0))
            
            if signal.get('signal_type') == 'LONG':
                # For LONG: SL hit if current_price <= sl_target
                if current_price <= sl_target:
                    pnl_usd = (current_price - entry_price) * 1
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    self.tracker.update_signal_execution(
                        signal_uuid=signal_uuid,
                        status='SL_HIT',
                        exit_price=sl_target,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct
                    )
                    print(f"[PEC-SL] {signal.get('symbol')} {signal.get('timeframe')} SL HIT @ {sl_target} | LOSS: ${pnl_usd:.2f} ({pnl_pct:.2f}%)", flush=True)
                    return True
            
            elif signal.get('signal_type') == 'SHORT':
                # For SHORT: SL hit if current_price >= sl_target
                if current_price >= sl_target:
                    pnl_usd = (entry_price - current_price) * 1
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    self.tracker.update_signal_execution(
                        signal_uuid=signal_uuid,
                        status='SL_HIT',
                        exit_price=sl_target,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct
                    )
                    print(f"[PEC-SL] {signal.get('symbol')} {signal.get('timeframe')} SL HIT @ {sl_target} | LOSS: ${pnl_usd:.2f} ({pnl_pct:.2f}%)", flush=True)
                    return True
        
        except Exception as e:
            print(f"[ERROR] Failed to check SL for {signal_uuid}: {e}", flush=True)
        
        return False
    
    def mark_as_timeout(self, signal_uuid: str, exit_price: float) -> bool:
        """
        Mark signal as TIMEOUT (didn't hit TP/SL within timeout period)
        
        Exit price = current market price when timeout triggered
        """
        try:
            signal = self._get_signal(signal_uuid)
            if not signal or signal.get('status') != 'OPEN':
                return False
            
            entry_price = float(signal.get('entry_price', 0))
            pnl_usd = (exit_price - entry_price) * 1
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            self.tracker.update_signal_execution(
                signal_uuid=signal_uuid,
                status='TIMEOUT',
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct
            )
            
            result = "PROFIT" if pnl_usd > 0 else "LOSS"
            print(f"[PEC-TIMEOUT] {signal.get('symbol')} {signal.get('timeframe')} TIMEOUT @ {exit_price} | {result}: ${pnl_usd:.2f} ({pnl_pct:.2f}%)", flush=True)
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to mark timeout for {signal_uuid}: {e}", flush=True)
        
        return False
    
    def get_open_signals(self) -> List[Dict]:
        """Get all OPEN signals waiting for execution"""
        return self.tracker.get_open_signals()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.tracker.get_summary_stats()
    
    def get_signals_by_status(self, status: str) -> List[Dict]:
        """Get signals filtered by status (OPEN, TP_HIT, SL_HIT, TIMEOUT)"""
        try:
            signals = []
            if os.path.exists(self.sent_signals_path):
                with open(self.sent_signals_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get('status') == status:
                                signals.append(record)
            return signals
        except Exception as e:
            print(f"[ERROR] Failed to filter signals: {e}", flush=True)
            return []
    
    def _get_signal(self, signal_uuid: str) -> Optional[Dict]:
        """Get a single signal by UUID"""
        try:
            if os.path.exists(self.sent_signals_path):
                with open(self.sent_signals_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get('uuid') == signal_uuid:
                                return record
        except Exception as e:
            print(f"[ERROR] Failed to get signal: {e}", flush=True)
        
        return None
    
    def print_execution_summary(self):
        """Print a summary of all signal executions"""
        stats = self.get_execution_stats()
        
        print("\n" + "="*70, flush=True)
        print("PEC EXECUTION SUMMARY:", flush=True)
        print(f"  Total Sent:      {stats.get('total_sent', 0)}", flush=True)
        print(f"  Open (Running):  {stats.get('open', 0)}", flush=True)
        print(f"  TP Hit:          {stats.get('tp_hit', 0)}", flush=True)
        print(f"  SL Hit:          {stats.get('sl_hit', 0)}", flush=True)
        print(f"  Timeout:         {stats.get('timeout', 0)}", flush=True)
        print(f"  Closed Trades:   {stats.get('closed', 0)}", flush=True)
        print(f"  Total P&L:       ${stats.get('total_pnl_usd', 0)}", flush=True)
        print(f"  Win Rate:        {stats.get('win_rate_pct', 0)}% ({stats.get('winning_trades', 0)}W/{stats.get('losing_trades', 0)}L)", flush=True)
        print("="*70 + "\n", flush=True)


def get_pec_execution_tracker(sent_signals_path: str = "SENT_SIGNALS.jsonl") -> PECExecutionTracker:
    """Factory function to get tracker instance"""
    return PECExecutionTracker(sent_signals_path)


# Example usage:
if __name__ == "__main__":
    tracker = get_pec_execution_tracker("SENT_SIGNALS.jsonl")
    
    # Get all open signals
    open_signals = tracker.get_open_signals()
    print(f"Open signals: {len(open_signals)}")
    
    # Example: Update a signal if TP hit
    for signal in open_signals:
        uuid = signal['uuid']
        symbol = signal['symbol']
        tp = signal['tp_target']
        sl = signal['sl_target']
        print(f"  {symbol}: TP={tp}, SL={sl}")
        
        # In real PEC, you'd get current price from market
        # current_price = get_market_price(symbol)
        # tracker.update_signal_if_tp_hit(uuid, current_price)
        # tracker.update_signal_if_sl_hit(uuid, current_price)
    
    # Print summary
    tracker.print_execution_summary()
