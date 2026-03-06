"""
PEC Executor: Auto-refresh signal execution status every 5 minutes

Strategy:
- Silent operation: Only announce when signals hit TP/SL/TIMEOUT
- Read SIGNALS_MASTER.jsonl (unified single source of truth)
- Check current prices via public KuCoin API
- Update signal status: OPEN → TP_HIT/SL_HIT/TIMEOUT
- Write back exit prices and P&L to SIGNALS_MASTER.jsonl
- Print summary of changes only

UPDATED (2026-03-06 14:59 GMT+7): 
- Changed from SENT_SIGNALS.jsonl to SIGNALS_MASTER.jsonl
- Now writes exit_price and pnl back to SIGNALS_MASTER.jsonl
- This provides P&L data needed by reporter
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import requests

# Import consolidated P&L calculation
from calculations import calculate_pnl

class PECExecutor:
    """Auto-execute and track PEC signals"""
    
    def __init__(self, signals_master_path: str = None):
        # Use absolute path to workspace root - read from SIGNALS_MASTER.jsonl (unified source)
        if signals_master_path is None:
            # pec_executor.py lives in workspace root, so __file__ dirname IS workspace root
            workspace_root = os.path.abspath(os.path.dirname(__file__))
            signals_master_path = os.path.join(workspace_root, 'SIGNALS_MASTER.jsonl')
        self.signals_master_path = signals_master_path
        self.kucoin_api_base = "https://api.kucoin.com"
        # MAX_BARS timeout per timeframe (bars since signal fired)
        self.max_bars = {
            "15min": 15,   # 15 bars × 15min = 225 min = 3.75 hours
            "30min": 10,   # 10 bars × 30min = 300 min = 5 hours
            "1h": 5        # 5 bars × 60min = 300 min = 5 hours
        }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from public KuCoin API (no auth needed)"""
        try:
            url = f"{self.kucoin_api_base}/api/v1/market/orderbook/level1?symbol={symbol}"
            response = requests.get(url, timeout=3)  # Reduced timeout from 5 to 3 sec
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    price = float(data['data'].get('price', 0))
                    return price
        except requests.exceptions.Timeout:
            pass  # Timeout - skip this symbol
        except requests.exceptions.ConnectionError:
            pass  # Network error - skip
        except Exception as e:
            pass  # Other errors - skip
        
        return None
    
    def check_signal_status(self, signal: Dict) -> Optional[Dict]:
        """
        Check if signal should be marked TP_HIT/SL_HIT/TIMEOUT/STALE_TIMEOUT
        Returns: {'status': 'TP_HIT'|'SL_HIT'|'TIMEOUT'|'STALE_TIMEOUT'|None, 'exit_price': float, 'pnl_usd': float, 'pnl_pct': float}
        
        **TIMEOUT CATEGORIES:**
        a) Normal TIMEOUT (max_bars ≤ bars_elapsed ≤ max_bars × 1.5)
           - pnl_usd calculated: positive = TIMEOUT_WIN, negative = TIMEOUT_LOSS
        b) STALE_TIMEOUT (bars_elapsed > max_bars × 1.5)
           - pnl_usd = 0.0 (excluded from P&L calculations)
           - Indicates system failure to process at deadline
        
        **CRITICAL:** Check TIMEOUT first, then TP/SL. A signal past max_bars should ALWAYS
        be TIMEOUT/STALE_TIMEOUT, even if price hit TP/SL. Otherwise 18-hour trades misclassify.
        
        With hourly runner, stale timeouts should be rare/zero.
        """
        if signal.get('status') != 'OPEN':
            return None  # Already closed
        
        symbol = signal.get('symbol')
        entry_price = signal.get('entry_price')
        tp_target = signal.get('tp_target')
        sl_target = signal.get('sl_target')
        fired_time_str = signal.get('fired_time_utc')
        timeframe = signal.get('timeframe', '')
        
        if not all([symbol, entry_price, tp_target, sl_target, fired_time_str]):
            return None
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return None
        
        # CRITICAL FIX: Check TIMEOUT **FIRST** before TP/SL
        # If signal is past max_bars, mark TIMEOUT regardless of price levels
        try:
            fired_time = datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
            
            # Convert timeframe to minutes
            tf_minutes = {
                '15min': 15,
                '30min': 30,
                '1h': 60
            }.get(timeframe, 60)
            
            # Get max bars for this timeframe
            max_bars = self.max_bars.get(timeframe, 5)
            
            # Calculate bars elapsed since signal fired
            now = datetime.utcnow().replace(tzinfo=timezone.utc) if fired_time.tzinfo else datetime.utcnow()
            time_delta_minutes = (now - fired_time).total_seconds() / 60
            bars_elapsed = int(time_delta_minutes / tf_minutes)
            
            # DEBUG: Log timeout calculation for signals near timeout
            if bars_elapsed >= max_bars - 2:
                print(f"[PEC-TIMEOUT-DEBUG] {symbol} {timeframe}: bars={bars_elapsed}/{max_bars}, "
                      f"fired={fired_time_str}, now={now}, delta_min={time_delta_minutes:.1f}", flush=True)
            
            # If bars exceed max, determine timeout type
            if bars_elapsed >= max_bars:
                # Check if STALE timeout (>150% overdue) - system failed to process at deadline
                if bars_elapsed > max_bars * 1.5:
                    hours_overdue = int((bars_elapsed - max_bars) * tf_minutes / 60)
                    print(f"[PEC-STALE-TIMEOUT] {symbol} {timeframe}: {bars_elapsed} bars ({hours_overdue}h overdue), "
                          f"marking as STALE_TIMEOUT (no P&L)", flush=True)
                    return {
                        'status': 'STALE_TIMEOUT',
                        'exit_price': current_price,
                        'pnl_usd': 0.0,  # Zero P&L for stale timeouts (excluded from calculations)
                        'pnl_pct': 0.0
                    }
                
                # Normal timeout (within reasonable window) - calculate P&L as TIMEOUT_WIN or TIMEOUT_LOSS
                print(f"[PEC-TIMEOUT-HIT] {symbol} {timeframe}: {bars_elapsed} bars >= {max_bars} max", flush=True)
                # FIXED POSITION SIZE: $100 with 10x leverage = $1,000 notional
                NOTIONAL_POSITION = 1000.0  # $100 position × 10x leverage
                # Use consolidated P&L calculation (use current_price at timeout)
                pnl_result = calculate_pnl(entry_price, current_price, signal.get('signal_type'), NOTIONAL_POSITION)
                
                return {
                    'status': 'TIMEOUT',
                    'exit_price': current_price,  # timeout_price = current_price
                    'pnl_usd': pnl_result['pnl_usd'],  # Can be positive (TIMEOUT_WIN) or negative (TIMEOUT_LOSS)
                    'pnl_pct': pnl_result['pnl_pct']
                }
        except Exception as e:
            print(f"[PEC-TIMEOUT-ERROR] {symbol}: {e}", flush=True)
        
        # ONLY check TP/SL if signal is STILL within timeout window
        # FIXED POSITION SIZE: $100 with 10x leverage = $1,000 notional exposure
        NOTIONAL_POSITION = 1000.0  # $100 position × 10x leverage
        signal_type = signal.get('signal_type')
        
        if signal_type == 'LONG':
            if current_price >= tp_target:
                # Use consolidated P&L calculation
                pnl_result = calculate_pnl(entry_price, current_price, 'LONG', NOTIONAL_POSITION)
                return {
                    'status': 'TP_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_result['pnl_usd'],
                    'pnl_pct': pnl_result['pnl_pct']
                }
            elif current_price <= sl_target:
                pnl_result = calculate_pnl(entry_price, current_price, 'LONG', NOTIONAL_POSITION)
                return {
                    'status': 'SL_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_result['pnl_usd'],
                    'pnl_pct': pnl_result['pnl_pct']
                }
        
        elif signal_type == 'SHORT':
            if current_price <= tp_target:
                pnl_result = calculate_pnl(entry_price, current_price, 'SHORT', NOTIONAL_POSITION)
                return {
                    'status': 'TP_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_result['pnl_usd'],
                    'pnl_pct': pnl_result['pnl_pct']
                }
            elif current_price >= sl_target:
                pnl_result = calculate_pnl(entry_price, current_price, 'SHORT', NOTIONAL_POSITION)
                return {
                    'status': 'SL_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_result['pnl_usd'],
                    'pnl_pct': pnl_result['pnl_pct']
                }
        
        return None  # Still open
    
    def update_signals(self) -> Dict:
        """
        Scan all OPEN signals in SIGNALS_MASTER.jsonl, check status, update file
        Writes exit prices and P&L back to SIGNALS_MASTER.jsonl
        Returns: summary of changes
        """
        summary = {
            'total_checked': 0,
            'tp_hits': [],
            'sl_hits': [],
            'timeouts': [],
            'stale_timeouts': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if not os.path.exists(self.signals_master_path):
            return summary
        
        try:
            records = []
            
            # Read all records from SIGNALS_MASTER.jsonl (unified source)
            with open(self.signals_master_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        records.append(record)
            
            # Check each OPEN signal
            for record in records:
                if record.get('status') == 'OPEN':
                    summary['total_checked'] += 1
                    result = self.check_signal_status(record)
                    
                    if result:
                        # Update record
                        record['status'] = result['status']
                        record['actual_exit_price'] = result['exit_price']
                        record['closed_at'] = datetime.utcnow().isoformat()
                        
                        # Set P&L (STALE_TIMEOUT will be 0, others calculated)
                        record['pnl_usd'] = round(result['pnl_usd'], 4)
                        record['pnl_pct'] = round(result['pnl_pct'], 2)
                        
                        # Track summary (include timestamp when signal fired)
                        fired_time = record.get('fired_time_utc', '')
                        # Convert UTC to local time (GMT+7)
                        if fired_time:
                            try:
                                dt = datetime.fromisoformat(fired_time.replace('Z', '+00:00'))
                                # Add 7 hours to convert UTC to GMT+7
                                gmt7_time = dt + timedelta(hours=7)
                                # Format as HH:MM:SS GMT+7
                                local_time = gmt7_time.strftime("%H:%M:%S")
                            except:
                                local_time = ''
                        else:
                            local_time = ''
                        
                        if result['status'] == 'TP_HIT':
                            summary['tp_hits'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'pnl_usd': record['pnl_usd'],
                                'pnl_pct': record['pnl_pct'],
                                'fired_time': local_time
                            })
                        elif result['status'] == 'SL_HIT':
                            summary['sl_hits'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'pnl_usd': record['pnl_usd'],
                                'pnl_pct': record['pnl_pct'],
                                'fired_time': local_time
                            })
                        elif result['status'] == 'TIMEOUT':
                            summary['timeouts'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'pnl_usd': record['pnl_usd'],
                                'pnl_pct': record['pnl_pct'],
                                'fired_time': local_time
                            })
                        elif result['status'] == 'STALE_TIMEOUT':
                            summary['stale_timeouts'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'fired_time': local_time
                            })
            
            # Write back updated records to SIGNALS_MASTER.jsonl
            with open(self.signals_master_path, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            
        except Exception as e:
            print(f"[ERROR] PEC update failed: {e}", flush=True)
        
        return summary
    
    def get_stats(self) -> Dict:
        """Get current PEC statistics"""
        stats = {
            'total': 0,
            'open': 0,
            'tp_hit': 0,
            'sl_hit': 0,
            'timeout': 0,
            'timeout_win': 0,
            'timeout_loss': 0,
            'stale_timeout': 0,
            'total_pnl_usd': 0.0,
            'win_rate_pct': 0.0
        }
        
        if not os.path.exists(self.sent_signals_path):
            return stats
        
        try:
            with open(self.sent_signals_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        stats['total'] += 1
                        
                        status = record.get('status', 'OPEN')
                        if status == 'OPEN':
                            stats['open'] += 1
                        elif status == 'TP_HIT':
                            stats['tp_hit'] += 1
                        elif status == 'SL_HIT':
                            stats['sl_hit'] += 1
                        elif status == 'TIMEOUT':
                            stats['timeout'] += 1
                            # Count TIMEOUT as win or loss based on P&L
                            pnl = record.get('pnl_usd')
                            if pnl is not None and float(pnl) > 0:
                                stats['timeout_win'] += 1
                            elif pnl is not None and float(pnl) < 0:
                                stats['timeout_loss'] += 1
                        elif status == 'STALE_TIMEOUT':
                            stats['stale_timeout'] += 1
                        
                        # Count P&L (STALE_TIMEOUT has pnl_usd=0.0, so it won't affect total)
                        pnl = record.get('pnl_usd')
                        if pnl is not None and status != 'STALE_TIMEOUT':
                            # Only add to total P&L if not stale and not None
                            stats['total_pnl_usd'] += float(pnl)
            
            # Win Rate = TP_HIT / (TP_HIT + SL_HIT) [exclude TIMEOUT and STALE_TIMEOUT from base calculation]
            closed_trades = stats['tp_hit'] + stats['sl_hit']
            if closed_trades > 0:
                stats['win_rate_pct'] = round((stats['tp_hit'] / closed_trades) * 100, 1)
            
            stats['total_pnl_usd'] = round(stats['total_pnl_usd'], 4)
        
        except Exception as e:
            print(f"[ERROR] Failed to get stats: {e}", flush=True)
        
        return stats
    
    def print_announcement(self, summary: Dict):
        """Print summary ONLY if something changed"""
        if not any([summary['tp_hits'], summary['sl_hits'], summary['timeouts'], summary['stale_timeouts']]):
            return  # Silent if nothing changed
        
        print(f"\n{'='*80}", flush=True)
        print(f"[PEC UPDATE] {summary['timestamp']}", flush=True)
        print(f"{'='*80}", flush=True)
        
        if summary['tp_hits']:
            print(f"\n✅ TP HIT ({len(summary['tp_hits'])} signals):", flush=True)
            for hit in summary['tp_hits']:
                fired = f" | Fired: {hit['fired_time']}" if hit.get('fired_time') else ""
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%){fired}", flush=True)
        
        if summary['sl_hits']:
            print(f"\n❌ SL HIT ({len(summary['sl_hits'])} signals):", flush=True)
            for hit in summary['sl_hits']:
                fired = f" | Fired: {hit['fired_time']}" if hit.get('fired_time') else ""
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%){fired}", flush=True)
        
        if summary['timeouts']:
            print(f"\n⏱️  TIMEOUT ({len(summary['timeouts'])} signals):", flush=True)
            for hit in summary['timeouts']:
                fired = f" | Fired: {hit['fired_time']}" if hit.get('fired_time') else ""
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%){fired}", flush=True)
        
        if summary['stale_timeouts']:
            print(f"\n⚠️  STALE_TIMEOUT ({len(summary['stale_timeouts'])} signals) - System failed to process at deadline:", flush=True)
            for hit in summary['stale_timeouts']:
                fired = f" | Fired: {hit['fired_time']}" if hit.get('fired_time') else ""
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | NO P&L (data quality issue){fired}", flush=True)
        
        # Print stats
        stats = self.get_stats()
        print(f"\n📊 PEC STATS:", flush=True)
        print(f"   Total: {stats['total']} | Open: {stats['open']} | TP: {stats['tp_hit']} | SL: {stats['sl_hit']} | TIMEOUT: {stats['timeout']} ({stats['timeout_win']}W/{stats['timeout_loss']}L) | STALE_TIMEOUT: {stats['stale_timeout']}", flush=True)
        print(f"   Win Rate (TP/SL only): {stats['win_rate_pct']:.1f}% | Total P&L: ${stats['total_pnl_usd']:+.2f}", flush=True)
        print(f"{'='*80}\n", flush=True)


def log_system_error(error_msg):
    """Log errors to health monitor log"""
    import sys
    # pec_executor.py lives in workspace root, so __file__ dirname IS workspace root
    workspace_root = os.path.abspath(os.path.dirname(__file__))
    health_log = os.path.join(workspace_root, 'pec_system_health.log')
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+7")
    line = f"[{ts}] 🔴 EXECUTOR        | RUNTIME_ERROR        | {error_msg}\n"
    
    try:
        with open(health_log, 'a') as f:
            f.write(line)
    except:
        pass  # If health log fails, at least we tried
    
    print(line.strip(), flush=True)

def run_pec_update():
    """Main entry point for cron job - executes signals from SIGNALS_MASTER.jsonl"""
    try:
        executor = PECExecutor()
        
        # Verify file exists before processing
        if not os.path.exists(executor.signals_master_path):
            error_msg = f"SIGNALS_MASTER.jsonl not found at {executor.signals_master_path}"
            log_system_error(error_msg)
            return
        
        summary = executor.update_signals()
        executor.print_announcement(summary)
        
    except FileNotFoundError as e:
        log_system_error(f"File access error: {e}")
    except PermissionError as e:
        log_system_error(f"Permission denied: {e}")
    except json.JSONDecodeError as e:
        log_system_error(f"SIGNALS_MASTER.jsonl corrupted: {e}")
    except Exception as e:
        log_system_error(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == '__main__':
    run_pec_update()
