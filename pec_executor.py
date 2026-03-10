"""
PEC Executor: Auto-refresh signal execution status every 5 minutes

CRITICAL FIX (2026-03-08):
- closed_at now set to ACTUAL timeout time (fired_time + window)
- Not current check time (prevents duration inflation)
- Properly tracks data_quality_flag for STALE_TIMEOUT

Strategy:
- Silent operation: Only announce when signals hit TP/SL/TIMEOUT
- Read SIGNALS_MASTER.jsonl (unified single source of truth)
- Check current prices via public KuCoin API
- Update signal status: OPEN → TP_HIT/SL_HIT/TIMEOUT/STALE_TIMEOUT
- Write back exit prices and P&L to SIGNALS_MASTER.jsonl
- Print summary of changes only
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
            # pec_executor.py lives in smart-filter-v14-main submodule, but reads from workspace root
            workspace_root = "/Users/geniustarigan/.openclaw/workspace"
            signals_master_path = os.path.join(workspace_root, 'SIGNALS_MASTER.jsonl')
        self.signals_master_path = signals_master_path
        self.kucoin_api_base = "https://api.kucoin.com"
        
        # CHAMPION: MAX_BARS timeout per timeframe (baseline, no tier)
        self.champion_max_bars = {
            "15min": 15,   # 15 bars × 15min = 225 min = 3.75 hours
            "30min": 10,   # 10 bars × 30min = 300 min = 5 hours
            "1h": 5        # 5 bars × 60min = 300 min = 5 hours
        }
        
        # CHALLENGER: Tier-based timeout windows (minutes, not bars)
        self.challenger_timeout_minutes = {
            "15min": {
                "TIER-1": 4 * 60,      # 4h (reward high conviction)
                "TIER-2": 3 * 60,      # 3h (standard)
                "TIER-3": 2 * 60,      # 2h (stop bleed)
            },
            "30min": {
                "TIER-1": 5 * 60,      # 5h
                "TIER-2": 4 * 60,      # 4h
                "TIER-3": 3 * 60,      # 3h
            },
            "1h": {
                "TIER-1": 6 * 60,      # 6h (extra runway for proven combos)
                "TIER-2": 5 * 60,      # 5h
                "TIER-3": 4 * 60,      # 4h (quick exit for losing combos)
            }
        }
        
        # Running Champion vs Challenger test (50/50 split)
        self.cc_test_enabled = True
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from public KuCoin API (no auth needed)"""
        try:
            url = f"{self.kucoin_api_base}/api/v1/market/orderbook/level1?symbol={symbol}"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    price = float(data['data']['price'])
                    return price
        except Exception as e:
            pass  # Return None on any error
        return None
    
    def check_signal_status(self, signal: Dict) -> Optional[Dict]:
        """
        Check if signal should be marked TP_HIT/SL_HIT/TIMEOUT/STALE_TIMEOUT
        Returns: {'status': ..., 'exit_price': float, 'pnl_usd': float, 'pnl_pct': float, 'actual_timeout_time': str (if timeout)}
        
        **CRITICAL FIX:** closed_at is now calculated as ACTUAL timeout time, not current check time
        """
        if signal.get('status') != 'OPEN':
            return None  # Already closed
        
        symbol = signal.get('symbol')
        entry_price = signal.get('entry_price')
        tp_target = signal.get('tp_target') or signal.get('tp_price')
        sl_target = signal.get('sl_target') or signal.get('sl_price')
        fired_time_str = signal.get('fired_time_utc')
        timeframe = signal.get('timeframe', '')
        
        if not all([symbol, entry_price, tp_target, sl_target, fired_time_str]):
            return None
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return None
        
        # CRITICAL: Check TIMEOUT FIRST before TP/SL
        try:
            fired_time = datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
            
            # Convert timeframe to minutes
            tf_minutes = {
                '15min': 15,
                '30min': 30,
                '1h': 60
            }.get(timeframe, 60)
            
            # CHAMPION vs CHALLENGER: Choose timeout strategy
            import random
            is_challenger = random.random() < 0.5
            
            if is_challenger and self.cc_test_enabled:
                # CHALLENGER: Tier-based timeout windows
                tier = signal.get('tier', 'TIER-3')
                timeout_minutes = self.challenger_timeout_minutes.get(timeframe, {}).get(tier, 5 * 60)
                max_bars = int(timeout_minutes / tf_minutes)  # Convert minutes back to bars for calculation
                cc_group = 'CHALLENGER'
                cc_window = timeout_minutes
            else:
                # CHAMPION: Standard (no tier distinction)
                max_bars = self.champion_max_bars.get(timeframe, 5)
                cc_group = 'CHAMPION'
                cc_window = max_bars * tf_minutes
            
            # Calculate bars elapsed since signal fired
            now = datetime.utcnow().replace(tzinfo=timezone.utc) if fired_time.tzinfo else datetime.utcnow()
            time_delta_minutes = (now - fired_time).total_seconds() / 60
            bars_elapsed = int(time_delta_minutes / tf_minutes)
            
            # If bars exceed max, determine timeout type and CALCULATE ACTUAL TIMEOUT TIME
            if bars_elapsed >= max_bars:
                # CRITICAL FIX: Calculate ACTUAL timeout time (when timeout should have occurred)
                actual_timeout_time = fired_time + timedelta(minutes=max_bars * tf_minutes)
                
                # Check if STALE timeout (>150% overdue)
                if bars_elapsed > max_bars * 1.5:
                    hours_overdue = int((bars_elapsed - max_bars) * tf_minutes / 60)
                    print(f"[PEC-STALE-TIMEOUT] {symbol} {timeframe}: {bars_elapsed} bars ({hours_overdue}h overdue), "
                          f"actual_timeout={actual_timeout_time.isoformat()}", flush=True)
                    return {
                        'status': 'STALE_TIMEOUT',
                        'exit_price': current_price,
                        'pnl_usd': 0.0,  # Zero P&L for stale timeouts
                        'pnl_pct': 0.0,
                        'actual_timeout_time': actual_timeout_time.isoformat(),
                        'hours_overdue': hours_overdue,
                        'is_stale': True
                    }
                
                # Normal timeout - calculate P&L
                print(f"[PEC-TIMEOUT-HIT] {symbol} {timeframe}: {bars_elapsed} bars >= {max_bars} max, "
                      f"actual_timeout={actual_timeout_time.isoformat()} [{cc_group}]", flush=True)
                NOTIONAL_POSITION = 1000.0
                pnl_result = calculate_pnl(entry_price, current_price, signal.get('signal_type'), NOTIONAL_POSITION)
                
                return {
                    'status': 'TIMEOUT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_result['pnl_usd'],
                    'pnl_pct': pnl_result['pnl_pct'],
                    'actual_timeout_time': actual_timeout_time.isoformat(),
                    'is_stale': False,
                    'cc_group': cc_group,
                    'timeout_window_minutes': cc_window
                }
        except Exception as e:
            print(f"[PEC-TIMEOUT-ERROR] {symbol}: {e}", flush=True)
        
        # ONLY check TP/SL if signal is STILL within timeout window
        NOTIONAL_POSITION = 1000.0
        signal_type = signal.get('signal_type')
        
        if signal_type == 'LONG':
            if current_price >= tp_target:
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
            
            # Read all records from SIGNALS_MASTER.jsonl
            with open(self.signals_master_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        records.append(record)
            
            # Check each OPEN signal
            for record in records:
                if record.get('status') == 'OPEN':
                    summary['total_checked'] += 1
                    
                    # SILENT TAGGING: Assign Champion/Challenger group before checking
                    import random
                    if self.cc_test_enabled and record.get('champion_challenger_group') is None:
                        record['champion_challenger_group'] = 'CHALLENGER' if random.random() < 0.5 else 'CHAMPION'
                    
                    result = self.check_signal_status(record)
                    
                    if result:
                        # Update record
                        record['status'] = result['status']
                        record['actual_exit_price'] = result['exit_price']
                        
                        # CRITICAL FIX: Use actual_timeout_time if available (timeout), else current time
                        if 'actual_timeout_time' in result:
                            record['closed_at'] = result['actual_timeout_time']
                        else:
                            record['closed_at'] = datetime.utcnow().isoformat()
                        
                        # Set P&L
                        record['pnl_usd'] = round(result['pnl_usd'], 4)
                        record['pnl_pct'] = round(result['pnl_pct'], 2)
                        
                        # Set data_quality_flag for STALE_TIMEOUT
                        if result['status'] == 'STALE_TIMEOUT':
                            hours_overdue = result.get('hours_overdue', 0)
                            record['data_quality_flag'] = f'STALE_TIMEOUT_{hours_overdue}h_overdue'
                        else:
                            record['data_quality_flag'] = None
                        
                        # SILENT TAGGING: Store timeout window info (internal only, not shown to traders)
                        if result['status'] in ['TIMEOUT', 'STALE_TIMEOUT']:
                            record['cc_timeout_window_minutes'] = result.get('timeout_window_minutes', 0)
                            record['cc_group'] = result.get('cc_group', 'CHAMPION')
                        
                        # Track summary
                        fired_time = record.get('fired_time_utc', '')
                        if fired_time:
                            try:
                                dt = datetime.fromisoformat(fired_time.replace('Z', '+00:00'))
                                gmt7_time = dt + timedelta(hours=7)
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
                                'fired_time': local_time,
                                'actual_timeout': result.get('actual_timeout_time', '')
                            })
                        elif result['status'] == 'STALE_TIMEOUT':
                            summary['stale_timeouts'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'fired_time': local_time,
                                'hours_overdue': result.get('hours_overdue', 0)
                            })
            
            # Write back updated records to SIGNALS_MASTER.jsonl
            with open(self.signals_master_path, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            
        except Exception as e:
            print(f"[ERROR] PEC update failed: {e}", flush=True)
        
        return summary
    
    def get_stats(self) -> Dict:
        """Get current signal stats"""
        stats = {
            'total': 0,
            'open': 0,
            'tp_hit': 0,
            'sl_hit': 0,
            'timeout': 0,
            'timeout_win': 0,
            'timeout_loss': 0,
            'stale_timeout': 0,
            'win_rate_pct': 0.0,
            'total_pnl_usd': 0.0
        }
        
        try:
            if not os.path.exists(self.signals_master_path):
                return stats
            
            with open(self.signals_master_path, 'r') as f:
                for line in f:
                    if line.strip():
                        signal = json.loads(line)
                        stats['total'] += 1
                        
                        status = signal.get('status', 'OPEN')
                        if status == 'OPEN':
                            stats['open'] += 1
                        elif status == 'TP_HIT':
                            stats['tp_hit'] += 1
                        elif status == 'SL_HIT':
                            stats['sl_hit'] += 1
                        elif status == 'TIMEOUT':
                            stats['timeout'] += 1
                            pnl = signal.get('pnl_usd')
                            if pnl is not None:
                                if pnl > 0:
                                    stats['timeout_win'] += 1
                                elif pnl < 0:
                                    stats['timeout_loss'] += 1
                        elif status == 'STALE_TIMEOUT':
                            stats['stale_timeout'] += 1
                        
                        # P&L
                        pnl = signal.get('pnl_usd')
                        if pnl is not None and status != 'STALE_TIMEOUT':
                            stats['total_pnl_usd'] += float(pnl)
            
            # Win Rate
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
                actual = f" | Timeout: {hit.get('actual_timeout', '')[:16]}" if hit.get('actual_timeout') else ""
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%){fired}{actual}", flush=True)
        
        if summary['stale_timeouts']:
            print(f"\n⚠️  STALE_TIMEOUT ({len(summary['stale_timeouts'])} signals):", flush=True)
            for hit in summary['stale_timeouts']:
                fired = f" | Fired: {hit['fired_time']}" if hit.get('fired_time') else ""
                hours = f" | {hit.get('hours_overdue', 0)}h overdue"
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | NO P&L (data quality issue){hours}{fired}", flush=True)
        
        # Print stats
        stats = self.get_stats()
        print(f"\n📊 PEC STATS:", flush=True)
        print(f"   Total: {stats['total']} | Open: {stats['open']} | TP: {stats['tp_hit']} | SL: {stats['sl_hit']} | TIMEOUT: {stats['timeout']} ({stats['timeout_win']}W/{stats['timeout_loss']}L) | STALE: {stats['stale_timeout']}", flush=True)
        print(f"   Win Rate (TP/SL only): {stats['win_rate_pct']:.1f}% | Total P&L: ${stats['total_pnl_usd']:+.2f}", flush=True)
        print(f"{'='*80}\n", flush=True)


def run_pec_update():
    """Main entry point"""
    try:
        executor = PECExecutor()
        
        if not os.path.exists(executor.signals_master_path):
            error_msg = f"SIGNALS_MASTER.jsonl not found"
            print(f"[ERROR] {error_msg}", flush=True)
            return
        
        summary = executor.update_signals()
        executor.print_announcement(summary)
        
    except Exception as e:
        print(f"[ERROR] PEC update failed: {type(e).__name__}: {e}", flush=True)


if __name__ == '__main__':
    run_pec_update()
