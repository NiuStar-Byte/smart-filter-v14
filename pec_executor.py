"""
PEC Executor: Auto-refresh signal execution status every 5 minutes

Strategy:
- Silent operation: Only announce when signals hit TP/SL/TIMEOUT
- Read SENT_SIGNALS.jsonl (signals sent to Telegram)
- Check current prices via public KuCoin API
- Update signal status: OPEN → TP_HIT/SL_HIT/TIMEOUT
- Print summary of changes only
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

class PECExecutor:
    """Auto-execute and track PEC signals"""
    
    def __init__(self, sent_signals_path: str = "SENT_SIGNALS.jsonl"):
        self.sent_signals_path = sent_signals_path
        self.kucoin_api_base = "https://api.kucoin.com"
        self.timeout_hours = 24  # Mark as TIMEOUT if open >24h
    
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
        Check if signal should be marked TP_HIT/SL_HIT/TIMEOUT
        Returns: {'status': 'TP_HIT'|'SL_HIT'|'TIMEOUT'|None, 'exit_price': float, 'pnl': float}
        """
        if signal.get('status') != 'OPEN':
            return None  # Already closed
        
        symbol = signal.get('symbol')
        entry_price = signal.get('entry_price')
        tp_target = signal.get('tp_target')
        sl_target = signal.get('sl_target')
        fired_time_str = signal.get('fired_time_utc')
        
        if not all([symbol, entry_price, tp_target, sl_target]):
            return None
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return None
        
        # Check TP hit
        if signal.get('signal_type') == 'LONG':
            if current_price >= tp_target:
                pnl_usd = (current_price - entry_price) * signal.get('confidence', 1.0)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                return {
                    'status': 'TP_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct
                }
            elif current_price <= sl_target:
                pnl_usd = (current_price - entry_price) * signal.get('confidence', 1.0)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                return {
                    'status': 'SL_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct
                }
        
        elif signal.get('signal_type') == 'SHORT':
            if current_price <= tp_target:
                pnl_usd = (entry_price - current_price) * signal.get('confidence', 1.0)
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                return {
                    'status': 'TP_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct
                }
            elif current_price >= sl_target:
                pnl_usd = (entry_price - current_price) * signal.get('confidence', 1.0)
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                return {
                    'status': 'SL_HIT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct
                }
        
        # Check TIMEOUT (>24 hours open)
        try:
            fired_time = datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
            if datetime.utcnow() - fired_time > timedelta(hours=self.timeout_hours):
                pnl_usd = (current_price - entry_price) * signal.get('confidence', 1.0)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                return {
                    'status': 'TIMEOUT',
                    'exit_price': current_price,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct
                }
        except:
            pass
        
        return None  # Still open
    
    def update_signals(self) -> Dict:
        """
        Scan all OPEN signals, check status, update file
        Returns: summary of changes
        """
        summary = {
            'total_checked': 0,
            'tp_hits': [],
            'sl_hits': [],
            'timeouts': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if not os.path.exists(self.sent_signals_path):
            return summary
        
        try:
            records = []
            
            # Read all records
            with open(self.sent_signals_path, 'r') as f:
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
                        record['pnl_usd'] = round(result['pnl_usd'], 2)
                        record['pnl_pct'] = round(result['pnl_pct'], 2)
                        record['closed_at'] = datetime.utcnow().isoformat()
                        
                        # Track summary
                        if result['status'] == 'TP_HIT':
                            summary['tp_hits'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'pnl_usd': record['pnl_usd'],
                                'pnl_pct': record['pnl_pct']
                            })
                        elif result['status'] == 'SL_HIT':
                            summary['sl_hits'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'pnl_usd': record['pnl_usd'],
                                'pnl_pct': record['pnl_pct']
                            })
                        elif result['status'] == 'TIMEOUT':
                            summary['timeouts'].append({
                                'symbol': record.get('symbol'),
                                'timeframe': record.get('timeframe'),
                                'pnl_usd': record['pnl_usd'],
                                'pnl_pct': record['pnl_pct']
                            })
            
            # Write back updated records
            with open(self.sent_signals_path, 'w') as f:
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
            'total_pnl_usd': 0.0,
            'win_rate_pct': 0.0
        }
        
        if not os.path.exists(self.sent_signals_path):
            return stats
        
        try:
            winning = 0
            losing = 0
            
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
                        
                        pnl = record.get('pnl_usd', 0)
                        if pnl:
                            stats['total_pnl_usd'] += pnl
                            if pnl > 0:
                                winning += 1
                            else:
                                losing += 1
            
            if winning + losing > 0:
                stats['win_rate_pct'] = round((winning / (winning + losing)) * 100, 1)
            
            stats['total_pnl_usd'] = round(stats['total_pnl_usd'], 2)
        
        except Exception as e:
            print(f"[ERROR] Failed to get stats: {e}", flush=True)
        
        return stats
    
    def print_announcement(self, summary: Dict):
        """Print summary ONLY if something changed"""
        if not any([summary['tp_hits'], summary['sl_hits'], summary['timeouts']]):
            return  # Silent if nothing changed
        
        print(f"\n{'='*80}", flush=True)
        print(f"[PEC UPDATE] {summary['timestamp']}", flush=True)
        print(f"{'='*80}", flush=True)
        
        if summary['tp_hits']:
            print(f"\n✅ TP HIT ({len(summary['tp_hits'])} signals):", flush=True)
            for hit in summary['tp_hits']:
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%)", flush=True)
        
        if summary['sl_hits']:
            print(f"\n❌ SL HIT ({len(summary['sl_hits'])} signals):", flush=True)
            for hit in summary['sl_hits']:
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%)", flush=True)
        
        if summary['timeouts']:
            print(f"\n⏱️  TIMEOUT ({len(summary['timeouts'])} signals):", flush=True)
            for hit in summary['timeouts']:
                print(f"   {hit['symbol']:12} {hit['timeframe']:6} | P&L: ${hit['pnl_usd']:8.2f} ({hit['pnl_pct']:+.2f}%)", flush=True)
        
        # Print stats
        stats = self.get_stats()
        print(f"\n📊 PEC STATS:", flush=True)
        print(f"   Total: {stats['total']} | Open: {stats['open']} | TP: {stats['tp_hit']} | SL: {stats['sl_hit']} | Timeout: {stats['timeout']}", flush=True)
        print(f"   Win Rate: {stats['win_rate_pct']:.1f}% | Total P&L: ${stats['total_pnl_usd']:+.2f}", flush=True)
        print(f"{'='*80}\n", flush=True)


def run_pec_update():
    """Main entry point for cron job"""
    executor = PECExecutor()
    summary = executor.update_signals()
    executor.print_announcement(summary)


if __name__ == '__main__':
    run_pec_update()
