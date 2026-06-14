#!/usr/bin/env python3
"""
ASTERDEX LIVE MONITOR - Jun 7+ Only
Auto-updates every 5 minutes with new closed positions.
Includes: WR, P&L, RR, TP/SL durations
Press Ctrl+C to stop.
"""
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path

class AsterdexLiveMonitor:
    def __init__(self, update_interval=300):
        self.update_interval = update_interval  # 5 minutes = 300 seconds
        self.last_count = 0
        self.last_pnl = 0.0
        self.last_wr = 0.0
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
    
    def load_positions(self):
        """Load and filter positions opened >= Jun 7"""
        positions = []
        file_path = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
        
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            pos = json.loads(line)
                            opened_str = pos.get('opened', '')
                            if 'T' in opened_str:
                                opened = datetime.fromisoformat(opened_str.replace('Z', '+00:00'))
                                if opened >= self.cutoff:
                                    positions.append(pos)
                        except:
                            continue
        except:
            pass
        
        return positions
    
    def calculate_duration_hours(self, opened_str, closed_str):
        """Calculate duration between opened and closed in hours"""
        try:
            opened = datetime.fromisoformat(opened_str.replace('Z', '+00:00'))
            closed = datetime.fromisoformat(closed_str.replace('Z', '+00:00'))
            duration = (closed - opened).total_seconds() / 3600  # Convert to hours
            return duration
        except:
            return 0
    
    def calculate_metrics(self, positions):
        """Calculate performance metrics"""
        if not positions:
            return {
                'total': 0, 'wins': 0, 'losses': 0, 'breakeven': 0,
                'wr': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0,
                'rr': 0.0, 'avg_tp_duration': 0.0, 'avg_sl_duration': 0.0
            }
        
        total = len(positions)
        wins = sum(1 for p in positions if p.get('pnl_usd', 0) > 0)
        losses = sum(1 for p in positions if p.get('pnl_usd', 0) < 0)
        breakeven = sum(1 for p in positions if p.get('pnl_usd', 0) == 0)
        
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        avg_pnl = total_pnl / total if total > 0 else 0
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Calculate TP/SL durations
        tp_positions = [p for p in positions if p.get('pnl_usd', 0) > 0]
        sl_positions = [p for p in positions if p.get('pnl_usd', 0) < 0]
        
        tp_durations = []
        for p in tp_positions:
            duration = self.calculate_duration_hours(p.get('opened', ''), p.get('closed', ''))
            if duration > 0:
                tp_durations.append(duration)
        
        sl_durations = []
        for p in sl_positions:
            duration = self.calculate_duration_hours(p.get('opened', ''), p.get('closed', ''))
            if duration > 0:
                sl_durations.append(duration)
        
        avg_tp_duration = sum(tp_durations) / len(tp_durations) if tp_durations else 0
        avg_sl_duration = sum(sl_durations) / len(sl_durations) if sl_durations else 0
        
        # Calculate Risk:Reward ratio
        avg_win = sum(p.get('pnl_usd', 0) for p in tp_positions) / len(tp_positions) if tp_positions else 0
        avg_loss = abs(sum(p.get('pnl_usd', 0) for p in sl_positions) / len(sl_positions)) if sl_positions else 0
        rr = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'wr': wr,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'rr': rr,
            'avg_tp_duration': avg_tp_duration,
            'avg_sl_duration': avg_sl_duration
        }
    
    def get_new_positions(self, positions):
        """Find positions not yet seen"""
        new = []
        for pos in positions:
            # Check if position is new (more recent than last run)
            if len(new) > 0 or pos not in [p for p in positions[:self.last_count]]:
                # Simple heuristic: if count increased, positions are new
                new.append(pos)
        
        return positions[self.last_count:] if self.last_count < len(positions) else []
    
    def print_header(self):
        """Print live monitor header"""
        os.system('clear' if os.name != 'nt' else 'cls')
        print("\n" + "="*80)
        print("🔴 ASTERDEX LIVE MONITOR - JUN 7+ (Auto-refresh every 5 minutes)")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        print(f"Update Interval: {self.update_interval} seconds")
        print("Press Ctrl+C to stop\n")
    
    def print_summary(self, metrics, positions):
        """Print summary metrics"""
        indicator = "📈" if metrics['total_pnl'] > 0 else "📉" if metrics['total_pnl'] < 0 else "⚖️"
        
        print(f"\n{indicator} LIVE METRICS (as of {datetime.now().strftime('%H:%M:%S GMT+7')})")
        print(f"   Total Positions: {metrics['total']}")
        print(f"   Wins: {metrics['wins']} ({metrics['wins']/metrics['total']*100:.1f}%)" if metrics['total'] > 0 else "   Wins: 0")
        print(f"   Losses: {metrics['losses']} ({metrics['losses']/metrics['total']*100:.1f}%)" if metrics['total'] > 0 else "   Losses: 0")
        print(f"   Breakeven: {metrics['breakeven']}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   Win Rate: {metrics['wr']*100:.1f}%")
        print(f"   Total P&L: ${metrics['total_pnl']:.2f} USD")
        print(f"   Avg P&L/trade: ${metrics['avg_pnl']:.2f} USD")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   Avg Risk:Reward: {metrics['rr']:.2f}:1.0")
        print(f"   Avg TP Duration: {metrics['avg_tp_duration']:.2f} hours")
        print(f"   Avg SL Duration: {metrics['avg_sl_duration']:.2f} hours")
        
        # Show recent closed positions (last 5)
        sorted_by_close = sorted(positions, 
                                key=lambda x: x.get('closed', ''), 
                                reverse=True)[:5]
        
        if sorted_by_close:
            print(f"\n📌 RECENT CLOSED (Last 5)")
            print(f"   {'Position':<20} {'P&L $':<12} {'Duration':<12} {'Closed Time':<20}")
            print(f"   " + "-"*70)
            for pos in sorted_by_close:
                pos_id = pos.get('position_id', 'UNKNOWN')[:20]
                pnl = pos.get('pnl_usd', 0)
                closed = pos.get('closed', 'N/A')[:20]
                duration = self.calculate_duration_hours(pos.get('opened', ''), pos.get('closed', ''))
                indicator = "✅" if pnl > 0 else "❌" if pnl < 0 else "⚖️"
                print(f"   {pos_id:<20} ${pnl:<11.2f} {duration:>6.2f}h {closed:<20} {indicator}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.print_header()
        
        try:
            while True:
                positions = self.load_positions()
                metrics = self.calculate_metrics(positions)
                
                # Check for changes
                if metrics['total'] != self.last_count:
                    self.print_header()
                    new_count = metrics['total'] - self.last_count
                    if new_count > 0:
                        print(f"\n✨ NEW POSITIONS DETECTED: +{new_count}")
                    self.last_count = metrics['total']
                
                self.print_summary(metrics, positions)
                
                # Next update notice
                print(f"\n⏳ Next update: {datetime.now().strftime('%H:%M:%S')} + {self.update_interval}s")
                print("   Press Ctrl+C to stop\n")
                
                # Sleep for update interval
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n\n⏹️ Live monitor stopped.")
            print(f"Final metrics: {metrics['total']} positions | WR: {metrics['wr']*100:.1f}% | P&L: ${metrics['total_pnl']:.2f} USD | RR: {metrics['rr']:.2f}:1.0")

def main():
    monitor = AsterdexLiveMonitor(update_interval=300)  # 5 minutes
    monitor.monitor_loop()

if __name__ == '__main__':
    main()
