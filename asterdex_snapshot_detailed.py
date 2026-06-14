#!/usr/bin/env python3
"""
Asterdex Detailed Snapshot with OPEN + CLOSED Positions
One-shot detailed account snapshot with comprehensive metrics including historical closed trades
"""

import json
import requests
from datetime import datetime, timedelta
from aster_v3_auth import AsterV3Auth
from asterdex_config import (
    ASTER_MAIN_ACCOUNT,
    ASTER_API_WALLET_ADDRESS,
    ASTER_API_WALLET_PRIVATE_KEY,
    ASTERDEX_ENDPOINT,
    API_VERSION
)

def get_account_info():
    """Fetch account info from Asterdex (OPEN positions)"""
    auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    
    try:
        params = {}
        signed_params = auth.sign_request_v3(params, main_account=ASTER_MAIN_ACCOUNT)
        
        url = f"{ASTERDEX_ENDPOINT}/fapi/{API_VERSION}/account"
        response = requests.get(url, params=signed_params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Failed to fetch account: {e}")
        return None

def get_closed_positions():
    """Fetch closed positions from History (allOrders endpoint)"""
    auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    
    try:
        # Get all orders (both filled and partially filled = closed)
        params = {
            'limit': 100  # Get last 100 orders
        }
        signed_params = auth.sign_request_v3(params, main_account=ASTER_MAIN_ACCOUNT)
        
        url = f"{ASTERDEX_ENDPOINT}/fapi/{API_VERSION}/allOrders"
        response = requests.get(url, params=signed_params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ Could not fetch closed positions: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️ Failed to fetch closed positions: {e}")
        return []

def analyze_positions(account_data, closed_orders):
    """Analyze OPEN + CLOSED positions in detail"""
    
    # ===== OPEN POSITIONS =====
    positions = account_data.get('positions', [])
    open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'open': {
            'total': len(open_positions),
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'total_pnl': 0,
            'long_positions': [],
            'short_positions': [],
        },
        'closed': {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'positions': []
        },
        'combined': {
            'total_positions': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'loss_rate': 0,
            'avg_pnl': 0
        }
    }
    
    # Analyze OPEN positions
    for pos in open_positions:
        symbol = pos.get('symbol')
        amount = float(pos.get('positionAmt', 0))
        entry = float(pos.get('entryPrice', 0))
        mark = float(pos.get('markPrice', 0))
        pnl = float(pos.get('unrealizedProfit', 0))
        pnl_pct = (pnl / (entry * abs(amount))) * 100 if entry and amount != 0 else 0
        
        # Get timestamp (Asterdex returns updateTime in milliseconds)
        update_time_ms = pos.get('updateTime', 0)
        if update_time_ms:
            update_time = datetime.utcfromtimestamp(update_time_ms / 1000).isoformat()
        else:
            update_time = 'N/A'
        
        position_detail = {
            'symbol': symbol,
            'quantity': abs(amount),
            'entry_price': entry,
            'mark_price': mark,
            'pnl_usd': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'time_opened': update_time
        }
        
        if amount > 0:
            analysis['open']['long_positions'].append(position_detail)
        else:
            analysis['open']['short_positions'].append(position_detail)
        
        analysis['open']['total_pnl'] += pnl
        
        if pnl > 0.01:
            analysis['open']['wins'] += 1
        elif pnl < -0.01:
            analysis['open']['losses'] += 1
        else:
            analysis['open']['breakeven'] += 1
    
    # Analyze CLOSED positions (from allOrders)
    closed_by_symbol = {}
    
    for order in closed_orders:
        if order.get('status') != 'FILLED':
            continue
        
        symbol = order.get('symbol')
        if symbol not in closed_by_symbol:
            closed_by_symbol[symbol] = {
                'entries': [],
                'exits': []
            }
        
        # Convert time (in milliseconds) to ISO format
        order_time_ms = order.get('time', order.get('updateTime', 0))
        if order_time_ms:
            order_time = datetime.utcfromtimestamp(order_time_ms / 1000).isoformat()
        else:
            order_time = 'N/A'
        
        # Identify if entry or exit (entry = LIMIT, exit = MARKET or opposite side)
        order_type = order.get('type')
        side = order.get('side')
        
        if order_type == 'LIMIT':
            closed_by_symbol[symbol]['entries'].append({
                'price': float(order.get('avgPrice', 0)),
                'qty': float(order.get('executedQty', 0)),
                'side': side,
                'time': order_time,
                'orderId': order.get('orderId')
            })
        elif order_type == 'MARKET':
            closed_by_symbol[symbol]['exits'].append({
                'price': float(order.get('avgPrice', 0)),
                'qty': float(order.get('executedQty', 0)),
                'side': side,
                'time': order_time,
                'orderId': order.get('orderId')
            })
    
    # Calculate closed position P&L - Match entries with exits properly
    for symbol, trades in closed_by_symbol.items():
        if trades['entries'] and trades['exits']:
            # Sort by time to match chronologically
            entries_buy = sorted([e for e in trades['entries'] if e['side'] == 'BUY'], 
                               key=lambda x: x['time'])
            entries_sell = sorted([e for e in trades['entries'] if e['side'] == 'SELL'], 
                                key=lambda x: x['time'])
            exits_sell = sorted([e for e in trades['exits'] if e['side'] == 'SELL'], 
                              key=lambda x: x['time'])
            exits_buy = sorted([e for e in trades['exits'] if e['side'] == 'BUY'], 
                             key=lambda x: x['time'])
            
            # Match BUY entries with SELL exits (LONG positions)
            used_exits_sell = set()
            for entry in entries_buy:
                for idx, exit_trade in enumerate(exits_sell):
                    if idx not in used_exits_sell:
                        # Validate exit happens AFTER entry
                        try:
                            entry_dt = datetime.fromisoformat(entry['time'])
                            exit_dt = datetime.fromisoformat(exit_trade['time'])
                            if exit_dt > entry_dt:  # Must be chronological
                                entry_price = entry['price']
                                exit_price = exit_trade['price']
                                qty = min(entry['qty'], exit_trade['qty'])
                                pnl = (exit_price - entry_price) * qty
                                pnl_pct = ((exit_price - entry_price) / entry_price * 100)
                                
                                duration = exit_dt - entry_dt
                                duration_hours = duration.total_seconds() / 3600
                                duration_str = f"{int(duration_hours)}h {int((duration.total_seconds() % 3600) / 60)}m"
                                
                                closed_pos = {
                                    'symbol': symbol,
                                    'side': 'LONG',
                                    'quantity': qty,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl_usd': round(pnl, 2),
                                    'pnl_pct': round(pnl_pct, 2),
                                    'time_opened': entry['time'],
                                    'time_closed': exit_trade['time'],
                                    'duration': duration_str
                                }
                                
                                analysis['closed']['positions'].append(closed_pos)
                                analysis['closed']['total_pnl'] += pnl
                                used_exits_sell.add(idx)
                                
                                if pnl > 0.01:
                                    analysis['closed']['wins'] += 1
                                elif pnl < -0.01:
                                    analysis['closed']['losses'] += 1
                                break
                        except:
                            pass
            
            # Match SELL entries with BUY exits (SHORT positions)
            used_exits_buy = set()
            for entry in entries_sell:
                for idx, exit_trade in enumerate(exits_buy):
                    if idx not in used_exits_buy:
                        # Validate exit happens AFTER entry
                        try:
                            entry_dt = datetime.fromisoformat(entry['time'])
                            exit_dt = datetime.fromisoformat(exit_trade['time'])
                            if exit_dt > entry_dt:  # Must be chronological
                                entry_price = entry['price']
                                exit_price = exit_trade['price']
                                qty = min(entry['qty'], exit_trade['qty'])
                                pnl = (entry_price - exit_price) * qty
                                pnl_pct = ((entry_price - exit_price) / entry_price * 100)
                                
                                duration = exit_dt - entry_dt
                                duration_hours = duration.total_seconds() / 3600
                                duration_str = f"{int(duration_hours)}h {int((duration.total_seconds() % 3600) / 60)}m"
                                
                                closed_pos = {
                                    'symbol': symbol,
                                    'side': 'SHORT',
                                    'quantity': qty,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl_usd': round(pnl, 2),
                                    'pnl_pct': round(pnl_pct, 2),
                                    'time_opened': entry['time'],
                                    'time_closed': exit_trade['time'],
                                    'duration': duration_str
                                }
                                
                                analysis['closed']['positions'].append(closed_pos)
                                analysis['closed']['total_pnl'] += pnl
                                used_exits_buy.add(idx)
                                
                                if pnl > 0.01:
                                    analysis['closed']['wins'] += 1
                                elif pnl < -0.01:
                                    analysis['closed']['losses'] += 1
                                break
                        except:
                            pass
    
    analysis['closed']['total'] = len(analysis['closed']['positions'])
    
    # Combined metrics
    total_positions = analysis['open']['total'] + analysis['closed']['total']
    total_wins = analysis['open']['wins'] + analysis['closed']['wins']
    total_losses = analysis['open']['losses'] + analysis['closed']['losses']
    total_pnl = analysis['open']['total_pnl'] + analysis['closed']['total_pnl']
    
    analysis['combined'] = {
        'total_positions': total_positions,
        'total_pnl': round(total_pnl, 2),
        'win_rate': round((total_wins / total_positions * 100), 2) if total_positions > 0 else 0,
        'loss_rate': round((total_losses / total_positions * 100), 2) if total_positions > 0 else 0,
        'avg_pnl': round(total_pnl / total_positions, 2) if total_positions > 0 else 0,
        'total_wins': total_wins,
        'total_losses': total_losses
    }
    
    return analysis

def display_detailed(analysis):
    """Display detailed analysis with OPEN + CLOSED positions"""
    print("\n" + "="*110)
    print(f"ASTERDEX COMPREHENSIVE SNAPSHOT - {analysis['timestamp']}")
    print("="*110)
    
    # COMBINED METRICS
    combined = analysis['combined']
    print(f"\n🎯 OVERALL METRICS (OPEN + CLOSED):")
    print(f"  Total All-Time Positions: {combined['total_positions']}")
    print(f"  Total All-Time P&L: ${combined['total_pnl']:.2f}")
    print(f"  Total Wins: {combined['total_wins']} | Total Losses: {combined['total_losses']}")
    print(f"  Win Rate: {combined['win_rate']}% | Loss Rate: {combined['loss_rate']}%")
    print(f"  Avg P&L/Position: ${combined['avg_pnl']:.2f}")
    
    # OPEN POSITIONS SECTION
    open_data = analysis['open']
    print(f"\n" + "-"*110)
    print(f"📊 OPEN POSITIONS ({open_data['total']}):")
    print(f"  P&L: ${open_data['total_pnl']:.2f} | Wins: {open_data['wins']} | Losses: {open_data['losses']} | Breakeven: {open_data['breakeven']}")
    
    if open_data['long_positions']:
        print(f"\n  📈 LONG ({len(open_data['long_positions'])}):")
        print(f"  {'Symbol':<12} {'Qty':<12} {'Entry':<12} {'Mark':<12} {'P&L USD':<12} {'P&L %':<8} {'Time Opened':<26}")
        print(f"  {'-'*140}")
        for pos in open_data['long_positions']:
            time_opened = pos.get('time_opened', 'N/A')
            print(f"  {pos['symbol']:<12} {pos['quantity']:<12.2f} ${pos['entry_price']:<11.2f} ${pos['mark_price']:<11.2f} ${pos['pnl_usd']:<11.2f} {pos['pnl_pct']:>6.2f}% {time_opened}")
    
    if open_data['short_positions']:
        print(f"\n  📉 SHORT ({len(open_data['short_positions'])}):")
        print(f"  {'Symbol':<12} {'Qty':<12} {'Entry':<12} {'Mark':<12} {'P&L USD':<12} {'P&L %':<8} {'Time Opened':<26}")
        print(f"  {'-'*140}")
        for pos in open_data['short_positions']:
            time_opened = pos.get('time_opened', 'N/A')
            print(f"  {pos['symbol']:<12} {pos['quantity']:<12.2f} ${pos['entry_price']:<11.2f} ${pos['mark_price']:<11.2f} ${pos['pnl_usd']:<11.2f} {pos['pnl_pct']:>6.2f}% {time_opened}")
    
    # CLOSED POSITIONS SECTION
    closed_data = analysis['closed']
    print(f"\n" + "-"*110)
    print(f"📜 CLOSED POSITIONS ({closed_data['total']} from History):")
    print(f"  Total P&L: ${closed_data['total_pnl']:.2f} | Wins: {closed_data['wins']} | Losses: {closed_data['losses']}")
    
    if closed_data['positions']:
        print(f"\n  {'Symbol':<12} {'Side':<6} {'Entry (UTC)':<27} {'Exit (UTC)':<27} {'Duration':<12} {'P&L $':<10}")
        print(f"  {'-'*180}")
        for pos in closed_data['positions'][:20]:  # Show last 20 closed positions
            time_opened = pos.get('time_opened', 'N/A')
            time_closed = pos.get('time_closed', 'N/A')
            duration = pos.get('duration', 'N/A')
            entry_price = pos.get('entry_price', 0)
            exit_price = pos.get('exit_price', 0)
            print(f"  {pos['symbol']:<12} {pos['side']:<6} {time_opened:<27} {time_closed:<27} {duration:<12} ${pos['pnl_usd']:<9.2f}")
            print(f"    Qty: {pos['quantity']:.2f} | Entry: ${entry_price:.6f} | Exit: ${exit_price:.6f} | P&L %: {pos['pnl_pct']:.2f}%")
        if len(closed_data['positions']) > 20:
            print(f"  ... and {len(closed_data['positions']) - 20} more closed positions")
    else:
        print(f"  No closed positions found in recent history")
    
    print("\n" + "="*110 + "\n")

def main():
    """Main - get comprehensive snapshot with OPEN + CLOSED positions"""
    print("🚀 Fetching Asterdex comprehensive snapshot...")
    print("   Fetching OPEN positions...")
    
    account_data = get_account_info()
    
    if not account_data:
        print("❌ Failed to fetch account data")
        return
    
    print("   Fetching CLOSED positions from History...")
    closed_orders = get_closed_positions()
    
    analysis = analyze_positions(account_data, closed_orders)
    display_detailed(analysis)
    
    # Save to file
    output_file = "/tmp/asterdex_snapshot_detailed.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✅ Comprehensive snapshot saved to {output_file}")

if __name__ == "__main__":
    main()
