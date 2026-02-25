# backtest_real.py
# Realistic historical backtesting for Smart-Filter signals
# Tests actual win rate vs. confidence correlation
# Uses TP/SL exit logic (not forward-looking)

import pandas as pd
import json
from datetime import datetime, timedelta
import pytz
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from pec_engine import find_realistic_exit

WIB = pytz.timezone('Asia/Jakarta')

def backtest_symbol_timeframe(symbol, tf, days=100, min_score=14):
    """
    Run realistic backtest over historical data.
    
    Args:
        symbol: str, e.g., "BTC-USDT"
        tf: str, e.g., "30min"
        days: int, lookback period
        min_score: int, minimum filter score to consider signal
    
    Returns:
        dict with results and metrics
    """
    print(f"\n[BACKTEST] Starting {symbol} {tf} ({days} days)...")
    
    try:
        # Fetch historical OHLCV
        df = get_ohlcv(symbol, tf, limit=1000)
        if df is None or len(df) < 300:
            print(f"[BACKTEST] ERROR: Not enough data for {symbol} {tf}")
            return None
        
        df = df.reset_index()
        if 'timestamp' not in df.columns and 0 in df.columns:
            df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].max() > 1e10 else 's')
        df = df.set_index('timestamp')
        
        signals = []
        results = []
        
        # Warmup: start after 200 bars (EMA200 needs data)
        for i in range(200, len(df) - 30):
            try:
                # Get OHLCV window up to current bar
                df_window = df.iloc[:i+1].copy()
                
                # Run SmartFilter
                sf = SmartFilter(
                    symbol=symbol,
                    df=df_window,
                    tf=tf,
                    min_score=min_score,
                    required_passed=None
                )
                
                signal = sf.scan()
                
                # Only record signals that pass min_score
                if signal and signal.get('score', 0) >= min_score:
                    entry_price = df.iloc[i]['close']
                    signal_type = signal['bias']  # LONG or SHORT
                    fired_idx = i
                    
                    # Find realistic exit with TP/SL
                    exit_result = find_realistic_exit(
                        df,
                        entry_idx=fired_idx,
                        entry_price=entry_price,
                        signal_type=signal_type,
                        max_bars=20,
                        tp_pct=1.5,
                        sl_pct=1.0,
                        maker_fee=0.001
                    )
                    
                    if exit_result:
                        pnl = (exit_result['exit_price'] - entry_price) if signal_type == "LONG" else (entry_price - exit_result['exit_price'])
                        pnl_pct = (pnl / entry_price) * 100
                        
                        result_record = {
                            'symbol': symbol,
                            'tf': tf,
                            'signal_time': str(df.index[fired_idx]),
                            'signal_type': signal_type,
                            'entry_price': float(entry_price),
                            'exit_price': float(exit_result['exit_price']),
                            'exit_reason': exit_result['exit_reason'],
                            'pnl': float(pnl),
                            'pnl_pct': float(pnl_pct),
                            'mfe': float(exit_result['mfe']),
                            'mae': float(exit_result['mae']),
                            'score': signal.get('score', 0),
                            'confidence': signal.get('confidence', 0),
                            'passed_gk': signal.get('passed_gk', 0),
                            'max_gk': signal.get('max_gk', 0)
                        }
                        
                        results.append(result_record)
                        signals.append(signal)
                
            except Exception as e:
                print(f"[BACKTEST] Error at bar {i}: {e}")
                continue
        
        if not results:
            print(f"[BACKTEST] No signals generated for {symbol} {tf}")
            return None
        
        df_results = pd.DataFrame(results)
        
        # Calculate metrics
        total_signals = len(df_results)
        winning_signals = len(df_results[df_results['pnl'] > 0])
        losing_signals = len(df_results[df_results['pnl'] < 0])
        break_even = total_signals - winning_signals - losing_signals
        
        win_rate = (winning_signals / total_signals * 100) if total_signals > 0 else 0
        avg_win = df_results[df_results['pnl'] > 0]['pnl_pct'].mean() if winning_signals > 0 else 0
        avg_loss = df_results[df_results['pnl'] < 0]['pnl_pct'].mean() if losing_signals > 0 else 0
        total_pnl = df_results['pnl'].sum()
        total_pnl_pct = (total_pnl / (df_results['entry_price'].mean() * total_signals)) * 100
        
        # Profit factor
        gross_profit = df_results[df_results['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_results[df_results['pnl'] < 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Maximum drawdown
        cumulative_pnl = df_results['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
        
        metrics = {
            'symbol': symbol,
            'tf': tf,
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'losing_signals': losing_signals,
            'break_even': break_even,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 3),
            'avg_loss': round(avg_loss, 3),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'avg_confidence': round(df_results['confidence'].mean(), 1),
            'avg_score': round(df_results['score'].mean(), 1),
        }
        
        return {
            'metrics': metrics,
            'results': df_results,
            'signals': signals
        }
    
    except Exception as e:
        print(f"[BACKTEST] Critical error for {symbol} {tf}: {e}")
        return None

def run_multi_backtest(symbols, timeframes, days=100):
    """
    Run backtest across multiple symbols and timeframes.
    
    Args:
        symbols: list, e.g., ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        timeframes: list, e.g., ["30min", "15min"]
        days: int, lookback period
    
    Returns:
        dict with all results and summary
    """
    print(f"\n========== SMART-FILTER BACKTEST ==========")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Period: {days} days")
    print(f"Entry: First bar of signal")
    print(f"Exit: TP (+1.5%) | SL (-1.0%) | Timeout (20 bars)")
    print(f"Fee: 0.1% (KuCoin maker)")
    print(f"==========================================\n")
    
    all_results = {}
    summary_metrics = []
    
    for symbol in symbols:
        for tf in timeframes:
            result = backtest_symbol_timeframe(symbol, tf, days)
            if result:
                key = f"{symbol}_{tf}"
                all_results[key] = result
                summary_metrics.append(result['metrics'])
                
                metrics = result['metrics']
                print(f"✅ {symbol:12} {tf:6} | "
                      f"Signals: {metrics['total_signals']:3} | "
                      f"Win Rate: {metrics['win_rate']:6.1f}% | "
                      f"P&L: {metrics['total_pnl_pct']:+7.2f}% | "
                      f"PF: {metrics['profit_factor']:6.2f}")
            else:
                print(f"❌ {symbol:12} {tf:6} | Insufficient data or error")
    
    # Summary table
    if summary_metrics:
        df_summary = pd.DataFrame(summary_metrics).sort_values('win_rate', ascending=False)
        
        print("\n========== SUMMARY (sorted by Win Rate) ==========")
        print(df_summary[['symbol', 'tf', 'total_signals', 'win_rate', 'total_pnl_pct', 'profit_factor']].to_string(index=False))
        
        # Export to JSON
        timestamp = datetime.now(WIB).strftime("%Y-%m-%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.json"
        
        export_data = {
            'timestamp': timestamp,
            'config': {
                'symbols': symbols,
                'timeframes': timeframes,
                'days': days,
                'exit_logic': 'TP: +1.5%, SL: -1.0%, Timeout: 20 bars',
                'fee': '0.1% (KuCoin maker)'
            },
            'summary': df_summary.to_dict('records'),
            'detailed': all_results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ JSON exported to: {filename}")
        
        # Export summary to Excel
        try:
            xlsx_filename = f"backtest_summary_{timestamp}.xlsx"
            
            with pd.ExcelWriter(xlsx_filename, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Get workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets['Summary']
                
                from openpyxl.styles import Font, PatternFill, Alignment
                
                # Header formatting
                header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF")
                
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                
                # Auto-width columns
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
            
            print(f"✅ Excel exported to: {xlsx_filename}")
        except ImportError:
            print(f"⚠️ openpyxl not installed. Install with: pip install openpyxl")
        except Exception as e:
            print(f"⚠️ Error saving Excel: {e}")
        
        return all_results, df_summary
    else:
        print("\n❌ No backtest results generated")
        return None, None

if __name__ == "__main__":
    # Example: Backtest BTC, ETH, SOL, XRP on 15-min, 30-min, and 1-hour
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"]
    timeframes = ["15min", "30min", "1h"]
    
    all_results, df_summary = run_multi_backtest(symbols, timeframes, days=100)
    
    if df_summary is not None:
        print("\n📊 KEY INSIGHTS:")
        best = df_summary.iloc[0]
        worst = df_summary.iloc[-1]
        print(f"   Best:  {best['symbol']} {best['tf']} ({best['win_rate']:.1f}% win rate, {best['total_pnl_pct']:+.2f}% P&L)")
        print(f"   Worst: {worst['symbol']} {worst['tf']} ({worst['win_rate']:.1f}% win rate, {worst['total_pnl_pct']:+.2f}% P&L)")
        print(f"\n💡 Use best-performing pairs/timeframes for live trading")
