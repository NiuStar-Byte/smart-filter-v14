#!/usr/bin/env python3
"""
Hourly Telegram Report v2: Clean summaries to Telegram (2 messages only)
- Message 1: pec_enhanced_reporter.py summary
- Message 2: pec_post_deployment_tracker.py summary
"""

import subprocess
import re
from datetime import datetime
import os
import sys

sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main')
from tg_config import BOT_TOKEN, CHAT_ID
import requests

def run_command(cmd):
    """Run command and capture output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return result.stdout
    except Exception as e:
        print(f"Error running command: {e}")
        return ""

def extract_section1_summary(output):
    """Extract and format Section 1 summary"""
    msg = "📊 SECTION 1: TOTAL SIGNALS (Foundation + New)\n"
    
    # Report generated timestamp
    match = re.search(r'Report Generated:\s*([\d\-\s:]+GMT\+\d+)', output)
    if match:
        msg += f"Report Generated: {match.group(1)}\n"
    
    # Total Signals Loaded
    match = re.search(r'Total Signals Loaded:\s*(\d+)', output)
    if match:
        msg += f"Total Signals Loaded: {match.group(1)}\n"
    
    msg += "\nSIGNAL BREAKDOWN (All signals shown for audit trail):\n"
    msg += "INCLUDED IN METRICS (TP/SL/TIMEOUT/OPEN):\n"
    
    # Extract counts
    for status in ['TP_HIT', 'SL_HIT', 'TIMEOUT', 'OPEN']:
        pattern = rf'•\s*{status}:\s*(\d+)'
        match = re.search(pattern, output)
        if match:
            msg += f"• {status}: {match.group(1)}\n"
    
    # Subtotal included
    pattern = r'Subtotal \(Counted in WR & P&L\):\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Subtotal (Counted in WR & P&L): {match.group(1)}\n"
    
    msg += "\nEXCLUDED FROM METRICS (Not counted in WR or P&L):\n"
    
    # Extract excluded
    pattern = r'•\s*REJECTED_NOT_SENT_TELEGRAM:\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"• REJECTED_NOT_SENT_TELEGRAM: {match.group(1)}\n"
    
    pattern = r'•\s*STALE_TIMEOUT:\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"• STALE_TIMEOUT: {match.group(1)}\n"
    
    msg += "\nCLOSED TRADES ANALYSIS (Backtest Signals Only):\n"
    
    # Closed trades count
    pattern = r'Closed Trades \(Clean Data\):\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Closed Trades (Clean Data): {match.group(1)}\n"
    
    # Status breakdown
    for status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
        pattern = rf'•\s*{status}:\s*(\d+)'
        match = re.search(pattern, output)
        if match:
            msg += f"• {status}: {match.group(1)}\n"
    
    # TIMEOUT breakdown
    pattern = r'- TimeOut Win:\s*([\d,]+)\s*\(approximate\)'
    match = re.search(pattern, output)
    if match:
        msg += f"  - TimeOut Win: {match.group(1)}\n"
    
    pattern = r'- TimeOut Loss:\s*([\d,]+)\s*\(approximate\)'
    match = re.search(pattern, output)
    if match:
        msg += f"  - TimeOut Loss: {match.group(1)}\n"
    
    msg += "\nOverall Win Rate:\n"
    
    # Overall WR
    pattern = r'Overall Win Rate:\s*([\d.]+)%'
    match = re.search(pattern, output)
    if match:
        msg += f"Overall WR: {match.group(1)}%\n"
    
    # WR TP&SL only
    pattern = r'Win Rate \(based on TP & SL ONLY\):\s*([\d.]+)%'
    match = re.search(pattern, output)
    if match:
        msg += f"WR (TP & SL ONLY): {match.group(1)}%\n"
    
    msg += "\nTotal P&L (Clean Data):\n"
    
    # Total P&L
    pattern = r'Total P&L \(Clean Data\):\s*\$?([-\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Total P&L: ${match.group(1)}\n"
    
    # Averages
    pattern = r'Avg P&L per Signal:\s*\$?([-\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg per Signal: ${match.group(1)}\n"
    
    pattern = r'Avg P&L per Closed Trade:\s*\$?([-\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg per Closed Trade: ${match.group(1)}\n"
    
    msg += "\nP&L BREAKDOWN:\n"
    
    # P&L by type - look for "INCLUDED IN TOTAL P&L" section with dollar amounts
    pattern = r'INCLUDED IN TOTAL P&L.*?•\s*TP_HIT:\s*\$?\s*([-+\d,.]+)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        msg += f"• TP_HIT: ${match.group(1)}\n"
    
    pattern = r'INCLUDED IN TOTAL P&L.*?•\s*SL_HIT:\s*\$?\s*([-+\d,.]+)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        msg += f"• SL_HIT: ${match.group(1)}\n"
    
    pattern = r'INCLUDED IN TOTAL P&L.*?•\s*TIMEOUT:\s*\$?\s*([-+\d,.]+)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        msg += f"• TIMEOUT: ${match.group(1)}\n"
    
    msg += "\nAverage P&L per Count:\n"
    
    # Avg per type - with full label (accounting for indentation)
    pattern = r'Avg P&L TP per Count TP:\s*\$?([-+\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg P&L TP per Count TP: ${match.group(1)}\n"
    
    pattern = r'Avg P&L SL per Count SL:\s*\$?([-+\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg P&L SL per Count SL: ${match.group(1)}\n"
    
    msg += "\nRisk:Reward (RR) Metrics:\n"
    
    # RR metrics
    pattern = r'Highest RR:\s*([-\d.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Highest RR: {match.group(1)}\n"
    
    pattern = r'Avg RR:\s*([-\d.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg RR: {match.group(1)}\n"
    
    pattern = r'Lowest RR:\s*([-\d.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Lowest RR: {match.group(1)}\n"
    
    return msg.strip()

def extract_post_deployment_summary(output):
    """Extract and format post-deployment tracker summary"""
    msg = "📊 POST-DEPLOYMENT TRACKER (3-Factor + 4-Factor Normalization)\n"
    
    # Deployment cut-off
    pattern = r'Deployment Cut-off:\s*([\d\-T:Z]+)\s*/\s*([\d:]+\s*GMT\+\d+)\s*\(\d{4}-\d{2}-\d{2}\)\s*—\s*onwards'
    match = re.search(pattern, output)
    if match:
        msg += f"Deployment Cut-off: {match.group(1)} / {match.group(2)} onwards\n"
    
    # Report generated
    pattern = r'Report Generated:\s*([\d\-\s:]+GMT\+\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Report Generated: {match.group(1)}\n"
    
    msg += "\n📋 SIGNAL BREAKDOWN\n"
    msg += "INCLUDED IN METRICS (TP/SL/TIMEOUT/OPEN - Counted in WR & P&L):\n"
    
    # Signal counts
    for status in ['TP_HIT', 'SL_HIT', 'TIMEOUT', 'OPEN']:
        pattern = rf'•\s*{status}:\s*(\d+)'
        match = re.search(pattern, output)
        if match:
            msg += f"• {status}: {match.group(1)}\n"
    
    # Subtotal included
    pattern = r'Subtotal \(Counted in WR & P&L\):\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Subtotal: {match.group(1)}\n"
    
    msg += "\nEXCLUDED FROM METRICS:\n"
    
    # Excluded
    pattern = r'•\s*REJECTED_NOT_SENT_TELEGRAM:\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"• REJECTED: {match.group(1)}\n"
    
    pattern = r'•\s*STALE_TIMEOUT:\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"• STALE_TIMEOUT: {match.group(1)}\n"
    
    # Add total loaded signals
    pattern = r'Loaded\s*(\d+)\s*post-deployment signals'
    match = re.search(pattern, output)
    if match:
        msg += f"\nTotal Loaded Signals: {match.group(1)}\n"
    
    msg += "\n🎯 CLOSED TRADES ANALYSIS (Backtest Signals Only)\n"
    
    # Closed trades
    pattern = r'Closed Trades \(Clean Data\):\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Closed Trades: {match.group(1)}\n"
    
    # Breakdown
    for status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
        pattern = rf'•\s*{status}:\s*(\d+)'
        match = re.search(pattern, output)
        if match:
            msg += f"• {status}: {match.group(1)}\n"
    
    # TIMEOUT breakdown
    pattern = r'- Timeout Win:\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"  - Timeout Win: {match.group(1)}\n"
    
    pattern = r'- Timeout Loss:\s*(\d+)'
    match = re.search(pattern, output)
    if match:
        msg += f"  - Timeout Loss: {match.group(1)}\n"
    
    msg += "\nOverall Win Rate:\n"
    
    # Win rates
    pattern = r'Overall Win Rate:\s*([\d.]+)%'
    match = re.search(pattern, output)
    if match:
        msg += f"Overall WR: {match.group(1)}%\n"
    
    pattern = r'Win Rate \(based on TP & SL ONLY\):\s*([\d.]+)%'
    match = re.search(pattern, output)
    if match:
        msg += f"WR (TP & SL ONLY): {match.group(1)}%\n"
    
    msg += "\n💰 P&L BREAKDOWN\n"
    
    # P&L by type - look for INCLUDED IN TOTAL P&L section with dollar amounts
    pattern = r'INCLUDED IN TOTAL P&L.*?•\s*TP_HIT:\s*\$?\s*([-+\d,.]+)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        msg += f"• TP_HIT: ${match.group(1)}\n"
    
    pattern = r'INCLUDED IN TOTAL P&L.*?•\s*SL_HIT:\s*\$?\s*([-+\d,.]+)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        msg += f"• SL_HIT: ${match.group(1)}\n"
    
    pattern = r'INCLUDED IN TOTAL P&L.*?•\s*TIMEOUT:\s*\$?\s*([-+\d,.]+)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        msg += f"• TIMEOUT: ${match.group(1)}\n"
    
    msg += "\nAverage P&L:\n"
    
    # Averages - order: Signal, Closed Trade, SL, TP
    pattern = r'Average P&L per Signal \(Included\):\s*\$?([-\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg per Signal: ${match.group(1)}\n"
    
    pattern = r'Average P&L per Closed Trade:\s*\$?([-\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg per Closed Trade: ${match.group(1)}\n"
    
    pattern = r'Average P&L per SL_HIT:\s*\$?([-\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg per SL: ${match.group(1)}\n"
    
    pattern = r'Average P&L per TP_HIT:\s*\$?([-+\d,.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg per TP: ${match.group(1)}\n"
    
    msg += "\n📊 OVERALL RISK:REWARD (RR) METRICS\n"
    
    # RR metrics
    pattern = r'Highest RR:\s*([-\d.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Highest RR: {match.group(1)}\n"
    
    pattern = r'Avg RR:\s*([-\d.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Avg RR: {match.group(1)}\n"
    
    pattern = r'Lowest RR:\s*([-\d.]+)'
    match = re.search(pattern, output)
    if match:
        msg += f"Lowest RR: {match.group(1)}\n"
    
    return msg.strip()

def send_telegram_message(message):
    """Send message to Telegram"""
    if not BOT_TOKEN or not CHAT_ID or "YOUR_BOT_TOKEN" in BOT_TOKEN:
        print("❌ Telegram not configured")
        return False
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    try:
        payload = {'chat_id': CHAT_ID, 'text': message}
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Telegram error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Send failed: {e}")
        return False

def main():
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    os.chdir(workspace)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📊 Starting hourly tracker report...")
    
    # Run both trackers
    print("  ⏳ Running pec_enhanced_reporter.py...")
    section1_output = run_command("python3 pec_enhanced_reporter.py 2>/dev/null")
    
    print("  ⏳ Running pec_post_deployment_tracker.py...")
    post_deploy_output = run_command("python3 pec_post_deployment_tracker.py 2>/dev/null")
    
    # Extract summaries
    print("  🔍 Extracting summaries...")
    msg1 = extract_section1_summary(section1_output)
    msg2 = extract_post_deployment_summary(post_deploy_output)
    
    # Send both messages
    print("  📤 Sending to Telegram...")
    
    success = True
    if send_telegram_message(msg1):
        print("✅ Message 1 sent (Section 1 Summary)")
    else:
        success = False
    
    if send_telegram_message(msg2):
        print("✅ Message 2 sent (Post-Deployment Summary)")
    else:
        success = False
    
    if success:
        print(f"✅ Hourly report sent to Telegram at {datetime.now().strftime('%Y-%m-%d %H:%M GMT+7')}")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
