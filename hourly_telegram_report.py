#!/usr/bin/env python3
"""
Hourly Telegram Report: Send tracker summaries to Telegram
- Runs both trackers  
- Keeps everything intact (full sections, no truncation)
- Sends to same Telegram channel as signals
"""

import subprocess
import re
from datetime import datetime
import os
import sys

# Import same Telegram config as signal sender
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

def extract_section_1(output):
    """Extract SECTION 1: TOTAL SIGNALS from pec_enhanced_reporter output"""
    # Look for the SECTION 1 marker and extract until next SECTION
    pattern = r'📊 SECTION 1: TOTAL SIGNALS.*?(?=📊 SECTION 2:|$)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(0).strip()
    return "Section 1 not found"

def extract_post_deployment_sections(output):
    """Extract specific sections from pec_post_deployment_tracker output - FULL SECTIONS"""
    sections = {}
    
    # Extract SIGNAL BREAKDOWN (full section)
    pattern = r'📋 SIGNAL BREAKDOWN\s*\n(.*?)(?=\nEXCLUDED FROM METRICS|---|\n[A-Z]{2,}|$)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        sections['breakdown'] = "📋 SIGNAL BREAKDOWN\n" + match.group(1).strip()
    
    # Extract CLOSED TRADES ANALYSIS (full section)
    pattern = r'🎯 CLOSED TRADES ANALYSIS.*?\n(.*?)(?=\n💰|---|\n[A-Z]{2,}|$)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        sections['closed'] = "🎯 CLOSED TRADES ANALYSIS\n" + match.group(1).strip()
    
    # Extract P&L BREAKDOWN (full section)
    pattern = r'💰 P&L BREAKDOWN\s*\n(.*?)(?=\n📊|---|\n[A-Z]{2,}|$)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        sections['pnl'] = "💰 P&L BREAKDOWN\n" + match.group(1).strip()
    
    # Extract OVERALL RISK:REWARD (full section)
    pattern = r'📊 OVERALL RISK:REWARD.*?\n(.*?)(?=\n---|\nRecommendation|$)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        sections['rr'] = "📊 OVERALL RISK:REWARD\n" + match.group(1).strip()
    
    return sections

def send_telegram_message(message):
    """Send message to same Telegram channel as signals"""
    if not BOT_TOKEN or not CHAT_ID or "YOUR_BOT_TOKEN" in BOT_TOKEN:
        print("❌ Telegram not configured in tg_config.py")
        return False
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    # Split message into chunks (Telegram limit is 4096 chars, use 3900 for safety)
    max_length = 3900
    messages = []
    
    if len(message) > max_length:
        # Split aggressively by newlines
        lines = message.split('\n')
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 < max_length:
                current += line + '\n'
            else:
                if current:
                    messages.append(current.strip())
                current = line + '\n'
        if current:
            messages.append(current.strip())
    else:
        messages = [message]
    
    success = True
    for i, msg in enumerate(messages):
        try:
            # Send without parse_mode to avoid formatting issues
            payload = {
                'chat_id': CHAT_ID,
                'text': msg
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ Telegram message {i+1}/{len(messages)} sent successfully")
            else:
                print(f"❌ Telegram error {response.status_code}: {response.text}")
                success = False
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            success = False
    
    return success

def main():
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    os.chdir(workspace)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📊 Starting hourly tracker report...")
    
    # Run both trackers
    print("  ⏳ Running pec_enhanced_reporter.py...")
    section1_output = run_command("python3 pec_enhanced_reporter.py 2>/dev/null")
    
    print("  ⏳ Running pec_post_deployment_tracker.py...")
    post_deploy_output = run_command("python3 pec_post_deployment_tracker.py 2>/dev/null")
    
    # Extract sections (FULL, not truncated)
    print("  🔍 Extracting sections...")
    section1 = extract_section_1(section1_output)
    post_deploy_sections = extract_post_deployment_sections(post_deploy_output)
    
    # Format message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M GMT+7")
    
    message = f"""
📊 **HOURLY TRACKER SUMMARY — {timestamp}**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**SECTION 1: TOTAL SIGNALS (Foundation + New)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{section1}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**POST-DEPLOYMENT TRACKER (3-Factor + 4-Factor)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{post_deploy_sections.get('breakdown', 'N/A')}

{post_deploy_sections.get('closed', 'N/A')}

{post_deploy_sections.get('pnl', 'N/A')}

{post_deploy_sections.get('rr', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Last Updated: {timestamp}
""".strip()
    
    # Save to file for reference
    report_file = os.path.join(workspace, f"hourly_telegram_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    with open(report_file, 'w') as f:
        f.write(message)
    
    print(f"  💾 Report saved: {report_file}")
    
    # Send to Telegram
    print("  📤 Sending to Telegram...")
    if send_telegram_message(message):
        print(f"✅ Hourly report sent to Telegram at {timestamp}")
        return True
    else:
        print(f"❌ Failed to send to Telegram")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
