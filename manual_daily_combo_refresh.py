#!/usr/bin/env python3
"""
MANUAL DAILY COMBO REFRESH - CORRECTED VERSION
Extracts combos from NEWEST yesterday's pec_post_deployment_tracker.py report
STEP-1: Dynamically finds NEWEST report from YESTERDAY (most recent end-of-day)
STEP-2: Applies dimensional filtering rules
STEP-3: Enforces max 4 combos per tier
STEP-4: Shows dimensional encodings for all locked combos
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import sys

# Add workspace to path for dimensional_encoder
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace')
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main')
try:
    from dimensional_encoder import encode_6d, encode_5d, encode_4d
    encoder_available = True
except ImportError:
    encoder_available = False

def main():
    print("="*80)
    print("MANUAL DAILY COMBO REFRESH")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GMT+7")
    print("="*80)
    
    # STEP 1: DYNAMICALLY FIND NEWEST REPORT FROM YESTERDAY
    print("\n" + "="*80)
    print("STEP 1: FIND NEWEST REPORT FROM YESTERDAY")
    print("="*80)
    
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    reports_dir = Path("/Users/geniustarigan/.openclaw/workspace/pec_reports")
    
    # Find all report files from YESTERDAY
    # Format: PEC_POST_DEPLOYMENT_TRACKER_v2_YYYY-MM-DD_HH-MM-SS.txt
    pattern = f"PEC_POST_DEPLOYMENT_TRACKER_v2_{yesterday.strftime('%Y-%m-%d')}_*.txt"
    yesterday_reports = sorted(reports_dir.glob(pattern))
    
    print(f"📋 Looking for reports from: {yesterday.strftime('%Y-%m-%d')}")
    print(f"   Found: {len(yesterday_reports)} reports from yesterday")
    
    if not yesterday_reports:
        print("❌ No reports found from yesterday")
        exit(1)
    
    # Select NEWEST (last in sorted list by filename)
    newest_report = yesterday_reports[-1]
    print(f"✅ NEWEST report selected: {newest_report.name}")
    
    # STEP 2: DYNAMICALLY EXTRACT COMBOS FROM TRACKER REPORT
    print("\n" + "="*80)
    print("STEP 2: DYNAMICALLY EXTRACT COMBOS FROM TRACKER REPORT (BOTH LONG & SHORT)")
    print("="*80)
    
    # Read the report and extract combos that start with "✓ TF_DIR"
    combos = []
    with open(newest_report, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("✓ TF_DIR"):
                # Format: "✓ TF_DIR_ROUTE_REGIME_SG_CONF_2h_SHORT_TREND CONTINUATION_RANGE_MAIN_BLOCKCHAIN_LOW | WR:  61.9% | P&L: $  -30.15 | Avg: $-1.44 | Closed: 21"
                # Extract: combo name, WR, P&L, avg_pnl, closed_trades
                parts = line.split(" | ")
                combo_name = parts[0][2:].strip()  # Remove "✓ "
                
                wr_str = parts[1].split(":")[1].strip().rstrip('%')
                pnl_str = parts[2].split(":")[1].strip().lstrip('$ ')
                avg_str = parts[3].split(":")[1].strip().lstrip('$ ')
                closed_str = parts[4].split(":")[1].strip()
                
                try:
                    wr = float(wr_str) / 100  # Convert percentage to decimal
                    pnl = float(pnl_str.replace('$', '').replace(',', '').strip())
                    avg_pnl = float(avg_str.replace('$', '').replace(',', '').strip())
                    closed_trades = int(closed_str)
                    
                    # Determine dimension from combo name
                    parts_count = combo_name.count('_')
                    if "CONF" in combo_name:
                        dim = "6D"
                    elif "SG" in combo_name:
                        dim = "5D"
                    elif parts_count >= 5:  # TF_DIR_ROUTE_REGIME = 4D
                        dim = "4D"
                        # VALIDATE 4D: Must have TF_DIR_ROUTE_REGIME_ header
                        if not combo_name.startswith("TF_DIR_ROUTE_REGIME_"):
                            continue  # Skip malformed 4D combos
                    else:
                        dim = "3D"
                        # Skip 3D combos (not valid for tier assignment)
                        continue
                    
                    combo = {
                        "combo": combo_name,
                        "tier": "Tier-X",  # Will be assigned later
                        "dimension": dim,
                        "wr": wr,
                        "pnl": pnl,
                        "avg_pnl": avg_pnl,
                        "closed_trades": closed_trades,
                        "valid_from": datetime.now().strftime("%Y-%m-%dT00:00:00Z"),
                        "generated_at": newest_report.name.split('_')[3] + "Z",  # Extract timestamp
                        "data_through": yesterday.strftime("%Y-%m-%dT16:59:59.999999Z"),
                    }
                    combos.append(combo)
                except:
                    pass  # Skip malformed lines
    
    print(f"✅ Dynamically extracted: {len(combos)} combos from {newest_report.name}")
    print(f"   (Includes BOTH LONG and SHORT combos)")
    
    # Separate by direction to show balance
    long_count = sum(1 for c in combos if "_LONG_" in c["combo"])
    short_count = sum(1 for c in combos if "_SHORT_" in c["combo"])
    print(f"   LONG: {long_count} | SHORT: {short_count}")
    
    # STEP 3: NO DIMENSIONAL FILTERING FOR NOW
    # (Dimensional rules will be enforced AFTER tier assignment based on tier rules)
    print("\n" + "="*80)
    print("STEP 3: KEEP ALL COMBOS FOR TIER ASSIGNMENT")
    print("="*80)
    
    filtered_combos = combos  # Keep all extracted combos
    print(f"✅ Keeping: {len(filtered_combos)} combos for tier assignment")
    
    # STEP 4: ASSIGN TIERS (SORTED BY WR DESC, THEN P&L DESC - TOP 4 PER TIER)
    print("\n" + "="*80)
    print("STEP 4: ASSIGN TIERS (TOP 4 COMBOS PER TIER - SORTED BY WR DESC → P&L DESC)")
    print("="*80)
    
    # Dimensional rules (STRICT)
    allowed_dims = {
        "Tier-1": ["6D"],           # 6D ONLY
        "Tier-2": ["6D", "5D"],     # 6D, 5D (no 4D)
        "Tier-3": ["6D", "5D", "4D"]  # 6D, 5D, 4D
    }
    
    # Sort ALL combos by WR (desc), then P&L (desc)
    all_sorted = sorted(filtered_combos, key=lambda x: (-x["wr"], -x["pnl"]))
    
    print(f"\n📊 Available: {len(all_sorted)} combos (LONG + SHORT combined)")
    
    # Assign tiers: take top 4 per tier that meet dimensional rules
    final_combos = []
    combo_idx = 0
    
    for tier_name in ["Tier-1", "Tier-2", "Tier-3"]:
        allowed = allowed_dims[tier_name]
        tier_combos = []
        
        # Find next 4 combos that meet dimensional rules for this tier
        while combo_idx < len(all_sorted) and len(tier_combos) < 4:
            combo = all_sorted[combo_idx]
            if combo["dimension"] in allowed:
                combo["tier"] = tier_name
                tier_combos.append(combo)
            combo_idx += 1
        
        final_combos.extend(tier_combos)
        
        long_cnt = sum(1 for c in tier_combos if "_LONG_" in c["combo"])
        short_cnt = sum(1 for c in tier_combos if "_SHORT_" in c["combo"])
        print(f"\n{tier_name} (allowed dims: {allowed}): {len(tier_combos)} combos (LONG: {long_cnt} | SHORT: {short_cnt})")
        for i, combo in enumerate(tier_combos, 1):
            direction = "LONG" if "_LONG_" in combo["combo"] else "SHORT"
            print(f"[{i}] {combo['dimension']} {direction} | WR={combo['wr']*100:.1f}% | P&L=${combo['pnl']:+.2f} | {combo['combo']}")
    
    print(f"\n" + "="*80)
    print(f"✅ FINAL RESULT: {len(final_combos)} COMBOS LOCKED FOR TODAY")
    print("="*80)
    
    # STEP 5: SHOW DIMENSIONAL ENCODINGS
    print("\n" + "="*80)
    print("STEP 5: DIMENSIONAL ENCODINGS FOR TODAY'S LOCKED COMBOS")
    print("="*80)
    
    if encoder_available:
        def parse_and_encode(combo_str, dimension):
            """Parse combo string and encode based on dimension"""
            # Strip prefix if present
            clean_str = combo_str
            for prefix in ["TF_DIR_ROUTE_REGIME_SG_CONF_", "TF_DIR_ROUTE_REGIME_SG_", "TF_DIR_ROUTE_REGIME_"]:
                if clean_str.startswith(prefix):
                    clean_str = clean_str[len(prefix):]
                    break
            
            parts = clean_str.split("_")
            try:
                if dimension == "6D":
                    # Example: 4h_LONG_REVERSAL_BULL_MAIN_BLOCKCHAIN_LOW
                    # or: 2h_LONG_TREND CONTINUATION_RANGE_MAIN_BLOCKCHAIN_LOW
                    tf = parts[0]
                    direction = parts[1]
                    
                    # Find regime (BULL/BEAR/RANGE) to locate where route ends
                    regime_idx = None
                    for i in range(2, len(parts)):
                        if parts[i] in ["BULL", "BEAR", "RANGE", "NEUTRAL"]:
                            regime_idx = i
                            break
                    
                    if regime_idx is None:
                        return None
                    
                    # Route is everything between direction and regime
                    route = " ".join(parts[2:regime_idx])
                    regime = parts[regime_idx]
                    
                    # Symbol group is next - could be MAIN_BLOCKCHAIN or TOP_ALTS, LOW_ALTS, MID_ALTS
                    # Valid symbol groups: LOW_ALTS, MID_ALTS, MAIN_BLOCKCHAIN, TOP_ALTS
                    # Handle multi-word symbol groups by reconstructing
                    sg_idx = regime_idx + 1
                    
                    # Look ahead to find symbol group
                    # Check if next 2 parts form a known symbol group
                    potential_sg = None
                    if sg_idx + 1 < len(parts):
                        potential_sg = parts[sg_idx] + "_" + parts[sg_idx + 1]
                    
                    if potential_sg in ["LOW_ALTS", "MID_ALTS", "MAIN_BLOCKCHAIN", "TOP_ALTS"]:
                        symbol_group = potential_sg
                        confidence = parts[sg_idx + 2]
                    else:
                        # Single-part symbol group (shouldn't happen in 6D, but handle it)
                        symbol_group = parts[sg_idx]
                        confidence = parts[sg_idx + 1] if sg_idx + 1 < len(parts) else "LOW"
                    
                    return encode_6d(tf, direction, route, regime, symbol_group, confidence)
                
                elif dimension == "5D":
                    # Example: After stripping "TF_DIR_ROUTE_REGIME_SG_"
                    # We have: "2h_LONG_TREND CONTINUATION_RANGE_TOP_ALTS"
                    # Format: <timeframe>_<direction>_<route>_<regime>_<symbol_group>
                    tf = parts[0]
                    direction = parts[1]
                    
                    # Find regime
                    regime_idx = None
                    for i in range(2, len(parts)):
                        if parts[i] in ["BULL", "BEAR", "RANGE", "NEUTRAL"]:
                            regime_idx = i
                            break
                    
                    if regime_idx is None:
                        return None
                    
                    route = " ".join(parts[2:regime_idx])
                    regime = parts[regime_idx]
                    
                    # Symbol group (no confidence in 5D)
                    sg_idx = regime_idx + 1
                    if sg_idx >= len(parts):
                        return None
                    
                    # Check if next 2 parts form a known multi-part symbol group
                    potential_sg = None
                    if sg_idx + 1 < len(parts):
                        potential_sg = parts[sg_idx] + "_" + parts[sg_idx + 1]
                    
                    if potential_sg in ["LOW_ALTS", "MID_ALTS", "MAIN_BLOCKCHAIN", "TOP_ALTS"]:
                        symbol_group = potential_sg
                    else:
                        # Single-part symbol group or entire remainder
                        symbol_group = "_".join(parts[sg_idx:])
                    
                    return encode_5d(tf, direction, route, regime, symbol_group)
                
                elif dimension == "4D":
                    # Example: After stripping prefix "TF_DIR_ROUTE_REGIME_"
                    # We have: "4h_SHORT_NONE_BEAR" → parts = ['4h', 'SHORT', 'NONE', 'BEAR']
                    # Format: <timeframe>_<direction>_<route>_<regime>
                    
                    if len(parts) < 4:
                        return None
                    
                    tf = parts[0]              # timeframe like 4h, 2h, etc.
                    direction = parts[1]      # LONG, SHORT
                    
                    # Find regime (BULL, BEAR, RANGE, NEUTRAL)
                    regime_idx = None
                    for i in range(2, len(parts)):
                        if parts[i] in ["BULL", "BEAR", "RANGE", "NEUTRAL"]:
                            regime_idx = i
                            break
                    
                    if regime_idx is None:
                        return None
                    
                    route = " ".join(parts[2:regime_idx])
                    regime = parts[regime_idx]
                    
                    return encode_4d(tf, direction, route, regime)
            except Exception as e:
                return None
        
        for tier_name in ["Tier-1", "Tier-2", "Tier-3"]:
            tier_combos = [c for c in final_combos if c["tier"] == tier_name]
            print(f"\n{tier_name}:")
            for i, combo in enumerate(tier_combos, 1):
                direction = "LONG" if "_LONG_" in combo["combo"] else "SHORT"
                encoded = parse_and_encode(combo["combo"], combo["dimension"])
                print(f"  [{i}] {combo['combo']}")
                if encoded:
                    print(f"      └─ ENCODED: {encoded}")
                else:
                    print(f"      └─ ⚠️  ENCODE FAILED (check combo format)")
    else:
        print("\n⚠️  DimensionalEncoder not available. Skipping encoding display.")
        print("   (dimensional_encoder functions not found in workspace)")
    
    print("\n" + "="*80)
    
    # Write to LOCKED_COMBOS_TODAY.py (as Python code, not JSONL)
    output_file = Path("/Users/geniustarigan/.openclaw/workspace/LOCKED_COMBOS_TODAY.py")
    submodule_file = Path("/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/LOCKED_COMBOS_TODAY.py")
    
    # Generate Python code
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    tomorrow_date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    generated_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')
    
    python_code = f"""\"\"\"
TODAY'S LOCKED ALLOWABLE COMBOS - {datetime.now().strftime('%B %d, %Y')}
Generated: {generated_datetime_str}
Source Report: {newest_report.name}

{len(final_combos)} COMBOS - LOCKED FOR THE DAY
Tier-1: {len([c for c in final_combos if c['tier'] == 'Tier-1'])} combos (6D only)
Tier-2: {len([c for c in final_combos if c['tier'] == 'Tier-2'])} combos (6D, 5D)
Tier-3: {len([c for c in final_combos if c['tier'] == 'Tier-3'])} combos (6D, 5D, 4D)

main.py should read from this file ONLY via get_locked_combos()
\"\"\"

# DATETIME VALIDATION: Ensures main.py is using TODAY's combos only
GENERATED_DATETIME = "{generated_datetime_str}"
GENERATED_DATE = "{today_date_str}"  # YYYY-MM-DD for date comparison
LOCK_EXPIRES_DATE = "{tomorrow_date_str}"  # Expires at midnight GMT+7

LOCKED_COMBOS = [
"""
    
    for combo in final_combos:
        python_code += f'    {{"combo": "{combo["combo"]}", "tier": "{combo["tier"]}", "dimension": "{combo["dimension"]}"}},\n'
    
    python_code += """]\n\ndef get_locked_combos():
    \"\"\"Return the locked combos for today\"\"\"
    return LOCKED_COMBOS
"""
    
    # Write to LOCKED_COMBOS_TODAY.py
    with open(output_file, 'w') as f:
        f.write(python_code)
    
    # Sync to submodule
    shutil.copy(output_file, submodule_file)
    
    # Save to .txt file for easy validation
    txt_file = Path(f"/Users/geniustarigan/.openclaw/workspace/TODAY_LOCKED_COMBOS_{today_date_str}.txt")
    
    with open(txt_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"TODAY'S LOCKED COMBOS - {today_date_str}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}\n")
        f.write(f"Source Report: {newest_report.name}\n")
        f.write("="*100 + "\n\n")
        f.write(f"TOTAL LOCKED COMBOS: {len(final_combos)}\n\n")
        
        for tier_name in ["Tier-1", "Tier-2", "Tier-3"]:
            tier_combos = [c for c in final_combos if c["tier"] == tier_name]
            f.write(f"\n{tier_name}: {len(tier_combos)} combos\n")
            f.write("-" * 100 + "\n")
            for i, combo in enumerate(tier_combos, 1):
                direction = "LONG" if "_LONG_" in combo["combo"] else "SHORT"
                f.write(f"{i}. {combo['dimension']} {direction} | WR={combo['wr']*100:5.1f}% | P&L=${combo['pnl']:+8.2f}\n")
                f.write(f"   {combo['combo']}\n")
    
    print(f"\n✅ Written to ROOT: {output_file}")
    print(f"✅ Synced to SUBMODULE: {submodule_file}")
    print(f"✅ Also saved to .txt: {txt_file}")
    print(f"\n🔒 Lock expires: {tomorrow_date_str} 00:00:00 GMT+7")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Source report: {newest_report.name}")
    print(f"✅ Dynamically extracted: {len(combos)} combos from report")
    print(f"✅ After dimensional filtering: {len(filtered_combos)} combos")
    print(f"✅ After tier assignment (top 4 each): {len(final_combos)} FINAL LOCKED COMBOS")
    long_total = sum(1 for c in final_combos if "_LONG_" in c["combo"])
    short_total = sum(1 for c in final_combos if "_SHORT_" in c["combo"])
    print(f"\nTotal: LONG={long_total} | SHORT={short_total}")
    print(f"Tier-1: 4 combos | Tier-2: 4 combos | Tier-3: 4 combos")
    print(f"\n✅ Report saved to: TODAY_LOCKED_COMBOS_{today_date_str}.txt")

if __name__ == "__main__":
    main()
