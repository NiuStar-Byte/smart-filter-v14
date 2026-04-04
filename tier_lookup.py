"""
Tier Lookup - Real-time tier assignment for signals
Loads latest SIGNAL_TIERS_*.json and provides tier lookup for signal combos
"""

import os
import json
from datetime import datetime
import glob

class TierLookup:
    def __init__(self):
        self.tier_data = None
        self.tier_file = None
        self.load_latest_tiers()
    
    def load_latest_tiers(self):
        """Load combos from SIGNAL_TIERS_APPEND.jsonl (LOCKED daily combos)
        
        File format: Individual combo records (one per line):
        {"combo": "15min|SHORT|TREND...", "tier": "Tier-3", "dimension": "6D", ...}
        
        Groups combos by tier and valid_from date.
        """
        try:
            # FIX: Use absolute path (main.py may run from any directory)
            workspace = "/Users/geniustarigan/.openclaw/workspace"
            filename = os.path.join(workspace, "smart-filter-v14-main/SIGNAL_TIERS_APPEND.jsonl")
            
            if not os.path.exists(filename):
                print(f"[TIER] {filename} not found. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                self.tier_file = None
                return
            
            # Read JSONL file (one combo record per line)
            tier1_combos = []
            tier2_combos = []
            tier3_combos = []
            tierx_combos = []
            latest_valid_from = None
            
            combo_count = 0
            with open(filename, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        combo_name = record.get('combo', '')
                        tier = record.get('tier', 'Tier-X')
                        valid_from = record.get('valid_from')
                        
                        if valid_from:
                            latest_valid_from = valid_from
                        
                        # Group by tier
                        if tier == 'Tier-1':
                            tier1_combos.append(combo_name)
                        elif tier == 'Tier-2':
                            tier2_combos.append(combo_name)
                        elif tier == 'Tier-3':
                            tier3_combos.append(combo_name)
                        else:
                            tierx_combos.append(combo_name)
                        
                        combo_count += 1
                    except json.JSONDecodeError:
                        continue
            
            if combo_count == 0:
                print(f"[TIER] {filename} has no valid combos. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                self.tier_file = None
                return
            
            # Build tier data structure
            self.tier_data = {
                "valid_from": latest_valid_from or "unknown",
                "tier1": tier1_combos,
                "tier2": tier2_combos,
                "tier3": tier3_combos,
                "tierx": tierx_combos,
            }
            
            print(f"[TIER] Loaded {combo_count} combos from {filename} (valid_from: {latest_valid_from})", flush=True)
            print(f"[TIER]   Tier-1: {len(tier1_combos)} | Tier-2: {len(tier2_combos)} | Tier-3: {len(tier3_combos)} | Tier-X: {len(tierx_combos)}", flush=True)
            self.tier_file = filename
            
        except Exception as e:
            print(f"[TIER-ERROR] Failed to load tier data: {e}. Defaulting to Tier-X.", flush=True)
            self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
            self.tier_file = None
    
    def _check_combo_variants(self, tier_key: str, tf: str, dir_val: str, route_val: str = None, regime_val: str = None, symbol_group_val: str = None, confidence_level_val: str = None) -> bool:
        """Helper: Check if ANY combo variant exists in tier list
        
        Checks formats:
        1. 6D with confidence: tf|DIR|ROUTE|REGIME|SG|CONF
        2. 5D variants: tf|DIR|ROUTE|REGIME|SG
        3. Underscore format: tf_DIR_ROUTE_REGIME_SG
        """
        tier_combos = self.tier_data.get(tier_key, [])
        
        # Build all possible combo variants
        variants = []
        
        # 6D variants (with confidence_level)
        if symbol_group_val and route_val and regime_val and confidence_level_val:
            variants.extend([
                f"{tf}|{dir_val}|{route_val}|{regime_val}|{symbol_group_val}|{confidence_level_val}",
                f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}_{confidence_level_val}",
            ])
        
        if symbol_group_val and route_val and regime_val:
            # 5D variants (without confidence, fallback)
            variants.extend([
                f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                f"TF_DIR_ROUTE_REGIME_SG_{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                f"{tf}|{dir_val}|{route_val}|{regime_val}|{symbol_group_val}",
            ])
        
        if route_val and regime_val and symbol_group_val:
            # 5D variants
            variants.extend([
                f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                f"TF_DIR_ROUTE_REGIME_SG_{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
            ])
        
        if route_val and regime_val:
            # 4D variants
            variants.extend([
                f"{tf}_{dir_val}_{route_val}_{regime_val}",
                f"TF_DIR_ROUTE_REGIME_{tf}_{dir_val}_{route_val}_{regime_val}",
            ])
        
        if route_val and regime_val:
            # 3D variants (various prefixes)
            variants.extend([
                f"TF_DIR_ROUTE_{tf}_{dir_val}_{route_val}",
                f"TF_ROUTE_REGIME_{tf}_{route_val}_{regime_val}",
                f"DIR_ROUTE_REGIME_{dir_val}_{route_val}_{regime_val}",
            ])
        
        # Check if ANY variant is in tier list
        for variant in variants:
            if variant in tier_combos:
                return True
        return False
    
    def get_tier(self, timeframe: str, direction: str, route: str = None, regime: str = None, symbol_group: str = None, confidence_level: str = None) -> str:
        """
        Look up tier for a signal combo with STRICT dimensional cascading rules:
        
        Supports 6D combos: tf|direction|route|regime|symbol_group|confidence_level
        
        TIER-1: 6D → 5D (STOP AT 5D, NO 4D/3D/2D)
        TIER-2: 6D → 5D → 4D (STOP AT 4D, NO 3D/2D)
        TIER-3: 6D → 5D → 4D → 3D (STOP AT 3D, NO 2D)
        TIER-X: 2D, 1D, or no match (silent)
        
        This prevents loose 2D/3D patterns from catching unrelated signals.
        Each tier level only uses combos specific enough to represent the performance criteria.
        
        Returns: "Tier-1", "Tier-2", "Tier-3", or "Tier-X"
        """
        if not self.tier_data:
            return "Tier-X"
        
        try:
            # Normalize inputs
            tf = str(timeframe).strip()
            dir_val = str(direction).strip().upper()
            route_val = str(route).strip() if route else None
            regime_val = str(regime).strip() if regime else None
            symbol_group_val = str(symbol_group).strip() if symbol_group else None
            confidence_level_val = str(confidence_level).strip() if confidence_level else None
            
            # ============================================================================
            # TIER-1: 6D → 5D (STOP AT 5D, NO 4D/3D/2D)
            # ============================================================================
            if self._check_combo_variants("tier1", tf, dir_val, route_val, regime_val, symbol_group_val, confidence_level_val):
                return "Tier-1"
            
            # ============================================================================
            # TIER-2: 6D → 5D → 4D (STOP AT 4D, NO 3D OR 2D)
            # ============================================================================
            if self._check_combo_variants("tier2", tf, dir_val, route_val, regime_val, symbol_group_val, confidence_level_val):
                return "Tier-2"
            
            # 4D (both formats)
            if route_val and regime_val:
                combo_4d = f"{tf}_{dir_val}_{route_val}_{regime_val}"
                combo_4d_prefixed = f"TF_DIR_ROUTE_REGIME_{tf}_{dir_val}_{route_val}_{regime_val}"
                if combo_4d in self.tier_data.get("tier2", []):
                    return "Tier-2"
            
            # If no 6D, 5D, or 4D match in Tier-2, continue to Tier-3 (do NOT check 3D/2D)
            
            # ============================================================================
            # TIER-3: 6D → 5D → 4D → 3D (STOP AT 3D, NO 2D)
            # ============================================================================
            
            # 6D with confidence_level (most specific)
            if symbol_group_val and route_val and regime_val and confidence_level_val:
                combos_tier3_6d = [
                    f"{tf}|{dir_val}|{route_val}|{regime_val}|{symbol_group_val}|{confidence_level_val}",
                    f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}_{confidence_level_val}",
                ]
                for combo in combos_tier3_6d:
                    if combo in self.tier_data.get("tier3", []):
                        return "Tier-3"
            
            # 6D without confidence (fallback)
            if symbol_group_val and route_val and regime_val:
                combos_tier3_6d = [
                    f"6D_{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                    f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                ]
                for combo in combos_tier3_6d:
                    if combo in self.tier_data.get("tier3", []):
                        return "Tier-3"
            
            # 5D
            if symbol_group_val and route_val and regime_val:
                combo_5d = f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}"
                if combo_5d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            # 4D
            if route_val and regime_val:
                combo_4d = f"{tf}_{dir_val}_{route_val}_{regime_val}"
                if combo_4d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            # 3D combos (WITH label prefix)
            if route_val and regime_val:
                combos_3d = [
                    f"TF_ROUTE_REGIME_{tf}_{route_val}_{regime_val}",
                    f"DIR_ROUTE_REGIME_{dir_val}_{route_val}_{regime_val}",
                ]
                for combo in combos_3d:
                    if combo in self.tier_data.get("tier3", []):
                        return "Tier-3"
            
            if route_val:
                combo_3d = f"TF_DIR_ROUTE_{tf}_{dir_val}_{route_val}"
                if combo_3d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            if regime_val:
                combo_3d = f"TF_DIR_REGIME_{tf}_{dir_val}_{regime_val}"
                if combo_3d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            if route_val:
                combo_3d = f"DIR_ROUTE_{dir_val}_{route_val}"
                if combo_3d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            if regime_val:
                combo_3d = f"DIR_REGIME_{dir_val}_{regime_val}"
                if combo_3d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            if route_val and regime_val:
                combo_3d = f"ROUTE_REGIME_{route_val}_{regime_val}"
                if combo_3d in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            # If no 6D, 5D, 4D, or 3D match, return Tier-X (do NOT check 2D/1D)
            return "Tier-X"
        
        except Exception as e:
            print(f"[TIER-ERROR] get_tier failed: {e}. Returning Tier-X.", flush=True)
            return "Tier-X"
    
    def reload(self):
        """Reload tier data from latest file"""
        self.load_latest_tiers()


# Global instance
_tier_lookup = None

def get_tier_lookup() -> TierLookup:
    """Get or create global tier lookup instance"""
    global _tier_lookup
    if _tier_lookup is None:
        _tier_lookup = TierLookup()
    return _tier_lookup

def get_signal_tier(timeframe: str, direction: str, route: str = None, regime: str = None, symbol_group: str = None) -> str:
    """Convenience function to get tier for a signal (reloads latest tier file each time)
    Optional symbol_group allows checking 5D combos for better tier matching"""
    lookup = get_tier_lookup()
    lookup.reload()  # Always reload latest tier data (SIGNAL_TIERS updates frequently)
    return lookup.get_tier(timeframe, direction, route, regime, symbol_group)
