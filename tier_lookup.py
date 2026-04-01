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
        """Load the latest entry from SIGNAL_TIERS_APPEND.jsonl (LOCKED daily combos)"""
        try:
            filename = "smart-filter-v14-main/SIGNAL_TIERS_APPEND.jsonl"
            
            if not os.path.exists(filename):
                print(f"[TIER] {filename} not found. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                self.tier_file = None
                return
            
            # Read JSONL file (one entry per line)
            entries = []
            with open(filename, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            if not entries:
                print(f"[TIER] {filename} has no valid entries. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                self.tier_file = None
                return
            
            # Get latest entry (last in file = today's locked combos)
            latest_entry = entries[-1]
            self.tier_data = latest_entry
            valid_from = latest_entry.get('valid_from', 'N/A')
            print(f"[TIER] Loaded latest entry from {filename} (valid_from: {valid_from})", flush=True)
            self.tier_file = filename
            
        except Exception as e:
            print(f"[TIER-ERROR] Failed to load tier data: {e}. Defaulting to Tier-X.", flush=True)
            self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
            self.tier_file = None
    
    def get_tier(self, timeframe: str, direction: str, route: str = None, regime: str = None, symbol_group: str = None) -> str:
        """
        Look up tier for a signal combo with STRICT dimensional cascading rules:
        
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
            
            # ============================================================================
            # TIER-1: 6D → 5D (STOP AT 5D, NO 4D/3D/2D)
            # ============================================================================
            
            # 6D: TF_DIR_ROUTE_REGIME_SG_CONFIDENCE (if all params present)
            # Note: confidence_level not passed, so we check 6D patterns with labels
            combos_tier1_6d = [
                f"6D_{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",  # Try labeled 6D
            ]
            if symbol_group_val and route_val and regime_val:
                # Also try unlabeled 6D format
                combos_tier1_6d.append(f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}")
            
            for combo in combos_tier1_6d:
                if combo in self.tier_data.get("tier1", []):
                    return "Tier-1"
            
            # 5D: TF_DIR_ROUTE_REGIME_SG (no confidence)
            if symbol_group_val and route_val and regime_val:
                combo_5d = f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}"
                if combo_5d in self.tier_data.get("tier1", []):
                    return "Tier-1"
            
            # If no 6D or 5D match in Tier-1, continue to Tier-2 (do NOT check 4D/3D/2D)
            
            # ============================================================================
            # TIER-2: 6D → 5D → 4D (STOP AT 4D, NO 3D OR 2D)
            # ============================================================================
            
            # 6D with labels (for future support)
            if symbol_group_val and route_val and regime_val:
                combos_tier2_6d = [
                    f"6D_{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                ]
                for combo in combos_tier2_6d:
                    if combo in self.tier_data.get("tier2", []):
                        return "Tier-2"
                
                # Unlabeled 6D
                combo_6d = f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}"
                if combo_6d in self.tier_data.get("tier2", []):
                    return "Tier-2"
            
            # 5D
            if symbol_group_val and route_val and regime_val:
                combo_5d = f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}"
                if combo_5d in self.tier_data.get("tier2", []):
                    return "Tier-2"
            
            # 4D
            if route_val and regime_val:
                combo_4d = f"{tf}_{dir_val}_{route_val}_{regime_val}"
                if combo_4d in self.tier_data.get("tier2", []):
                    return "Tier-2"
            
            # If no 6D, 5D, or 4D match in Tier-2, continue to Tier-3 (do NOT check 3D/2D)
            
            # ============================================================================
            # TIER-3: 6D → 5D → 4D → 3D (STOP AT 3D, NO 2D)
            # ============================================================================
            
            # 6D with labels
            if symbol_group_val and route_val and regime_val:
                combos_tier3_6d = [
                    f"6D_{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}",
                ]
                for combo in combos_tier3_6d:
                    if combo in self.tier_data.get("tier3", []):
                        return "Tier-3"
                
                # Unlabeled 6D
                combo_6d = f"{tf}_{dir_val}_{route_val}_{regime_val}_{symbol_group_val}"
                if combo_6d in self.tier_data.get("tier3", []):
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
