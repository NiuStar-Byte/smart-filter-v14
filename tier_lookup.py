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
        """Load the latest entry from SIGNAL_TIERS.json (append-only file)"""
        try:
            filename = "SIGNAL_TIERS.json"
            
            if not os.path.exists(filename):
                print(f"[TIER] {filename} not found. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                self.tier_file = None
                return
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Handle both old format (single dict) and new format (list)
            if isinstance(data, list):
                if not data:
                    print(f"[TIER] {filename} is empty. All signals will be Tier-X.", flush=True)
                    self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                else:
                    # Get latest entry (last in list)
                    latest_entry = data[-1]
                    self.tier_data = latest_entry
                    print(f"[TIER] Loaded latest entry from {filename} (timestamp: {latest_entry.get('timestamp', 'N/A')})", flush=True)
            else:
                # Old single-dict format
                self.tier_data = data
                print(f"[TIER] Loaded {filename} (legacy format)", flush=True)
            
            self.tier_file = filename
            
        except Exception as e:
            print(f"[TIER-ERROR] Failed to load tier data: {e}. Defaulting to Tier-X.", flush=True)
            self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
            self.tier_file = None
    
    def get_tier(self, timeframe: str, direction: str, route: str = None, regime: str = None) -> str:
        """
        Look up tier for a signal combo (ALL parent combos, not just TF-anchored).
        
        Checks combinations in order:
        1. 4D: TF_DIR_ROUTE_REGIME
        2. 3D: TF_DIR_ROUTE, TF_DIR_REGIME, TF_ROUTE_REGIME, DIR_ROUTE_REGIME, DIR_ROUTE, DIR_REGIME, ROUTE_REGIME
        3. 2D: TF_DIR, TF_ROUTE, TF_REGIME, DIR_ROUTE, DIR_REGIME, ROUTE_REGIME
        4. 1D: TF, DIR, ROUTE, REGIME
        5. Default to Tier-X if not found
        
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
            
            # Build ALL possible combo names to search for (most specific to least)
            combos_to_check = []
            
            # 4D combo (most specific)
            if route_val and regime_val:
                combos_to_check.append(f"TF_DIR_ROUTE_REGIME_{tf}_{dir_val}_{route_val}_{regime_val}")
            
            # 3D combos (7 possibilities)
            if route_val and regime_val:
                combos_to_check.append(f"TF_ROUTE_REGIME_{tf}_{route_val}_{regime_val}")
                combos_to_check.append(f"DIR_ROUTE_REGIME_{dir_val}_{route_val}_{regime_val}")
            if route_val:
                combos_to_check.append(f"TF_DIR_ROUTE_{tf}_{dir_val}_{route_val}")
            if regime_val:
                combos_to_check.append(f"TF_DIR_REGIME_{tf}_{dir_val}_{regime_val}")
            if route_val:
                combos_to_check.append(f"DIR_ROUTE_{dir_val}_{route_val}")
            if regime_val:
                combos_to_check.append(f"DIR_REGIME_{dir_val}_{regime_val}")
            if route_val and regime_val:
                combos_to_check.append(f"ROUTE_REGIME_{route_val}_{regime_val}")
            
            # 2D combos
            combos_to_check.append(f"TF_DIR_{tf}_{dir_val}")
            if route_val:
                combos_to_check.append(f"TF_ROUTE_{tf}_{route_val}")
            if regime_val:
                combos_to_check.append(f"TF_REGIME_{tf}_{regime_val}")
            if route_val:
                combos_to_check.append(f"DIR_ROUTE_{dir_val}_{route_val}")
            if regime_val:
                combos_to_check.append(f"DIR_REGIME_{dir_val}_{regime_val}")
            if route_val and regime_val:
                combos_to_check.append(f"ROUTE_REGIME_{route_val}_{regime_val}")
            
            # Search for each combo in tier lists (stop at first match)
            for combo in combos_to_check:
                if combo in self.tier_data.get("tier1", []):
                    return "Tier-1"
                elif combo in self.tier_data.get("tier2", []):
                    return "Tier-2"
                elif combo in self.tier_data.get("tier3", []):
                    return "Tier-3"
            
            # Not found in any tier = Tier-X
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

def get_signal_tier(timeframe: str, direction: str, route: str = None, regime: str = None) -> str:
    """Convenience function to get tier for a signal (reloads latest tier file each time)"""
    lookup = get_tier_lookup()
    lookup.reload()  # Always reload latest tier data (SIGNAL_TIERS updates frequently)
    return lookup.get_tier(timeframe, direction, route, regime)
