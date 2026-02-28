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
        """Load the latest SIGNAL_TIERS_*.json file"""
        try:
            # Find all tier files
            tier_files = glob.glob("SIGNAL_TIERS_*.json")
            
            if not tier_files:
                print("[TIER] No SIGNAL_TIERS files found. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                self.tier_file = None
                return
            
            # Sort by modification time, get latest
            tier_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = tier_files[0]
            
            with open(latest_file, 'r') as f:
                self.tier_data = json.load(f)
            
            self.tier_file = latest_file
            print(f"[TIER] Loaded: {latest_file}", flush=True)
            
        except Exception as e:
            print(f"[TIER-ERROR] Failed to load tier data: {e}. Defaulting to Tier-X.", flush=True)
            self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
            self.tier_file = None
    
    def get_tier(self, timeframe: str, direction: str, route: str = None, regime: str = None) -> str:
        """
        Look up tier for a signal combo.
        
        Checks combinations in order:
        1. TF_DIR_ROUTE_REGIME (4D)
        3. TF_DIR_ROUTE (3D) 
        3. TF_DIR (2D)
        4. Default to Tier-X if not found
        
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
            
            # Build combo names to search for
            combos_to_check = []
            
            # 4D combo (if both route and regime available)
            if route_val and regime_val:
                combos_to_check.append(f"TF_DIR_ROUTE_REGIME_{tf}_{dir_val}_{route_val}_{regime_val}")
            
            # 3D combos
            if route_val:
                combos_to_check.append(f"TF_DIR_ROUTE_{tf}_{dir_val}_{route_val}")
            
            if regime_val:
                combos_to_check.append(f"TF_DIR_REGIME_{tf}_{dir_val}_{regime_val}")
            
            # 2D combo (always try this)
            combos_to_check.append(f"TF_DIR_{tf}_{dir_val}")
            
            # Search for each combo in tier lists
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
    """Convenience function to get tier for a signal"""
    lookup = get_tier_lookup()
    return lookup.get_tier(timeframe, direction, route, regime)
