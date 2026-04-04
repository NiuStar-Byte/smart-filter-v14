"""
COMPLETE TIER LOOKUP - ALL DIMENSIONAL COMBOS
=============================================

Dimension levels (strict cascading, no gaps):

TIER-1: 6D → 5D (STOP at 5D, NO 4D/3D/2D)
TIER-2: 6D → 5D → 4D (STOP at 4D, NO 3D/2D)  
TIER-3: 6D → 5D → 4D → 3D (STOP at 3D, NO 2D)
TIER-X: 2D, 1D, no match

Dimensions Defined:
- 6D: TF × DIR × ROUTE × REGIME × SG × CONF
- 5D: TF × DIR × ROUTE × REGIME × SG
- 4D: TF × DIR × ROUTE × REGIME
- 3D: TF_DIR_ROUTE, TF_DIR_REGIME, DIR_ROUTE_REGIME, TF_ROUTE_REGIME
- 2D: TF_DIR, TF_REGIME, DIR_REGIME, DIR_ROUTE, ROUTE_REGIME
"""

import os
import json
from datetime import datetime

class TierLookupComplete:
    def __init__(self):
        self.tier_data = None
        self.tier_file = None
        self.load_latest_tiers()
    
    def load_latest_tiers(self):
        """Load combos from SIGNAL_TIERS_APPEND.jsonl"""
        try:
            workspace = "/Users/geniustarigan/.openclaw/workspace"
            filename = os.path.join(workspace, "smart-filter-v14-main/SIGNAL_TIERS_APPEND.jsonl")
            
            if not os.path.exists(filename):
                print(f"[TIER] {filename} not found. All signals will be Tier-X.", flush=True)
                self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
                return
            
            tier_combos = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
            
            with open(filename, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        combo = record.get('combo', '')
                        tier = record.get('tier', 'Tier-X')
                        
                        # Normalize tier name
                        tier_key = tier.lower().replace('-', '')  # 'Tier-1' → 'tier1'
                        if tier_key in tier_combos:
                            tier_combos[tier_key].append(combo)
                        else:
                            tier_combos['tierx'].append(combo)
                    except:
                        continue
            
            self.tier_data = tier_combos
            total = sum(len(v) for v in tier_combos.values())
            print(f"[TIER] Loaded {total} combos from {filename}", flush=True)
            print(f"[TIER]   Tier-1: {len(tier_combos['tier1'])} | Tier-2: {len(tier_combos['tier2'])} | Tier-3: {len(tier_combos['tier3'])} | Tier-X: {len(tier_combos['tierx'])}", flush=True)
            
        except Exception as e:
            print(f"[TIER-ERROR] Failed to load tier data: {e}. Defaulting to Tier-X.", flush=True)
            self.tier_data = {"tier1": [], "tier2": [], "tier3": [], "tierx": []}
    
    def _generate_all_6d_variants(self, tf, direction, route, regime, symbol_group, confidence_level):
        """Generate all 6D format variants"""
        variants = []
        
        # Pipe-separated (canonical)
        variants.append(f"{tf}|{direction}|{route}|{regime}|{symbol_group}|{confidence_level}")
        
        # Underscore-separated
        variants.append(f"{tf}_{direction}_{route}_{regime}_{symbol_group}_{confidence_level}")
        
        # With prefix
        variants.append(f"6D_{tf}_{direction}_{route}_{regime}_{symbol_group}_{confidence_level}")
        variants.append(f"TF_DIR_ROUTE_REGIME_SG_CONF_{tf}_{direction}_{route}_{regime}_{symbol_group}_{confidence_level}")
        
        return variants
    
    def _generate_all_5d_variants(self, tf, direction, route, regime, symbol_group):
        """Generate all 5D format variants"""
        variants = []
        
        # Pipe-separated
        variants.append(f"{tf}|{direction}|{route}|{regime}|{symbol_group}")
        
        # Underscore-separated
        variants.append(f"{tf}_{direction}_{route}_{regime}_{symbol_group}")
        
        # With prefix
        variants.append(f"5D_{tf}_{direction}_{route}_{regime}_{symbol_group}")
        variants.append(f"TF_DIR_ROUTE_REGIME_SG_{tf}_{direction}_{route}_{regime}_{symbol_group}")
        
        return variants
    
    def _generate_all_4d_variants(self, tf, direction, route, regime):
        """Generate all 4D format variants"""
        variants = []
        
        # Main variants
        variants.append(f"{tf}_{direction}_{route}_{regime}")
        variants.append(f"TF_DIR_ROUTE_REGIME_{tf}_{direction}_{route}_{regime}")
        
        return variants
    
    def _generate_all_3d_variants(self, tf, direction, route, regime, symbol_group=None):
        """Generate all 3D format variants - MANY COMBINATIONS"""
        variants = []
        
        # TF_DIR_ROUTE (first 3 dimensions)
        variants.append(f"TF_DIR_ROUTE_{tf}_{direction}_{route}")
        variants.append(f"{tf}_{direction}_{route}")
        
        # TF_DIR_REGIME
        variants.append(f"TF_DIR_REGIME_{tf}_{direction}_{regime}")
        
        # DIR_ROUTE_REGIME
        variants.append(f"DIR_ROUTE_REGIME_{direction}_{route}_{regime}")
        
        # TF_ROUTE_REGIME
        variants.append(f"TF_ROUTE_REGIME_{tf}_{route}_{regime}")
        
        # With symbol_group if available
        if symbol_group:
            variants.append(f"TF_DIR_ROUTE_SG_{tf}_{direction}_{route}_{symbol_group}")
            variants.append(f"TF_REGIME_SG_{tf}_{regime}_{symbol_group}")
        
        return variants
    
    def _generate_all_2d_variants(self, tf, direction, route, regime, symbol_group=None):
        """Generate all 2D format variants - MANY COMBINATIONS"""
        variants = []
        
        # TF_DIR
        variants.append(f"TF_DIR_{tf}_{direction}")
        variants.append(f"{tf}_{direction}")
        
        # TF_REGIME
        variants.append(f"TF_REGIME_{tf}_{regime}")
        
        # DIR_REGIME
        variants.append(f"DIR_REGIME_{direction}_{regime}")
        
        # DIR_ROUTE
        variants.append(f"DIR_ROUTE_{direction}_{route}")
        
        # ROUTE_REGIME
        variants.append(f"ROUTE_REGIME_{route}_{regime}")
        
        # With symbol_group if available
        if symbol_group:
            variants.append(f"TF_SG_{tf}_{symbol_group}")
            variants.append(f"DIR_SG_{direction}_{symbol_group}")
        
        return variants
    
    def _check_tier(self, tier_key, tf, direction, route, regime, symbol_group=None, confidence_level=None):
        """Check if ANY combo variant exists in tier list"""
        tier_combos = self.tier_data.get(tier_key, [])
        if not tier_combos:
            return False
        
        tier_combos_set = set(tier_combos)  # Faster lookup
        
        # For each dimensional level, try all variants
        if symbol_group and confidence_level:
            # 6D check
            for variant in self._generate_all_6d_variants(tf, direction, route, regime, symbol_group, confidence_level):
                if variant in tier_combos_set:
                    return True
        
        if symbol_group:
            # 5D check
            for variant in self._generate_all_5d_variants(tf, direction, route, regime, symbol_group):
                if variant in tier_combos_set:
                    return True
        
        # 4D check
        for variant in self._generate_all_4d_variants(tf, direction, route, regime):
            if variant in tier_combos_set:
                return True
        
        return False
    
    def _check_tier_3d_only(self, tier_key, tf, direction, route, regime, symbol_group=None):
        """Check only 3D combos"""
        tier_combos_set = set(self.tier_data.get(tier_key, []))
        
        for variant in self._generate_all_3d_variants(tf, direction, route, regime, symbol_group):
            if variant in tier_combos_set:
                return True
        
        return False
    
    def get_tier(self, timeframe, direction, route=None, regime=None, symbol_group=None, confidence_level=None):
        """
        COMPLETE tier lookup with STRICT dimensional cascading
        
        TIER-1: 6D → 5D (STOP at 5D)
        TIER-2: 6D → 5D → 4D (STOP at 4D)
        TIER-3: 6D → 5D → 4D → 3D (STOP at 3D)
        TIER-X: Everything else
        """
        if not self.tier_data:
            return "Tier-X"
        
        try:
            # Normalize inputs
            tf = str(timeframe).strip()
            dir_val = str(direction).strip().upper()
            route_val = str(route).strip() if route else None
            regime_val = str(regime).strip() if regime else None
            sg_val = str(symbol_group).strip() if symbol_group else None
            conf_val = str(confidence_level).strip() if confidence_level else None
            
            # ====== TIER-1: 6D → 5D (STOP at 5D) ======
            if self._check_tier("tier1", tf, dir_val, route_val, regime_val, sg_val, conf_val):
                return "Tier-1"
            
            # ====== TIER-2: 6D → 5D → 4D (STOP at 4D) ======
            if self._check_tier("tier2", tf, dir_val, route_val, regime_val, sg_val, conf_val):
                return "Tier-2"
            
            # ====== TIER-3: 6D → 5D → 4D → 3D (STOP at 3D) ======
            if self._check_tier("tier3", tf, dir_val, route_val, regime_val, sg_val, conf_val):
                return "Tier-3"
            
            # Check 3D for Tier-3 (below 4D)
            if self._check_tier_3d_only("tier3", tf, dir_val, route_val, regime_val, sg_val):
                return "Tier-3"
            
            # ====== TIER-X: Everything else ======
            return "Tier-X"
        
        except Exception as e:
            print(f"[TIER-ERROR] get_tier failed: {e}. Returning Tier-X.", flush=True)
            return "Tier-X"


# Aliases for backward compatibility
TierLookup = TierLookupComplete

# Global instance
_tier_lookup_complete = None

def get_tier_lookup() -> TierLookupComplete:
    """Get or create global tier lookup instance (backward compatible name)"""
    return get_tier_lookup_complete()

def get_tier_lookup_complete() -> TierLookupComplete:
    """Get or create global tier lookup instance"""
    global _tier_lookup_complete
    if _tier_lookup_complete is None:
        _tier_lookup_complete = TierLookupComplete()
    return _tier_lookup_complete


def get_signal_tier(timeframe: str, direction: str, route: str = None, regime: str = None, symbol_group: str = None, confidence_level: str = None) -> str:
    """Convenience function - get tier for a signal (reloads latest tier file each time)"""
    lookup = get_tier_lookup_complete()
    lookup.load_latest_tiers()  # Always reload latest tier data
    return lookup.get_tier(timeframe, direction, route, regime, symbol_group, confidence_level)
