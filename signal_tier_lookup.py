#!/usr/bin/env python3
"""
Signal Tier Lookup - Maps signals to their tier based on latest SIGNAL_TIERS.json
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


class SignalTierLookup:
    def __init__(self):
        self.tiers = None
        self.tier_file = None
        self.load_latest_tiers()
    
    def load_latest_tiers(self):
        """Load the latest SIGNAL_TIERS_*.json file"""
        try:
            # Find all SIGNAL_TIERS files
            files = list(Path('.').glob('SIGNAL_TIERS_*.json'))
            if not files:
                print("[WARN] No SIGNAL_TIERS_*.json files found. Tier lookup disabled.")
                return False
            
            # Get the most recent one
            latest = max(files, key=os.path.getctime)
            self.tier_file = latest
            
            with open(latest, 'r') as f:
                self.tiers = json.load(f)
            
            return True
        except Exception as e:
            print(f"[WARN] Error loading tier file: {e}")
            return False
    
    def get_tier_emoji(self, tier_name):
        """Get emoji for tier"""
        emojis = {
            'tier1': '🥇',
            'tier2': '🥈',
            'tier3': '🥉',
            'tierx': '⚙️'
        }
        return emojis.get(tier_name, '❓')
    
    def get_tier_label(self, tier_name):
        """Get human-readable label for tier"""
        labels = {
            'tier1': 'Tier-1',
            'tier2': 'Tier-2',
            'tier3': 'Tier-3',
            'tierx': 'Tier-X'
        }
        return labels.get(tier_name, 'Unknown')
    
    def get_signal_tier(self, signal_dict):
        """
        Get tier for a signal based on its dimensions.
        
        Returns: {
            'tier': 'tier1',
            'emoji': '🥇',
            'label': 'Tier-1'
        }
        """
        if not self.tiers:
            return {'tier': 'tierx', 'emoji': '⚙️', 'label': 'Tier-X'}
        
        tf = signal_dict.get('timeframe', 'N/A')
        direction = signal_dict.get('signal_type', 'N/A')
        route = signal_dict.get('route', 'N/A')
        regime = signal_dict.get('regime', 'N/A')
        
        # Try to match combo patterns (in order of priority)
        # 3D combos first (most specific)
        patterns_3d = [
            f"TF_DIR_ROUTE_{tf}_{direction}_{route}",
            f"TF_DIR_REGIME_{tf}_{direction}_{regime}",
            f"DIR_ROUTE_REGIME_{direction}_{route}_{regime}",
            f"TF_ROUTE_REGIME_{tf}_{route}_{regime}",
        ]
        
        for pattern in patterns_3d:
            if pattern in self.tiers.get('tier1', []):
                return {'tier': 'tier1', 'emoji': '🥇', 'label': 'Tier-1'}
            elif pattern in self.tiers.get('tier2', []):
                return {'tier': 'tier2', 'emoji': '🥈', 'label': 'Tier-2'}
            elif pattern in self.tiers.get('tier3', []):
                return {'tier': 'tier3', 'emoji': '🥉', 'label': 'Tier-3'}
            elif pattern in self.tiers.get('tierx', []):
                return {'tier': 'tierx', 'emoji': '⚙️', 'label': 'Tier-X'}
        
        # Then 2D combos (less specific)
        patterns_2d = [
            f"TF_DIR_{tf}_{direction}",
            f"TF_REGIME_{tf}_{regime}",
            f"DIR_REGIME_{direction}_{regime}",
            f"DIR_ROUTE_{direction}_{route}",
            f"ROUTE_REGIME_{route}_{regime}",
        ]
        
        for pattern in patterns_2d:
            if pattern in self.tiers.get('tier1', []):
                return {'tier': 'tier1', 'emoji': '🥇', 'label': 'Tier-1'}
            elif pattern in self.tiers.get('tier2', []):
                return {'tier': 'tier2', 'emoji': '🥈', 'label': 'Tier-2'}
            elif pattern in self.tiers.get('tier3', []):
                return {'tier': 'tier3', 'emoji': '🥉', 'label': 'Tier-3'}
            elif pattern in self.tiers.get('tierx', []):
                return {'tier': 'tierx', 'emoji': '⚙️', 'label': 'Tier-X'}
        
        # Default: Tier-X (unknown)
        return {'tier': 'tierx', 'emoji': '⚙️', 'label': 'Tier-X'}
    
    def cleanup_old_tier_files(self, keep_days=7):
        """Delete SIGNAL_TIERS files older than keep_days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            deleted_count = 0
            
            for file in Path('.').glob('SIGNAL_TIERS_*.json'):
                file_time = datetime.fromtimestamp(os.path.getctime(file))
                if file_time < cutoff_time:
                    os.remove(file)
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"[INFO] Deleted {deleted_count} old SIGNAL_TIERS files (>{keep_days} days)")
            
            return deleted_count
        except Exception as e:
            print(f"[WARN] Error cleaning up tier files: {e}")
            return 0


if __name__ == "__main__":
    # Test
    lookup = SignalTierLookup()
    test_signal = {
        'timeframe': '1h',
        'signal_type': 'LONG',
        'route': 'TREND CONTINUATION',
        'regime': 'BULL'
    }
    tier_info = lookup.get_signal_tier(test_signal)
    print(f"Signal tier: {tier_info['emoji']} {tier_info['label']}")
