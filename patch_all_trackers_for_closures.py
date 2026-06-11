#!/usr/bin/env python3
"""
PATCH ALL TRACKERS - Add closure reconciliation support
Built: May 19 2026, 11:56 GMT+7

This script patches ALL tracker files to use signal_loader.py (universal signal loader with closures)
Replaces direct SIGNALS_MASTER.jsonl loads with signal_loader.load_all_signals()

Safe: Creates backups before patching
"""

import os
import re
import shutil
from pathlib import Path

TRACKER_PATTERNS = [
    '/Users/geniustarigan/.openclaw/workspace/*tracker*.py',
    '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/*tracker*.py',
]

EXCLUDE_DIRS = [
    '_retired', '.venv', 'smart-filter-clean', 'smart-filter-v14-clean',
    'smart-filter-v14-main.broken', 'polymarket-copytrade'
]

def is_excluded(filepath):
    """Check if file is in excluded directories"""
    for exclude in EXCLUDE_DIRS:
        if exclude in filepath:
            return True
    return False

def find_tracker_files():
    """Find all active tracker files"""
    trackers = []
    workspace = Path('/Users/geniustarigan/.openclaw/workspace')
    
    # Find all *tracker*.py files
    for pattern in ['*tracker*.py', '*analysis*.py']:
        for root_dir in [workspace, workspace / 'smart-filter-v14-main']:
            for file in root_dir.glob(pattern):
                if file.is_file() and not is_excluded(str(file)):
                    trackers.append(str(file))
    
    return sorted(list(set(trackers)))  # Deduplicate

def patch_file(filepath):
    """Patch a single tracker file to use signal_loader"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Skip if already patched
        if 'from signal_loader import' in content or 'import signal_loader' in content:
            return (filepath, 'SKIP', 'Already patched')
        
        # Add import at top
        if 'import json' in content:
            content = content.replace(
                'import json',
                'import json\nfrom signal_loader import load_all_signals, get_signal_metrics, load_signals_filtered'
            )
        elif 'import os' in content:
            content = content.replace(
                'import os',
                'import os\nfrom signal_loader import load_all_signals, get_signal_metrics, load_signals_filtered'
            )
        else:
            # Add import at top of file
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if not line.startswith('#') and not line.startswith('#!/'):
                    insert_idx = i
                    break
            lines.insert(insert_idx, 'from signal_loader import load_all_signals, get_signal_metrics, load_signals_filtered')
            content = '\n'.join(lines)
        
        # Replace common signal loading patterns
        patterns = [
            # Pattern 1: Loading from SIGNALS_MASTER.jsonl directly
            (r'with\s+open\([\'"].*?SIGNALS_MASTER\.jsonl[\'"][,\)][^:]*:\s*for\s+line\s+in\s+f:\s*(?:if|try).*?\n.*?signals\.append\(json\.loads\(line\)\)', 
             'signals = load_all_signals()'),
            
            # Pattern 2: Assigning to self.signals_file
            (r'self\.signals_file\s*=\s*[\'"].*?SIGNALS_MASTER\.jsonl[\'"]',
             'self.signals_file = None  # Use load_all_signals() instead'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                # Found pattern, add comment
                if replacement not in content:
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # If content changed, save backup and write new version
        if content != original_content:
            # Backup
            backup_path = filepath + '.backup.closure-patch'
            shutil.copy(filepath, backup_path)
            
            # Write patched version
            with open(filepath, 'w') as f:
                f.write(content)
            
            return (filepath, 'PATCHED', f'Backup: {backup_path}')
        else:
            # Manually add the import if automatic patching didn't work
            return (filepath, 'NEEDS_REVIEW', 'Add: from signal_loader import load_all_signals')
    
    except Exception as e:
        return (filepath, 'ERROR', str(e))

def main():
    """Patch all tracker files"""
    print("=" * 80)
    print("PATCHING ALL TRACKERS FOR CLOSURE RECONCILIATION")
    print("=" * 80)
    
    trackers = find_tracker_files()
    print(f"\nFound {len(trackers)} active tracker files\n")
    
    results = {
        'PATCHED': [],
        'SKIP': [],
        'NEEDS_REVIEW': [],
        'ERROR': [],
    }
    
    for filepath in trackers:
        filename = os.path.basename(filepath)
        filepath_short = '/'.join(filepath.split('/')[-2:])  # Last 2 path components
        
        status, result_code, msg = patch_file(filepath)
        results[result_code].append((filepath_short, msg))
        
        print(f"[{result_code:12}] {filepath_short:60} {msg}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ PATCHED: {len(results['PATCHED'])}")
    print(f"⏭️  SKIPPED: {len(results['SKIP'])} (already updated)")
    print(f"⚠️  NEEDS_REVIEW: {len(results['NEEDS_REVIEW'])}")
    print(f"❌ ERRORS: {len(results['ERROR'])}")
    
    if results['NEEDS_REVIEW']:
        print(f"\nFiles needing manual review:")
        for filepath, msg in results['NEEDS_REVIEW']:
            print(f"  - {filepath}: {msg}")
    
    print(f"\n✅ All trackers can now use:")
    print(f"   from signal_loader import load_all_signals")
    print(f"   signals = load_all_signals()  # Returns signals with closures applied")

if __name__ == '__main__':
    main()
