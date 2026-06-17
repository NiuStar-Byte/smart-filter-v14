#!/usr/bin/env python3
"""
Multi-Timeframe (MTF) Alignment Analyzer
Analyzes signal quality by checking alignment with higher timeframes
Completely isolated from main signal generation pipeline
Last updated: 2026-04-07
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from mtf_alignment_config import (
    MTF_ALIGNMENT_ENABLED,
    ALIGNMENT_WEIGHTS,
    CONFIDENCE_ADJUSTMENTS,
    ALIGNMENT_THRESHOLDS,
    ALIGNMENT_LABELS,
    TF_CHECK_CASCADE,
    LOOKBACK_MODE,
    MATCH_SCORES,
    MTF_ALIGNMENT_RESULTS_FILE,
    MTF_ALIGNMENT_LOG_LEVEL
)
from mtf_1d_fetcher import get_1d_context  # For 4h→1d confirmation


class MTFAlignmentAnalyzer:
    """Analyzes multi-timeframe alignment for signal quality assessment"""
    
    def __init__(self):
        self.enabled = MTF_ALIGNMENT_ENABLED
        self.weights = ALIGNMENT_WEIGHTS
        self.adjustments = CONFIDENCE_ADJUSTMENTS
        self.thresholds = ALIGNMENT_THRESHOLDS
        self.cascade = TF_CHECK_CASCADE
        self.results_file = MTF_ALIGNMENT_RESULTS_FILE
        self.signals_master = 'SIGNALS_MASTER.jsonl'
        
        # Cache recent signals for quick lookups
        self._signal_cache = {}
        self._cache_loaded = False
    
    def _load_signal_cache(self):
        """Load recent signals from SIGNALS_MASTER.jsonl into memory for fast lookups"""
        if self._cache_loaded:
            return
        
        try:
            if not os.path.exists(self.signals_master):
                print(f"[MTF] Warning: {self.signals_master} not found", flush=True)
                return
            
            # Load last 5000 signals (recent data for current/active candles)
            signals = []
            with open(self.signals_master, 'r') as f:
                for line in f:
                    if line.strip():
                        signals.append(json.loads(line))
            
            # Cache by (symbol, timeframe) → signal data
            for sig in signals[-5000:]:  # Keep recent 5000
                key = (sig.get('symbol'), sig.get('timeframe'))
                self._signal_cache[key] = sig
            
            self._cache_loaded = True
            print(f"[MTF] Cached {len(self._signal_cache)} recent signals for TF lookups", flush=True)
        except Exception as e:
            print(f"[MTF] Error loading signal cache: {e}", flush=True)
    
    def get_latest_signal_for_tf(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the latest signal for a given symbol + timeframe"""
        
        # Special case: 1d (daily) and 1w (weekly) timeframes are fetched from KuCoin API (not from signals)
        if timeframe in ['1d', '1w']:
            try:
                # Map internal names to KuCoin API timeframes
                kucoin_tf = "1day" if timeframe == "1d" else "1week"
                context = get_1d_context(symbol, timeframe=kucoin_tf)
                if context and 'candle_data' in context:
                    candle = context['candle_data']
                    return {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'signal_type': candle.get('direction', 'NONE'),
                        'direction': candle.get('direction', 'NONE'),
                        'regime': context.get('regime', 'NONE'),
                        'route': 'NONE',
                        'source': 'kucoin_api'
                    }
            except Exception as e:
                print(f"[MTF] Error fetching {timeframe} context for {symbol}: {e}", flush=True)
                return None
        
        # For other timeframes, use cached signals
        self._load_signal_cache()
        key = (symbol, timeframe)
        if key in self._signal_cache:
            return self._signal_cache[key]
        
        return None
    
    def _match_score(self, value1: any, value2: any, dimension: str) -> Tuple[float, str]:
        """
        Score how well two values match
        Returns: (score, detail_string)
        """
        if value1 is None or value2 is None:
            return MATCH_SCORES['unknown'], f"{dimension}: one value missing"
        
        v1 = str(value1).upper().strip()
        v2 = str(value2).upper().strip()
        
        if v1 == v2:
            return MATCH_SCORES['full_match'], f"{dimension}: {v1} ✓"
        elif v1 == 'NONE' or v2 == 'NONE':
            # NONE = waiting/no signal, treat as partial
            return MATCH_SCORES['partial_match'], f"{dimension}: {v1} vs {v2} (partial)"
        else:
            return MATCH_SCORES['no_match'], f"{dimension}: {v1} vs {v2} ✗"
    
    def analyze(self, signal_data: Dict) -> Dict:
        """
        Main analysis function
        Analyzes multi-TF alignment for a given signal
        
        Args:
            signal_data: Signal object with keys like symbol, timeframe, direction, regime, route, confidence
        
        Returns:
            {
                'alignment_score': 0-100,
                'alignment_band': 'strong'|'partial'|'weak'|'conflict',
                'alignment_label': 'Strong Alignment',
                'alignment_icon': '🔗',
                'base_confidence': 57.6,
                'adjusted_confidence': 66.2,
                'confidence_multiplier': 1.15,
                'mtf_checks': {...},  # Detail of what was checked
                'warning': None or string
            }
        """
        
        if not self.enabled:
            # Feature disabled, return neutral result (v2)
            return {
                'alignment_score': 50,
                'alignment_band': 'neutral',  # v2: neutral for disabled/no-data state
                'alignment_label': 'Feature Disabled',
                'alignment_icon': '⊘',
                'base_confidence': signal_data.get('confidence', 0),
                'adjusted_confidence': signal_data.get('confidence', 0),
                'confidence_multiplier': 1.0,
                'mtf_checks': {},
                'warning': None,
                'enabled': False
            }
        
        # Extract signal data
        symbol = signal_data.get('symbol', 'UNKNOWN')
        timeframe = signal_data.get('timeframe', '15min')
        direction = signal_data.get('signal_type') or signal_data.get('direction', 'LONG')
        regime = signal_data.get('regime', 'NONE')
        route = signal_data.get('route', 'NONE')
        base_confidence = float(signal_data.get('confidence', 50))
        
        # Determine which TFs to check
        tfs_to_check = self.cascade.get(timeframe, [])
        
        if not tfs_to_check:
            # No TF to check (e.g., 4h is highest), return neutral (v2)
            return {
                'alignment_score': 50,
                'alignment_band': 'neutral',  # v2: neutral when no higher TF exists to check
                'alignment_label': 'No Higher TF',
                'alignment_icon': '⊘',
                'base_confidence': base_confidence,
                'adjusted_confidence': base_confidence,
                'confidence_multiplier': 1.0,
                'mtf_checks': {'note': f'{timeframe} is highest TF, no checks performed'},
                'warning': None,
                'enabled': True
            }
        
        # Collect match scores from all checked TFs
        all_matches = []
        mtf_details = {}
        
        for check_tf in tfs_to_check:
            check_signal = self.get_latest_signal_for_tf(symbol, check_tf)
            
            if not check_signal:
                # No signal data for this TF
                mtf_details[check_tf] = {
                    'status': 'NO_DATA',
                    'detail': f'No signal found for {symbol} {check_tf}'
                }
                continue
            
            check_direction = check_signal.get('signal_type') or check_signal.get('direction', 'NONE')
            check_regime = check_signal.get('regime', 'NONE')
            check_route = check_signal.get('route', 'NONE')
            
            # Score each dimension
            dir_score, dir_detail = self._match_score(direction, check_direction, 'Direction')
            reg_score, reg_detail = self._match_score(regime, check_regime, 'Regime')
            rte_score, rte_detail = self._match_score(route, check_route, 'Route')
            
            # Store details
            mtf_details[check_tf] = {
                'status': 'CHECKED',
                'direction_match': (dir_score, dir_detail),
                'regime_match': (reg_score, reg_detail),
                'route_match': (rte_score, rte_detail),
                'signal_direction': check_direction,
                'signal_regime': check_regime,
                'signal_route': check_route
            }
            
            # Collect all scores
            all_matches.append({
                'tf': check_tf,
                'direction': dir_score,
                'regime': reg_score,
                'route': rte_score
            })
        
        # Calculate overall alignment score
        alignment_score = self._calculate_alignment_score(all_matches)
        
        # Determine alignment band
        alignment_band = self._get_alignment_band(alignment_score)
        
        # Get label and icon
        band_info = ALIGNMENT_LABELS.get(alignment_band, ALIGNMENT_LABELS['weak'])
        
        # Calculate adjusted confidence
        multiplier = self.adjustments.get(alignment_band, 1.0)
        adjusted_confidence = round(base_confidence * multiplier, 2)
        
        # Generate warning if needed
        warning = None
        if alignment_band == 'conflict':
            warning = f"⛔ Counter-trend risk: {timeframe} {direction} conflicts with higher TF signals ({alignment_score}/100 - severe penalty: 0.60x confidence)"
        
        result = {
            'alignment_score': alignment_score,
            'alignment_band': alignment_band,
            'alignment_label': band_info['label'],
            'alignment_icon': band_info['icon'],
            'base_confidence': base_confidence,
            'adjusted_confidence': adjusted_confidence,
            'confidence_multiplier': multiplier,
            'mtf_checks': mtf_details,
            'warning': warning,
            'enabled': True
        }
        
        return result
    
    def _calculate_alignment_score(self, matches: List[Dict]) -> int:
        """
        Calculate overall alignment score from individual match scores
        Returns 0-100 where 100 is perfect alignment
        """
        if not matches:
            return 50  # No data = neutral
        
        # Apply weights to each dimension
        weighted_scores = []
        
        for match in matches:
            dir_weighted = match['direction'] * self.weights['direction']
            reg_weighted = match['regime'] * self.weights['regime']
            rte_weighted = match['route'] * self.weights['route']
            
            # Average across TFs
            match_score = (dir_weighted + reg_weighted + rte_weighted) * 100
            weighted_scores.append(match_score)
        
        # Overall is average of all TF checks
        overall = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 50
        
        return int(round(overall))
    
    def _get_alignment_band(self, alignment_score: int) -> str:
        """
        Determine which band a score falls into (v2 system).
        
        v2 bands (ONLY 4 values, no 'partial'):
        - strong: Full alignment (≥75)
        - weak: Partial alignment (50-74) - this is where OLD 'partial' maps to
        - conflict: Misalignment (<50 but approaching weak threshold)
        - neutral: No data / fallback
        """
        if alignment_score >= self.thresholds['strong']:
            return 'strong'
        elif alignment_score >= self.thresholds['weak']:
            # v2: 'weak' replaces old 'partial' (50-74 range)
            return 'weak'
        elif alignment_score > 0:
            # Below weak threshold but has some score = conflict (opposing TF signals)
            return 'conflict'
        else:
            # No data / neutral state
            return 'neutral'
    
    def store_result(self, signal_uuid: str, symbol: str, timeframe: str, 
                     base_confidence: float, mtf_result: Dict) -> bool:
        """Store MTF analysis result to JSONL file for historical tracking"""
        try:
            result_record = {
                'signal_uuid': signal_uuid,
                'symbol': symbol,
                'timeframe': timeframe,
                'base_confidence': base_confidence,
                'alignment_score': mtf_result.get('alignment_score'),
                'alignment_band': mtf_result.get('alignment_band'),
                'alignment_label': mtf_result.get('alignment_label'),
                'adjusted_confidence': mtf_result.get('adjusted_confidence'),
                'confidence_multiplier': mtf_result.get('confidence_multiplier'),
                'warning': mtf_result.get('warning'),
                'analyzed_at_utc': datetime.utcnow().isoformat(),
                'mtf_checks_summary': {k: v.get('status') for k, v in mtf_result.get('mtf_checks', {}).items()}
            }
            
            # Append to results file
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result_record) + '\n')
            
            return True
        except Exception as e:
            print(f"[MTF] Error storing result: {e}", flush=True)
            return False


# Global instance
analyzer = MTFAlignmentAnalyzer()


def analyze_mtf_alignment(signal_data: Dict) -> Dict:
    """Public function to analyze MTF alignment"""
    return analyzer.analyze(signal_data)


def store_mtf_result(signal_uuid: str, symbol: str, timeframe: str, 
                     base_confidence: float, mtf_result: Dict) -> bool:
    """Public function to store MTF result"""
    return analyzer.store_result(signal_uuid, symbol, timeframe, base_confidence, mtf_result)


if __name__ == '__main__':
    print("MTF Alignment Analyzer initialized")
    print(f"Feature enabled: {MTF_ALIGNMENT_ENABLED}")
