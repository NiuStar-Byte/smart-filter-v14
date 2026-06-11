"""Safe wrapper for MTF alignment analysis with error handling"""

from mtf_alignment_analyzer import analyze_mtf_alignment, store_mtf_result

def safe_analyze_mtf(signal_data: dict, confidence: float) -> dict:
    """
    Safely analyze MTF alignment with fallback defaults
    
    Args:
        signal_data: Signal data dict
        confidence: Base confidence level
        
    Returns:
        Dict with MTF analysis results (never None, always has required fields)
    """
    try:
        result = analyze_mtf_alignment(signal_data)
        
        # Validate result is a dict
        if not isinstance(result, dict):
            print(f"[MTF-WRAPPER] Unexpected result type: {type(result)}", flush=True)
            raise ValueError(f"analyze_mtf_alignment returned {type(result)}, expected dict")
        
        # Validate required fields
        required = ['alignment_score', 'alignment_band', 'adjusted_confidence', 'alignment_label', 'alignment_icon', 'warning']
        for field in required:
            if field not in result:
                print(f"[MTF-WRAPPER] Missing field: {field}", flush=True)
                raise ValueError(f"Missing field: {field}")
        
        return result
        
    except Exception as e:
        print(f"[MTF-WRAPPER] Error in MTF analysis: {e}", flush=True)
        
        # Return safe defaults
        return {
            'alignment_score': 50,
            'alignment_band': 'partial',
            'alignment_label': 'Analysis Failed',
            'alignment_icon': '⚠️',
            'base_confidence': confidence,
            'adjusted_confidence': confidence,
            'confidence_multiplier': 1.0,
            'mtf_checks': {},
            'warning': f'MTF analysis error: {str(e)[:50]}',
            'enabled': False
        }

def safe_store_mtf(signal_uuid: str, symbol: str, timeframe: str, 
                    confidence: float, mtf_result: dict) -> bool:
    """Safely store MTF result with error handling"""
    try:
        if not isinstance(mtf_result, dict):
            print(f"[MTF-WRAPPER] Cannot store non-dict result: {type(mtf_result)}", flush=True)
            return False
        
        return store_mtf_result(signal_uuid, symbol, timeframe, confidence, mtf_result)
    except Exception as e:
        print(f"[MTF-WRAPPER] Error storing MTF result: {e}", flush=True)
        return False
