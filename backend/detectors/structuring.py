"""
Detection Module 8: Transaction Structuring (Threshold Avoidance)
Detects repeated near-threshold amounts (smurfing/structuring).
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


# Common regulatory thresholds
REGULATORY_THRESHOLDS = [10000, 5000, 3000, 15000, 50000]
THRESHOLD_MARGIN = 0.15  # 15% below threshold


def detect_structuring(tg, min_count: int = 3) -> List[Dict[str, Any]]:
    """
    Detect transaction structuring / threshold avoidance.
    
    Pattern: Multiple transactions just below reporting thresholds.
    Also checks for round-number avoidance.
    """
    results = []
    
    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        amounts = temporal.get("amounts", [])
        
        if len(amounts) < min_count:
            continue
        
        structured_patterns = []
        
        for threshold in REGULATORY_THRESHOLDS:
            lower = threshold * (1 - THRESHOLD_MARGIN)
            upper = threshold * 0.999  # Just below
            
            near_threshold = [a for a in amounts if lower <= a <= upper]
            
            if len(near_threshold) >= min_count:
                # Check if these cluster tightly
                if near_threshold:
                    mean_val = np.mean(near_threshold)
                    std_val = np.std(near_threshold) if len(near_threshold) > 1 else 0
                    cv = std_val / max(mean_val, 0.01)
                    
                    structured_patterns.append({
                        "threshold": threshold,
                        "count": len(near_threshold),
                        "avg_amount": round(mean_val, 2),
                        "clustering": round(max(0, 1 - cv), 4),
                        "amounts": sorted(near_threshold)[:10],
                    })
        
        if not structured_patterns:
            continue
        
        # Also check for repeated identical amounts
        amount_counts = Counter([round(a, 2) for a in amounts])
        repeated = {amt: cnt for amt, cnt in amount_counts.items() if cnt >= min_count}
        
        best_pattern = max(structured_patterns, key=lambda p: p["count"])
        
        risk = _calculate_structuring_risk(
            best_pattern["count"],
            best_pattern["clustering"],
            len(repeated),
            min_count
        )
        
        results.append({
            "account_id": node,
            "type": "structuring",
            "threshold": best_pattern["threshold"],
            "near_threshold_count": best_pattern["count"],
            "avg_amount": best_pattern["avg_amount"],
            "amount_clustering": best_pattern["clustering"],
            "repeated_amounts": len(repeated),
            "patterns": structured_patterns,
            "risk_score": risk,
            "explanation": (
                f"Account {node} has {best_pattern['count']} transactions near "
                f"${best_pattern['threshold']:,} threshold (avg ${best_pattern['avg_amount']:,.2f}). "
                f"Clustering: {best_pattern['clustering']:.0%}."
            )
        })
    
    return results


def _calculate_structuring_risk(count, clustering, repeated_count, min_count):
    count_score = min(1.0, count / (min_count * 5))
    cluster_score = clustering
    repeat_score = min(1.0, repeated_count / 5)
    return round(count_score * 0.4 + cluster_score * 0.35 + repeat_score * 0.25, 4)
