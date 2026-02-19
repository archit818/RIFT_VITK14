"""
Detection Module 7: Dormant Account Activation
Detects sudden activation after prolonged inactivity.
"""

from typing import List, Dict, Any
from datetime import timedelta
import numpy as np


def detect_dormant_activation(tg, dormancy_days: int = 30, burst_threshold: int = 5) -> List[Dict[str, Any]]:
    """
    Detect dormant accounts that suddenly become active.
    
    Pattern: Long silence -> sudden burst of activity.
    This is common when mule accounts are activated.
    """
    results = []
    dormancy_period = timedelta(days=dormancy_days)
    
    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        timestamps = temporal.get("timestamps", [])
        
        if len(timestamps) < burst_threshold:
            continue
        
        ts_sorted = sorted(timestamps)
        
        # Find gaps in activity
        gaps = []
        for i in range(1, len(ts_sorted)):
            gap = ts_sorted[i] - ts_sorted[i - 1]
            if gap >= dormancy_period:
                # Count activity after the gap
                post_gap = [t for t in ts_sorted[i:] if t <= ts_sorted[i] + timedelta(days=7)]
                gaps.append({
                    "gap_start": ts_sorted[i - 1],
                    "gap_end": ts_sorted[i],
                    "gap_days": gap.days,
                    "post_gap_activity": len(post_gap),
                })
        
        if not gaps:
            continue
        
        # Find the most suspicious gap
        for gap in gaps:
            if gap["post_gap_activity"] >= burst_threshold:
                risk = _calculate_dormant_risk(
                    gap["gap_days"], gap["post_gap_activity"], dormancy_days, burst_threshold
                )
                
                results.append({
                    "account_id": node,
                    "type": "dormant_activation",
                    "dormancy_days": gap["gap_days"],
                    "gap_start": str(gap["gap_start"]),
                    "gap_end": str(gap["gap_end"]),
                    "post_activation_txns": gap["post_gap_activity"],
                    "risk_score": risk,
                    "explanation": (
                        f"Account {node} was dormant for {gap['gap_days']} days then had "
                        f"{gap['post_gap_activity']} transactions within 7 days."
                    )
                })
                break  # One flag per account
    
    return results


def _calculate_dormant_risk(gap_days, post_activity, dormancy_threshold, burst_threshold):
    dormancy_score = min(1.0, gap_days / (dormancy_threshold * 6))
    burst_score = min(1.0, post_activity / (burst_threshold * 5))
    return round(dormancy_score * 0.5 + burst_score * 0.5, 4)
