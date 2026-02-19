"""
Detection Module 5: Transaction Burst Detection
Detects sudden spikes in account activity using rolling windows.
"""

import numpy as np
from typing import List, Dict, Any
from datetime import timedelta


def detect_transaction_bursts(tg, window_hours: int = 6, spike_multiplier: float = 3.0) -> List[Dict[str, Any]]:
    """
    Detect accounts with sudden bursts of transaction activity.
    
    Uses rolling window comparison: if activity in short window
    exceeds long-term average by spike_multiplier, flag as burst.
    """
    results = []
    
    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        tx_count = temporal.get("tx_count", 0)
        
        if tx_count < 5:
            continue
        
        timestamps = temporal.get("timestamps", [])
        if not timestamps:
            continue
        
        timestamps_sorted = sorted(timestamps)
        
        # Calculate overall rate
        total_span = (timestamps_sorted[-1] - timestamps_sorted[0]).total_seconds() / 3600
        if total_span < 1:
            continue
        
        overall_rate = tx_count / total_span  # tx per hour
        
        # Sliding window burst detection
        window = timedelta(hours=window_hours)
        max_burst_rate = 0
        burst_window = None
        
        for i, ts in enumerate(timestamps_sorted):
            end = ts + window
            window_count = sum(1 for t in timestamps_sorted[i:] if t <= end)
            rate = window_count / window_hours
            
            if rate > max_burst_rate:
                max_burst_rate = rate
                burst_window = {
                    "start": ts,
                    "end": end,
                    "count": window_count,
                    "rate": rate,
                }
        
        if not burst_window or max_burst_rate <= overall_rate * spike_multiplier:
            continue
        
        spike_ratio = max_burst_rate / max(overall_rate, 0.001)
        
        risk = _calculate_burst_risk(spike_ratio, burst_window["count"], spike_multiplier)
        
        results.append({
            "account_id": node,
            "type": "transaction_burst",
            "burst_count": burst_window["count"],
            "burst_rate": round(max_burst_rate, 2),
            "baseline_rate": round(overall_rate, 4),
            "spike_ratio": round(spike_ratio, 2),
            "window_start": str(burst_window["start"]),
            "window_end": str(burst_window["end"]),
            "risk_score": risk,
            "explanation": (
                f"Account {node} had {burst_window['count']} transactions in {window_hours}h "
                f"({spike_ratio:.1f}x the baseline rate of {overall_rate:.2f}/h)."
            )
        })
    
    return results


def _calculate_burst_risk(spike_ratio, count, multiplier):
    ratio_score = min(1.0, spike_ratio / (multiplier * 5))
    count_score = min(1.0, count / 50)
    return round(ratio_score * 0.6 + count_score * 0.4, 4)
