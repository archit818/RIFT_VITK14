"""
Detection Module 3: Fan-out Dispersal Detection
Detects rapid dispersal to 10+ receivers from a single account.
"""

import numpy as np
from typing import List, Dict, Any
from datetime import timedelta


def detect_fan_out(tg, threshold_receivers: int = 10, window_hours: int = 24) -> List[Dict[str, Any]]:
    """
    Detect fan-out dispersal patterns.
    
    An account sending to many unique receivers rapidly
    may be dispersing aggregated illicit funds.
    """
    results = []
    window = timedelta(hours=window_hours)
    
    for node in tg.G.nodes():
        out_deg = tg.out_degree.get(node, 0)
        if out_deg < threshold_receivers:
            continue
        
        out_txns = [
            t for t in tg.G.nodes[node]["transactions"]
            if t["type"] == "out"
        ]
        
        if len(out_txns) < threshold_receivers:
            continue
        
        out_txns.sort(key=lambda x: x["timestamp"])
        
        # Sliding window
        fan_out_windows = []
        for i, txn in enumerate(out_txns):
            window_end = txn["timestamp"] + window
            window_txns = [t for t in out_txns[i:] if t["timestamp"] <= window_end]
            
            unique_receivers = set(t["counterparty"] for t in window_txns)
            if len(unique_receivers) >= threshold_receivers:
                amounts = [t["amount"] for t in window_txns]
                
                # Velocity analysis: transactions per hour
                time_span = (window_txns[-1]["timestamp"] - window_txns[0]["timestamp"]).total_seconds() / 3600
                velocity = len(window_txns) / max(time_span, 0.01)
                
                fan_out_windows.append({
                    "start": txn["timestamp"],
                    "end": window_end,
                    "receiver_count": len(unique_receivers),
                    "receivers": list(unique_receivers),
                    "total_amount": sum(amounts),
                    "avg_amount": np.mean(amounts),
                    "velocity": velocity,
                })
        
        if not fan_out_windows:
            continue
        
        # Skip if looks like payroll
        if _is_likely_payroll(tg, node, fan_out_windows):
            continue
        
        best_window = max(fan_out_windows, key=lambda w: w["receiver_count"])
        
        risk = _calculate_fan_out_risk(
            best_window["receiver_count"],
            best_window["total_amount"],
            best_window["velocity"],
            threshold_receivers
        )
        
        results.append({
            "account_id": node,
            "type": "fan_out_dispersal",
            "receiver_count": best_window["receiver_count"],
            "receivers": best_window["receivers"][:20],
            "window_start": str(best_window["start"]),
            "window_end": str(best_window["end"]),
            "total_amount": round(best_window["total_amount"], 2),
            "avg_amount": round(best_window["avg_amount"], 2),
            "velocity": round(best_window["velocity"], 2),
            "risk_score": risk,
            "explanation": (
                f"Account {node} dispersed funds to {best_window['receiver_count']} receivers "
                f"within {window_hours}h. Total: ${best_window['total_amount']:,.2f}. "
                f"Velocity: {best_window['velocity']:.1f} tx/hour."
            )
        })
    
    return results


def _is_likely_payroll(tg, node, windows):
    """Check if fan-out looks like payroll (regular timing, fixed amounts)."""
    temporal = tg.node_temporal.get(node, {})
    
    if not windows:
        return False
    
    # Payroll: low amount variance, mostly outgoing, specific day patterns
    amount_cv = temporal.get("std_amount", 0) / max(temporal.get("avg_amount", 1), 0.01)
    out_ratio = temporal.get("out_count", 0) / max(temporal.get("tx_count", 1), 1)
    
    if amount_cv < 0.1 and out_ratio > 0.9:
        return True
    
    return False


def _calculate_fan_out_risk(receiver_count, total_amount, velocity, threshold):
    """Calculate risk score for fan-out pattern."""
    count_score = min(1.0, receiver_count / (threshold * 3))
    amount_score = min(1.0, total_amount / 500000)
    velocity_score = min(1.0, velocity / 20)
    
    risk = count_score * 0.35 + amount_score * 0.3 + velocity_score * 0.35
    return round(min(1.0, risk), 4)
