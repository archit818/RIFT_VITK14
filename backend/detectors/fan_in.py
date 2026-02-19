"""
Detection Module 2: Fan-in Aggregator Detection
Detects accounts receiving from 10+ senders within 72 hours.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import timedelta


def detect_fan_in(tg, threshold_senders: int = 10, window_hours: int = 72) -> List[Dict[str, Any]]:
    """
    Detect fan-in aggregation patterns.
    
    An account receiving from many unique senders in a short window
    may be aggregating illicit funds before moving them.
    
    Enhancements:
    - Amount clustering analysis
    - Merchant exclusion (high diversity + stable behavior)
    """
    results = []
    window = timedelta(hours=window_hours)
    
    for node in tg.G.nodes():
        in_deg = tg.in_degree.get(node, 0)
        if in_deg < threshold_senders:
            continue
        
        # Get all incoming transactions
        in_txns = [
            t for t in tg.G.nodes[node]["transactions"]
            if t["type"] == "in"
        ]
        
        if len(in_txns) < threshold_senders:
            continue
        
        # Sort by timestamp
        in_txns.sort(key=lambda x: x["timestamp"])
        
        # Sliding window analysis
        fan_in_windows = []
        for i, txn in enumerate(in_txns):
            window_end = txn["timestamp"] + window
            window_txns = [
                t for t in in_txns[i:]
                if t["timestamp"] <= window_end
            ]
            
            unique_senders = set(t["counterparty"] for t in window_txns)
            if len(unique_senders) >= threshold_senders:
                amounts = [t["amount"] for t in window_txns]
                fan_in_windows.append({
                    "start": txn["timestamp"],
                    "end": window_end,
                    "sender_count": len(unique_senders),
                    "senders": list(unique_senders),
                    "total_amount": sum(amounts),
                    "avg_amount": np.mean(amounts),
                    "amount_std": np.std(amounts) if len(amounts) > 1 else 0,
                })
        
        if not fan_in_windows:
            continue
        
        # Merchant check: skip if behavior is stable over long term
        if _is_likely_merchant(tg, node):
            continue
        
        # Take the most suspicious window
        best_window = max(fan_in_windows, key=lambda w: w["sender_count"])
        
        # Amount clustering
        cluster_score = _amount_clustering_score(best_window)
        
        risk = _calculate_fan_in_risk(
            best_window["sender_count"],
            best_window["total_amount"],
            cluster_score,
            threshold_senders
        )
        
        results.append({
            "account_id": node,
            "type": "fan_in_aggregation",
            "sender_count": best_window["sender_count"],
            "senders": best_window["senders"][:20],  # Limit for output
            "window_start": str(best_window["start"]),
            "window_end": str(best_window["end"]),
            "total_amount": round(best_window["total_amount"], 2),
            "avg_amount": round(best_window["avg_amount"], 2),
            "amount_clustering": round(cluster_score, 4),
            "risk_score": risk,
            "explanation": (
                f"Account {node} received funds from {best_window['sender_count']} unique senders "
                f"within {window_hours}h window. Total: ${best_window['total_amount']:,.2f}. "
                f"Amount clustering: {cluster_score:.2%}."
            )
        })
    
    return results


def _is_likely_merchant(tg, node: str) -> bool:
    """
    Heuristic to check if a node is likely a merchant (legitimate fan-in).
    Merchants have: high incoming diversity, stable over time, regular patterns.
    """
    temporal = tg.node_temporal.get(node, {})
    
    # High counterparty diversity
    unique_cp = temporal.get("unique_counterparties", 0)
    total_tx = temporal.get("tx_count", 0)
    
    if total_tx == 0:
        return False
    
    diversity_ratio = unique_cp / total_tx
    
    # If very high diversity ratio + mostly incoming + low amount variance
    in_ratio = temporal.get("in_count", 0) / max(total_tx, 1)
    amount_cv = temporal.get("std_amount", 0) / max(temporal.get("avg_amount", 1), 0.01)
    
    # Merchant pattern: high diversity, mostly receiving, moderate variance
    if diversity_ratio > 0.6 and in_ratio > 0.8 and amount_cv < 0.5:
        return True
    
    return False


def _amount_clustering_score(window: Dict) -> float:
    """Check if amounts cluster around similar values (sign of structuring)."""
    if window["amount_std"] == 0:
        return 1.0  # All same amount = highly suspicious
    
    cv = window["amount_std"] / max(window["avg_amount"], 0.01)
    return max(0, 1 - cv)


def _calculate_fan_in_risk(sender_count, total_amount, cluster_score, threshold):
    """Calculate risk score for fan-in pattern."""
    count_score = min(1.0, sender_count / (threshold * 3))
    amount_score = min(1.0, total_amount / 500000)
    
    risk = count_score * 0.4 + cluster_score * 0.3 + amount_score * 0.3
    return round(min(1.0, risk), 4)
