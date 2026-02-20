"""
Detection Module 2: Fan-in Aggregator Detection
Detects accounts receiving from 3+ senders within 72 hours.
Can also detect combined fan-in/fan-out mule hub pattern.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set
from datetime import timedelta


def detect_smurfing(
    tg,
    threshold_senders: int = 5,
    window_hours: int = 48,
    exclude_nodes: Set[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect smurfing (fan-in aggregation) patterns.
    5+ accounts sending to the SAME destination within 48 hours.
    """
    results = []
    window = timedelta(hours=window_hours)
    exclude = exclude_nodes or set()

    for node in tg.G.nodes():
        if node in exclude:
            continue

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
            unique_senders -= exclude

            if len(unique_senders) >= threshold_senders:
                amounts = [t["amount"] for t in window_txns]
                fan_in_windows.append({
                    "start": txn["timestamp"],
                    "end": window_end,
                    "sender_count": len(unique_senders),
                    "senders": list(unique_senders),
                    "total_amount": sum(amounts),
                    "avg_amount": np.mean(amounts),
                })

        if not fan_in_windows:
            continue

        # Take the most suspicious window
        best_window = max(fan_in_windows, key=lambda w: w["sender_count"])
        
        # --- NEW: Automation & Sequential Detection (Task 7 Improvement) ---
        # Detect robot-like cadence (e.g. 60 min intervals)
        window_txns = [t for t in in_txns if best_window["start"] <= t["timestamp"] <= best_window["end"]]
        intervals = []
        for i in range(1, len(window_txns)):
            delta = (window_txns[i]["timestamp"] - window_txns[i-1]["timestamp"]).total_seconds() / 60
            if delta > 0: intervals.append(delta)
            
        automation_score = 0.0
        if len(intervals) >= 3:
            cv = np.std(intervals) / np.mean(intervals)
            if cv < 0.01: automation_score = 1.0 # Perfect rhythmic cadence
            elif cv < 0.05: automation_score = 0.7
            
        # Detect sequential account IDs
        senders = sorted(best_window["senders"])
        sequential_count = 0
        for i in range(1, len(senders)):
            try:
                # Try to extract numbers from IDs like ACC_02000, ACC_02001
                n1 = int(''.join(filter(str.isdigit, senders[i-1])))
                n2 = int(''.join(filter(str.isdigit, senders[i])))
                if n2 == n1 + 1: sequential_count += 1
            except: pass
        
        sequential_ratio = sequential_count / max(1, len(senders)-1)
        if sequential_ratio > 0.8: automation_score = max(automation_score, 0.9)

        risk = _calculate_fan_in_risk(
            best_window["sender_count"],
            best_window["total_amount"],
            1.0, # Default cluster score for smurfing
            threshold_senders
        )
        
        # Boost risk for automated/sequential patterns to bypass MERCHANT suppression
        if automation_score > 0.5:
            risk = max(risk, 0.85)

        results.append({
            "account_id": node,
            "type": "fan_in_aggregation", # Use consistent naming
            "nodes": [node] + best_window["senders"],
            "sender_count": best_window["sender_count"],
            "automation_score": automation_score,
            "sequential_ratio": round(sequential_ratio, 2),
            "window_start": str(best_window["start"]),
            "window_end": str(best_window["end"]),
            "total_amount": round(best_window["total_amount"], 2),
            "risk_score": risk,
            "explanation": (
                f"Aggregated Fan-in: Account {node} received funds from {best_window['sender_count']} "
                f"senders within {window_hours}h. Automation confidence: {automation_score:.0%}. "
                f"Sequential ID ratio: {sequential_ratio:.0%}."
            )
        })

    return results


def _is_likely_merchant(tg, node: str) -> bool:
    """
    Heuristic to check if a node is likely a merchant (legitimate fan-in).
    Merchants have: high incoming diversity, stable over time, regular patterns.
    """
    temporal = tg.node_temporal.get(node, {})

    unique_cp = temporal.get("unique_counterparties", 0)
    total_tx = temporal.get("tx_count", 0)

    if total_tx == 0:
        return False

    diversity_ratio = unique_cp / total_tx

    in_ratio = temporal.get("in_count", 0) / max(total_tx, 1)
    amount_cv = temporal.get("std_amount", 0) / max(temporal.get("avg_amount", 1), 0.01)

    # Merchant pattern: high diversity, mostly receiving, moderate variance
    if diversity_ratio > 0.6 and in_ratio > 0.8 and amount_cv < 0.5:
        return True

    return False


def _amount_clustering_score(window: Dict) -> float:
    """Check if amounts cluster around similar values (sign of structuring)."""
    if window["amount_std"] == 0:
        return 1.0
    cv = window["amount_std"] / max(window["avg_amount"], 0.01)
    return max(0, 1 - cv)


def _calculate_fan_in_risk(sender_count, total_amount, cluster_score, threshold):
    """Calculate risk score for fan-in pattern."""
    count_score = min(1.0, sender_count / (threshold * 5))
    amount_score = min(1.0, total_amount / 500000)

    risk = count_score * 0.4 + cluster_score * 0.3 + amount_score * 0.3
    return round(min(1.0, risk), 4)
