"""
Detection Module 3: Fan-out Dispersal Detection
Detects rapid dispersal to 3+ receivers from a single account.
"""

import numpy as np
from typing import List, Dict, Any, Set
from datetime import timedelta


def detect_dispersal(
    tg,
    threshold_receivers: int = 5,
    window_hours: int = 48,
    exclude_nodes: Set[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect fan-out dispersal patterns.
    1 account sending to 5+ distinct accounts within 48 hours.
    """
    results = []
    window = timedelta(hours=window_hours)
    exclude = exclude_nodes or set()

    for node in tg.G.nodes():
        if node in exclude:
            continue

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
            unique_receivers -= exclude

            if len(unique_receivers) >= threshold_receivers:
                amounts = [t["amount"] for t in window_txns]

                fan_out_windows.append({
                    "start": txn["timestamp"],
                    "end": window_end,
                    "receiver_count": len(unique_receivers),
                    "receivers": list(unique_receivers),
                    "total_amount": sum(amounts),
                })

        if not fan_out_windows:
            continue

        best_window = max(fan_out_windows, key=lambda w: w["receiver_count"])

        risk = _calculate_fan_out_risk(
            best_window["receiver_count"],
            best_window["total_amount"],
            10.0, # default velocity
            threshold_receivers
        )

        results.append({
            "account_id": node,
            "type": "fan_in_fan_out",
            "nodes": [node] + best_window["receivers"],
            "receiver_count": best_window["receiver_count"],
            "window_start": str(best_window["start"]),
            "window_end": str(best_window["end"]),
            "total_amount": round(best_window["total_amount"], 2),
            "risk_score": risk,
            "explanation": (
                f"Dispersal detected: Account {node} sent funds to {best_window['receiver_count']} receivers within {window_hours}h."
            )
        })

    return results


def _is_likely_payroll(tg, node, windows):
    """Check if fan-out looks like payroll (regular timing, fixed amounts)."""
    temporal = tg.node_temporal.get(node, {})

    if not windows:
        return False

    amount_cv = temporal.get("std_amount", 0) / max(temporal.get("avg_amount", 1), 0.01)
    out_ratio = temporal.get("out_count", 0) / max(temporal.get("tx_count", 1), 1)

    if amount_cv < 0.1 and out_ratio > 0.9:
        return True

    return False


def _calculate_fan_out_risk(receiver_count, total_amount, velocity, threshold):
    """Calculate risk score for fan-out pattern."""
    count_score = min(1.0, receiver_count / (threshold * 5))
    amount_score = min(1.0, total_amount / 500000)
    velocity_score = min(1.0, velocity / 20)

    risk = count_score * 0.35 + amount_score * 0.3 + velocity_score * 0.35
    return round(min(1.0, risk), 4)
