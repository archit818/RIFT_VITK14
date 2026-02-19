"""
Suspicion Scoring Engine.
Aggregates results from all detection modules into per-account scores.
Uses RingManager for dedup/priority, legitimacy scores for FP control.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict

from ring_manager import RingManager, get_priority


# Module weights (tunable)
MODULE_WEIGHTS = {
    "circular_routing": 0.18,
    "fan_in_fan_out": 0.15,
    "fan_in_aggregation": 0.10,
    "fan_out_dispersal": 0.10,
    "shell_chain": 0.10,
    "transaction_burst": 0.08,
    "rapid_movement": 0.08,
    "dormant_activation": 0.06,
    "structuring": 0.08,
    "amount_consistency_ring": 0.05,
    "diversity_shift": 0.04,
    "centrality_spike": 0.03,
    "community_suspicion": 0.02,
}


def compute_scores(
    tg,
    all_detections: Dict[str, List[Dict]],
    ring_manager: RingManager,
    legitimate_accounts: Set[str] = None,
    legitimacy_scores: Dict[str, float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-account suspicion scores from all detection results.

    Args:
        tg: TransactionGraph
        all_detections: dict of module_name -> list of detections
        ring_manager: RingManager with rings already populated
        legitimate_accounts: set of account IDs marked as legitimate
        legitimacy_scores: dict of account_id -> legitimacy score (0-1)

    Returns:
        suspicious_accounts: sorted list of accounts with scores
        fraud_rings: consolidated ring/group detections from ring_manager
        summary: overall analysis summary
    """
    legit = legitimate_accounts or set()
    legit_scores = legitimacy_scores or {}

    account_data = defaultdict(lambda: {
        "score_components": [],
        "triggered_patterns": [],
        "ring_ids": set(),
        "total_weighted_risk": 0.0,
        "detection_count": 0,
        "explanations": [],
    })

    # Process each detection module's results
    for module_type, detections in all_detections.items():
        weight = MODULE_WEIGHTS.get(module_type, 0.03)

        for detection in detections:
            risk_score = detection.get("risk_score", 0)

            # Get accounts involved
            accounts = []
            if "account_id" in detection:
                accounts.append(detection["account_id"])
            if "nodes" in detection:
                accounts.extend(detection["nodes"])

            accounts = list(set(str(a) for a in accounts))

            for account in accounts:
                # Skip legitimate accounts entirely
                if account in legit:
                    continue

                ring_id = ring_manager.get_account_ring(account)

                ad = account_data[account]
                ad["score_components"].append({
                    "module": module_type,
                    "weight": weight,
                    "risk": risk_score,
                    "weighted": weight * risk_score,
                })
                ad["total_weighted_risk"] += weight * risk_score
                ad["detection_count"] += 1
                ad["triggered_patterns"].append(module_type)
                ad["explanations"].append(detection.get("explanation", ""))

                if ring_id:
                    ad["ring_ids"].add(ring_id)

    # --- Normalize scores to 0-100 ---
    suspicious_accounts = []

    if account_data:
        max_raw = max(ad["total_weighted_risk"] for ad in account_data.values()) or 1

        for account_id, ad in account_data.items():
            raw_score = ad["total_weighted_risk"]
            unique_patterns = set(ad["triggered_patterns"])

            # --- Score calculation ---

            # 1. Base: normalize raw weighted risk
            base = (raw_score / max_raw) * 60

            # 2. Multi-pattern bonus (strong signal)
            pattern_bonus = min(25, len(unique_patterns) * 5)

            # 3. Ring involvement bonus
            ring_bonus = 0
            for rid in ad["ring_ids"]:
                ring_info = ring_manager.rings.get(rid)
                if ring_info:
                    ring_type = ring_info.get("type", "")
                    priority = get_priority(ring_type)
                    # Higher priority (lower number) = more bonus
                    ring_bonus = max(ring_bonus, {1: 15, 2: 12, 3: 8, 4: 3}.get(priority, 2))

            normalized = base + pattern_bonus + ring_bonus
            normalized = min(100, max(0, normalized))

            # 4. Penalize accounts only flagged by very broad detectors
            broad_only = unique_patterns <= {"community_suspicion", "centrality_spike", "diversity_shift"}
            if broad_only:
                normalized *= 0.2

            # 5. Penalize single-pattern detections
            if len(unique_patterns) == 1:
                normalized *= 0.5

            # 6. Apply legitimacy-based reduction
            account_legit = legit_scores.get(account_id, 0)
            if account_legit > 0.5:
                normalized *= max(0.2, 1.0 - account_legit)

            # 7. Apply behavioral FP reduction
            normalized = _apply_fp_reduction(tg, account_id, normalized)

            if normalized >= 15:
                suspicious_accounts.append({
                    "account_id": account_id,
                    "risk_score": round(normalized, 2),
                    "triggered_patterns": list(unique_patterns),
                    "pattern_count": len(unique_patterns),
                    "detection_count": ad["detection_count"],
                    "ring_ids": list(ad["ring_ids"]),
                    "explanations": ad["explanations"][:5],
                    "score_breakdown": ad["score_components"][:10],
                    "legitimacy_score": round(legit_scores.get(account_id, 0), 4),
                })

    # Sort descending by score
    suspicious_accounts.sort(key=lambda x: x["risk_score"], reverse=True)

    # --- Get rings from RingManager ---
    fraud_rings = ring_manager.get_rings()

    # Update ring scores with member account scores
    for ring in fraud_rings:
        member_scores = [
            sa["risk_score"] for sa in suspicious_accounts
            if ring["ring_id"] in sa.get("ring_ids", [])
        ]
        if member_scores:
            ring["risk_score"] = round(min(100, np.mean(member_scores) + 10), 2)

    # Re-sort
    fraud_rings.sort(key=lambda x: x["risk_score"], reverse=True)

    # --- Summary ---
    summary = {
        "total_accounts_analyzed": tg.node_count,
        "total_transactions_analyzed": len(tg.df),
        "suspicious_accounts_found": len(suspicious_accounts),
        "fraud_rings_detected": len(fraud_rings),
        "legitimate_accounts_excluded": len(legit),
        "high_risk_accounts": len([a for a in suspicious_accounts if a["risk_score"] >= 70]),
        "medium_risk_accounts": len([a for a in suspicious_accounts if 40 <= a["risk_score"] < 70]),
        "low_risk_accounts": len([a for a in suspicious_accounts if a["risk_score"] < 40]),
        "detection_modules_triggered": list(set(
            p for a in suspicious_accounts for p in a["triggered_patterns"]
        )),
        "top_patterns": _get_top_patterns(all_detections),
        "total_suspicious_amount": round(
            sum(r.get("total_amount", 0) for r in fraud_rings), 2
        ),
    }

    return suspicious_accounts, fraud_rings, summary


def _apply_fp_reduction(tg, account_id: str, score: float) -> float:
    """
    Reduce false positives using behavior stability analysis.
    Stable, long-term accounts get score reductions.
    """
    temporal = tg.node_temporal.get(account_id, {})

    tx_count = temporal.get("tx_count", 0)
    first_seen = temporal.get("first_seen")
    last_seen = temporal.get("last_seen")

    if first_seen and last_seen:
        account_age_days = (last_seen - first_seen).days

        # Very old stable accounts get reduction
        if account_age_days > 180 and tx_count > 100:
            score *= 0.85

        # High regularity -> likely legitimate
        amount_cv = temporal.get("std_amount", 0) / max(temporal.get("avg_amount", 1), 0.01)
        if amount_cv < 0.1:
            score *= 0.9

    return score


def _get_top_patterns(all_detections: Dict[str, List]) -> List[Dict[str, Any]]:
    """Get stats on which patterns fired most."""
    pattern_stats = []
    for module_type, detections in all_detections.items():
        if detections:
            pattern_stats.append({
                "pattern": module_type,
                "count": len(detections),
                "avg_risk": round(np.mean([d.get("risk_score", 0) for d in detections]), 4),
            })
    pattern_stats.sort(key=lambda x: x["count"], reverse=True)
    return pattern_stats
