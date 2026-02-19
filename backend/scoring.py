"""
Suspicion Scoring Engine.
Aggregates results from all detection modules into per-account scores.
Implements weighted scoring, normalization, and false positive control.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict


# Module weights (tunable)
MODULE_WEIGHTS = {
    "circular_routing": 0.15,
    "fan_in_aggregation": 0.12,
    "fan_out_dispersal": 0.12,
    "shell_chain": 0.10,
    "transaction_burst": 0.08,
    "rapid_movement": 0.10,
    "dormant_activation": 0.08,
    "structuring": 0.10,
    "amount_consistency_ring": 0.05,
    "diversity_shift": 0.04,
    "centrality_spike": 0.03,
    "community_suspicion": 0.03,
}


def compute_scores(
    tg,
    all_detections: Dict[str, List[Dict]],
    fp_whitelist: set = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-account suspicion scores from all detection results.
    
    Returns:
        suspicious_accounts: sorted list of accounts with scores
        fraud_rings: consolidated ring/group detections
        summary: overall analysis summary
    """
    
    account_data = defaultdict(lambda: {
        "score_components": [],
        "triggered_patterns": [],
        "ring_ids": set(),
        "total_weighted_risk": 0.0,
        "detection_count": 0,
        "explanations": [],
    })
    
    ring_data = {}  # ring_id -> ring info
    
    # Process each detection module's results
    for module_type, detections in all_detections.items():
        weight = MODULE_WEIGHTS.get(module_type, 0.05)
        
        for detection in detections:
            risk_score = detection.get("risk_score", 0)
            
            # Get accounts involved - be selective about which accounts get scored
            accounts = []
            if "account_id" in detection:
                accounts.append(detection["account_id"])
            if "nodes" in detection:
                # For rings/chains, only score nodes directly involved
                accounts.extend(detection["nodes"])
            # Don't add senders/receivers of fan-in/fan-out to avoid
            # scoring innocent counterparties
            
            accounts = list(set(str(a) for a in accounts))
            
            ring_id = detection.get("ring_id")
            
            for account in accounts:
                # Skip whitelisted accounts
                if fp_whitelist and account in fp_whitelist:
                    continue
                
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
            
            # Track rings
            if ring_id:
                if ring_id not in ring_data:
                    ring_data[ring_id] = {
                        "ring_id": ring_id,
                        "type": detection.get("type", "unknown"),
                        "nodes": [],
                        "risk_score": 0,
                        "patterns": set(),
                        "total_amount": 0,
                        "explanations": [],
                    }
                rd = ring_data[ring_id]
                if "nodes" in detection:
                    for n in detection["nodes"]:
                        if n not in rd["nodes"]:
                            rd["nodes"].append(n)
                rd["risk_score"] = max(rd["risk_score"], risk_score)
                rd["patterns"].add(module_type)
                rd["total_amount"] += detection.get("total_amount", 0)
                rd["explanations"].append(detection.get("explanation", ""))
    
    # --- Normalize scores to 0-100 ---
    suspicious_accounts = []
    
    if account_data:
        max_raw = max(ad["total_weighted_risk"] for ad in account_data.values()) or 1
        
        for account_id, ad in account_data.items():
            # Normalize
            raw_score = ad["total_weighted_risk"]
            unique_patterns = set(ad["triggered_patterns"])
            
            # Multi-pattern bonus: accounts flagged by multiple modules are more suspicious
            pattern_bonus = min(0.3, len(unique_patterns) * 0.05)
            
            # Normalize to 0-100
            normalized = (raw_score / max_raw) * 70 + pattern_bonus * 100
            normalized = min(100, max(0, normalized))
            
            # Penalize accounts only flagged by very broad detectors
            broad_only = unique_patterns <= {"community_suspicion", "centrality_spike"}
            if broad_only:
                normalized *= 0.3  # Heavy reduction for broad-only flags
            
            # Penalize single-pattern detections
            if len(unique_patterns) == 1:
                normalized *= 0.6  # 40% reduction for single-pattern
            
            # Apply false positive reduction
            normalized = _apply_fp_reduction(tg, account_id, normalized)
            
            if normalized >= 15:  # Min threshold (raised for precision)
                suspicious_accounts.append({
                    "account_id": account_id,
                    "risk_score": round(normalized, 2),
                    "triggered_patterns": list(unique_patterns),
                    "pattern_count": len(unique_patterns),
                    "detection_count": ad["detection_count"],
                    "ring_ids": list(ad["ring_ids"]),
                    "explanations": ad["explanations"][:5],
                    "score_breakdown": ad["score_components"][:10],
                })
    
    # Sort descending by score
    suspicious_accounts.sort(key=lambda x: x["risk_score"], reverse=True)
    
    # --- Build fraud rings ---
    fraud_rings = []
    suspicious_ids = {sa["account_id"] for sa in suspicious_accounts}
    
    for ring_id, rd in ring_data.items():
        # Compute ring-level score: average of member scores
        member_scores = [
            sa["risk_score"] for sa in suspicious_accounts
            if ring_id in sa.get("ring_ids", [])
        ]
        ring_score = np.mean(member_scores) if member_scores else rd["risk_score"] * 100
        
        # Only include rings with at least one suspicious member or high own risk
        has_suspicious = any(n in suspicious_ids for n in rd["nodes"])
        if not has_suspicious and rd["risk_score"] < 0.3:
            continue
        
        if ring_score < 10:
            continue
        
        fraud_rings.append({
            "ring_id": ring_id,
            "type": rd["type"],
            "nodes": rd["nodes"][:50],
            "node_count": len(rd["nodes"]),
            "risk_score": round(min(100, ring_score), 2),
            "patterns": list(rd["patterns"]),
            "total_amount": round(rd["total_amount"], 2),
            "explanations": rd["explanations"][:3],
        })
    
    fraud_rings.sort(key=lambda x: x["risk_score"], reverse=True)
    
    # --- Summary ---
    summary = {
        "total_accounts_analyzed": tg.node_count,
        "total_transactions_analyzed": len(tg.df),
        "suspicious_accounts_found": len(suspicious_accounts),
        "fraud_rings_detected": len(fraud_rings),
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
    
    # Long-term consistency check
    tx_count = temporal.get("tx_count", 0)
    first_seen = temporal.get("first_seen")
    last_seen = temporal.get("last_seen")
    
    if first_seen and last_seen:
        account_age_days = (last_seen - first_seen).days
        
        # Very old stable accounts get reduction
        if account_age_days > 180 and tx_count > 100:
            score *= 0.85  # 15% reduction
        
        # High regularity -> likely legitimate
        amount_cv = temporal.get("std_amount", 0) / max(temporal.get("avg_amount", 1), 0.01)
        if amount_cv < 0.1:  # Very regular amounts
            score *= 0.9  # 10% reduction
    
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
