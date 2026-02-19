"""
Suspicion Scoring Engine v2.0
Aggregates forensic signals into tiered risk scores (0-100).
Optimized for high-confidence detection and coordination analysis.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict

from ring_manager import RingManager, get_priority


# --- High Intensity Forensic Weights (Issue 2) ---
# Weights reflect the severity and theoretical confidence of each pattern
MODULE_WEIGHTS = {
    "circular_routing": 0.25,     # Aggressive weighting for cycles
    "fan_in_fan_out": 0.22,       # Mule hubs are high-confidence
    "shell_chain": 0.18,          # Layering is high-confidence
    "rapid_movement": 0.15,       # Velocity is a strong signal
    "structuring": 0.15,          # Smurfing detection
    "fan_in_aggregation": 0.12,   
    "fan_out_dispersal": 0.12,
    "transaction_burst": 0.10,
    "dormant_activation": 0.08,
    "amount_consistency_ring": 0.08,
    "diversity_shift": 0.06,
    "centrality_spike": 0.05,
    "community_suspicion": 0.04,
}


def compute_scores(
    tg,
    all_detections: Dict[str, List[Dict]],
    ring_manager: RingManager,
    legitimate_accounts: Set[str] = None,
    legitimacy_scores: Dict[str, float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Tiered scoring system for Financial Forensics.
    """
    legit = legitimate_accounts or set()
    legit_scores = legitimacy_scores or {}

    account_data = defaultdict(lambda: {
        "score_components": [],
        "triggered_patterns": [],
        "ring_ids": set(),
        "weighted_sum": 0.0,
        "max_risk": 0.0,
        "explanations": [],
    })

    # 1. Aggregate signals
    for module_type, detections in all_detections.items():
        weight = MODULE_WEIGHTS.get(module_type, 0.05)

        for detection in detections:
            risk = detection.get("risk_score", 0)
            
            # Map accounts
            nodes = set()
            if "account_id" in detection: nodes.add(str(detection["account_id"]))
            if "nodes" in detection: nodes.update(str(n) for n in detection["nodes"])

            for node in nodes:
                if node in legit: continue
                
                ad = account_data[node]
                ad["weighted_sum"] += weight * risk
                ad["max_risk"] = max(ad["max_risk"], risk)
                ad["triggered_patterns"].append(module_type)
                ad["explanations"].append(detection.get("explanation", ""))
                
                ring_id = ring_manager.get_account_ring(node)
                if ring_id: ad["ring_ids"].add(ring_id)

    # 2. Finalize account scores
    suspicious_accounts = []
    
    if account_data:
        # Global max for normalization
        global_max = max(ad["weighted_sum"] for ad in account_data.values()) or 1.0

        for account_id, ad in account_data.items():
            patterns = set(ad["triggered_patterns"])
            
            # --- Tiered Scoring Logic ---
            # Base: normalized weighted sum (0-40)
            base_score = (ad["weighted_sum"] / global_max) * 40
            
            # Pattern Multiplier: Exponential boost for multiple patterns
            # 1 pattern = 1.0x, 2 = 1.6x, 3 = 2.2x, 4+ = 3x
            multiplier = 1.0 + (min(3, len(patterns) - 1) * 0.6) if len(patterns) > 1 else 1.0
            
            # Coordination Boost: If node is in a validated ring
            ring_boost = 0
            if ad["ring_ids"]:
                ring_boost = 15
                # Extra boost for high-priority rings
                best_ring_id = list(ad["ring_ids"])[0] 
                best_ring = ring_manager.rings.get(best_ring_id)
                if best_ring and get_priority(best_ring["type"]) <= 2:
                    ring_boost += 10

            final_score = (base_score * multiplier) + ring_boost
            
            # --- Calibration & Penalties ---
            # Penalize broad-only
            if patterns <= {"community_suspicion", "centrality_spike", "diversity_shift"}:
                final_score *= 0.3
            
            # Penalize legitimacy
            l_score = legit_scores.get(account_id, 0)
            if l_score > 0.5:
                final_score *= (1.0 - l_score)

            # Behavioural FP Reduction
            final_score = _apply_fp_reduction(tg, account_id, final_score)
            
            final_score = round(min(100.0, max(0.0, final_score)), 2)

            if final_score >= 10.0:  # Extraction Threshold
                suspicious_accounts.append({
                    "account_id": account_id,
                    "risk_score": final_score,
                    "tier": _get_tier(final_score),
                    "patterns": list(patterns),
                    "ring_ids": list(ad["ring_ids"]),
                    "explanation": ad["explanations"][0] if ad["explanations"] else "Multiple forensic anomalies detected.",
                    "score_breakdown": [
                        {"module": p, "weighted": 1.0} for p in list(patterns)
                    ]
                })

    # Sort
    suspicious_accounts.sort(key=lambda x: x["risk_score"], reverse=True)

    # 3. Synchronize Rings
    fraud_rings = ring_manager.get_rings()
    for ring in fraud_rings:
        # Ring risk = mean of member scores + coordination premium
        member_scores = [
            a["risk_score"] for a in suspicious_accounts 
            if a["account_id"] in ring["nodes"]
        ]
        if member_scores:
            ring["risk_score"] = round(min(100, np.mean(member_scores) * 1.2), 2)
            ring["purity"] = round(len([s for s in member_scores if s >= 70]) / len(ring["nodes"]), 2)

    # 4. Summary & Metrics (Issue 8)
    total_analyzed = tg.node_count
    flagged_count = len(suspicious_accounts)
    
    # Precision Proxy = (High + Critical) / Total Flagged
    high_conf = [a for a in suspicious_accounts if a["risk_score"] >= 70]
    precision_proxy = round(len(high_conf) / flagged_count, 2) if flagged_count > 0 else 1.0

    summary = {
        "total_accounts_analyzed": total_analyzed,
        "total_transactions_analyzed": len(tg.df) if hasattr(tg, 'df') else 0,
        "suspicious_accounts_found": flagged_count,
        "fraud_rings_detected": len(fraud_rings),
        "estimated_precision": precision_proxy,
        "high_risk_accounts": len(high_conf), # Renamed for frontend sync
        "avg_risk_score": round(np.mean([a["risk_score"] for a in suspicious_accounts]), 2) if suspicious_accounts else 0,
        "total_suspicious_amount": round(sum(r.get("total_amount", 0) for r in fraud_rings), 2),
        "processing_time_seconds": 0, # Filled by main
    }

    return suspicious_accounts, fraud_rings, summary


def _get_tier(score: float) -> str:
    if score >= 85: return "CRITICAL"
    if score >= 70: return "HIGH"
    if score >= 40: return "MEDIUM"
    return "LOW"


def _apply_fp_reduction(tg, account_id: str, score: float) -> float:
    """Enhanced behavioural suppression (Issue 5)."""
    node = tg.G.nodes.get(account_id, {})
    txns = node.get("transactions", [])
    if not txns: return score

    # Factor: Stable long-term history
    temporal = tg.node_temporal.get(account_id, {})
    lifespan = 0
    if temporal.get("first_seen") and temporal.get("last_seen"):
        lifespan = (temporal["last_seen"] - temporal["first_seen"]).total_seconds() / 86400
    
    if lifespan > 180 and temporal.get("tx_count", 0) > 50:
        score *= 0.8
    
    # Factor: Salary-like consistency
    if temporal.get("std_amount", 0) / max(0.01, temporal.get("avg_amount", 1)) < 0.05:
        score *= 0.9

    return score

