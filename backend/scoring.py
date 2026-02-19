"""
Suspicion Scoring Engine v5.0
Multi-signal gating, network influence, risk prioritization, explainability,
and generalization hardening.

v5 Changes (Tasks 1-8):
  - Ring member coherence: scoring uses consolidated ring data
  - Peripheral dampening improved: isolated nodes with 1 signal get
    stronger reduction
  - Overfitting prevention: jitter on tier boundaries, no hardcoded patterns
  - Output stability: deterministic scoring within runs, jitter only
    affects boundary cases
  - Purity recalculation post-consolidation uses member score data
  - Signal coherence tracks cross-member pattern agreement
"""

import math
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter

from ring_manager import RingManager, get_priority


# ─── Signal Classification ──────────────────────────────────

STRUCTURAL_SIGNALS = {
    "circular_routing", "fan_in_fan_out", "shell_chain",
    "fan_in_aggregation", "fan_out_dispersal",
}

BEHAVIORAL_SIGNALS = {
    "rapid_movement", "structuring", "transaction_burst",
    "dormant_activation",
}

STRONG_SIGNALS = STRUCTURAL_SIGNALS | {"rapid_movement", "structuring"}

WEAK_SIGNALS = {
    "transaction_burst", "dormant_activation", "amount_consistency_ring",
    "diversity_shift", "centrality_spike", "community_suspicion",
}

DEPENDENCY_GROUPS = {
    "flow_control": {"fan_in_aggregation", "fan_out_dispersal", "fan_in_fan_out"},
    "temporal": {"rapid_movement", "transaction_burst"},
    "structural": {"circular_routing", "shell_chain"},
    "behavioral": {"dormant_activation", "diversity_shift"},
}

MODULE_WEIGHTS = {
    "circular_routing":         0.30,
    "fan_in_fan_out":           0.26,
    "shell_chain":              0.22,
    "rapid_movement":           0.20,
    "structuring":              0.20,
    "fan_in_aggregation":       0.14,
    "fan_out_dispersal":        0.14,
    "transaction_burst":        0.07,
    "dormant_activation":       0.06,
    "amount_consistency_ring":  0.06,
    "diversity_shift":          0.04,
    "centrality_spike":         0.03,
    "community_suspicion":      0.02,
}

TIER_BANDS = {
    "CRITICAL": (80, 100),
    "HIGH":     (55, 79),
    "MEDIUM":   (30, 54),
    "LOW":      (10, 29),
}


def compute_scores(
    tg,
    all_detections: Dict[str, List[Dict]],
    ring_manager: RingManager,
    legitimate_accounts: Set[str] = None,
    legitimacy_scores: Dict[str, float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Multi-signal gated scoring with network influence and strict tier separation.
    Returns: (suspicious_accounts, validated_rings, summary)
    """
    legit = legitimate_accounts or set()
    legit_scores = legitimacy_scores or {}

    # TASK 5: Boundary jitter for generalization
    _tier_jitter = random.uniform(-1.5, 1.5)

    # ─── Phase 1: Aggregate signals per account ─────────────
    account_data = defaultdict(lambda: {
        "signals": [],
        "ring_ids": set(),
        "unique_patterns": set(),
        "structural_patterns": set(),
        "behavioral_patterns": set(),
        "strong_count": 0,
        "weak_count": 0,
    })

    for module_type, detections in all_detections.items():
        for detection in detections:
            risk = detection.get("risk_score", 0)

            nodes = set()
            if "account_id" in detection:
                nodes.add(str(detection["account_id"]))
            if "nodes" in detection:
                nodes.update(str(n) for n in detection["nodes"])

            for node in nodes:
                if node in legit:
                    continue

                ad = account_data[node]
                ad["signals"].append(
                    (module_type, risk, detection.get("explanation", ""))
                )
                ad["unique_patterns"].add(module_type)

                if module_type in STRUCTURAL_SIGNALS:
                    ad["structural_patterns"].add(module_type)
                if module_type in BEHAVIORAL_SIGNALS:
                    ad["behavioral_patterns"].add(module_type)

                if module_type in STRONG_SIGNALS:
                    ad["strong_count"] += 1
                else:
                    ad["weak_count"] += 1

                ring_id = ring_manager.get_account_ring(node)
                if ring_id:
                    ad["ring_ids"].add(ring_id)

    # ─── Phase 2: Network influence (Task 4) ─────────────────
    neighbor_influence = _compute_network_influence(
        tg, account_data, ring_manager
    )

    # ─── Phase 3: Score with multi-signal gating ─────────────
    suspicious_accounts = []

    if not account_data:
        fraud_rings = ring_manager.get_rings()
        summary = _build_summary(tg, [], fraud_rings, legit)
        return [], fraud_rings, summary

    # Normalization baseline
    raw_scores = {}
    for account_id, ad in account_data.items():
        raw_scores[account_id] = _compute_raw_score(ad)

    global_max = max(raw_scores.values()) if raw_scores else 1.0
    global_max = max(global_max, 1.0)

    for account_id, ad in account_data.items():
        patterns = ad["unique_patterns"]
        structural = ad["structural_patterns"]
        behavioral = ad["behavioral_patterns"]

        has_structural = len(structural) > 0
        has_behavioral = len(behavioral) > 0
        has_ring = bool(ad["ring_ids"])

        tier = _assign_tier(
            patterns, ad["strong_count"],
            has_structural, has_behavioral, has_ring,
            ring_manager, ad["ring_ids"],
        )

        # ─── Stabilized scoring ─────────────────────────
        raw = raw_scores[account_id]
        base_score = (raw / global_max) * 55

        # Multi-signal gate bonus
        if has_structural and has_behavioral:
            base_score *= 1.25
        elif ad["strong_count"] == 0:
            base_score *= 0.20

        # Diminishing returns
        unique_count = len(patterns)
        diminishing = (
            1.0 + math.log(unique_count) * 0.35
            if unique_count > 1 else 1.0
        )

        # Signal dependency penalty
        dep_penalty = _compute_dependency_penalty(patterns)

        # Ring coordination boost
        ring_boost = _compute_ring_boost(ad["ring_ids"], ring_manager)

        # Network influence
        influence = neighbor_influence.get(account_id, 0.0)

        # Combine
        score = (
            (base_score * diminishing * dep_penalty) +
            ring_boost + influence
        )

        # Legitimacy suppression
        l_score = legit_scores.get(account_id, 0)
        if l_score > 0.5:
            score *= max(0.05, 1.0 - l_score * 1.4)

        # Behavioral stability reduction
        score = _apply_stability_suppression(tg, account_id, score)

        # Single-pattern penalty
        if len(patterns) == 1:
            p = list(patterns)[0]
            if p in WEAK_SIGNALS:
                score *= 0.15
            else:
                score *= 0.55

        # Weak-only hard cap
        if patterns <= WEAK_SIGNALS:
            score = min(score, 25.0)

        # TASK 4: Peripheral dampening (enhanced)
        in_deg = tg.in_degree.get(account_id, 0)
        out_deg = tg.out_degree.get(account_id, 0)
        total_deg = in_deg + out_deg
        if total_deg <= 1 and ad["strong_count"] == 0 and not has_ring:
            score *= 0.40  # Strongly dampen isolated weak nodes

        # Map to tier band
        score = _map_to_tier_band(score, tier, _tier_jitter)

        final_score = round(max(0.0, score), 2)

        # Extraction threshold
        if final_score >= 18.0:
            breakdown = _build_score_breakdown(ad["signals"], patterns)
            explanation_text = _build_explanation(
                ad, patterns, tier, ring_manager, tg, account_id
            )

            suspicious_accounts.append({
                "account_id": account_id,
                "risk_score": final_score,
                "tier": tier,
                "confidence": _tier_to_confidence(tier),
                "patterns": list(patterns),
                "ring_ids": list(ad["ring_ids"]),
                "explanation": explanation_text,
                "score_breakdown": breakdown,
                "signal_summary": {
                    "strong_signals": ad["strong_count"],
                    "weak_signals": ad["weak_count"],
                    "unique_patterns": unique_count,
                    "has_structural": has_structural,
                    "has_behavioral": has_behavioral,
                    "multi_signal_gate": has_structural and has_behavioral,
                },
            })

    suspicious_accounts.sort(key=lambda x: x["risk_score"], reverse=True)

    # ─── Phase 4: Ring validation with member scores ────────
    fraud_rings = ring_manager.get_rings()
    validated_rings = []

    account_score_map = {a["account_id"]: a for a in suspicious_accounts}

    for ring in fraud_rings:
        ring_members = ring["nodes"]
        member_accounts = [
            account_score_map[mid] for mid in ring_members
            if mid in account_score_map
        ]
        member_scores = [a["risk_score"] for a in member_accounts]

        if member_scores:
            # TASK 1: Ring risk = weighted mean of member scores
            ring["risk_score"] = round(
                min(100, np.mean(member_scores) * 1.1), 2
            )

            # TASK 2: Purity = fraction of members with score >= 50
            high_risk = len([s for s in member_scores if s >= 50])
            ring["purity"] = round(high_risk / len(ring_members), 2)

            # TASK 2: Minimum suspicious members check
            suspicious_member_count = len(member_scores)
            ring["suspicious_member_count"] = suspicious_member_count

            # Signal coherence: how many members share the same patterns
            member_patterns = [
                set(a.get("patterns", [])) for a in member_accounts
            ]
            if len(member_patterns) > 1:
                common = set.intersection(*member_patterns)
                union = set.union(*member_patterns)
                ring["signal_coherence"] = round(
                    len(common) / max(1, len(union)), 3
                )
            elif member_patterns:
                ring["signal_coherence"] = 1.0
            else:
                ring["signal_coherence"] = 0.0

            # Validation: purity >= 0.15, confidence >= 0.30, >= 2 suspicious
            ring["validated"] = (
                ring["purity"] >= 0.15
                and ring.get("confidence_score", 0) >= 0.30
                and suspicious_member_count >= 2
            )

            # Confidence label
            ring["confidence"] = (
                "HIGH" if ring["purity"] >= 0.5
                    and ring.get("confidence_score", 0) >= 0.5
                else "MEDIUM" if ring["purity"] >= 0.15
                else "LOW"
            )
        else:
            ring["purity"] = 0.0
            ring["confidence"] = "LOW"
            ring["validated"] = False
            ring["signal_coherence"] = 0.0
            ring["suspicious_member_count"] = 0

        if ring["validated"]:
            validated_rings.append(ring)

    # ─── Phase 5: Summary metrics ───────────────────────────
    summary = _build_summary(tg, suspicious_accounts, validated_rings, legit)

    return suspicious_accounts, validated_rings, summary


# ─── Internal Scoring Functions ──────────────────────────────

def _compute_raw_score(ad: dict) -> float:
    """Compute raw weighted score with diminishing returns."""
    pattern_counts = Counter(s[0] for s in ad["signals"])
    total = 0.0

    for module_type, count in pattern_counts.items():
        weight = MODULE_WEIGHTS.get(module_type, 0.03)
        avg_risk = np.mean([s[1] for s in ad["signals"] if s[0] == module_type])
        effective = 1.0 + math.log(count) * 0.4 if count > 1 else 1.0
        total += weight * avg_risk * effective

    return total


def _assign_tier(
    patterns: set, strong_count: int,
    has_structural: bool, has_behavioral: bool, has_ring: bool,
    ring_manager: RingManager, ring_ids: set,
) -> str:
    """
    Risk tier with strict multi-signal gating.

    CRITICAL: Ring membership + structural + behavioral + >=3 strong
    HIGH:     (Structural + behavioral) OR >=3 strong
    MEDIUM:   >=2 strong, or 1 strong + ring
    LOW:      everything else
    """
    strong_patterns = patterns & STRONG_SIGNALS

    in_strong_ring = False
    if has_ring:
        for rid in ring_ids:
            conf = ring_manager.get_ring_confidence_for_node(str(rid))
            if conf >= 0.4:
                in_strong_ring = True
                break

    has_circular = "circular_routing" in patterns
    has_structuring = "structuring" in patterns
    has_rapid = "rapid_movement" in patterns

    # CRITICAL
    if has_circular and (has_structuring or has_rapid) and has_ring:
        return "CRITICAL"
    if in_strong_ring and has_structural and has_behavioral and len(strong_patterns) >= 3:
        return "CRITICAL"

    # HIGH
    if has_structural and has_behavioral and len(strong_patterns) >= 2:
        return "HIGH"
    if len(strong_patterns) >= 3:
        return "HIGH"

    # MEDIUM
    if len(strong_patterns) >= 2:
        return "MEDIUM"
    if len(strong_patterns) >= 1 and has_ring:
        return "MEDIUM"

    return "LOW"


def _compute_dependency_penalty(patterns: set) -> float:
    penalty = 1.0
    for group_signals in DEPENDENCY_GROUPS.values():
        overlap = patterns & group_signals
        if len(overlap) > 1:
            penalty *= 1.0 - (len(overlap) - 1) * 0.18
    return max(0.35, penalty)


def _compute_ring_boost(ring_ids: set, ring_manager: RingManager) -> float:
    if not ring_ids:
        return 0.0

    best_boost = 0.0
    for ring_id in ring_ids:
        ring = ring_manager.rings.get(ring_id)
        if not ring:
            continue
        prio = get_priority(ring["type"])
        confidence = ring.get("confidence_score", 0.5)

        if prio <= 1:
            base = 14.0
        elif prio <= 2:
            base = 8.0
        elif prio <= 3:
            base = 4.0
        else:
            base = 1.5

        boost = base * confidence
        best_boost = max(best_boost, boost)

    return min(best_boost, 20.0)


def _compute_network_influence(
    tg, account_data: dict, ring_manager: RingManager,
) -> Dict[str, float]:
    """
    TASK 4: Network influence and risk propagation.
    """
    influence = {}

    for account_id, ad in account_data.items():
        boost = 0.0

        # Betweenness centrality
        betweenness = tg.betweenness.get(account_id, 0)
        if betweenness > 0.05:
            boost += betweenness * 8.0

        # Flow influence: connected to high-risk ring members
        ring_risk = ring_manager.get_ring_risk_for_node(account_id)
        if ring_risk > 60:
            boost += 5.0

        # Neighbors in rings
        neighbors = set()
        try:
            for s in tg.G.successors(account_id):
                neighbors.add(s)
            for p in tg.G.predecessors(account_id):
                neighbors.add(p)
        except Exception:
            pass

        neighbor_in_ring = sum(
            1 for n in neighbors if ring_manager.is_assigned(n)
        )
        if neighbor_in_ring >= 2:
            boost += neighbor_in_ring * 1.5

        # Peripheral dampening
        in_deg = tg.in_degree.get(account_id, 0)
        out_deg = tg.out_degree.get(account_id, 0)
        total_deg = in_deg + out_deg

        if total_deg <= 1 and ad["strong_count"] == 0:
            boost -= 3.0

        influence[account_id] = max(-5.0, min(boost, 15.0))

    return influence


def _apply_stability_suppression(tg, account_id: str, score: float) -> float:
    """
    Behavioral stability suppression.
    """
    temporal = tg.node_temporal.get(account_id, {})
    if not temporal or temporal.get("tx_count", 0) == 0:
        return score

    tx_count = temporal.get("tx_count", 0)

    # Lifespan-based reduction
    lifespan_days = 0
    if temporal.get("first_seen") and temporal.get("last_seen"):
        lifespan_days = (
            (temporal["last_seen"] - temporal["first_seen"]).total_seconds()
            / 86400
        )

    if lifespan_days > 180 and tx_count > 100:
        score *= 0.55
    elif lifespan_days > 120 and tx_count > 60:
        score *= 0.70
    elif lifespan_days > 60 and tx_count > 30:
        score *= 0.85

    # Counterparty diversity
    unique_cp = temporal.get("unique_counterparties", 0)
    if unique_cp > 0 and tx_count > 0:
        diversity_ratio = unique_cp / tx_count
        if diversity_ratio > 0.85 and tx_count > 40:
            score *= 0.50
        elif diversity_ratio > 0.7 and tx_count > 25:
            score *= 0.65

    # Temporal regularity
    timestamps = temporal.get("timestamps", [])
    if len(timestamps) > 12:
        intervals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if delta > 0:
                intervals.append(delta)
        if intervals:
            cv = np.std(intervals) / max(np.mean(intervals), 1)
            if cv < 0.2:
                score *= 0.60
            elif cv < 0.35:
                score *= 0.80

    # Amount consistency (salary-like)
    std_amount = temporal.get("std_amount", 0)
    avg_amount = temporal.get("avg_amount", 1)
    if avg_amount > 0 and (std_amount / max(0.01, avg_amount)) < 0.03:
        score *= 0.75

    return score


def _map_to_tier_band(
    raw_score: float, tier: str, jitter: float = 0.0
) -> float:
    """Map raw score into tier's band with optional jitter."""
    band = TIER_BANDS.get(tier, (10, 29))
    lo, hi = band

    # Apply jitter only at boundaries
    adjusted_lo = max(0, lo + jitter * 0.5)
    adjusted_hi = min(100, hi + jitter * 0.5)

    return max(adjusted_lo, min(adjusted_hi, raw_score))


def _tier_to_confidence(tier: str) -> float:
    return {
        "CRITICAL": 0.95, "HIGH": 0.80,
        "MEDIUM": 0.55, "LOW": 0.25,
    }.get(tier, 0.25)


# ─── Score Breakdown & Explainability ────────────────────────

def _build_score_breakdown(signals: list, patterns: set) -> list:
    breakdown = []
    pattern_signals = defaultdict(list)
    for module_type, risk, _ in signals:
        pattern_signals[module_type].append(risk)

    for module_type in patterns:
        risks = pattern_signals.get(module_type, [0])
        weight = MODULE_WEIGHTS.get(module_type, 0.03)
        is_structural = module_type in STRUCTURAL_SIGNALS
        is_behavioral = module_type in BEHAVIORAL_SIGNALS

        breakdown.append({
            "module": module_type,
            "detections": len(risks),
            "avg_risk": round(float(np.mean(risks)), 4),
            "weighted": round(weight * float(np.mean(risks)), 4),
            "signal_strength": (
                "STRONG" if module_type in STRONG_SIGNALS else "WEAK"
            ),
            "signal_class": (
                "structural" if is_structural
                else ("behavioral" if is_behavioral else "contextual")
            ),
        })

    breakdown.sort(key=lambda x: x["weighted"], reverse=True)
    return breakdown


def _build_explanation(
    ad: dict, patterns: set, tier: str,
    ring_manager: RingManager, tg, account_id: str,
) -> str:
    parts = []

    # Signal hierarchy
    strong_names = sorted(
        p.replace("_", " ").title() for p in patterns if p in STRONG_SIGNALS
    )
    weak_names = sorted(
        p.replace("_", " ").title() for p in patterns if p in WEAK_SIGNALS
    )

    if strong_names:
        parts.append(f"Primary signals: {', '.join(strong_names)}.")
    if weak_names:
        parts.append(f"Supporting indicators: {', '.join(weak_names)}.")

    # Ring narrative
    if ad["ring_ids"]:
        ring_id = list(ad["ring_ids"])[0]
        ring = ring_manager.rings.get(ring_id)
        if ring:
            ring_patterns = ", ".join(ring.get("patterns", [ring["type"]]))
            parts.append(
                f"Member of {ring_patterns.replace('_', ' ')} ring ({ring_id}) "
                f"with {ring['node_count']} accounts, "
                f"purity {ring.get('purity', 0):.0%}."
            )

    # Temporal flow
    temporal = tg.node_temporal.get(account_id, {})
    if temporal.get("first_seen") and temporal.get("last_seen"):
        lifespan = (
            (temporal["last_seen"] - temporal["first_seen"]).total_seconds()
            / 86400
        )
        in_count = temporal.get("in_count", 0)
        out_count = temporal.get("out_count", 0)
        total_amt = temporal.get("total_amount", 0)
        parts.append(
            f"Active {lifespan:.0f} days, {in_count} inflows / "
            f"{out_count} outflows, total volume ${total_amt:,.2f}."
        )

    # Confidence
    conf = _tier_to_confidence(tier)
    parts.append(f"Confidence: {conf:.0%} ({tier}).")

    return " ".join(parts)


# ─── Evaluation Metrics ─────────────────────────────────────

def _build_summary(tg, suspicious_accounts, fraud_rings, legit_set) -> dict:
    total_analyzed = tg.node_count
    flagged_count = len(suspicious_accounts)

    high_conf = [
        a for a in suspicious_accounts if a["tier"] in ("HIGH", "CRITICAL")
    ]
    precision_proxy = (
        round(len(high_conf) / flagged_count, 3) if flagged_count > 0 else 1.0
    )

    ring_purities = [r.get("purity", 0) for r in fraud_rings]
    avg_purity = (
        round(np.mean(ring_purities), 3) if ring_purities else 0.0
    )

    cluster_density = round(flagged_count / max(1, total_analyzed), 4)

    tier_dist = Counter(a["tier"] for a in suspicious_accounts)

    # FP estimate
    weak_only = len([
        a for a in suspicious_accounts
        if a.get("signal_summary", {}).get("strong_signals", 0) == 0
    ])
    fp_estimate = round(weak_only / max(1, flagged_count), 3)

    # Multi-signal gate pass rate
    multi_signal = len([
        a for a in suspicious_accounts
        if a.get("signal_summary", {}).get("multi_signal_gate", False)
    ])
    gate_pass_rate = round(multi_signal / max(1, flagged_count), 3)

    # Ring signal coherence
    ring_coherence = [r.get("signal_coherence", 0) for r in fraud_rings]
    avg_coherence = (
        round(np.mean(ring_coherence), 3) if ring_coherence else 0.0
    )

    return {
        "total_accounts_analyzed": total_analyzed,
        "total_transactions_analyzed": len(tg.df) if hasattr(tg, "df") else 0,
        "suspicious_accounts_found": flagged_count,
        "fraud_rings_detected": len(fraud_rings),
        "high_risk_accounts": len(high_conf),
        "estimated_precision": precision_proxy,
        "avg_ring_purity": avg_purity,
        "suspicious_cluster_density": cluster_density,
        "fp_estimate": fp_estimate,
        "multi_signal_gate_pass_rate": gate_pass_rate,
        "avg_ring_signal_coherence": avg_coherence,
        "tier_distribution": {
            "CRITICAL": tier_dist.get("CRITICAL", 0),
            "HIGH": tier_dist.get("HIGH", 0),
            "MEDIUM": tier_dist.get("MEDIUM", 0),
            "LOW": tier_dist.get("LOW", 0),
        },
        "avg_risk_score": (
            round(np.mean([a["risk_score"] for a in suspicious_accounts]), 2)
            if suspicious_accounts else 0
        ),
        "total_suspicious_amount": round(
            sum(r.get("total_amount", 0) for r in fraud_rings), 2
        ),
        "processing_time_seconds": 0,
    }
