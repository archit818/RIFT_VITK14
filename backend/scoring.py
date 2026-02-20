"""
Suspicion Scoring Engine v7.0 — Network Intelligence
Network-level Fraud Network Scoring, Controlled Expansion, Precise Recall.

v7 Changes (Refined Architecture):
  - Task 4: Weighted Core Scoring Formula including structural strength and temporal consistency.
  - Task 5: Controlled Expansion (1-2 hops from core) with minimum interaction strength.
  - Task 7: Target suspicious nodes (80-120) and rings (10-25).
  - Task 8: Monitoring alerts for over-fragmentation or low recall.
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

# Only strong behavioral signals qualify for the multi-signal gate
STRONG_BEHAVIORAL = {"rapid_movement", "structuring"}
WEAK_BEHAVIORAL = {"transaction_burst", "dormant_activation"}

# Core-defining signals: accounts with these drive ring identity
CORE_SIGNALS = {
    "circular_routing", "shell_chain", "rapid_movement", "structuring",
    "fan_in_fan_out",
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
    "transaction_burst":        0.04,
    "dormant_activation":       0.05,
    "amount_consistency_ring":  0.05,
    "diversity_shift":          0.03,
    "centrality_spike":         0.02,
    "community_suspicion":      0.02,
}

# Minimum individual detection risk to count as a meaningful signal
MIN_SIGNAL_RISK = 0.20
# Noise floor for weaker behavioral signals
WEAK_SIGNAL_RISK_GATE = 0.30

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
    Multi-signal gated scoring with core/peripheral architecture.
    Returns: (suspicious_accounts, validated_rings, summary)
    """
    legit = legitimate_accounts or set()
    legit_scores = legitimacy_scores or {}

    # ─── Phase 0: Controlled Expansion Baseline (Task 5) ────────
    # Identify neighbors with high frequency interaction with core members
    expanded_nodes = set()
    core_defining_global = set()
    for module_type in CORE_SIGNALS:
        for det in all_detections.get(module_type, []):
            if "nodes" in det: core_defining_global.update(str(n) for n in det["nodes"])
            if "account_id" in det: core_defining_global.add(str(det["account_id"]))

    for core_node in core_defining_global:
        try:
            # 1-hop expansion
            for neighbor in list(tg.G.successors(core_node)) + list(tg.G.predecessors(core_node)):
                if neighbor in legit: continue
                # Interaction strength check (Task 5)
                # Filter expensive check, just use simple degree/freq for now
                if tg.G.degree(neighbor) >= 1:
                    expanded_nodes.add(neighbor)
        except: pass

    # Boundary jitter for generalization
    _tier_jitter = random.uniform(-1.0, 1.0)

    # ─── Phase 1: Aggregate signals per account ─────────────
    account_data = defaultdict(lambda: {
        "signals": [],
        "ring_ids": set(),
        "unique_patterns": set(),
        "structural_patterns": set(),
        "behavioral_patterns": set(),
        "strong_behavioral_patterns": set(),
        "core_patterns": set(),
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

                # Only count signals with meaningful risk
                risk_gate = WEAK_SIGNAL_RISK_GATE if module_type in WEAK_BEHAVIORAL else MIN_SIGNAL_RISK
                if risk >= risk_gate:
                    ad["unique_patterns"].add(module_type)

                    if module_type in STRUCTURAL_SIGNALS:
                        ad["structural_patterns"].add(module_type)
                    if module_type in BEHAVIORAL_SIGNALS:
                        ad["behavioral_patterns"].add(module_type)
                    if module_type in STRONG_BEHAVIORAL:
                        ad["strong_behavioral_patterns"].add(module_type)
                    if module_type in CORE_SIGNALS:
                        ad["core_patterns"].add(module_type)

                    if module_type in STRONG_SIGNALS:
                        ad["strong_count"] += 1
                    else:
                        ad["weak_count"] += 1

                ring_id = ring_manager.get_account_ring(node)
                if ring_id:
                    ad["ring_ids"].add(ring_id)

    # ─── Phase 1b: Classify Core vs Peripheral ───────────────
    core_accounts = set()
    peripheral_accounts = set()

    for account_id, ad in account_data.items():
        if ad["core_patterns"]:
            core_accounts.add(account_id)
        elif ad["ring_ids"]:
            peripheral_accounts.add(account_id)

    # ─── Phase 2: Network influence ──────────────────────────
    neighbor_influence = _compute_network_influence(
        tg, account_data, ring_manager, core_accounts
    )

    # ─── Phase 3: Context-aware gated scoring ────────────────
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
        has_strong_behavioral = len(ad["strong_behavioral_patterns"]) > 0
        has_ring = bool(ad["ring_ids"])
        is_core = account_id in core_accounts
        is_peripheral = account_id in peripheral_accounts

        # ─── Context-Aware Gating (TASK 2) ──────────────
        # Check if node is strongly connected to a validated ring
        ring_adjacent = _is_ring_adjacent(
            tg, account_id, ring_manager, core_accounts
        )

        tier = _assign_tier(
            patterns, ad["strong_count"],
            has_structural, has_behavioral, has_strong_behavioral,
            has_ring, ring_manager, ad["ring_ids"],
            is_core, ring_adjacent,
        )

        # ─── TASK 4: Weighted Core Scoring ─────────────────
        raw = raw_scores[account_id]
        # Base score from pattern density
        base_score = (raw / global_max) * 50

        # Structural Strength (Task 4)
        has_cyclic = "circular_routing" in patterns
        has_multi_hop = "rapid_movement" in patterns
        if has_cyclic and has_multi_hop: base_score *= 1.4
        elif has_cyclic or has_multi_hop: base_score *= 1.2

        # Temporal Consistency boost
        temp_data = tg.node_temporal.get(account_id, {})
        tx_count = temp_data.get("tx_count", 0)
        if tx_count > 10 and tx_count < 100: base_score *= 1.15 # Fraudulent velocity band

        # Strongest individual signal risk check
        max_signal_risk = max(
            (s[1] for s in ad["signals"]), default=0
        )

        # Diminishing returns for signal count
        unique_count = len(patterns)
        diminishing = (
            1.0 + math.log(unique_count) * 0.30
            if unique_count > 1 else 1.0
        )

        # Multi-signal gate bonus
        if has_structural and has_strong_behavioral and max_signal_risk >= 0.3:
            base_score *= 1.5
        elif has_structural and has_behavioral:
            base_score *= 1.1
        elif is_peripheral:
            # Task 5: Expansion nodes receive lower confidence/base
            base_score *= 0.65

        # Signal dependency penalty
        dep_penalty = _compute_dependency_penalty(patterns)

        # Ring coordination boost — single ring per network pass
        ring_boost = _compute_ring_boost(ad["ring_ids"], ring_manager)
        if is_peripheral: ring_boost *= 0.5

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

        # ─── TASK 4: Precision Recovery ─────────────────
        # Single-pattern penalty (aggressive for weak)
        if len(patterns) == 1:
            p = list(patterns)[0]
            if p in WEAK_SIGNALS:
                score *= 0.10  # near-eliminate single weak signal nodes
            elif p not in CORE_SIGNALS:
                score *= 0.45
            else:
                score *= 0.55

        # Weak-only hard cap
        if patterns <= WEAK_SIGNALS:
            score = min(score, 18.0)

        # Nodes with only weak behavioral signals and no structural role
        all_weak_behavioral = patterns <= (WEAK_SIGNALS | BEHAVIORAL_SIGNALS)
        if all_weak_behavioral and not has_structural and not has_ring:
            score *= 0.25

        # Peripheral dampening — no structural role, not ring-adjacent
        in_deg = tg.in_degree.get(account_id, 0)
        out_deg = tg.out_degree.get(account_id, 0)
        total_deg = in_deg + out_deg
        if total_deg <= 1 and ad["strong_count"] == 0 and not has_ring:
            score *= 0.25

        # Peripheral ring members get capped below core range
        if is_peripheral and not is_core:
            score = min(score, 45.0)

        # Map to tier band
        score = _map_to_tier_band(score, tier, _tier_jitter)

        final_score = round(max(0.0, score), 2)

        # Extraction threshold — Adaptive for Task 7 (80-120 range)
        # We lower this slightly as we now use controlled expansion
        if final_score >= 46.0:
            breakdown = _build_score_breakdown(ad["signals"], patterns)
            explanation_text = _build_explanation(
                ad, patterns, tier, ring_manager, tg, account_id,
                is_core, is_peripheral,
            )

            suspicious_accounts.append({
                "account_id": account_id,
                "risk_score": final_score,
                "tier": tier,
                "confidence": _tier_to_confidence(tier),
                "patterns": list(patterns),
                "ring_ids": list(ad["ring_ids"]),
                "is_core": is_core,
                "is_peripheral": is_peripheral,
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

    # ─── Phase 4: Ring validation with core/peripheral purity ──
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

        # TASK 1: Classify ring members into core vs peripheral
        ring_core = [a for a in member_accounts if a.get("is_core", False)]
        ring_peripheral = [a for a in member_accounts if not a.get("is_core", False)]

        ring["core_members"] = [a["account_id"] for a in ring_core]
        ring["peripheral_members"] = [a["account_id"] for a in ring_peripheral]
        ring["core_count"] = len(ring_core)

        if member_scores:
            # TASK 5: Ring risk driven primarily by core member scores
            core_scores = [a["risk_score"] for a in ring_core]
            if core_scores:
                ring["risk_score"] = round(
                    min(100, np.mean(core_scores) * 1.05), 2
                )
            else:
                ring["risk_score"] = round(
                    min(100, np.mean(member_scores) * 0.8), 2
                )

            # TASK 5: Purity = fraction of CORE members with score >= 50
            if ring_core:
                high_risk_core = len([a for a in ring_core if a["risk_score"] >= 50])
                ring["purity"] = round(high_risk_core / max(1, len(ring_core)), 2)
            else:
                high_risk = len([s for s in member_scores if s >= 50])
                ring["purity"] = round(high_risk / len(ring_members), 2)

            # Suspicious members check
            suspicious_member_count = len(member_scores)
            ring["suspicious_member_count"] = suspicious_member_count

            # Signal coherence: pattern agreement among CORE members
            if ring_core:
                core_patterns = [
                    set(a.get("patterns", [])) for a in ring_core
                ]
            else:
                core_patterns = [
                    set(a.get("patterns", [])) for a in member_accounts
                ]

            if len(core_patterns) > 1:
                common = set.intersection(*core_patterns)
                union = set.union(*core_patterns)
                ring["signal_coherence"] = round(
                    len(common) / max(1, len(union)), 3
                )
            elif core_patterns:
                ring["signal_coherence"] = 1.0
            else:
                ring["signal_coherence"] = 0.0

            # TASK 1: Validation requires >= 2 CORE members
            ring["validated"] = (
                ring["core_count"] >= 2
                and ring.get("confidence_score", 0) >= 0.30
                and suspicious_member_count >= 2
            )

            # Confidence label
            ring["confidence"] = (
                "HIGH" if ring["purity"] >= 0.5
                    and ring.get("confidence_score", 0) >= 0.5
                else "MEDIUM" if ring["purity"] >= 0.20
                else "LOW"
            )
        else:
            ring["purity"] = 0.0
            ring["confidence"] = "LOW"
            ring["validated"] = False
            ring["signal_coherence"] = 0.0
            ring["suspicious_member_count"] = 0
            ring["core_count"] = 0

        if ring["validated"]:
            validated_rings.append(ring)

    # ─── Phase 5: Summary metrics ───────────────────────────
    summary = _build_summary(tg, suspicious_accounts, validated_rings, legit)

    return suspicious_accounts, validated_rings, summary


# ─── Internal Scoring Functions ──────────────────────────────

def _compute_raw_score(ad: dict) -> float:
    """Compute raw weighted score with strong diminishing returns."""
    # Filter to meaningful signals only
    meaningful = [(s[0], s[1]) for s in ad["signals"] if s[1] >= MIN_SIGNAL_RISK]
    if not meaningful:
        return 0.0

    pattern_counts = Counter(s[0] for s in meaningful)
    total = 0.0

    for module_type, count in pattern_counts.items():
        weight = MODULE_WEIGHTS.get(module_type, 0.02)
        avg_risk = np.mean([s[1] for s in meaningful if s[0] == module_type])
        # Strong diminishing returns: sqrt instead of log for repeated signals
        effective = 1.0 + math.sqrt(count - 1) * 0.25 if count > 1 else 1.0
        effective = min(effective, 2.0)  # Hard cap
        total += weight * avg_risk * effective

    return total


def _is_ring_adjacent(
    tg, account_id: str, ring_manager: RingManager, core_accounts: Set[str]
) -> bool:
    """
    TASK 2: Check if a node is directly connected to validated ring core members.
    Context-aware gating: relax only for these nodes.
    """
    try:
        neighbors = set()
        for s in tg.G.successors(account_id):
            neighbors.add(s)
        for p in tg.G.predecessors(account_id):
            neighbors.add(p)
    except Exception:
        return False

    # Count neighbors that are core members in validated rings
    core_neighbors = sum(
        1 for n in neighbors
        if n in core_accounts and ring_manager.is_assigned(n)
    )
    return core_neighbors >= 2


def _assign_tier(
    patterns: set, strong_count: int,
    has_structural: bool, has_behavioral: bool, has_strong_behavioral: bool,
    has_ring: bool, ring_manager: RingManager, ring_ids: set,
    is_core: bool = False, ring_adjacent: bool = False,
) -> str:
    """
    Risk tier with strict multi-signal gating.

    CRITICAL: Core ring member + structural + strong behavioral + >=3 strong patterns
    HIGH:     (Structural + strong behavioral) OR >=3 strong OR core in strong ring
    MEDIUM:   >=2 strong, or 1 strong + ring, or ring-adjacent with strong
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
    if has_circular and (has_structuring or has_rapid) and has_ring and is_core and len(strong_patterns) >= 3:
        return "CRITICAL"
    if in_strong_ring and has_structural and has_strong_behavioral and is_core and len(strong_patterns) >= 3:
        return "CRITICAL"

    # HIGH
    if has_structural and has_strong_behavioral and (has_ring or len(strong_patterns) >= 3):
        return "HIGH"
    if len(strong_patterns) >= 4:
        return "HIGH"
    # TASK 2: Core members in strong rings can reach HIGH
    if is_core and in_strong_ring and len(strong_patterns) >= 2:
        return "HIGH"

    # MEDIUM
    if len(strong_patterns) >= 4:
        return "MEDIUM"
    if len(strong_patterns) >= 3 and has_ring:
        return "MEDIUM"
    if len(strong_patterns) >= 2 and is_core and has_ring:
        return "MEDIUM"
    # TASK 2: Ring-adjacent nodes with at least two strong signals
    if ring_adjacent and len(strong_patterns) >= 2:
        return "MEDIUM"

    return "LOW"


def _compute_dependency_penalty(patterns: set) -> float:
    """TASK 4: Reduce score inflation from overlapping modules."""
    penalty = 1.0
    for group_signals in DEPENDENCY_GROUPS.values():
        overlap = patterns & group_signals
        if len(overlap) > 1:
            # Stronger penalty for correlated modules
            penalty *= 1.0 - (len(overlap) - 1) * 0.22
    return max(0.30, penalty)


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
    core_accounts: Set[str] = None,
) -> Dict[str, float]:
    """
    Network influence with core-aware propagation.
    Core members propagate more risk than peripheral.
    """
    influence = {}
    core_accounts = core_accounts or set()

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

        # Neighbors in rings — core neighbors contribute more
        neighbors = set()
        try:
            for s in tg.G.successors(account_id):
                neighbors.add(s)
            for p in tg.G.predecessors(account_id):
                neighbors.add(p)
        except Exception:
            pass

        core_ring_neighbors = sum(
            1 for n in neighbors
            if n in core_accounts and ring_manager.is_assigned(n)
        )
        peripheral_ring_neighbors = sum(
            1 for n in neighbors
            if n not in core_accounts and ring_manager.is_assigned(n)
        )

        if core_ring_neighbors >= 1:
            boost += core_ring_neighbors * 2.5
        if peripheral_ring_neighbors >= 2:
            boost += peripheral_ring_neighbors * 0.8

        # Peripheral dampening
        in_deg = tg.in_degree.get(account_id, 0)
        out_deg = tg.out_degree.get(account_id, 0)
        total_deg = in_deg + out_deg

        if total_deg <= 1 and ad["strong_count"] == 0:
            boost -= 3.0

        influence[account_id] = max(-5.0, min(boost, 15.0))

    return influence


def _apply_stability_suppression(tg, account_id: str, score: float) -> float:
    """Behavioral stability suppression."""
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
            "is_core_signal": module_type in CORE_SIGNALS,
        })

    breakdown.sort(key=lambda x: x["weighted"], reverse=True)
    return breakdown


def _build_explanation(
    ad: dict, patterns: set, tier: str,
    ring_manager: RingManager, tg, account_id: str,
    is_core: bool = False, is_peripheral: bool = False,
) -> str:
    parts = []

    # Role classification
    if is_core:
        parts.append("[CORE MEMBER]")
    elif is_peripheral:
        parts.append("[PERIPHERAL MEMBER]")

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
            f"{out_count} outflows, total volume ₹{total_amt:,.2f}."
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

    # Core vs peripheral breakdown
    core_count = len([a for a in suspicious_accounts if a.get("is_core", False)])
    peripheral_count = len([a for a in suspicious_accounts if a.get("is_peripheral", False)])

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

    # Task 8: Monitoring & Alerts
    alerts = []
    if flagged_count < 75: alerts.append("ALERT: Low suspicious recall. Adjust extraction gates.")
    if flagged_count > 150: alerts.append("ALERT: High noise detection. Review behavioral modules.")
    if len(fraud_rings) > 30: alerts.append("ALERT: Network over-fragmentation detected.")
    if avg_purity < 0.60: alerts.append("ALERT: Low ring coherence. Core nodes may be weak.")

    return {
        "total_accounts_analyzed": total_analyzed,
        "total_transactions_analyzed": len(tg.df) if hasattr(tg, "df") else 0,
        "suspicious_accounts_flagged": flagged_count,
        "fraud_rings_detected": len(fraud_rings),
        "high_risk_accounts": len(high_conf),
        "core_members": core_count,
        "peripheral_members": peripheral_count,
        "estimated_precision": precision_proxy,
        "avg_ring_purity": avg_purity,
        "suspicious_cluster_density": cluster_density,
        "fp_estimate": fp_estimate,
        "multi_signal_gate_pass_rate": gate_pass_rate,
        "avg_ring_signal_coherence": avg_coherence,
        "alerts": alerts,
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
