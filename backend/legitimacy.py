"""
Legitimacy Scoring Module v4.0
Tasks 3 & 7: Stability Suppression + False Positive Control.

Stability Scoring (Task 3):
  - Long-term behavioral consistency
  - Low behavioral change rate
  - Regular counterparties
  - Stable account reduces suspicion

Merchant Suppression (Task 7):
  - High transaction diversity
  - Stable daily behavior
  - No cyclic routing / structuring
  - Primarily incoming

Payroll Suppression (Task 7):
  - Recurring fixed-amount payments
  - Regular intervals
  - Primarily outgoing
  - Revoke on sudden behavior change or redistribution

Platform/Exchange Suppression (Task 7):
  - Very high counterparty diversity
  - Balanced bidirectional flow
  - High volume with consistent throughput
  - Only flag abnormal behavior deviations
"""

import math
import numpy as np
from typing import Dict, Set, Tuple, List
from datetime import timedelta
from collections import Counter


def compute_legitimacy_scores(tg) -> Dict[str, float]:
    """
    Multi-factor legitimacy scoring (0-1).
    High scores = high confidence of legitimate activity.
    """
    scores = {}
    cycle_nodes = _find_short_cycle_nodes(tg)
    structuring_suspects = _detect_structuring_patterns(tg)

    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        tx_count = temporal.get("tx_count", 0)

        if tx_count == 0:
            scores[node] = 0.3  # Unknown = slight suspicion bias
            continue

        # ─── Merchant Profile (Task 7) ──────────────────
        merchant_score = _compute_merchant_score(tg, node, temporal, cycle_nodes, structuring_suspects)

        # ─── Payroll Profile (Task 7) ───────────────────
        payroll_score = _compute_payroll_score(tg, node, temporal)

        # ─── Platform/Exchange Profile (Task 7) ─────────
        platform_score = _compute_platform_score(tg, node, temporal, cycle_nodes)

        # ─── Business Hub Profile ───────────────────────
        hub_score = _compute_hub_score(tg, node, temporal)

        # ─── Stability Score (Task 3) ───────────────────
        stability_score = _compute_stability_score(tg, node, temporal)

        # ─── Operational Longevity ──────────────────────
        lifespan_score = 0.0
        if temporal.get("first_seen") and temporal.get("last_seen"):
            days = (temporal["last_seen"] - temporal["first_seen"]).total_seconds() / 86400
            lifespan_score = min(1.0, days / 180) # Slower maturity for high scores

        # ─── Robot Automation Detection (Task 7) ────────
        automation_penalty = 0.0
        timestamps = temporal.get("timestamps", [])
        if len(timestamps) >= 8:
            intervals = []
            for i in range(1, len(timestamps)):
                d = (timestamps[i] - timestamps[i-1]).total_seconds()
                if d > 0: intervals.append(d)
            if intervals:
                cv = np.std(intervals) / np.mean(intervals)
                if cv < 0.005: automation_penalty = 0.8  # Near-perfect rhythmic script
                elif cv < 0.02: automation_penalty = 0.4

        # ─── Temporal Regularity ────────────────────────
        regularity_score = _compute_temporal_regularity(temporal)

        # ─── Diversity Entropy ──────────────────────────
        entropy_score = _compute_diversity_entropy(temporal)
        malicious_penalty = 0.0
        
        # Mule Hub Penalty: High-volume passthrough behavior
        if _has_immediate_redistribution(tg, node, temporal):
            malicious_penalty += 0.45
            
        if node in cycle_nodes:
            malicious_penalty += 0.7
        if node in structuring_suspects:
            malicious_penalty += 0.35

        # ─── Combine (Task 7: strengthened) ─────────────
        best_profile = max(merchant_score, payroll_score, hub_score, platform_score)

        legitimacy = (
            best_profile * 0.35 +
            stability_score * 0.25 +
            lifespan_score * 0.15 +
            regularity_score * 0.10 +
            entropy_score * 0.10 +
            min(1.0, tx_count / 100) * 0.05
        )

        final_score = max(0.0, min(1.0, legitimacy - malicious_penalty - automation_penalty))
        scores[node] = round(final_score, 4)

    return scores


def _compute_merchant_score(tg, node, temporal, cycle_nodes, structuring_suspects) -> float:
    """
    Task 7: Merchant detection with strict validation.
    Merchants: high diversity, mostly incoming, stable, no cycles, no structuring.
    """
    tx_count = temporal.get("tx_count", 0)
    unique_cp = temporal.get("unique_counterparties", 0)
    in_count = temporal.get("in_count", 0)

    if tx_count < 15:
        return 0.0

    diversity_ratio = unique_cp / max(1, tx_count)
    if diversity_ratio < 0.45:
        return 0.0

    in_ratio = in_count / max(1, tx_count)
    if in_ratio < 0.65:
        return 0.0

    if node in cycle_nodes:
        return 0.0
    if node in structuring_suspects:
        return 0.0

    # Stable amounts
    std_amount = temporal.get("std_amount", 0)
    avg_amount = temporal.get("avg_amount", 1)
    cv = std_amount / max(0.01, avg_amount)
    stability_factor = 1.0 if cv < 2.0 else max(0.0, 1.0 - (cv - 2.0) * 0.3)

    rapid_penalty = _check_rapid_movement_indicator(tg, node, temporal)
    if rapid_penalty > 0.2:
        return 0.0

    return min(1.0,
        (diversity_ratio * 0.35) +
        (in_ratio * 0.30) +
        (stability_factor * 0.35)
    )


def _compute_payroll_score(tg, node, temporal) -> float:
    """
    Task 7: Payroll detection.
    Payroll: recurring outgoing, fixed amounts, regular intervals.
    """
    tx_count = temporal.get("tx_count", 0)
    out_count = temporal.get("out_count", 0)

    if tx_count < 10 or out_count < 5:
        return 0.0

    out_ratio = out_count / max(1, tx_count)
    if out_ratio < 0.6:
        return 0.0

    std_amount = temporal.get("std_amount", 0)
    avg_amount = temporal.get("avg_amount", 1)
    cv = std_amount / max(0.01, avg_amount)
    amount_consistency = max(0.0, 1.0 - cv * 5)

    timestamps = temporal.get("timestamps", [])
    interval_regularity = _check_interval_regularity(timestamps)

    # Abnormality: sudden behavior change
    amounts = temporal.get("amounts", [])
    if len(amounts) > 10:
        recent = amounts[-5:]
        historical = amounts[:-5]
        if historical:
            hist_avg = np.mean(historical)
            recent_avg = np.mean(recent)
            if hist_avg > 0 and abs(recent_avg - hist_avg) / hist_avg > 0.5:
                return 0.0

    if _has_immediate_redistribution(tg, node, temporal):
        return 0.0

    return min(1.0,
        (amount_consistency * 0.40) +
        (interval_regularity * 0.30) +
        (out_ratio * 0.30)
    )


def _compute_platform_score(tg, node, temporal, cycle_nodes) -> float:
    """
    Task 7: Platform/exchange detection.
    Platforms: very high diversity, balanced flow, high volume, consistent throughput.
    """
    tx_count = temporal.get("tx_count", 0)
    unique_cp = temporal.get("unique_counterparties", 0)
    in_count = temporal.get("in_count", 0)
    out_count = temporal.get("out_count", 0)

    if tx_count < 50:
        return 0.0

    # Very high diversity
    diversity_ratio = unique_cp / max(1, tx_count)
    if diversity_ratio < 0.6:
        return 0.0

    # Balanced flow (not purely one direction)
    balance = min(in_count, out_count) / max(1, max(in_count, out_count))
    if balance < 0.2:
        return 0.0

    # No cyclic routing
    if node in cycle_nodes:
        return 0.0

    # Consistent throughput: check if daily volumes are stable
    timestamps = temporal.get("timestamps", [])
    if len(timestamps) < 20:
        return 0.0

    lifespan_days = 0
    if temporal.get("first_seen") and temporal.get("last_seen"):
        lifespan_days = (temporal["last_seen"] - temporal["first_seen"]).total_seconds() / 86400

    if lifespan_days < 14:
        return 0.0  # Must be active for at least 2 weeks

    daily_rate = tx_count / max(1, lifespan_days)
    consistency = min(1.0, daily_rate / 2)  # At least 2 tx/day on average

    return min(1.0,
        (diversity_ratio * 0.30) +
        (balance * 0.25) +
        (consistency * 0.25) +
        (min(1.0, tx_count / 200) * 0.20)
    )


def _compute_hub_score(tg, node, temporal) -> float:
    """Business hub: balanced flow + high diversity."""
    tx_count = temporal.get("tx_count", 0)
    in_count = temporal.get("in_count", 0)
    out_count = temporal.get("out_count", 0)
    unique_cp = temporal.get("unique_counterparties", 0)

    if tx_count < 30:
        return 0.0

    balance = min(in_count, out_count) / max(1, max(in_count, out_count))
    if balance < 0.3:
        return 0.0

    diversity = unique_cp / max(1, tx_count)
    if diversity < 0.4:
        return 0.0

    return min(1.0, balance * 0.5 + diversity * 0.5)


def _compute_stability_score(tg, node, temporal) -> float:
    """
    Task 3: Compute behavioral stability score.
    High stability = low behavioral change = likely legitimate.
    """
    tx_count = temporal.get("tx_count", 0)
    if tx_count < 8:
        return 0.0

    factors = []

    # Factor 1: Amount stability (low coefficient of variation)
    std_amount = temporal.get("std_amount", 0)
    avg_amount = temporal.get("avg_amount", 1)
    if avg_amount > 0:
        cv = std_amount / avg_amount
        if cv < 0.1:
            factors.append(1.0)
        elif cv < 0.3:
            factors.append(0.7)
        elif cv < 0.6:
            factors.append(0.4)
        else:
            factors.append(0.1)
    else:
        factors.append(0.0)

    # Factor 2: Temporal regularity
    regularity = _compute_temporal_regularity(temporal)
    factors.append(regularity)

    # Factor 3: Counterparty consistency
    # Stable accounts reuse the same counterparties
    unique_cp = temporal.get("unique_counterparties", 0)
    if tx_count > 0:
        reuse_ratio = 1.0 - (unique_cp / max(1, tx_count))
        # Moderate reuse is stable (not too many new counterparties)
        if reuse_ratio > 0.7:
            factors.append(0.9)
        elif reuse_ratio > 0.4:
            factors.append(0.6)
        elif reuse_ratio > 0.2:
            factors.append(0.3)
        else:
            factors.append(0.1)
    else:
        factors.append(0.0)

    # Factor 4: Behavioral change rate (compare halves)
    amounts = temporal.get("amounts", [])
    if len(amounts) >= 10:
        mid = len(amounts) // 2
        first_half_avg = np.mean(amounts[:mid])
        second_half_avg = np.mean(amounts[mid:])
        if first_half_avg > 0:
            change_rate = abs(second_half_avg - first_half_avg) / first_half_avg
            if change_rate < 0.1:
                factors.append(1.0)
            elif change_rate < 0.3:
                factors.append(0.6)
            elif change_rate < 0.5:
                factors.append(0.3)
            else:
                factors.append(0.0)
        else:
            factors.append(0.0)
    else:
        factors.append(0.0)

    # Factor 5: Lifespan maturity
    if temporal.get("first_seen") and temporal.get("last_seen"):
        days = (temporal["last_seen"] - temporal["first_seen"]).total_seconds() / 86400
        if days > 120:
            factors.append(1.0)
        elif days > 60:
            factors.append(0.6)
        elif days > 14:
            factors.append(0.3)
        else:
            factors.append(0.0)
    else:
        factors.append(0.0)

    return round(np.mean(factors), 4) if factors else 0.0


def _compute_temporal_regularity(temporal) -> float:
    """Measure temporal regularity (CV of intervals)."""
    timestamps = temporal.get("timestamps", [])
    if len(timestamps) < 5:
        return 0.0

    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i-1]).total_seconds()
        if delta > 0:
            intervals.append(delta)

    if not intervals:
        return 0.0

    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    if mean_interval == 0:
        return 0.0

    cv = std_interval / mean_interval
    if cv < 0.2:
        return 1.0
    elif cv < 0.5:
        return 0.7
    elif cv < 1.0:
        return 0.3
    return 0.0


def _compute_diversity_entropy(temporal) -> float:
    """Shannon entropy proxy for counterparty diversity."""
    unique_cp = temporal.get("unique_counterparties", 0)
    tx_count = temporal.get("tx_count", 0)

    if unique_cp <= 1 or tx_count <= 1:
        return 0.0

    ratio = unique_cp / tx_count
    if ratio > 0.8:
        return 1.0
    elif ratio > 0.5:
        return 0.7
    elif ratio > 0.3:
        return 0.4
    return 0.1


def _check_interval_regularity(timestamps) -> float:
    """Check for weekly/monthly interval patterns."""
    if len(timestamps) < 4:
        return 0.0

    intervals_hours = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
        if delta > 0:
            intervals_hours.append(delta)

    if not intervals_hours:
        return 0.0

    mean_h = np.mean(intervals_hours)
    std_h = np.std(intervals_hours)

    if mean_h == 0:
        return 0.0

    cv = std_h / mean_h
    if cv < 0.3:
        return 1.0
    elif cv < 0.6:
        return 0.5
    return 0.1


def _has_immediate_redistribution(tg, node, temporal) -> bool:
    """Check if incoming funds are immediately redistributed (within 30 min)."""
    txns = tg.G.nodes.get(node, {}).get("transactions", [])
    if len(txns) < 4:
        return False

    in_txns = sorted([t for t in txns if t["type"] == "in"], key=lambda t: t["timestamp"])
    out_txns = sorted([t for t in txns if t["type"] == "out"], key=lambda t: t["timestamp"])

    quick_redistributions = 0
    threshold = timedelta(minutes=30)

    for in_tx in in_txns:
        for out_tx in out_txns:
            if timedelta(0) < (out_tx["timestamp"] - in_tx["timestamp"]) < threshold:
                if abs(in_tx["amount"] - out_tx["amount"]) / max(in_tx["amount"], 1) < 0.15:
                    quick_redistributions += 1

    return quick_redistributions > len(in_txns) * 0.3


def _check_rapid_movement_indicator(tg, node, temporal) -> float:
    """Check for rapid in → out fund movement patterns."""
    txns = tg.G.nodes.get(node, {}).get("transactions", [])
    if len(txns) < 4:
        return 0.0

    in_txns = [t for t in txns if t["type"] == "in"]
    out_txns = [t for t in txns if t["type"] == "out"]

    if not in_txns or not out_txns:
        return 0.0

    rapid_count = 0
    threshold = timedelta(hours=2)

    for in_tx in in_txns[-10:]:
        for out_tx in out_txns:
            if timedelta(0) < (out_tx["timestamp"] - in_tx["timestamp"]) < threshold:
                rapid_count += 1
                break

    ratio = rapid_count / max(1, len(in_txns[-10:]))
    if ratio > 0.5:
        return 0.35
    elif ratio > 0.3:
        return 0.18
    return 0.0


def _detect_structuring_patterns(tg) -> Set[str]:
    """Detect nodes showing structuring behavior."""
    suspects = set()
    thresholds = [10000, 5000, 3000]

    for node in tg.G.nodes():
        amounts = tg.node_temporal.get(node, {}).get("amounts", [])
        if len(amounts) < 5:
            continue

        for threshold in thresholds:
            below = [a for a in amounts if threshold * 0.8 <= a < threshold]
            if len(below) >= 3 and len(below) / len(amounts) > 0.3:
                suspects.add(node)
                break

    return suspects


def filter_legitimate_accounts(tg, threshold: float = 0.75) -> Tuple[Set[str], Dict[str, float]]:
    """Filter out accounts scoring above the legitimacy threshold."""
    legitimacy_scores = compute_legitimacy_scores(tg)
    legitimate_set = {n for n, s in legitimacy_scores.items() if s >= threshold}
    return legitimate_set, legitimacy_scores


def _find_short_cycle_nodes(tg) -> Set[str]:
    """Find nodes involved in short cycles (length <= 5)."""
    import networkx as nx
    cycle_nodes = set()
    candidates = [
        n for n in tg.G.nodes()
        if tg.in_degree.get(n, 0) >= 1 and tg.out_degree.get(n, 0) >= 1
    ]
    if not candidates:
        return cycle_nodes

    subgraph = tg.G.subgraph(candidates)
    try:
        count = 0
        for cycle in nx.simple_cycles(subgraph, length_bound=5):
            cycle_nodes.update(cycle)
            count += 1
            if count >= 300:
                break
    except Exception:
        pass
    return cycle_nodes
