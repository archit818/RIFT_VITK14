"""
Legitimacy Scoring Module.
Computes a legitimacy score for each account BEFORE detection.
Accounts exceeding the threshold are excluded from the fraud detection graph,
preventing merchants, payroll, and stable business accounts from being flagged.
"""

import numpy as np
from typing import Dict, Set, Tuple


def compute_legitimacy_scores(tg) -> Dict[str, float]:
    """
    Compute a 0-1 legitimacy score for each account.
    Higher = more legitimate.

    Criteria:
    - high transaction count (consistent activity)
    - many unique counterparties (diverse business)
    - no short cycles (not part of circular routing)
    - transactions spread over time (long lifespan)
    - consistent amounts (regular payments)
    """
    scores = {}

    # Pre-check which nodes participate in short cycles (3-5)
    cycle_nodes = _find_short_cycle_nodes(tg)

    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        tx_count = temporal.get("tx_count", 0)

        if tx_count == 0:
            scores[node] = 0.0
            continue

        # --- Factor 1: Transaction volume (high = legitimate) ---
        # Normalized: 50+ txns → 1.0
        volume_score = min(1.0, tx_count / 50.0)

        # --- Factor 2: Counterparty diversity (high = legitimate) ---
        unique_cp = temporal.get("unique_counterparties", 0)
        diversity_ratio = unique_cp / max(tx_count, 1)
        # High diversity of counterparties → more likely a real business
        diversity_score = min(1.0, diversity_ratio * 2.0)

        # --- Factor 3: No short cycles (absence = legitimate) ---
        cycle_score = 0.0 if node in cycle_nodes else 1.0

        # --- Factor 4: Temporal spread (long lifespan = legitimate) ---
        first_seen = temporal.get("first_seen")
        last_seen = temporal.get("last_seen")
        if first_seen and last_seen:
            lifespan_days = (last_seen - first_seen).days
            # 60+ days of activity → 1.0
            lifespan_score = min(1.0, lifespan_days / 60.0)
        else:
            lifespan_score = 0.0

        # --- Factor 5: Amount consistency (low variance = regular business) ---
        avg_amount = temporal.get("avg_amount", 0)
        std_amount = temporal.get("std_amount", 0)
        if avg_amount > 0:
            cv = std_amount / avg_amount
            # CV < 0.3 → very consistent → 1.0
            consistency_score = max(0.0, 1.0 - cv / 0.5)
        else:
            consistency_score = 0.0

        # --- Combine ---
        legitimacy = (
            volume_score * 0.20
            + diversity_score * 0.25
            + cycle_score * 0.25
            + lifespan_score * 0.15
            + consistency_score * 0.15
        )
        scores[node] = round(min(1.0, max(0.0, legitimacy)), 4)

    return scores


def filter_legitimate_accounts(
    tg,
    threshold: float = 0.70
) -> Tuple[Set[str], Dict[str, float]]:
    """
    Returns:
        legitimate_set: set of account IDs considered legitimate (excluded from detection)
        legitimacy_scores: full dict of account_id → legitimacy score
    """
    legitimacy_scores = compute_legitimacy_scores(tg)

    legitimate_set = set()
    for account, score in legitimacy_scores.items():
        if score >= threshold:
            legitimate_set.add(account)

    return legitimate_set, legitimacy_scores


def _find_short_cycle_nodes(tg) -> Set[str]:
    """
    Quick check for nodes in short cycles (length 3-5).
    Uses limited DFS — not full enumeration, just membership check.
    """
    import networkx as nx

    cycle_nodes = set()

    # Only check nodes with both in/out edges (can participate in cycles)
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
            if len(cycle) >= 3:
                cycle_nodes.update(cycle)
            count += 1
            if count >= 500:
                break
    except Exception:
        pass

    return cycle_nodes
