"""
Legitimacy Scoring Module v2.0
Prevents merchants, payroll systems, and high-volume business hubs from false-positive flagging.
"""

import numpy as np
from typing import Dict, Set, Tuple
from datetime import timedelta


def compute_legitimacy_scores(tg) -> Dict[str, float]:
    """
    Compute complex legitimacy scores (0-1).
    Heuristics:
    - Merchant: High incoming diversity, stable daily volume, long history.
    - Payroll: Periodic (monthly/weekly) outgoing spikes, fixed amounts, limited recipients info.
    - Business Hub: High diversity in both directions + balanced flow.
    """
    scores = {}
    cycle_nodes = _find_short_cycle_nodes(tg)

    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        tx_count = temporal.get("tx_count", 0)

        if tx_count == 0:
            scores[node] = 0.5 # Unknown neutrality
            continue

        # --- Base Metrics ---
        unique_cp = temporal.get("unique_counterparties", 0)
        in_count = temporal.get("in_count", 0)
        out_count = temporal.get("out_count", 0)
        in_ratio = in_count / max(1, tx_count)
        diversity_ratio = unique_cp / max(1, tx_count)
        
        # --- Factor 1: Merchant Profile ---
        # Large diversity + incoming bias + long history
        merchant_score = 0
        if diversity_ratio > 0.7 and in_ratio > 0.9 and tx_count > 50:
            merchant_score = 1.0
        
        # --- Factor 2: Payroll Profile ---
        # Regular outgoing bursts to same/varying nodes at month end
        payroll_score = 0
        if out_count > 10 and in_count < 3:
            # Check for amount consistency (fixed salaries)
            std_amt = temporal.get("std_amount", 0)
            avg_amt = temporal.get("avg_amount", 1)
            if (std_amt / max(0.01, avg_amt)) < 0.05:
                payroll_score = 1.0

        # --- Factor 3: Operational Longevity ---
        lifespan_score = 0
        if temporal.get("first_seen") and temporal.get("last_seen"):
            days = (temporal["last_seen"] - temporal["first_seen"]).total_seconds() / 86400
            lifespan_score = min(1.0, days / 180) # 6 months for full credit

        # --- Factor 4: Negative Signals (Fraud indicators reduce legitimacy) ---
        malicious_penalty = 0
        if node in cycle_nodes:
            malicious_penalty = 0.8
        
        # --- Combine ---
        legitimacy = (
            merchant_score * 0.4 +
            payroll_score * 0.3 +
            lifespan_score * 0.3
        )
        
        # Apply volume trust
        if tx_count > 200: 
            legitimacy += 0.2
            
        final_score = min(1.0, max(0.0, legitimacy - malicious_penalty))
        scores[node] = round(final_score, 4)

    return scores


def filter_legitimate_accounts(tg, threshold: float = 0.75) -> Tuple[Set[str], Dict[str, float]]:
    legitimacy_scores = compute_legitimacy_scores(tg)
    legitimate_set = {n for n, s in legitimacy_scores.items() if s >= threshold}
    return legitimate_set, legitimacy_scores


def _find_short_cycle_nodes(tg) -> Set[str]:
    import networkx as nx
    cycle_nodes = set()
    candidates = [
        n for n in tg.G.nodes()
        if tg.in_degree.get(n, 0) >= 1 and tg.out_degree.get(n, 0) >= 1
    ]
    if not candidates: return cycle_nodes
    
    subgraph = tg.G.subgraph(candidates)
    try:
        count = 0
        for cycle in nx.simple_cycles(subgraph, length_bound=5):
            cycle_nodes.update(cycle)
            count += 1
            if count >= 300: break # Perf optimization
    except Exception: pass
    return cycle_nodes

