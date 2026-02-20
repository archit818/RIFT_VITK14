"""
Detection Module 1: Circular Routing (Cycle Detection)
Detects cycles of length 3-5 in the transaction graph.
Uses bounded DFS with degree filtering for performance.
"""

import networkx as nx
from typing import List, Dict, Any
from collections import defaultdict


def detect_circular_routing(tg) -> List[Dict[str, Any]]:
    """
    Detect circular routing patterns (money cycles).
    
    Strategy:
    1. Filter to nodes with degree >= 2 (cycle candidates only).
    2. Use bounded DFS to find simple cycles of length 3-5.
    3. Assign Ring IDs to each detected cycle.
    
    Returns list of detected cycles with metadata.
    """
    results = []
    
    # Filter subgraph to nodes that can participate in cycles
    candidate_nodes = [
        n for n in tg.G.nodes()
        if tg.in_degree.get(n, 0) >= 1 and tg.out_degree.get(n, 0) >= 1
    ]
    
    if not candidate_nodes:
        return results
    
    subgraph = tg.G.subgraph(candidate_nodes)
    
    # Find simple cycles with length bound
    seen_cycles = set()
    cycle_id = 0
    
    try:
        # Use Johnson's algorithm with length limit
        for cycle in nx.simple_cycles(subgraph, length_bound=5):
            if len(cycle) < 3:
                continue
            
            # Normalize cycle for dedup (start from smallest node)
            normalized = _normalize_cycle(cycle)
            cycle_key = tuple(normalized)
            
            if cycle_key in seen_cycles:
                continue
            seen_cycles.add(cycle_key)
            
            # Calculate cycle metadata
            total_amount = 0
            amounts = []
            timestamps = []
            
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if tg.G.has_edge(u, v):
                    edge = tg.G[u][v]
                    total_amount += edge["total_amount"]
                    for tx in edge["transactions"]:
                        amounts.append(tx["amount"])
                        timestamps.append(tx["timestamp"])
            
            # Amount consistency check
            amount_consistency = 0.0
            if amounts:
                mean_amt = sum(amounts) / len(amounts)
                if mean_amt > 0:
                    variance = sum((a - mean_amt) ** 2 for a in amounts) / len(amounts)
                    cv = (variance ** 0.5) / mean_amt
                    amount_consistency = max(0, 1 - cv)
            
            cycle_id += 1
            results.append({
                "ring_id": f"RING-CYC-{cycle_id:04d}",
                "type": "circular_routing",
                "nodes": cycle,
                "cycle_length": len(cycle),
                "total_amount": round(total_amount, 2),
                "amount_consistency": round(amount_consistency, 4),
                "transaction_count": len(amounts),
                "risk_score": _calculate_cycle_risk(len(cycle), amount_consistency, total_amount),
                "explanation": f"Circular fund flow detected through {len(cycle)} accounts with "
                              f"₹{total_amount:,.2f} total and {amount_consistency:.0%} amount consistency."
            })
            
            # Limit cycles to prevent explosion
            if len(results) >= 500:
                break
                
    except Exception:
        # Fallback: manual bounded DFS
        results = _fallback_cycle_detection(tg, subgraph, max_length=5, max_results=500)
    
    return results


def _normalize_cycle(cycle: list) -> list:
    """Normalize cycle to start from the smallest node for dedup."""
    if not cycle:
        return cycle
    min_idx = cycle.index(min(cycle))
    return cycle[min_idx:] + cycle[:min_idx]


def _calculate_cycle_risk(length: int, consistency: float, total_amount: float) -> float:
    """Calculate risk score for a cycle."""
    # Shorter cycles are more suspicious
    length_score = {3: 0.9, 4: 0.7, 5: 0.5}.get(length, 0.3)
    # High consistency = more suspicious
    consistency_score = consistency
    # Higher amounts = more risk (Scale for Rupee)
    amount_score = min(1.0, total_amount / 8000000)
    
    return round(
        length_score * 0.4 + consistency_score * 0.4 + amount_score * 0.2,
        4
    )


def _fallback_cycle_detection(tg, subgraph, max_length=5, max_results=500):
    """Fallback DFS-based cycle detection."""
    results = []
    seen = set()
    cycle_id = 0
    
    for start_node in subgraph.nodes():
        stack = [(start_node, [start_node])]
        while stack and len(results) < max_results:
            node, path = stack.pop()
            for neighbor in subgraph.successors(node):
                if neighbor == start_node and len(path) >= 3:
                    normalized = tuple(_normalize_cycle(path))
                    if normalized not in seen:
                        seen.add(normalized)
                        cycle_id += 1
                        results.append({
                            "ring_id": f"RING-CYC-{cycle_id:04d}",
                            "type": "circular_routing",
                            "nodes": path,
                            "cycle_length": len(path),
                            "total_amount": 0,
                            "amount_consistency": 0,
                            "transaction_count": 0,
                            "risk_score": 0.5,
                            "explanation": f"Cycle of length {len(path)} detected (fallback)."
                        })
                elif neighbor not in path and len(path) < max_length:
                    stack.append((neighbor, path + [neighbor]))
    
    return results
