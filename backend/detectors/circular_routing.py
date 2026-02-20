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
    
    # Find simple cycles with length bound 3-6
    seen_cycles = set()
    cycle_id = 0
    
    try:
        # Use simple_cycles with length_bound=6 as per CRITICAL requirement
        for cycle in nx.simple_cycles(subgraph, length_bound=6):
            if len(cycle) < 3:
                continue
            
            # REJECT any ring with >50 members (indicates graph connectivity bug)
            if len(cycle) > 50:
                continue

            # Normalize cycle for dedup
            normalized = _normalize_cycle(cycle)
            cycle_key = tuple(normalized)
            
            if cycle_key in seen_cycles:
                continue
            
            # --- Strict Forensic Validation ---
            # Sequential Timestamps + Amount Consistency
            
            valid_cycle = True
            current_cycle_txs = []
            
            # Start search from first edge
            u_start, v_start = cycle[0], cycle[1]
            first_edge_txs = sorted(tg.G[u_start][v_start]["transactions"], key=lambda x: x["timestamp"])
            
            # We try to find ANY sequential path through the cycle
            # Cycle can start at any node, so check all cyclic shifts
            valid_cycle_overall = False
            total_vol = 0
            best_explanation = ""
            
            for start_idx in range(len(cycle)):
                shifted_cycle = cycle[start_idx:] + cycle[:start_idx]
                
                valid_cycle = True
                current_cycle_txs = []
                prev_ts = None
                cycle_amounts = []
                cycle_start_ts = None
                
                for i in range(len(shifted_cycle)):
                    u = shifted_cycle[i]
                    v = shifted_cycle[(i + 1) % len(shifted_cycle)]
                    
                    edge_txs = sorted(tg.G[u][v]["transactions"], key=lambda x: x["timestamp"])
                    
                    best_tx = None
                    if prev_ts is None:
                        best_tx = edge_txs[0] if edge_txs else None
                    else:
                        for tx in edge_txs:
                            if tx["timestamp"] > prev_ts:
                                best_tx = tx
                                break
                    
                    if not best_tx:
                        valid_cycle = False
                        break
                    
                    if i == 0:
                        cycle_start_ts = best_tx["timestamp"]
                    elif (best_tx["timestamp"] - cycle_start_ts).total_seconds() > 259200:
                        valid_cycle = False
                        break

                    prev_ts = best_tx["timestamp"]
                    current_cycle_txs.append(best_tx)
                    cycle_amounts.append(best_tx["amount"])

                if not valid_cycle:
                    continue

                for j in range(1, len(cycle_amounts)):
                    prev_amt = cycle_amounts[j-1]
                    curr_amt = cycle_amounts[j]
                    if abs(curr_amt - prev_amt) / max(prev_amt, 1) > 0.20:
                        valid_cycle = False
                        break

                if valid_cycle:
                    valid_cycle_overall = True
                    total_vol = sum(cycle_amounts)
                    best_explanation = (
                        f"Validated {len(cycle)}-node cycle. Funds moved sequentially through nodes "
                        f"within {round((prev_ts - cycle_start_ts).total_seconds()/3600, 1)} hours."
                    )
                    break

            if not valid_cycle_overall:
                continue

            seen_cycles.add(cycle_key)
            cycle_id += 1
            
            total_vol = sum(cycle_amounts)
            
            results.append({
                "ring_id": f"RING-CYC-{cycle_id:04d}",
                "type": "cycle",
                "nodes": cycle,
                "cycle_length": len(cycle),
                "total_amount": round(total_vol, 2),
                "risk_score": _calculate_cycle_risk(len(cycle), 0.95, total_vol),
                "explanation": (
                    f"Validated {len(cycle)}-node cycle. Funds moved sequentially through nodes "
                    f"within {round((prev_ts - cycle_start_ts).total_seconds()/3600, 1)} hours."
                )
            })
            
            if len(results) >= 500:
                break
                
    except Exception:
        # Fallback manual DFS still respects bounds
        results = _fallback_cycle_detection(tg, subgraph, max_length=6, max_results=500)
    
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
