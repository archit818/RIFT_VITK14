"""
Detection Module 4: Layered Shell Chain Detection
Detects paths >= 3 where intermediate accounts have low activity.
These are "pass-through" or "shell" accounts used to layer funds.
"""

import networkx as nx
from typing import List, Dict, Any
from collections import deque


def detect_shell_chains(tg, min_path_length: int = 3, max_path_length: int = 6) -> List[Dict[str, Any]]:
    """
    Detect layered shell chains.
    
    Shell accounts:
    - Low total degree (≤ 2)
    - Low transaction count
    - Act as intermediaries in longer paths
    """
    results = []
    chain_id = 0
    seen_chains = set()
    
    # Identify shell candidates (low activity nodes)
    shell_candidates = set()
    for node in tg.G.nodes():
        total_degree = tg.in_degree.get(node, 0) + tg.out_degree.get(node, 0)
        tx_count = tg.node_temporal.get(node, {}).get("tx_count", 0)
        if total_degree <= 3 and tx_count <= 5:
            shell_candidates.add(node)
    
    if not shell_candidates:
        return results
    
    # Find paths through shell nodes using BFS
    # Start from high-out-degree nodes
    source_nodes = [
        n for n in tg.G.nodes()
        if tg.out_degree.get(n, 0) >= 2 and n not in shell_candidates
    ]
    
    for source in source_nodes:
        # BFS to find paths through shell accounts
        queue = deque([(source, [source], 0)])
        visited = set()
        
        while queue and len(results) < 300:
            current, path, shell_count = queue.popleft()
            
            if len(path) > max_path_length:
                continue
            
            state = (current, len(path))
            if state in visited:
                continue
            visited.add(state)
            
            for successor in tg.G.successors(current):
                if successor in path:
                    continue
                
                new_path = path + [successor]
                new_shell = shell_count + (1 if successor in shell_candidates else 0)
                
                # Check if this is a valid shell chain
                if len(new_path) >= min_path_length + 1:
                    # Count intermediate shell nodes
                    intermediates = new_path[1:-1]
                    shell_intermediates = [n for n in intermediates if n in shell_candidates]
                    
                    if len(shell_intermediates) >= max(1, len(intermediates) // 2):
                        chain_key = tuple(sorted(new_path))
                        if chain_key not in seen_chains:
                            seen_chains.add(chain_key)
                            chain_id += 1
                            
                            # Calculate chain metadata
                            total_amount = 0
                            for i in range(len(new_path) - 1):
                                u, v = new_path[i], new_path[i + 1]
                                if tg.G.has_edge(u, v):
                                    total_amount += tg.G[u][v]["total_amount"]
                            
                            risk = _calculate_shell_risk(
                                len(new_path), len(shell_intermediates), total_amount
                            )
                            
                            results.append({
                                "ring_id": f"RING-SHELL-{chain_id:04d}",
                                "type": "shell_chain",
                                "nodes": new_path,
                                "chain_length": len(new_path),
                                "shell_nodes": shell_intermediates,
                                "shell_count": len(shell_intermediates),
                                "total_amount": round(total_amount, 2),
                                "risk_score": risk,
                                "explanation": (
                                    f"Layered chain of {len(new_path)} accounts with "
                                    f"{len(shell_intermediates)} shell intermediaries. "
                                    f"Total flow: ${total_amount:,.2f}."
                                )
                            })
                
                if len(new_path) <= max_path_length:
                    queue.append((successor, new_path, new_shell))
    
    return results


def _calculate_shell_risk(path_length, shell_count, total_amount):
    """Calculate risk score for shell chain."""
    length_score = min(1.0, path_length / 6)
    shell_ratio = shell_count / max(path_length - 2, 1)
    amount_score = min(1.0, total_amount / 200000)
    
    risk = length_score * 0.3 + shell_ratio * 0.4 + amount_score * 0.3
    return round(min(1.0, risk), 4)
