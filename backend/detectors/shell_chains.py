"""
Detection Module 4: Layered Shell Chain Detection
Detects paths >= 3 where intermediate accounts have low activity.
These are "pass-through" or "shell" accounts used to layer funds.
"""

import networkx as nx
from typing import List, Dict, Any
from collections import deque


def detect_layering(tg, min_hops: int = 3) -> List[Dict[str, Any]]:
    """
    Detect layering chains: A->B->C->D.
    Each hop passes >60% of received amount forward.
    """
    results = []
    chain_id = 0
    seen_chains = set()

    # Start from any node that has an outgoing transaction
    for start_node in tg.G.nodes():
        # State: (current_node, path, received_amount, last_timestamp)
        # For the source node, we treat "received_amount" as infinite to start
        queue = deque([(start_node, [start_node], float('inf'), None)])
        
        while queue:
            current, path, received_amt, last_ts = queue.popleft()
            
            if len(path) > 8: # Safety limit
                continue
                
            for successor in tg.G.successors(current):
                if successor in path:
                    continue
                
                # Get the transfer amount (sum of all txs)
                edge_data = tg.G[current][successor]
                sent_amt = edge_data["total_amount"]
                
                # Each hop passes >60% of received amount forward
                if received_amt != float('inf') and sent_amt < (received_amt * 0.6):
                    continue
                
                new_path = path + [successor]
                
                if len(new_path) >= min_hops + 1: # hops = len(nodes) - 1
                    chain_key = tuple(new_path)
                    if chain_key not in seen_chains:
                        seen_chains.add(chain_key)
                        chain_id += 1
                        
                        results.append({
                            "ring_id": f"RING-LAY-{chain_id:04d}",
                            "type": "layering",
                            "nodes": new_path,
                            "hops": len(new_path) - 1,
                            "total_amount": round(sent_amt, 2),
                            "risk_score": 0.8,
                            "explanation": f"Layering chain of {len(new_path)-1} hops where each step passes >60% of funds forward."
                        })
                
                # Continue BFS
                queue.append((successor, new_path, sent_amt, None))

    return results


def _calculate_shell_risk(path_length, shell_count, total_amount):
    """Calculate risk score for shell chain."""
    length_score = min(1.0, path_length / 6)
    shell_ratio = shell_count / max(path_length - 2, 1)
    amount_score = min(1.0, total_amount / 16000000)
    
    risk = length_score * 0.3 + shell_ratio * 0.4 + amount_score * 0.3
    return round(min(1.0, risk), 4)
