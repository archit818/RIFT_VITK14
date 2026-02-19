"""
Detection Module 6: Rapid Fund Movement
Detects multi-hop transfers occurring within hours using temporal BFS.
"""

from typing import List, Dict, Any
from datetime import timedelta
from collections import deque


def detect_rapid_movement(tg, max_hours: int = 4, min_hops: int = 3) -> List[Dict[str, Any]]:
    """
    Detect rapid multi-hop fund movement.
    
    Uses temporal BFS: follow edges where each successive transaction
    occurs within max_hours of the previous one.
    """
    results = []
    chain_id = 0
    seen_chains = set()
    
    # Start from nodes with both in and out
    candidate_starts = [
        n for n in tg.G.nodes()
        if tg.out_degree.get(n, 0) >= 1
    ]
    
    time_window = timedelta(hours=max_hours)
    
    for start in candidate_starts:
        # Temporal BFS
        # State: (node, path, last_timestamp, total_amount)
        out_txns = [
            t for t in tg.G.nodes[start]["transactions"]
            if t["type"] == "out"
        ]
        
        for initial_tx in out_txns:
            queue = deque([(
                initial_tx["counterparty"],
                [start, initial_tx["counterparty"]],
                initial_tx["timestamp"],
                initial_tx["amount"],
                [initial_tx["amount"]]
            )])
            
            visited = set()
            
            while queue and len(results) < 200:
                current, path, last_ts, total_amt, amounts = queue.popleft()
                
                state_key = (current, len(path))
                if state_key in visited:
                    continue
                visited.add(state_key)
                
                if len(path) > 8:
                    continue
                
                # Check outgoing from current within time window
                out_from_current = [
                    t for t in tg.G.nodes[current]["transactions"]
                    if t["type"] == "out"
                    and t["counterparty"] not in path
                    and timedelta(0) <= (t["timestamp"] - last_ts) <= time_window
                ]
                
                for tx in out_from_current:
                    new_path = path + [tx["counterparty"]]
                    new_amounts = amounts + [tx["amount"]]
                    
                    if len(new_path) >= min_hops + 1:
                        chain_key = tuple(sorted(new_path))
                        if chain_key not in seen_chains:
                            seen_chains.add(chain_key)
                            chain_id += 1
                            
                            # Amount preservation ratio
                            preservation = min(new_amounts) / max(max(new_amounts), 0.01)
                            
                            risk = _calculate_rapid_risk(
                                len(new_path), preservation, sum(new_amounts)
                            )
                            
                            results.append({
                                "ring_id": f"RING-RAPID-{chain_id:04d}",
                                "type": "rapid_movement",
                                "nodes": new_path,
                                "hop_count": len(new_path) - 1,
                                "total_amount": round(sum(new_amounts), 2),
                                "amount_preservation": round(preservation, 4),
                                "time_span_hours": max_hours,
                                "risk_score": risk,
                                "explanation": (
                                    f"Rapid {len(new_path)-1}-hop fund movement within {max_hours}h. "
                                    f"Amount preservation: {preservation:.0%}."
                                )
                            })
                    
                    if len(new_path) <= 8:
                        queue.append((
                            tx["counterparty"], new_path,
                            tx["timestamp"], total_amt + tx["amount"],
                            new_amounts
                        ))
    
    return results


def _calculate_rapid_risk(hops, preservation, total_amount):
    hop_score = min(1.0, hops / 6)
    pres_score = preservation  # Higher preservation = more suspicious
    amt_score = min(1.0, total_amount / 200000)
    return round(hop_score * 0.35 + pres_score * 0.35 + amt_score * 0.3, 4)
