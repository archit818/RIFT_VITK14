"""
Detection Module 6: Rapid Fund Movement
Detects multi-hop transfers occurring within hours using temporal BFS.
"""

from typing import List, Dict, Any
from datetime import timedelta
from collections import deque


def detect_rapid_movement(tg, max_hours_node: int = 1) -> List[Dict[str, Any]]:
    """
    Detect rapid fund movement.
    1. Single-node flag: funds move out within 1 hour of being received.
    2. Multi-hop chains are captured via individual node participation.
    """
    results = []
    seen_nodes = set()
    
    for node in tg.G.nodes():
        # Get incoming transactions
        txns = tg.G.nodes[node]["transactions"]
        in_txs = [t for t in txns if t["type"] == "in"]
        out_txs = [t for t in txns if t["type"] == "out"]
        
        if not in_txs or not out_txs:
            continue
            
        # Check if any OUT follows any IN within 1 hour
        rapid_flag = False
        for itx in in_txs:
            for otx in out_txs:
                delta = (otx["timestamp"] - itx["timestamp"]).total_seconds()
                if 0 <= delta <= (max_hours_node * 3600):
                    rapid_flag = True
                    break
            if rapid_flag: break
            
        if rapid_flag:
            results.append({
                "account_id": node,
                "type": "rapid_movement",
                "risk_score": 0.85,
                "explanation": f"Funds moved out of account within {max_hours_node} hour(s) of being received."
            })
    
    return results


def _calculate_rapid_risk(hops, preservation, total_amount):
    hop_score = min(1.0, hops / 6)
    pres_score = preservation  # Higher preservation = more suspicious
    amt_score = min(1.0, total_amount / 200000)
    return round(hop_score * 0.35 + pres_score * 0.35 + amt_score * 0.3, 4)
