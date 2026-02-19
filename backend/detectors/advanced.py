"""
Detection Module 9: Amount Consistency in Rings
Detects equal or near-equal transfer amounts within cycles.

Detection Module 10: Counterparty Diversity Shift
Detects sudden increase in unique counterparties.

Detection Module 11: Centrality Spike
Detects nodes becoming sudden hubs (centrality anomalies).

Detection Module 12: Community Suspicion
Cluster-level risk propagation using community detection.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any
from datetime import timedelta
from collections import defaultdict


# ─── Module 9: Amount Consistency in Rings ───────────────────

def detect_amount_consistency(tg, cycles: list) -> List[Dict[str, Any]]:
    """
    Analyze detected cycles for amount consistency.
    Cycles where all transfers are the same amount are highly suspicious.
    """
    results = []
    
    for cycle_info in cycles:
        nodes = cycle_info.get("nodes", [])
        if len(nodes) < 3:
            continue
        
        edge_amounts = []
        for i in range(len(nodes)):
            u = nodes[i]
            v = nodes[(i + 1) % len(nodes)]
            if tg.G.has_edge(u, v):
                for tx in tg.G[u][v]["transactions"]:
                    edge_amounts.append(tx["amount"])
        
        if not edge_amounts:
            continue
        
        mean_amt = np.mean(edge_amounts)
        if mean_amt == 0:
            continue
        
        std_amt = np.std(edge_amounts) if len(edge_amounts) > 1 else 0
        consistency = max(0, 1 - (std_amt / mean_amt))
        
        if consistency >= 0.8:
            results.append({
                "ring_id": cycle_info.get("ring_id", "UNKNOWN"),
                "type": "amount_consistency_ring",
                "nodes": nodes,
                "avg_amount": round(mean_amt, 2),
                "consistency": round(consistency, 4),
                "transaction_count": len(edge_amounts),
                "risk_score": round(consistency * 0.9, 4),
                "explanation": (
                    f"Ring {cycle_info.get('ring_id')} has {consistency:.0%} amount consistency "
                    f"(avg ${mean_amt:,.2f} across {len(edge_amounts)} transactions)."
                )
            })
    
    return results


# ─── Module 10: Counterparty Diversity Shift ──────────────────

def detect_diversity_shift(tg, window_days: int = 7, multiplier: float = 3.0) -> List[Dict[str, Any]]:
    """
    Detect sudden increase in unique counterparties.
    A mule being activated will suddenly interact with many new entities.
    """
    results = []
    window = timedelta(days=window_days)
    
    for node in tg.G.nodes():
        temporal = tg.node_temporal.get(node, {})
        txns = tg.G.nodes[node]["transactions"]
        
        if len(txns) < 5:
            continue
        
        # Split timeline into windows
        txns_sorted = sorted(txns, key=lambda x: x["timestamp"])
        
        if not txns_sorted:
            continue
        
        start = txns_sorted[0]["timestamp"]
        end = txns_sorted[-1]["timestamp"]
        total_span = (end - start).days
        
        if total_span < window_days * 2:
            continue
        
        # Calculate diversity per window
        window_diversities = []
        current_start = start
        
        while current_start + window <= end:
            window_end = current_start + window
            window_txns = [
                t for t in txns_sorted
                if current_start <= t["timestamp"] < window_end
            ]
            unique_cp = len(set(t["counterparty"] for t in window_txns))
            window_diversities.append({
                "start": current_start,
                "diversity": unique_cp,
                "tx_count": len(window_txns),
            })
            current_start += timedelta(days=1)
        
        if len(window_diversities) < 3:
            continue
        
        diversities = [w["diversity"] for w in window_diversities]
        mean_div = np.mean(diversities)
        
        if mean_div == 0:
            continue
        
        # Find max spike
        max_div_window = max(window_diversities, key=lambda x: x["diversity"])
        
        if max_div_window["diversity"] > mean_div * multiplier and max_div_window["diversity"] >= 5:
            spike_ratio = max_div_window["diversity"] / max(mean_div, 0.01)
            
            risk = min(1.0, spike_ratio / (multiplier * 3)) * 0.7 + 0.2
            
            results.append({
                "account_id": node,
                "type": "diversity_shift",
                "baseline_diversity": round(mean_div, 2),
                "spike_diversity": max_div_window["diversity"],
                "spike_ratio": round(spike_ratio, 2),
                "window_start": str(max_div_window["start"]),
                "risk_score": round(risk, 4),
                "explanation": (
                    f"Account {node} counterparty diversity spiked to "
                    f"{max_div_window['diversity']} ({spike_ratio:.1f}x baseline of {mean_div:.1f})."
                )
            })
    
    return results


# ─── Module 11: Centrality Spike ──────────────────────────────

def detect_centrality_spike(tg, top_percentile: float = 0.05) -> List[Dict[str, Any]]:
    """
    Detect nodes with anomalously high centrality.
    These could be newly formed hubs for fund routing.
    """
    results = []
    
    if not tg.betweenness:
        return results
    
    centrality_values = list(tg.betweenness.values())
    if not centrality_values:
        return results
    
    # Find threshold for top percentile
    threshold = np.percentile(centrality_values, (1 - top_percentile) * 100)
    
    if threshold <= 0:
        return results
    
    for node, centrality in tg.betweenness.items():
        if centrality >= threshold and centrality > 0:
            pagerank = tg.pagerank.get(node, 0)
            temporal = tg.node_temporal.get(node, {})
            
            # More suspicious if high centrality but low total transactions
            tx_count = temporal.get("tx_count", 0)
            suspicion_boost = 1.0
            if tx_count < 20:
                suspicion_boost = 1.3  # Low-tx hubs are more suspicious
            
            risk = min(1.0, (centrality / max(threshold, 0.001)) * 0.5 * suspicion_boost)
            
            results.append({
                "account_id": node,
                "type": "centrality_spike",
                "betweenness": round(centrality, 6),
                "pagerank": round(pagerank, 6),
                "threshold": round(threshold, 6),
                "tx_count": tx_count,
                "risk_score": round(risk, 4),
                "explanation": (
                    f"Account {node} has betweenness centrality {centrality:.4f} "
                    f"(top {top_percentile:.0%}, threshold: {threshold:.4f}). "
                    f"PageRank: {pagerank:.4f}."
                )
            })
    
    return results


# ─── Module 12: Community Suspicion ───────────────────────────

def detect_community_suspicion(tg, min_community_size: int = 3) -> List[Dict[str, Any]]:
    """
    Detect suspicious communities using graph clustering.
    Risk propagates within communities: if multiple members
    are flagged, the entire group becomes more suspicious.
    """
    results = []
    
    # Convert to undirected for community detection
    undirected = tg.G.to_undirected()
    
    # Use connected components as base communities
    communities = list(nx.connected_components(undirected))
    
    # Also try Louvain-style greedy modularity
    try:
        louvain_communities = list(nx.community.greedy_modularity_communities(undirected))
        communities = louvain_communities if louvain_communities else communities
    except Exception:
        pass
    
    community_id = 0
    for community in communities:
        if len(community) < min_community_size:
            continue
        
        community_id += 1
        community_nodes = list(community)
        
        # Community-level metrics
        subgraph = tg.G.subgraph(community_nodes)
        internal_edges = subgraph.number_of_edges()
        
        # Density
        max_edges = len(community_nodes) * (len(community_nodes) - 1)
        density = internal_edges / max(max_edges, 1)
        
        # Average centrality
        avg_centrality = np.mean([
            tg.betweenness.get(n, 0) for n in community_nodes
        ])
        
        # Total flow
        total_flow = sum(
            d.get("total_amount", 0)
            for _, _, d in subgraph.edges(data=True)
        )
        
        # Internal flow ratio
        total_out_flow = sum(
            d.get("total_amount", 0)
            for n in community_nodes
            for _, _, d in tg.G.out_edges(n, data=True)
        )
        
        internal_ratio = total_flow / max(total_out_flow, 0.01)
        
        # Risk: dense, isolated communities with high internal flow are suspicious
        risk = min(1.0, density * 0.3 + internal_ratio * 0.3 + min(1.0, avg_centrality * 10) * 0.2 + min(1.0, total_flow / 500000) * 0.2)
        
        if risk >= 0.2:
            results.append({
                "ring_id": f"RING-COMM-{community_id:04d}",
                "type": "community_suspicion",
                "nodes": community_nodes[:50],
                "community_size": len(community_nodes),
                "density": round(density, 4),
                "internal_flow": round(total_flow, 2),
                "internal_ratio": round(internal_ratio, 4),
                "avg_centrality": round(avg_centrality, 6),
                "risk_score": round(risk, 4),
                "explanation": (
                    f"Community of {len(community_nodes)} accounts with density {density:.2%} "
                    f"and ${total_flow:,.2f} internal flow ({internal_ratio:.0%} of total)."
                )
            })
    
    return results
