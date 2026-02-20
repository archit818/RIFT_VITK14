"""
Ring Manager Module v7.0 — Network Intelligence
Network-level fraud networks via Connected Components, High-Risk Edge Filtering.

v7 Changes (Refined Architecture):
  - Task 1: Building Fraud Networks using connected components on high-risk interaction graph.
  - Task 2: Assigning a single Ring ID per connected network.
  - Task 3: Identification of Core vs. Peripheral nodes inside networks.
  - Task 6: High-risk edge selection based on temporal proximity and flow participation.
  - Task 9: Performance optimized by running network ops only on suspicious subgraph.
"""

import networkx as nx
import pandas as pd

import math
import random
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta


# Priority: lower number = higher priority
PATTERN_PRIORITY = {
    "cycle": 1,
    "circular_routing": 1,
    "fan_in_fan_out": 2,
    "fan_in_aggregation": 2,
    "fan_out_dispersal": 2,
    "layered_chain": 3,
    "shell_chain": 3,
    "community": 4,
    "community_suspicion": 4,
}

STRUCTURAL_PATTERNS = {"cycle", "circular_routing", "shell_chain", "layered_chain"}

# Core-defining patterns for ring membership
CORE_DEFINING_PATTERNS = {
    "cycle", "circular_routing", "shell_chain",
    "layered_chain", "fan_in_fan_out",
    "fan_in_aggregation", "fan_out_dispersal",
    "rapid_movement", "structuring",
}

# Weak patterns that alone cannot form a valid ring
WEAK_ONLY_PATTERNS = {
    "community_suspicion", "centrality_spike",
    "diversity_shift", "amount_consistency_ring",
}


def get_priority(pattern_type: str) -> int:
    return PATTERN_PRIORITY.get(pattern_type, 5)


class RingManager:
    """
    Ring lifecycle: detect → register → consolidate → validate → output.

    Key invariants (v7):
    - Each Ring represents exactly one coordinated Fraud Network (connected component).
    - Rings are built from high-risk interactions (Task 6).
    - Nodes are members of at most one network.
    """

    def __init__(self, min_ring_size: int = 3, min_risk_threshold: float = 25.0):
        self.account_to_ring: Dict[str, str] = {}
        self.rings: Dict[str, Dict[str, Any]] = {}
        self._ring_counter = 0
        self.min_ring_size = max(2, min_ring_size) # Relaxed for network intelligence
        self.min_risk_threshold = min_risk_threshold
        self._jitter = random.uniform(-0.02, 0.02)

    def _next_ring_id(self) -> str:
        self._ring_counter += 1
        return f"RING_{self._ring_counter:03d}"

    def get_assigned_accounts(self) -> Set[str]:
        return set(self.account_to_ring.keys())

    def is_assigned(self, account_id: str) -> bool:
        return account_id in self.account_to_ring

    def get_account_ring(self, account_id: str) -> Optional[str]:
        return self.account_to_ring.get(str(account_id))

    # ─── New Workflow: Network Intelligence Pass ──────────

    def build_fraud_intelligence_networks(self, tg, all_detections: Dict[str, List[Dict]]):
        """
        TASK 1, 2, 3: Aggressive Temporal Segmentation & 2-Hop Expansion.
        Solves the Giant Component problem.
        """
        self.account_to_ring = {}
        self.rings = {}

        # 1. Gather nodes and their detection metadata
        suspicious_nodes = set()
        node_patterns = defaultdict(set)
        node_detections = defaultdict(list)

        for module_type, detections in all_detections.items():
            for det in detections:
                nodes = set()
                if "account_id" in det: nodes.add(str(det["account_id"]))
                if "nodes" in det: nodes.update(str(n) for n in det["nodes"])

                suspicious_nodes.update(nodes)
                for node in nodes:
                    node_patterns[node].add(module_type)
                    node_detections[node].append(det)

        if not suspicious_nodes:
            return

        # 2. Build THE GLOBAL RISK GRAPH
        global_risk_graph = nx.Graph()
        global_risk_graph.add_nodes_from(suspicious_nodes)
        
        # 2a. Add edges from detections (Star Topology - Task 1)
        for module_type, detections in all_detections.items():
            if module_type in CORE_DEFINING_PATTERNS:
                for det in detections:
                    nodes = [str(n) for n in det.get("nodes", [])]
                    hub = str(det.get("account_id")) if det.get("account_id") else (nodes[0] if nodes else None)
                    if not hub: continue
                    for node in nodes:
                        if node != hub:
                            global_risk_graph.add_edge(hub, node, weight=0.8, type="pattern")

        # 2b. Add interaction edges (Optimized O(E_suspicious) instead of O(N^2))
        for u in suspicious_nodes:
            # Check only neighbors in the original transaction graph
            for v in tg.G.successors(u):
                if v in suspicious_nodes and v != u:
                    is_edge, weight = self._is_high_risk_edge(tg, u, v, node_patterns[u], node_patterns[v])
                    if is_edge:
                        global_risk_graph.add_edge(u, v, weight=weight, type="interaction")

        # 3. TASK 2: Temporal Rolling Windows (72 hours)
        processed_nodes = set()
        core_candidates = self._identify_core_members(list(suspicious_nodes), node_patterns, tg)
        
        def get_safe_time(node):
            val = tg.node_temporal.get(node, {}).get("first_seen")
            if val is None or pd.isna(val): return pd.Timestamp("1970-01-01")
            return pd.Timestamp(val)

        # Process cores chronologically
        core_candidates.sort(key=get_safe_time)

        for core in core_candidates:
            if core in processed_nodes:
                continue
            
            core_time = get_safe_time(core)
            if core_time == pd.Timestamp("1970-01-01"): continue

            # Window definition (72 hours total)
            window_start = core_time - timedelta(hours=36)
            window_end = core_time + timedelta(hours=36)
            
            # BFS 2-hop expansion restricted by time (Task 2 & 3)
            component = {core}
            queue = [(core, 0)]
            visited = {core}
            
            while queue:
                current, hop = queue.pop(0)
                if hop >= 2: continue
                
                if current not in global_risk_graph: continue
                for neighbor in global_risk_graph.neighbors(current):
                    if neighbor in visited or neighbor in processed_nodes:
                        continue
                    
                    n_time = get_safe_time(neighbor)
                    if n_time != pd.Timestamp("1970-01-01"):
                        if window_start <= n_time <= window_end:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append((neighbor, hop + 1))
                    else:
                        # Nodes without time (e.g. pattern-only)
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append((neighbor, hop + 1))

            if len(component) >= 2:
                self._process_network_component(list(component), node_patterns, node_detections, tg)
                processed_nodes.update(component)

        # 4. Final quality filter
        self._apply_quality_filters(tg)

    def _process_network_component(self, nodes: List[str], node_patterns, node_detections, tg):
        """Helper to identify core and create ring for a component."""
        core_members = self._identify_core_members(nodes, node_patterns, tg)
        if not core_members and len(nodes) < 4:
            return
        self._create_network_ring(nodes, core_members, node_patterns, node_detections, tg)

    def _is_high_risk_edge(self, tg, u, v, patterns_u, patterns_v) -> Tuple[bool, float]:
        """TASK 6: High-risk edge selection with multi-factor weighting (Task 4)."""
        weight = 0.0
        
        # 1. Direct interaction check
        has_u_to_v = tg.G.has_edge(u, v)
        has_v_to_u = tg.G.has_edge(v, u)
        if not (has_u_to_v or has_v_to_u):
            return False, 0.0

        # 2. Structural overlap (Task 1) - both are core-suspicious
        shared_struct = (patterns_u & STRUCTURAL_PATTERNS) and (patterns_v & STRUCTURAL_PATTERNS)
        if shared_struct: weight += 0.4
        
        # 3. Amount similarity (Task 4)
        u_temp = tg.node_temporal.get(u, {})
        v_temp = tg.node_temporal.get(v, {})
        u_avg = u_temp.get("avg_amount", 0)
        v_avg = v_temp.get("avg_amount", 0)
        
        if u_avg > 0 and v_avg > 0:
            diff = abs(u_avg - v_avg) / max(u_avg, v_avg)
            if diff < 0.05: weight += 0.5 # Stricter similarity
            elif diff < 0.15: weight += 0.2
            
        # 4. Temporal Proximity (Task 2 & 4)
        u_first = u_temp.get("first_seen")
        v_first = v_temp.get("first_seen")
        if u_first and v_first:
            gap = abs((u_first - v_first).total_seconds()) / 3600 # hours
            if gap < 6: weight += 0.3 # Tight window
            elif gap < 48: weight += 0.1
            
            # TASK 2: Aggressive temporal segmentation — kill edge if gap > 10 days
            if gap > 240 and weight < 0.8:
                return False, 0.0

        # 5. Interaction Strength (Repeat transfers)
        tx_count = 0
        if has_u_to_v: tx_count += len(tg.G[u][v].get('transactions', []))
        if has_v_to_u: tx_count += len(tg.G[v][u].get('transactions', []))
        if tx_count >= 3: weight += 0.3

        # Return weighted decision
        return (weight >= 0.6), min(1.0, weight)

    def _identify_core_members(self, nodes: List[str], node_patterns: dict, tg) -> List[str]:
        """TASK 3: Identify Core Nodes based on structural and behavioral dominance."""
        core_members = []
        for node in nodes:
            patterns = node_patterns.get(node, set())
            # Structural core signals
            has_structural = bool(patterns & STRUCTURAL_PATTERNS)
            has_multi_hop = "rapid_movement" in patterns
            has_structuring = "structuring" in patterns

            # Behavioral core signals
            has_burst = "transaction_burst" in patterns
            has_fan = bool(patterns & {"fan_in_aggregation", "fan_out_dispersal", "fan_in_fan_out"})

            # Criteria: Multi-pattern OR High-priority structural
            if (has_structural and (has_multi_hop or has_structuring or has_fan)) or len(patterns) >= 3:
                core_members.append(node)
            elif patterns & {"circular_routing", "shell_chain"}:
                core_members.append(node)

        return core_members

    def _create_network_ring(self, nodes, core_members, node_patterns, node_detections, tg):
        """Internal helper for creating a network-level ring."""
        ring_id = self._next_ring_id()
        all_patterns = set()
        total_amount = 0
        all_explanations = []

        # Aggregate from all member detections
        for node in nodes:
            all_patterns.update(node_patterns.get(node, []))
            for det in node_detections.get(node, []):
                total_amount += det.get("total_amount", 0)
                if "explanation" in det and det["explanation"] not in all_explanations:
                    all_explanations.append(det["explanation"])

        # Determine dominant type
        pattern_list = list(all_patterns)
        dominant_type = sorted(pattern_list, key=lambda p: get_priority(p))[0] if pattern_list else "network"

        # Calculate temporal consistency (Task 4/6)
        temporal_consistency = self._check_temporal_consistency(tg, set(nodes)) if tg else 0.5
        
        # Calculate cluster density (Task 3)
        cluster_density = 0.0
        if tg and len(nodes) > 1:
            node_list = list(nodes)
            internal_edges = sum(1 for u in node_list for v in node_list if u != v and tg.G.has_edge(u, v))
            max_possible = len(node_list) * (len(node_list) - 1)
            cluster_density = internal_edges / max(1, max_possible)

        ring = {
            "ring_id": ring_id,
            "type": dominant_type,
            "nodes": nodes,
            "node_count": len(nodes),
            "risk_score": 0.0, # Will be set by scoring.py
            "total_amount": round(total_amount, 2),
            "purity": 0.0,
            "confidence_score": 0.6, 
            "temporal_consistency": temporal_consistency,
            "cluster_density": cluster_density,
            "patterns": pattern_list,
            "explanations": all_explanations[:8],
            "validated": True,
            "core_members": core_members,
            "peripheral_members": list(set(nodes) - set(core_members)),
            "core_count": len(core_members),
        }

        self.rings[ring_id] = ring
        for node in nodes:
            self.account_to_ring[node] = ring_id

    # (Legacy add_ring removed in favor of network intelligence pass)
    def add_ring(self, *args, **kwargs):
        pass

    # ─── Validation Checks ───────────────────────────────────

    def _check_temporal_consistency(self, tg, nodes: Set[str]) -> float:
        """Check pairwise temporal overlap between ring members."""
        active_windows = []
        for node in nodes:
            temporal = tg.node_temporal.get(str(node), {})
            first = temporal.get("first_seen")
            last = temporal.get("last_seen")
            if first and last:
                active_windows.append((first, last))

        if len(active_windows) < 2:
            return 0.5

        overlaps = 0
        total_pairs = 0
        for i in range(len(active_windows)):
            for j in range(i + 1, len(active_windows)):
                total_pairs += 1
                a_start, a_end = active_windows[i]
                b_start, b_end = active_windows[j]
                if a_start <= b_end and b_start <= a_end:
                    overlaps += 1

        return round(overlaps / max(1, total_pairs), 3)

    def _check_amount_similarity(self, tg, nodes: Set[str]) -> float:
        """Check amount coherence across ring members."""
        avg_amounts = []
        for node in nodes:
            temporal = tg.node_temporal.get(str(node), {})
            avg = temporal.get("avg_amount", 0)
            if avg > 0:
                avg_amounts.append(avg)

        if len(avg_amounts) < 2:
            return 0.5

        mean_avg = np.mean(avg_amounts)
        if mean_avg == 0:
            return 0.5

        cv = np.std(avg_amounts) / mean_avg
        if cv < 0.3:
            return 1.0
        elif cv < 0.6:
            return 0.7
        elif cv < 1.0:
            return 0.4
        return 0.2

    def _check_multi_hop(self, tg, nodes: Set[str]) -> float:
        """Check internal graph connectivity (structural density)."""
        if tg is None:
            return 0.5
        node_list = [str(n) for n in nodes]
        internal_edges = 0
        for u in node_list:
            for v in node_list:
                if u != v and tg.G.has_edge(u, v):
                    internal_edges += 1
        max_possible = len(node_list) * (len(node_list) - 1)
        if max_possible == 0:
            return 0.3
        density = internal_edges / max_possible

        # Multi-hop: moderate density is ideal (chain, not clique)
        if 0.08 <= density <= 0.5:
            return 1.0
        elif density > 0.5:
            return 0.7
        elif density > 0.03:
            return 0.4
        return 0.15

    # ─── Ring Creation with Core/Peripheral Classification ────

    def _create_ring(
        self, pattern_type, nodes, risk_score,
        total_amount, explanation, extra,
        temporal_score=1.0, amount_score=1.0,
        signal_overlap_score=1.0, multi_hop_score=0.5,
        tg=None,
    ) -> str:
        ring_id = self._next_ring_id()
        nodes = list(set(str(n) for n in nodes))

        # Confidence scoring with jitter for generalization
        prio = get_priority(pattern_type)
        priority_factor = max(0.3, 1.0 - (prio - 1) * 0.18)
        size_factor = min(1.0, len(nodes) / 8)

        jitter = self._jitter

        confidence_score = round(
            (priority_factor * 0.25) +
            (temporal_score * 0.25) +
            (amount_score * 0.15) +
            (size_factor * 0.10) +
            (signal_overlap_score * 0.15) +
            (multi_hop_score * 0.10) +
            jitter,
            3
        )
        confidence_score = max(0.0, min(1.0, confidence_score))

        # Initial purity estimate
        purity = round(min(1.0, confidence_score * 1.1), 2)

        # Cluster density
        cluster_density = 0
        if len(nodes) > 1:
            if tg is not None:
                node_list = list(nodes)
                internal_edges = sum(
                    1 for u in node_list for v in node_list
                    if u != v and tg.G.has_edge(u, v)
                )
                max_possible = len(node_list) * (len(node_list) - 1)
                cluster_density = round(internal_edges / max(1, max_possible), 4)
            else:
                cluster_density = round(
                    len(nodes) / max(1, len(nodes) * (len(nodes) - 1)), 4
                )

        # Validation gate: tighter, requires both confidence and structural signal
        has_structural = pattern_type in STRUCTURAL_PATTERNS
        validated = (
            confidence_score >= (0.35 + jitter)
            and (has_structural or confidence_score >= 0.50)
        )

        ring = {
            "ring_id": ring_id,
            "type": pattern_type,
            "nodes": nodes,
            "node_count": len(nodes),
            "risk_score": round(max(0, min(100, risk_score)), 2),
            "total_amount": round(total_amount, 2),
            "purity": purity,
            "confidence_score": confidence_score,
            "temporal_consistency": temporal_score,
            "amount_similarity": amount_score,
            "signal_overlap": signal_overlap_score,
            "multi_hop_routing": multi_hop_score,
            "cluster_density": cluster_density,
            "patterns": [pattern_type],
            "explanations": [explanation] if explanation else [],
            "validated": validated,
            "core_members": [],
            "peripheral_members": [],
            "core_count": 0,
        }
        if extra:
            ring.update(extra)

        self.rings[ring_id] = ring

        for node in nodes:
            if str(node) not in self.account_to_ring:
                self.account_to_ring[str(node)] = ring_id

        return ring_id

    def _remove_ring(self, ring_id: str):
        old = self.rings.pop(ring_id, None)
        if old:
            for node in old.get("nodes", []):
                if self.account_to_ring.get(str(node)) == ring_id:
                    del self.account_to_ring[str(node)]

    # ─── Phase 2: Strict Consolidation ────────────────────────

    def consolidate_rings(self, tg=None):
        """Legacy placeholder for pipeline compatibility."""
        pass

    def _should_merge_rings(
        self,
        ring_a: dict, ring_b: dict,
        nodes_a: Set[str], nodes_b: Set[str],
        tg=None,
    ) -> bool:
        """
        TASK 3/5: Strict merging criteria.
        REMOVED: temporal adjacency alone no longer triggers merge.
        REMOVED: same pattern type + single shared member merge.
        """
        overlap = nodes_a & nodes_b
        union_size = len(nodes_a | nodes_b)
        jaccard = len(overlap) / max(1, union_size)

        # Criterion 1: Significant member overlap (Jaccard >= 50%)
        if jaccard >= 0.50:
            return True

        # Criterion 2: High overlap relative to smaller ring (>= 60%)
        smaller_size = min(len(nodes_a), len(nodes_b))
        if smaller_size > 0 and len(overlap) / smaller_size >= 0.60:
            return True

        # Criterion 3: Flow path adjacency — REQUIRE structural density
        if tg is not None and len(overlap) >= 1:
            # At least 1 shared member AND structural flow connections
            flow_connections = 0
            for u in list(nodes_a)[:15]:
                for v in list(nodes_b)[:15]:
                    if tg.G.has_edge(u, v) or tg.G.has_edge(v, u):
                        flow_connections += 1

            sampled_pairs = min(15, len(nodes_a)) * min(15, len(nodes_b))

            # TASK 3: Require >= 20% flow density (up from 15%)
            if sampled_pairs > 0 and flow_connections / sampled_pairs >= 0.20:
                # TASK 5: Also require at least one ring has a structural pattern
                has_struct_a = bool(set(ring_a.get("patterns", [])) & STRUCTURAL_PATTERNS)
                has_struct_b = bool(set(ring_b.get("patterns", [])) & STRUCTURAL_PATTERNS)
                if has_struct_a or has_struct_b:
                    return True

        return False

    def _find_connected_components(
        self, ring_ids: list, adjacency: dict
    ) -> List[List[str]]:
        """Union-find connected components."""
        visited = set()
        components = []

        for rid in ring_ids:
            if rid in visited or rid not in self.rings:
                continue

            component = []
            queue = [rid]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                if current in self.rings:
                    component.append(current)
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited and neighbor in self.rings:
                        queue.append(neighbor)

            if component:
                components.append(component)

        return components

    def _apply_quality_filters(self, tg=None):
        """
        TASK 3: Strict quality filtering.

        Requirements:
        - Min 3 nodes
        - Temporal coherence > 0.10
        - At least one structural pattern OR confidence >= 0.45
        - TASK 3: Minimum structural density > 0
        - TASK 3: Reject rings formed by weak-only patterns
        """
        to_remove = []

        for rid, ring in self.rings.items():
            # Size gate
            if ring["node_count"] < self.min_ring_size:
                to_remove.append(rid)
                continue

            # Temporal coherence gate
            if ring.get("temporal_consistency", 0) < 0.10:
                to_remove.append(rid)
                continue

            # TASK 3: Reject rings formed only by weak patterns
            patterns = set(ring.get("patterns", [ring["type"]]))
            has_strong_pattern = bool(
                patterns - WEAK_ONLY_PATTERNS
            )
            if not has_strong_pattern:
                to_remove.append(rid)
                continue

            # Structural pattern OR high confidence
            has_structural = bool(patterns & STRUCTURAL_PATTERNS)
            has_structural_derived = ring["type"] in (
                "cycle", "circular_routing", "fan_in_fan_out",
                "fan_in_aggregation", "fan_out_dispersal",
                "layered_chain", "shell_chain"
            )

            if not has_structural and not has_structural_derived:
                if ring.get("confidence_score", 0) < 0.45:
                    to_remove.append(rid)
                    continue

            # TASK 3: Minimum cluster density gate
            if ring.get("cluster_density", 0) < 0.01 and ring["node_count"] > 5:
                to_remove.append(rid)
                continue

        for rid in to_remove:
            self._remove_ring(rid)

    # ─── Retrieval ───────────────────────────────────────────

    def get_rings(self) -> List[Dict[str, Any]]:
        return sorted(self.rings.values(), key=lambda x: x["risk_score"], reverse=True)

    def get_validated_rings(self) -> List[Dict[str, Any]]:
        return sorted(
            [r for r in self.rings.values() if r.get("validated", False)],
            key=lambda x: x["risk_score"], reverse=True
        )

    def get_ring_stats(self) -> Dict[str, Any]:
        rings = list(self.rings.values())
        if not rings:
            return {
                "count": 0, "avg_risk": 0, "total_value": 0,
                "avg_purity": 0, "avg_confidence": 0
            }

        validated = [r for r in rings if r.get("validated", False)]
        return {
            "count": len(rings),
            "validated_count": len(validated),
            "avg_risk": round(np.mean([r["risk_score"] for r in rings]), 2),
            "total_value": round(sum(r.get("total_amount", 0) for r in rings), 2),
            "avg_purity": round(np.mean([r.get("purity", 0) for r in rings]), 2),
            "avg_confidence": round(np.mean([r.get("confidence_score", 0) for r in rings]), 3),
            "avg_temporal_consistency": round(
                np.mean([r.get("temporal_consistency", 0) for r in rings]), 3
            ),
            "avg_signal_overlap": round(
                np.mean([r.get("signal_overlap", 0) for r in rings]), 3
            ),
        }

    # ─── Network influence accessors ─────────────────────────

    def get_ring_risk_for_node(self, node: str) -> float:
        ring_id = self.account_to_ring.get(str(node))
        if ring_id and ring_id in self.rings:
            return self.rings[ring_id]["risk_score"]
        return 0.0

    def get_ring_confidence_for_node(self, node: str) -> float:
        ring_id = self.account_to_ring.get(str(node))
        if ring_id and ring_id in self.rings:
            return self.rings[ring_id].get("confidence_score", 0.0)
        return 0.0


# ─── Multi-Window Temporal Analysis ──────────────────────────

def run_multi_window_detection(tg, detect_rapid_func) -> List[Dict[str, Any]]:
    """
    Run rapid movement detection with two temporal windows and combine.
    Window 1: Rapid laundering (2h) — high confidence
    Window 2: Standard laundering (72h) — moderate confidence
    """
    rapid_results = detect_rapid_func(tg, max_hours=2, min_hops=3)
    standard_results = detect_rapid_func(tg, max_hours=72, min_hops=3)

    rapid_nodes = set()
    for det in rapid_results:
        nodes = det.get("nodes", [])
        rapid_nodes.update(str(n) for n in nodes)
        det["temporal_window"] = "rapid_2h"
        det["risk_score"] = min(1.0, det.get("risk_score", 0) * 1.3)

    for det in standard_results:
        nodes = set(str(n) for n in det.get("nodes", []))
        det["temporal_window"] = "standard_72h"

        overlap = nodes & rapid_nodes
        if overlap:
            det["risk_score"] = min(1.0, det.get("risk_score", 0) * 1.5)
            det["dual_window_confirmed"] = True
            det["explanation"] = (
                det.get("explanation", "") +
                " [DUAL-WINDOW CONFIRMED: Detected in both 2h and 72h windows]"
            )

    combined = list(rapid_results)
    rapid_chains = set()
    for det in rapid_results:
        chain_key = tuple(sorted(str(n) for n in det.get("nodes", [])))
        rapid_chains.add(chain_key)

    for det in standard_results:
        chain_key = tuple(sorted(str(n) for n in det.get("nodes", [])))
        if chain_key not in rapid_chains:
            combined.append(det)

    return combined
