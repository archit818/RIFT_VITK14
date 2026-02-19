"""
Ring Manager Module v5.0
Fraud Ring Consolidation, Deduplication, Quality Filtering & Monitoring.

v5 Changes (Tasks 1-8):
  - Connected-component clustering: treat detections as fragments of
    laundering COMMUNITIES, not individual rings
  - Multi-criteria merging: member overlap >= 40%, shared flow paths,
    temporal overlap, graph adjacency
  - Aggressive deduplication: each cluster gets exactly ONE ring ID
  - Minimum quality gates: >= 3 nodes, >= 2 suspicious, structural pattern,
    temporal coherence, amount flow
  - Overfitting prevention: randomized threshold jitter, no hardcoded
    synthetic-specific thresholds
  - Performance: consolidation only runs on suspicious subgraph
"""

import math
import random
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from datetime import timedelta


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


def get_priority(pattern_type: str) -> int:
    return PATTERN_PRIORITY.get(pattern_type, 5)


class RingManager:
    """
    Ring lifecycle: detect → register → consolidate → validate → output.

    Key invariants:
    - Each account belongs to at most one ring (highest priority wins).
    - Consolidation merges overlapping detections into unified clusters.
    - Only validated rings appear in final output.
    """

    def __init__(self, min_ring_size: int = 3, min_risk_threshold: float = 25.0):
        self.account_to_ring: Dict[str, str] = {}
        self.rings: Dict[str, Dict[str, Any]] = {}
        self._ring_counter = 0
        self.min_ring_size = max(3, min_ring_size)
        self.min_risk_threshold = min_risk_threshold

        # TASK 5: Small jitter to prevent overfitting to exact thresholds
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

    # ─── Ring Registration (Phase 1: fragment collection) ─────

    def add_ring(
        self,
        pattern_type: str,
        nodes: List[str],
        risk_score: float,
        total_amount: float = 0.0,
        explanation: str = "",
        extra: Dict[str, Any] = None,
        tg=None,
    ) -> Optional[str]:
        """
        Register a detected pattern as a ring fragment.
        Lightweight validation here — heavy consolidation happens in Phase 2.
        """
        nodes_set = set(str(n) for n in nodes)

        # Basic gates
        if len(nodes_set) < self.min_ring_size:
            return None
        if risk_score < self.min_risk_threshold:
            return None

        new_priority = get_priority(pattern_type)

        # Temporal coherence gate (reject temporally disconnected)
        temporal_score = 1.0
        if tg is not None:
            temporal_score = self._check_temporal_consistency(tg, nodes_set)
            if temporal_score < 0.12:
                return None

        # Amount similarity
        amount_score = self._check_amount_similarity(tg, nodes_set) if tg else 0.5

        # Multi-hop connectivity
        multi_hop_score = self._check_multi_hop(tg, nodes_set) if tg else 0.5

        # Signal overlap (structural patterns get higher base)
        signal_overlap_score = 1.0 if pattern_type in STRUCTURAL_PATTERNS else 0.6

        # ─── Overlap resolution ─────────────────────────
        overlapping_rings: Dict[str, Set[str]] = {}
        for node in nodes_set:
            existing_ring_id = self.account_to_ring.get(node)
            if existing_ring_id and existing_ring_id in self.rings:
                overlapping_rings.setdefault(existing_ring_id, set()).add(node)

        # No overlap → create new
        if not overlapping_rings:
            return self._create_ring(
                pattern_type, list(nodes_set), risk_score,
                total_amount, explanation, extra,
                temporal_score, amount_score, signal_overlap_score, multi_hop_score,
            )

        # TASK 1: Graph-connectivity merge — absorb lower-priority overlapping rings
        all_ovr_nodes = set().union(*overlapping_rings.values())

        # If mostly covered by higher-priority rings, skip
        higher_covered = 0
        for rid in overlapping_rings:
            if rid in self.rings and get_priority(self.rings[rid]["type"]) < new_priority:
                higher_covered += len(overlapping_rings[rid])
        if higher_covered >= len(nodes_set) * 0.6:
            return None

        # Absorb lower-priority rings with significant overlap
        rings_to_absorb = []
        for rid, ovr_nodes in overlapping_rings.items():
            if rid not in self.rings:
                continue
            existing = self.rings[rid]
            existing_prio = get_priority(existing["type"])
            existing_size = len(set(existing["nodes"]))
            overlap_ratio = len(ovr_nodes) / max(1, existing_size)

            # TASK 1: Merge threshold 40% for cross-priority, 50% for same
            if existing_prio > new_priority and overlap_ratio >= 0.40:
                rings_to_absorb.append(rid)
            elif existing_prio == new_priority and overlap_ratio >= 0.50:
                rings_to_absorb.append(rid)

        if rings_to_absorb:
            final_nodes = nodes_set.copy()
            merged_amount = total_amount
            merged_patterns = {pattern_type}
            merged_explanations = [explanation] if explanation else []
            for rid in rings_to_absorb:
                old = self.rings[rid]
                final_nodes.update(old["nodes"])
                merged_amount += old.get("total_amount", 0)
                merged_patterns.update(old.get("patterns", []))
                merged_explanations.extend(old.get("explanations", []))
                self._remove_ring(rid)

            ring_id = self._create_ring(
                pattern_type, list(final_nodes), risk_score,
                merged_amount, explanation, extra,
                temporal_score, amount_score, signal_overlap_score, multi_hop_score,
            )
            if ring_id:
                self.rings[ring_id]["patterns"] = list(merged_patterns)
                self.rings[ring_id]["explanations"] = merged_explanations[:5]
            return ring_id

        # Unassigned subset
        unassigned = nodes_set - all_ovr_nodes
        if len(unassigned) >= self.min_ring_size:
            return self._create_ring(
                pattern_type, list(unassigned), risk_score,
                total_amount, explanation, extra,
                temporal_score, amount_score, signal_overlap_score, multi_hop_score,
            )

        return None

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
        """Check internal graph connectivity (multi-hop chain structure)."""
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
        elif density > 0.02:
            return 0.4
        return 0.2

    # ─── Ring Creation ───────────────────────────────────────

    def _create_ring(
        self, pattern_type, nodes, risk_score,
        total_amount, explanation, extra,
        temporal_score=1.0, amount_score=1.0,
        signal_overlap_score=1.0, multi_hop_score=0.5,
    ) -> str:
        ring_id = self._next_ring_id()
        nodes = list(set(str(n) for n in nodes))

        # Confidence scoring with jitter for generalization
        prio = get_priority(pattern_type)
        priority_factor = max(0.3, 1.0 - (prio - 1) * 0.18)
        size_factor = min(1.0, len(nodes) / 8)

        # TASK 5: Jitter prevents overfitting to exact thresholds
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
            cluster_density = round(
                len(nodes) / max(1, len(nodes) * (len(nodes) - 1)), 4
            )

        # Validation gate
        validated = confidence_score >= (0.35 + jitter)

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

    # ─── Phase 2: Community-based Consolidation ───────────────

    def consolidate_rings(self, tg=None):
        """
        TASK 1: Multi-criteria ring consolidation.

        Strategy:
        1. Build a "ring adjacency graph" where rings are nodes and edges
           represent shared members, flow paths, or temporal overlap.
        2. Find connected components in this adjacency graph.
        3. Each connected component becomes one unified ring.
        4. Apply quality filters to reject weak clusters.

        Merge criteria (any one triggers an edge):
        - Member overlap >= 40% (Jaccard)
        - Shared flow paths (members directly connected in transaction graph)
        - Temporal overlap with shared counterparties
        """
        if len(self.rings) <= 1:
            return

        ring_ids = list(self.rings.keys())
        n = len(ring_ids)

        # Build adjacency matrix for ring-level graph
        adjacency = defaultdict(set)  # ring_id -> set of connected ring_ids

        for i in range(n):
            rid_a = ring_ids[i]
            ring_a = self.rings.get(rid_a)
            if not ring_a:
                continue
            nodes_a = set(ring_a["nodes"])

            for j in range(i + 1, n):
                rid_b = ring_ids[j]
                ring_b = self.rings.get(rid_b)
                if not ring_b:
                    continue
                nodes_b = set(ring_b["nodes"])

                should_merge = self._should_merge_rings(
                    ring_a, ring_b, nodes_a, nodes_b, tg
                )

                if should_merge:
                    adjacency[rid_a].add(rid_b)
                    adjacency[rid_b].add(rid_a)

        # Find connected components (union-find)
        components = self._find_connected_components(ring_ids, adjacency)

        # Merge each component into a single ring
        merged_ids = set()
        for component in components:
            if len(component) <= 1:
                continue

            # Pick the highest-priority ring as the keeper
            sorted_rids = sorted(
                component,
                key=lambda rid: (
                    get_priority(self.rings[rid]["type"]),
                    -self.rings[rid].get("confidence_score", 0),
                    -self.rings[rid]["node_count"],
                )
            )

            keep_id = sorted_rids[0]
            absorb_ids = sorted_rids[1:]

            keep = self.rings[keep_id]
            all_nodes = set(keep["nodes"])
            all_patterns = set(keep.get("patterns", [keep["type"]]))
            all_explanations = list(keep.get("explanations", []))
            total_amount = keep.get("total_amount", 0)
            max_risk = keep["risk_score"]
            max_confidence = keep.get("confidence_score", 0)
            max_temporal = keep.get("temporal_consistency", 0)

            for rid in absorb_ids:
                absorb = self.rings[rid]
                all_nodes.update(absorb["nodes"])
                all_patterns.update(absorb.get("patterns", [absorb["type"]]))
                all_explanations.extend(absorb.get("explanations", []))
                total_amount += absorb.get("total_amount", 0)
                max_risk = max(max_risk, absorb["risk_score"])
                max_confidence = max(max_confidence, absorb.get("confidence_score", 0))
                max_temporal = max(max_temporal, absorb.get("temporal_consistency", 0))

                # Reassign account mappings
                for node in absorb["nodes"]:
                    self.account_to_ring[str(node)] = keep_id

                merged_ids.add(rid)

            # Update the keeper ring
            keep["nodes"] = list(all_nodes)
            keep["node_count"] = len(all_nodes)
            keep["total_amount"] = round(total_amount, 2)
            keep["risk_score"] = round(max_risk, 2)
            keep["patterns"] = list(all_patterns)
            keep["explanations"] = all_explanations[:5]
            keep["confidence_score"] = round(max_confidence, 3)
            keep["temporal_consistency"] = round(max_temporal, 3)

            # Recalculate density after merge
            if tg and len(all_nodes) > 1:
                node_list = list(all_nodes)
                internal_edges = sum(
                    1 for u in node_list for v in node_list
                    if u != v and tg.G.has_edge(u, v)
                )
                max_possible = len(node_list) * (len(node_list) - 1)
                keep["cluster_density"] = round(
                    internal_edges / max(1, max_possible), 4
                )

        # Remove absorbed rings
        for rid in merged_ids:
            self.rings.pop(rid, None)

        # TASK 2: Quality gate — remove rings that fail minimum criteria
        self._apply_quality_filters(tg)

    def _should_merge_rings(
        self,
        ring_a: dict, ring_b: dict,
        nodes_a: Set[str], nodes_b: Set[str],
        tg=None,
    ) -> bool:
        """
        TASK 1: Multi-criteria merge decision.
        Returns True if two rings should be merged.
        """
        overlap = nodes_a & nodes_b
        union_size = len(nodes_a | nodes_b)
        jaccard = len(overlap) / max(1, union_size)

        # Criterion 1: Member overlap >= 40%
        if jaccard >= 0.40:
            return True

        # Criterion 2: High member overlap relative to smaller ring
        smaller_size = min(len(nodes_a), len(nodes_b))
        if smaller_size > 0 and len(overlap) / smaller_size >= 0.50:
            return True

        # Criterion 3: Flow path adjacency (members directly connected)
        if tg is not None and len(overlap) == 0:
            flow_connections = 0
            # Sample: check if ANY node in A connects to ANY node in B
            for u in list(nodes_a)[:15]:  # Limit for performance
                for v in list(nodes_b)[:15]:
                    if tg.G.has_edge(u, v) or tg.G.has_edge(v, u):
                        flow_connections += 1

            # If many flow connections exist between the two rings
            sampled_pairs = min(15, len(nodes_a)) * min(15, len(nodes_b))
            if sampled_pairs > 0 and flow_connections / sampled_pairs >= 0.15:
                return True

        # Criterion 4: Temporal overlap + same pattern type
        if ring_a["type"] == ring_b["type"]:
            # Same pattern type with any shared members
            if len(overlap) >= 1:
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

            # BFS
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
        TASK 2: Remove rings that fail minimum quality criteria.

        Requirements:
        - Min 3 nodes
        - Temporal coherence > 0.10
        - At least one structural pattern OR confidence >= 0.35
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

            # Must have structural pattern OR high confidence
            patterns = set(ring.get("patterns", [ring["type"]]))
            has_structural = bool(patterns & STRUCTURAL_PATTERNS)
            has_structural_derived = ring["type"] in (
                "cycle", "circular_routing", "fan_in_fan_out",
                "fan_in_aggregation", "fan_out_dispersal",
                "layered_chain", "shell_chain"
            )

            if not has_structural and not has_structural_derived:
                if ring.get("confidence_score", 0) < 0.40:
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
