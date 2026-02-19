"""
Ring Manager Module.
Handles ring priority hierarchy, deduplication, and merging.

Priority order:
  1. cycle           (highest priority)
  2. fan_in_fan_out
  3. layered_chain
  4. community       (lowest priority)

Rules:
  - An account already assigned to a higher-priority ring is NOT reassigned.
  - Rings of the SAME priority merge ONLY if they share ≥50% of nodes (dedup).
  - Higher-priority rings absorb lower-priority rings.
  - Community rings are only created for accounts NOT already in any ring.
"""

from typing import Dict, List, Set, Any, Optional, Tuple


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


def get_priority(pattern_type: str) -> int:
    """Get priority rank for a pattern type (lower = higher priority)."""
    return PATTERN_PRIORITY.get(pattern_type, 5)


class RingManager:
    """
    Manages ring assignments with priority hierarchy and deduplication.
    """

    def __init__(self):
        # account_id → ring_id (each account belongs to at most one ring)
        self.account_to_ring: Dict[str, str] = {}
        # ring_id → ring info dict
        self.rings: Dict[str, Dict[str, Any]] = {}
        # Sequential ID counter
        self._ring_counter = 0

    def _next_ring_id(self, prefix: str) -> str:
        self._ring_counter += 1
        return f"RING_{self._ring_counter:03d}"

    def get_assigned_accounts(self) -> Set[str]:
        """Return set of all accounts already assigned to a ring."""
        return set(self.account_to_ring.keys())

    def is_assigned(self, account_id: str) -> bool:
        return account_id in self.account_to_ring

    def get_account_ring(self, account_id: str) -> Optional[str]:
        return self.account_to_ring.get(account_id)

    def add_ring(
        self,
        pattern_type: str,
        nodes: List[str],
        risk_score: float,
        total_amount: float = 0.0,
        explanation: str = "",
        extra: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Try to add a ring with priority and dedup rules.

        Same-priority merging only if ≥50% overlap (true duplicate).
        Higher-priority absorbs lower-priority.
        """
        new_priority = get_priority(pattern_type)
        nodes_set = set(str(n) for n in nodes)

        if len(nodes_set) < 3:
            return None

        # Check overlap with existing rings
        overlapping_rings: Dict[str, Set[str]] = {}  # ring_id → overlapping nodes
        for node in nodes_set:
            existing_ring = self.account_to_ring.get(node)
            if existing_ring:
                overlapping_rings.setdefault(existing_ring, set()).add(node)

        # Case 1: No overlap → create new ring
        if not overlapping_rings:
            return self._create_ring(
                pattern_type, list(nodes_set), risk_score,
                total_amount, explanation, extra
            )

        # Classify overlapping rings by priority relationship
        already_assigned = set()
        for ovr_nodes in overlapping_rings.values():
            already_assigned |= ovr_nodes

        higher_priority_assigned = set()  # nodes in rings with STRICTLY higher prio
        same_priority_assigned = set()     # nodes in rings with SAME prio
        lower_priority_assigned = set()    # nodes in rings with LOWER prio

        for ring_id, ovr_nodes in overlapping_rings.items():
            existing_ring = self.rings[ring_id]
            existing_priority = get_priority(existing_ring["type"])

            if existing_priority < new_priority:
                higher_priority_assigned |= ovr_nodes
            elif existing_priority == new_priority:
                same_priority_assigned |= ovr_nodes
            else:
                lower_priority_assigned |= ovr_nodes

        unassigned = nodes_set - already_assigned

        # Case 2: All nodes in HIGHER-priority rings → skip entirely
        if higher_priority_assigned >= nodes_set:
            return None

        # Case 3: Same-priority overlap → only merge if it's a TRUE DUPLICATE
        # (≥50% of nodes shared with a single same-priority ring)
        for ring_id, ovr_nodes in overlapping_rings.items():
            existing_ring = self.rings[ring_id]
            existing_priority = get_priority(existing_ring["type"])

            if existing_priority == new_priority:
                existing_nodes = set(existing_ring.get("nodes", []))
                union_size = len(existing_nodes | nodes_set)
                overlap_size = len(existing_nodes & nodes_set)

                # Jaccard overlap ≥ 0.5 → treat as duplicate
                if union_size > 0 and overlap_size / union_size >= 0.5:
                    # Merge: keep the one with higher risk score
                    if risk_score > existing_ring.get("risk_score", 0):
                        # Remove old ring, create new merged one
                        merged_nodes = list(existing_nodes | nodes_set)
                        self._remove_ring(ring_id)
                        return self._create_ring(
                            pattern_type, merged_nodes, risk_score,
                            max(total_amount, existing_ring.get("total_amount", 0)),
                            explanation, extra
                        )
                    else:
                        # Existing ring is stronger — just add unassigned nodes to it
                        for node in unassigned:
                            self.account_to_ring[str(node)] = ring_id
                            existing_ring.setdefault("nodes", []).append(node)
                        existing_ring["node_count"] = len(set(existing_ring["nodes"]))
                        return ring_id

        # Case 4: Higher-priority over lower-priority → absorb lower-priority rings
        if lower_priority_assigned:
            rings_to_absorb = []
            for ring_id, ovr_nodes in overlapping_rings.items():
                existing_ring = self.rings[ring_id]
                existing_priority = get_priority(existing_ring["type"])
                if existing_priority > new_priority:
                    rings_to_absorb.append(ring_id)

            absorbed_nodes = set()
            merged_amount = total_amount
            for ring_id in rings_to_absorb:
                old_ring = self.rings.get(ring_id)
                if old_ring:
                    absorbed_nodes |= set(old_ring.get("nodes", []))
                    merged_amount = max(merged_amount, old_ring.get("total_amount", 0))
                self._remove_ring(ring_id)

            final_nodes = nodes_set | absorbed_nodes
            return self._create_ring(
                pattern_type, list(final_nodes), risk_score,
                merged_amount, explanation, extra
            )

        # Case 5: Partial overlap with same-priority but low overlap (<50%)
        # → DON'T merge. Just assign unassigned nodes as a separate ring.
        if len(unassigned) >= 3:
            return self._create_ring(
                pattern_type, list(unassigned), risk_score,
                total_amount, explanation, extra
            )

        # Not enough unassigned nodes for a standalone ring → skip
        return None

    def _create_ring(
        self, pattern_type, nodes, risk_score,
        total_amount, explanation, extra
    ) -> str:
        ring_id = self._next_ring_id(pattern_type)
        nodes = list(set(str(n) for n in nodes))
        ring = {
            "ring_id": ring_id,
            "type": pattern_type,
            "nodes": nodes,
            "node_count": len(nodes),
            "risk_score": round(risk_score, 2),
            "total_amount": round(total_amount, 2),
            "patterns": [pattern_type],
            "explanations": [explanation] if explanation else [],
        }
        if extra:
            ring.update(extra)
        self.rings[ring_id] = ring

        # Assign accounts
        for node in nodes:
            self.account_to_ring[str(node)] = ring_id

        return ring_id

    def _remove_ring(self, ring_id: str):
        """Remove a ring and unassign its accounts."""
        old_ring = self.rings.pop(ring_id, None)
        if old_ring:
            for node in old_ring.get("nodes", []):
                if self.account_to_ring.get(str(node)) == ring_id:
                    del self.account_to_ring[str(node)]

    def get_rings(self) -> List[Dict[str, Any]]:
        """Return all rings, sorted by risk score descending."""
        rings = list(self.rings.values())
        rings.sort(key=lambda r: r["risk_score"], reverse=True)
        return rings

    def get_ring_for_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get the ring info for an account, if any."""
        ring_id = self.account_to_ring.get(account_id)
        if ring_id:
            return self.rings.get(ring_id)
        return None
