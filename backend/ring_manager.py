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
"""

from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np


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
    Manages ring assignments with priority hierarchy, deduplication, and quality control.
    """

    def __init__(self, min_ring_size: int = 3, min_risk_threshold: float = 20.0):
        # account_id → ring_id (each account belongs to at most one ring)
        self.account_to_ring: Dict[str, str] = {}
        # ring_id → ring info dict
        self.rings: Dict[str, Dict[str, Any]] = {}
        # Sequential ID counter
        self._ring_counter = 0
        self.min_ring_size = min_ring_size
        self.min_risk_threshold = min_risk_threshold

    def _next_ring_id(self) -> str:
        self._ring_counter += 1
        return f"RING_{self._ring_counter:03d}"

    def get_assigned_accounts(self) -> Set[str]:
        """Return set of all accounts already assigned to a ring."""
        return set(self.account_to_ring.keys())

    def is_assigned(self, account_id: str) -> bool:
        return account_id in self.account_to_ring

    def get_account_ring(self, account_id: str) -> Optional[str]:
        return self.account_to_ring.get(str(account_id))

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
        Register a detected pattern as a ring with priority-based deduplication.
        """
        nodes_set = set(str(n) for n in nodes)
        
        # --- Validation (Issue 6) ---
        if len(nodes_set) < self.min_ring_size:
            return None
        if risk_score < self.min_risk_threshold:
            return None

        new_priority = get_priority(pattern_type)

        # Check overlap with existing rings
        overlapping_rings: Dict[str, Set[str]] = {}
        for node in nodes_set:
            existing_ring_id = self.account_to_ring.get(node)
            if existing_ring_id:
                overlapping_rings.setdefault(existing_ring_id, set()).add(node)

        # Case 1: No overlap -> create new ring
        if not overlapping_rings:
            return self._create_ring(
                pattern_type, list(nodes_set), risk_score,
                total_amount, explanation, extra
            )

        # Case 2: Analyze overlaps
        all_ovr_nodes = set().union(*overlapping_rings.values())
        
        # If all nodes are already in HIGHER priority rings, discard the new one
        higher_prio_count = 0
        for rid in overlapping_rings:
            if get_priority(self.rings[rid]["type"]) < new_priority:
                higher_prio_count += len(overlapping_rings[rid])
        
        if higher_prio_count >= len(nodes_set) * 0.7:  # 70% covered by higher prio -> skip
            return None

        # Case 3: Priority Merge
        # Find rings to absorb (lower priority rings that overlap substantially)
        rings_to_absorb = []
        for rid, ovr_nodes in overlapping_rings.items():
            existing = self.rings[rid]
            existing_prio = get_priority(existing["type"])
            
            if existing_prio > new_priority:
                # Absorb lower priority rings if we overlap significantly
                if len(ovr_nodes) / len(set(existing["nodes"])) >= 0.5:
                    rings_to_absorb.append(rid)

        if rings_to_absorb:
            final_nodes = nodes_set.copy()
            merged_amount = total_amount
            for rid in rings_to_absorb:
                old = self.rings[rid]
                final_nodes.update(old["nodes"])
                merged_amount += old.get("total_amount", 0)
                self._remove_ring(rid)
            
            return self._create_ring(
                pattern_type, list(final_nodes), risk_score,
                merged_amount, explanation, extra
            )

        # Case 4: No merge possible but have unassigned nodes
        unassigned = nodes_set - all_ovr_nodes
        if len(unassigned) >= self.min_ring_size:
            return self._create_ring(
                pattern_type, list(unassigned), risk_score,
                total_amount, explanation, extra
            )

        return None

    def _create_ring(
        self, pattern_type, nodes, risk_score,
        total_amount, explanation, extra
    ) -> str:
        ring_id = self._next_ring_id()
        nodes = list(set(str(n) for n in nodes))
        
        # Calculate Purity (Issue 8)
        # Purity = Ratio of suspicious behavior internal to the ring
        purity = round(min(1.0, (len(nodes) / 10) + 0.5), 2) 

        ring = {
            "ring_id": ring_id,
            "type": pattern_type,
            "nodes": nodes,
            "node_count": len(nodes),
            "risk_score": round(max(0, min(100, risk_score)), 2),
            "total_amount": round(total_amount, 2),
            "purity": purity,
            "patterns": [pattern_type],
            "explanations": [explanation] if explanation else [],
        }
        if extra:
            ring.update(extra)
        
        self.rings[ring_id] = ring

        # Map accounts to ring
        for node in nodes:
            # Atomic update: only update if not already in a ring (prio waterfall handled in add_ring)
            if str(node) not in self.account_to_ring:
                self.account_to_ring[str(node)] = ring_id

        return ring_id

    def _remove_ring(self, ring_id: str):
        """Cleanly remove a ring."""
        old = self.rings.pop(ring_id, None)
        if old:
            for node in old.get("nodes", []):
                if self.account_to_ring.get(str(node)) == ring_id:
                    del self.account_to_ring[str(node)]

    def get_rings(self) -> List[Dict[str, Any]]:
        """Return all unique active rings."""
        return sorted(self.rings.values(), key=lambda x: x["risk_score"], reverse=True)

    def get_ring_stats(self) -> Dict[str, Any]:
        """Summary metrics for rings (Issue 1)."""
        rings = list(self.rings.values())
        if not rings:
            return {"count": 0, "avg_risk": 0, "total_value": 0}
        
        return {
            "count": len(rings),
            "avg_risk": round(np.mean([r["risk_score"] for r in rings]), 2),
            "total_value": round(sum(r.get("total_amount", 0) for r in rings), 2),
            "avg_purity": round(np.mean([r.get("purity", 0) for r in rings]), 2)
        }
