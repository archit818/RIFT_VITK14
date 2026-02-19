"""
Graph Construction Engine.
Converts validated transaction DataFrame into a directed weighted temporal graph.
Precomputes node features for detection modules.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from collections import defaultdict


class TransactionGraph:
    """
    Encapsulates the transaction graph with precomputed features.
    Serves as the central data structure for all detection modules.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.G = nx.DiGraph()
        self._build_graph()
        self._precompute_features()

    def _build_graph(self):
        """Build directed graph from transaction DataFrame."""
        for _, row in self.df.iterrows():
            sender = str(row["sender_id"])
            receiver = str(row["receiver_id"])
            amount = float(row["amount"])
            ts = row["timestamp"]

            # Add nodes if not present
            if not self.G.has_node(sender):
                self.G.add_node(sender, transactions=[])
            if not self.G.has_node(receiver):
                self.G.add_node(receiver, transactions=[])

            # Append transaction metadata to node
            self.G.nodes[sender]["transactions"].append({
                "type": "out", "counterparty": receiver,
                "amount": amount, "timestamp": ts
            })
            self.G.nodes[receiver]["transactions"].append({
                "type": "in", "counterparty": sender,
                "amount": amount, "timestamp": ts
            })

            # Add or update edge
            if self.G.has_edge(sender, receiver):
                edge_data = self.G[sender][receiver]
                edge_data["transactions"].append({
                    "amount": amount, "timestamp": ts
                })
                edge_data["total_amount"] += amount
                edge_data["count"] += 1
            else:
                self.G.add_edge(sender, receiver,
                                transactions=[{"amount": amount, "timestamp": ts}],
                                total_amount=amount,
                                count=1)

    def _precompute_features(self):
        """Precompute node-level features for efficient detection."""
        # Degree features
        self.in_degree = dict(self.G.in_degree())
        self.out_degree = dict(self.G.out_degree())

        # Betweenness centrality (approximate for large graphs)
        n = self.G.number_of_nodes()
        if n > 500:
            k = min(100, n)
            self.betweenness = nx.betweenness_centrality(self.G, k=k)
        else:
            self.betweenness = nx.betweenness_centrality(self.G)

        # PageRank
        try:
            self.pagerank = nx.pagerank(self.G, max_iter=100)
        except Exception:
            self.pagerank = {node: 1.0 / n for node in self.G.nodes()}

        # Clustering coefficient (undirected version)
        undirected = self.G.to_undirected()
        self.clustering = nx.clustering(undirected)

        # Temporal features per node
        self.node_temporal = {}
        for node in self.G.nodes():
            txns = self.G.nodes[node]["transactions"]
            if txns:
                timestamps = sorted([t["timestamp"] for t in txns])
                amounts = [t["amount"] for t in txns]
                self.node_temporal[node] = {
                    "first_seen": timestamps[0],
                    "last_seen": timestamps[-1],
                    "tx_count": len(txns),
                    "total_amount": sum(amounts),
                    "avg_amount": np.mean(amounts),
                    "std_amount": np.std(amounts) if len(amounts) > 1 else 0,
                    "in_count": sum(1 for t in txns if t["type"] == "in"),
                    "out_count": sum(1 for t in txns if t["type"] == "out"),
                    "unique_counterparties": len(set(t["counterparty"] for t in txns)),
                    "timestamps": timestamps,
                    "amounts": amounts,
                }
            else:
                self.node_temporal[node] = {
                    "first_seen": None, "last_seen": None,
                    "tx_count": 0, "total_amount": 0, "avg_amount": 0,
                    "std_amount": 0, "in_count": 0, "out_count": 0,
                    "unique_counterparties": 0, "timestamps": [], "amounts": [],
                }

        # Precompute degree sets for efficient filtering
        self.high_degree_nodes = set(
            n for n in self.G.nodes()
            if self.in_degree.get(n, 0) + self.out_degree.get(n, 0) >= 2
        )
        self.low_degree_nodes = set(
            n for n in self.G.nodes()
            if self.in_degree.get(n, 0) + self.out_degree.get(n, 0) <= 2
        )

        # Precompute edge temporal index
        self.edge_temporal = {}
        for u, v, data in self.G.edges(data=True):
            self.edge_temporal[(u, v)] = sorted(
                data["transactions"], key=lambda x: x["timestamp"]
            )

    def get_node_features(self, node: str) -> Dict[str, Any]:
        """Get all precomputed features for a node."""
        return {
            "in_degree": self.in_degree.get(node, 0),
            "out_degree": self.out_degree.get(node, 0),
            "betweenness": self.betweenness.get(node, 0),
            "pagerank": self.pagerank.get(node, 0),
            "clustering": self.clustering.get(node, 0),
            **self.node_temporal.get(node, {}),
        }

    def get_nodes(self):
        return list(self.G.nodes())

    def get_edges(self):
        return list(self.G.edges(data=True))

    def get_subgraph(self, nodes):
        return self.G.subgraph(nodes)

    @property
    def node_count(self):
        return self.G.number_of_nodes()

    @property
    def edge_count(self):
        return self.G.number_of_edges()
