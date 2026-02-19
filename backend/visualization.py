"""
PyVis Graph Visualization Generator.
Produces an interactive HTML graph with color-coded nodes and ring highlighting.
"""

from pyvis.network import Network
import os
import json
from typing import List, Dict, Any


def generate_visualization(
    tg,
    suspicious_accounts: List[Dict[str, Any]],
    fraud_rings: List[Dict[str, Any]],
    output_dir: str = "static"
) -> str:
    """
    Generate an interactive PyVis graph visualization.
    
    Node coloring:
    - Red: High risk (≥ 70)
    - Orange: Medium risk (40-70)
    - Yellow: Low risk (10-40)
    - Green: Clean (< 10)
    
    Ring highlighting:
    - Nodes in the same ring share a ring color outline
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build score lookup
    score_map = {a["account_id"]: a for a in suspicious_accounts}
    
    # Ring membership lookup
    ring_membership = {}
    ring_colors_list = [
        "#FF1744", "#D500F9", "#651FFF", "#2979FF", "#00E5FF",
        "#00E676", "#FFEA00", "#FF9100", "#FF3D00", "#C51162",
    ]
    ring_color_map = {}
    for i, ring in enumerate(fraud_rings):
        color = ring_colors_list[i % len(ring_colors_list)]
        ring_color_map[ring["ring_id"]] = color
        for node in ring["nodes"]:
            if node not in ring_membership:
                ring_membership[node] = ring["ring_id"]
    
    # Create PyVis network
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0a0a1a",
        font_color="#ffffff",
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )
    
    # Physics configuration for better layout
    net.set_options(json.dumps({
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09,
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "barnesHut",
            "stabilization": {
                "enabled": True,
                "iterations": 150,
                "updateInterval": 25
            }
        },
        "nodes": {
            "font": {"size": 12, "color": "#ffffff"},
            "borderWidth": 2,
            "borderWidthSelected": 4,
        },
        "edges": {
            "color": {"color": "#444466", "highlight": "#ffffff"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "zoomView": True,
        }
    }))
    
    # Add nodes
    # Limit to top N accounts for readability
    all_nodes = set(tg.G.nodes())
    
    # Always include suspicious accounts and their neighbors
    important_nodes = set()
    for acc in suspicious_accounts[:100]:
        account_id = acc["account_id"]
        important_nodes.add(account_id)
        # Add neighbors
        if account_id in tg.G:
            for neighbor in tg.G.predecessors(account_id):
                important_nodes.add(neighbor)
            for neighbor in tg.G.successors(account_id):
                important_nodes.add(neighbor)
    
    # Add ring nodes
    for ring in fraud_rings[:50]:
        for node in ring["nodes"]:
            important_nodes.add(node)
    
    # If still small, add more
    if len(important_nodes) < 200:
        for node in all_nodes:
            important_nodes.add(node)
            if len(important_nodes) >= 500:
                break
    
    display_nodes = important_nodes
    
    for node in display_nodes:
        acc_info = score_map.get(node)
        score = acc_info["risk_score"] if acc_info else 0
        patterns = acc_info["triggered_patterns"] if acc_info else []
        ring_ids = acc_info["ring_ids"] if acc_info else []
        
        # Color based on risk score
        if score >= 70:
            color = "#FF1744"
            group = "high_risk"
        elif score >= 40:
            color = "#FF9100"
            group = "medium_risk"
        elif score >= 10:
            color = "#FFEA00"
            group = "low_risk"
        else:
            color = "#00E676"
            group = "clean"
        
        # Size based on score
        size = max(10, min(40, 10 + score * 0.3))
        
        # Border color for ring membership
        border_color = ring_color_map.get(
            ring_membership.get(node), color
        )
        
        # Tooltip
        tooltip_lines = [
            f"<b>Account:</b> {node}",
            f"<b>Risk Score:</b> {score:.1f}/100",
        ]
        if patterns:
            tooltip_lines.append(f"<b>Patterns:</b> {', '.join(patterns[:5])}")
        if ring_ids:
            tooltip_lines.append(f"<b>Ring:</b> {', '.join(ring_ids[:3])}")
        
        tooltip = "<br>".join(tooltip_lines)
        
        net.add_node(
            node,
            label=f"{node}\n({score:.0f})",
            title=tooltip,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": "#FFFFFF", "border": border_color}
            },
            size=size,
            group=group,
        )
    
    # Add edges
    for u, v, data in tg.G.edges(data=True):
        if u in display_nodes and v in display_nodes:
            count = data.get("count", 1)
            total = data.get("total_amount", 0)
            
            width = max(1, min(5, count * 0.5))
            
            # Edge color intensity based on amount
            edge_color = "#444466"
            if u in score_map or v in score_map:
                edge_color = "#FF6B6B"
            
            net.add_edge(
                u, v,
                title=f"${total:,.2f} ({count} tx)",
                width=width,
                color=edge_color,
            )
    
    # Save
    output_path = os.path.join(output_dir, "graph.html")
    net.save_graph(output_path)
    
    # Inject custom CSS for dark theme
    _inject_custom_styles(output_path)
    
    return output_path


def _inject_custom_styles(filepath: str):
    """Inject custom dark-theme styles into the generated HTML."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        custom_css = """
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: #0a0a1a;
                overflow: hidden;
            }
            #mynetwork {
                background-color: #0a0a1a !important;
                border: none !important;
            }
            .vis-tooltip {
                background-color: #1a1a2e !important;
                color: #ffffff !important;
                border: 1px solid #333366 !important;
                border-radius: 8px !important;
                padding: 10px !important;
                font-family: 'Inter', sans-serif !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
            }
        </style>
        """
        
        content = content.replace("</head>", custom_css + "</head>")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass
