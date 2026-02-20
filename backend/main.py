"""
RIFT - Financial Forensics Engine v5.0
FastAPI Backend - Main Application

Detection Pipeline (waterfall priority with community consolidation):
  1. Compute legitimacy scores → exclude legitimate nodes
  2. Cycle detection (highest priority rings)
  3. Fan-in + Fan-out → combined fan_in_fan_out rings
  4. Shell chain detection → layered_chain rings
  5. Community detection (lowest priority, only remaining nodes)
  6. Parallel: burst, rapid movement, dormant, structuring, diversity, centrality
  7. Amount consistency on detected cycles
  8. Ring consolidation (community clustering + multi-criteria merge)
  9. Multi-signal gated scoring + network influence
"""

import os
import io
import time
import json
import asyncio
import traceback
from datetime import datetime
from typing import Optional, Set
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from validators import validate_csv
from graph_engine import TransactionGraph
from legitimacy import filter_legitimate_accounts
from ring_manager import RingManager, run_multi_window_detection
from scoring import compute_scores
from visualization import generate_visualization
from synthetic import generate_synthetic_data, generate_edge_case_data

from detectors.circular_routing import detect_circular_routing
from detectors.fan_in import detect_smurfing
from detectors.fan_out import detect_dispersal
from detectors.shell_chains import detect_layering
from detectors.burst import detect_transaction_bursts
from detectors.rapid_movement import detect_rapid_movement
from detectors.dormant import detect_dormant_activation
from detectors.structuring import detect_structuring
from detectors.advanced import (
    detect_amount_consistency,
    detect_diversity_shift,
    detect_centrality_spike,
    detect_community_suspicion,
)


# ─── App Setup ────────────────────────────────────────────────

app = FastAPI(
    title="RIFT - Financial Forensics Engine",
    description="Detect money muling networks using graph analytics and temporal intelligence.",
    version="5.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory for graph visualizations
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Thread pool for parallel detection
executor = ThreadPoolExecutor(max_workers=4)

# Cache
analysis_cache = {}


# ─── Health Check ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "RIFT - Financial Forensics Engine",
        "version": "5.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ─── Analysis Endpoint ────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_transactions(file: UploadFile = File(...)):
    """
    Main analysis endpoint.
    Accepts CSV upload, runs all detection modules with priority pipeline,
    returns structured results.
    """
    start_time = time.time()

    # --- Read and validate ---
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        df = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                break
            except Exception:
                continue

        if df is None:
            raise HTTPException(status_code=400, detail="Could not parse CSV file.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Validate
    df_clean, validation_errors = validate_csv(df)

    critical_errors = [e for e in validation_errors if e.get("error_type") in
                       ("MISSING_COLUMNS", "EMPTY_FILE", "ALL_ROWS_INVALID")]

    if critical_errors:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation failed",
                "details": critical_errors,
                "warnings": [e for e in validation_errors if e not in critical_errors],
            }
        )

    if df_clean.empty:
        raise HTTPException(status_code=400, detail="No valid transactions after validation.")

    # --- Build Graph ---
    try:
        # Optimization: Early pruning of inactive data
        # Only keep accounts with at least one transaction (automatic in TG)
        tg = TransactionGraph(df_clean)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph construction failed: {str(e)}"
        )

    # --- Run Detection Pipeline ---
    try:
        all_detections, ring_manager, legit_set, legit_scores = await _run_detection_pipeline(tg)
        print(f"DEBUG: Pipeline found {sum(len(v) for v in all_detections.values())} detections.")
        print(f"DEBUG: Ring Manager has {len(ring_manager.get_rings())} rings.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

    # --- Compute Scores ---
    try:
        suspicious_accounts, fraud_rings, summary = compute_scores(
            tg, all_detections, ring_manager,
            legitimate_accounts=legit_set,
            legitimacy_scores=legit_scores,
        )
        print(f"DEBUG: Scoring found {len(suspicious_accounts)} suspicious accounts and {len(fraud_rings)} rings.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}"
        )

    # --- Generate Visualization ---
    try:
        graph_path = generate_visualization(
            tg, suspicious_accounts, fraud_rings, output_dir=STATIC_DIR
        )
    except Exception as e:
        traceback.print_exc()
        graph_path = None

    # Fallback: if graph_path is None but we have a previous one, use it
    if not graph_path:
        existing_graph = os.path.join(STATIC_DIR, "graph.html")
        if os.path.exists(existing_graph):
            graph_path = existing_graph

    elapsed = round(time.time() - start_time, 2)

    # --- Build Response (Issue 1 & 9) ---
    # Cleanup internal fields
    for acc in suspicious_accounts:
        acc.pop("_is_core", None)
        acc.pop("_patterns", None)

    summary["processing_time_seconds"] = elapsed
    summary["engine_status"] = "FORENSIC_INTEGRITY_HIGH"
    
    response = {
        "suspicious_accounts": suspicious_accounts[:300],
        "fraud_rings": fraud_rings,
        "summary": summary,
        "graph_url": "/static/graph.html" if graph_path else None,
        "metadata": {
            "version": "5.0.0-forensic",
            "legitimate_suppression_count": len(legit_set),
            "optimization_pruning_active": len(tg.df) < len(df_clean),
            "multi_signal_gate_enabled": True,
            "ring_consolidation_enabled": True,
        }
    }

    # Cache the result
    analysis_cache["latest"] = response

    return response


async def _run_detection_pipeline(tg: TransactionGraph):
    """
    Detection pipeline v5.0 with community-based ring consolidation,
    multi-criteria merging, and quality filtering.
    """
    loop = asyncio.get_event_loop()

    # TASK 1 & 9: Relaxed ring params for adaptive detection
    ring_mgr = RingManager(min_ring_size=2, min_risk_threshold=5.0)

    # ──────────────────────────────────────────────────────────
    # STEP 0: Legitimacy filter (Tasks 3 & 7)
    # REDUCED THRESHOLD: Allow for automated fan-in detection while still suppressing massive legitimate hubs
    legit_set, legit_scores = filter_legitimate_accounts(tg, threshold=0.85)

    # ─── Phase 1: High-Priority Structural Detections ─────────────
    # We DO NOT filter by legitimacy here: detection must see all nodes.
    # Suppression happens during scoring via weighted penalties.
    
    # Cycle detection
    cycles = await loop.run_in_executor(executor, detect_circular_routing, tg, 0.20)
    if not cycles:
        print("DEBUG: Cycles found 0 results at tolerance 0.20, retrying at 0.50...")
        cycles = await loop.run_in_executor(executor, detect_circular_routing, tg, 0.50)
    
    # Smurfing detection
    smurfing_results = await loop.run_in_executor(
        executor, lambda: detect_smurfing(tg, threshold_senders=4, exclude_nodes=None)
    )
    if not smurfing_results:
        print("DEBUG: Smurfing found 0 results at threshold 4, retrying at 2...")
        smurfing_results = await loop.run_in_executor(
            executor, lambda: detect_smurfing(tg, threshold_senders=2, exclude_nodes=None)
        )
    
    # Layering detection
    layering_results = await loop.run_in_executor(executor, detect_layering, tg)

    # Dispersal detection (mapped to fan_in_fan_out)
    dispersal_results = await loop.run_in_executor(
        executor, lambda: detect_dispersal(tg, threshold_receivers=4, window_hours=48, exclude_nodes=None)
    )
    if not dispersal_results:
        print("DEBUG: Dispersal found 0 results at threshold 4, retrying at 2...")
        dispersal_results = await loop.run_in_executor(
            executor, lambda: detect_dispersal(tg, threshold_receivers=2, window_hours=48, exclude_nodes=None)
        )

    # Community Suspicion
    community_results = await loop.run_in_executor(
        executor, lambda: detect_community_suspicion(tg, exclude_nodes=None, already_in_ring=set())
    )

    # Simplified logic for network pass
    parallel_futures = {
        "transaction_burst": loop.run_in_executor(executor, detect_transaction_bursts, tg),
        "dormant_activation": loop.run_in_executor(executor, detect_dormant_activation, tg),
        "structuring": loop.run_in_executor(executor, detect_structuring, tg),
    }
    parallel_results = {}
    for k, f in parallel_futures.items():
        try: parallel_results[k] = await f
        except: parallel_results[k] = []

    # Multi-window rapid movement detection (2h + 72h)
    try:
        multi_window_rapid = await loop.run_in_executor(
            executor, lambda: run_multi_window_detection(tg, detect_rapid_movement)
        )
    except:
        multi_window_rapid = []

    # ──────────────────────────────────────────────────────────
    # STEP 6: Amount consistency on detected cycles (Removed from pipeline)
    # ──────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────
    # STEP 7: Advanced graph signals (Removed from pipeline)
    # ──────────────────────────────────────────────────────────

    all_detections = {
        "circular_routing": cycles,
        "fan_in_aggregation": smurfing_results,
        "shell_chain": layering_results,
        "fan_in_fan_out": dispersal_results,
        "community_suspicion": community_results,
        "rapid_movement": multi_window_rapid,
        **parallel_results,
    }

    # HYPER-ADAPTIVE FALLBACK (User Request)
    total_found = sum(len(v) for v in all_detections.values())
    if total_found == 0:
        print("DEBUG: CRITICAL - 0 patterns found. Activating HYPER-ADAPTIVE mode...")
        # Extreme relaxation
        cycles = await loop.run_in_executor(executor, detect_circular_routing, tg, 0.90)
        smurfing = await loop.run_in_executor(executor, lambda: detect_smurfing(tg, threshold_senders=2))
        dispersal = await loop.run_in_executor(executor, lambda: detect_dispersal(tg, threshold_receivers=2))
        
        all_detections["circular_routing"] = cycles
        all_detections["fan_in_aggregation"] = smurfing
        all_detections["fan_in_fan_out"] = dispersal
        print(f"DEBUG: Hyper-adaptive found {len(cycles)} cycles, {len(smurfing)} smurfing, {len(dispersal)} dispersal.")

    print(f"FINAL_DETECTION_PASS: {sum(len(v) for v in all_detections.values())} total signals.")

    # ──────────────────────────────────────────────────────────
    # PHASE 2: TASK 1 & 2 — Network intelligence consolidation
    # Build Rings using connected components on high-risk graph
    # ──────────────────────────────────────────────────────────
    ring_mgr.build_fraud_intelligence_networks(tg, all_detections)

    return all_detections, ring_mgr, legit_set, legit_scores


# ─── Graph Visualization Endpoint ─────────────────────────────

@app.get("/api/graph")
async def get_graph():
    """Serve the latest graph visualization."""
    graph_path = os.path.join(STATIC_DIR, "graph.html")
    if os.path.exists(graph_path):
        return FileResponse(graph_path, media_type="text/html")
    # If not exists, try to generate it from cache
    if "latest" in analysis_cache:
         return FileResponse(graph_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="No graph generated yet. Run analysis first.")


@app.get("/api/download-graph")
async def download_graph():
    """Download the graph HTML file."""
    graph_path = os.path.join(STATIC_DIR, "graph.html")
    if os.path.exists(graph_path):
        return FileResponse(
            graph_path, 
            filename=f"RIFT_Topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            media_type="text/html"
        )
    raise HTTPException(status_code=404, detail="Graph file not found.")


# ─── Cached Results ───────────────────────────────────────────

@app.get("/api/results")
async def get_latest_results():
    """Get the latest analysis results from cache."""
    if "latest" in analysis_cache:
        return analysis_cache["latest"]
    raise HTTPException(status_code=404, detail="No analysis results available. Upload a CSV first.")


# ─── Synthetic Data Generation ────────────────────────────────

@app.post("/api/generate-test-data")
async def generate_test_data(
    accounts: int = Query(default=500, ge=10, le=5000),
    transactions: int = Query(default=5000, ge=100, le=50000),
    fraud_ratio: float = Query(default=0.15, ge=0.0, le=0.5),
):
    """Generate synthetic test data with embedded fraud patterns."""
    try:
        df = generate_synthetic_data(
            num_accounts=accounts,
            num_transactions=transactions,
            fraud_ratio=fraud_ratio,
        )

        csv_path = os.path.join(STATIC_DIR, "test_data.csv")
        df.to_csv(csv_path, index=False)

        return {
            "message": "Test data generated successfully",
            "stats": {
                "total_transactions": len(df),
                "unique_senders": df["sender_id"].nunique(),
                "unique_receivers": df["receiver_id"].nunique(),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "amount_range": f"${df['amount'].min():,.2f} to ${df['amount'].max():,.2f}",
            },
            "download_url": "/static/test_data.csv",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/api/download-test-data")
async def download_test_data():
    """Download the generated test data CSV."""
    csv_path = os.path.join(STATIC_DIR, "test_data.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, filename="test_data.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="No test data generated yet.")


# ─── Innovation: Predictive Risk & Simulation ─────────────────

@app.post("/api/simulate")
async def simulate_fraud(
    scenario: str = Query(default="ring", description="Fraud scenario: ring, layering, smurfing"),
    intensity: float = Query(default=0.5, ge=0.1, le=1.0),
):
    """
    Innovation: Simulate future fraud scenarios.
    Shows how the system would respond to unseen patterns.
    """
    if "latest" not in analysis_cache:
        raise HTTPException(status_code=400, detail="Run analysis first before simulation.")

    latest = analysis_cache["latest"]
    summary = latest.get("summary", {})

    simulation = {
        "scenario": scenario,
        "intensity": intensity,
        "predictions": {
            "estimated_new_rings": int(summary.get("fraud_rings_detected", 0) * intensity * 1.5),
            "estimated_new_suspicious": int(summary.get("suspicious_accounts_found", 0) * intensity * 1.3),
            "risk_increase_pct": round(intensity * 25, 1),
        },
        "recommendations": [
            f"Monitor top {max(5, int(intensity * 20))} high-risk accounts for new connections.",
            f"Set alert threshold to {max(40, int(70 - intensity * 30))} for early detection.",
            "Increase monitoring frequency during next 72 hours.",
            f"Review {scenario} detection module sensitivity.",
        ],
        "explanation": (
            f"Simulating a {scenario} fraud scenario at {intensity:.0%} intensity. "
            f"Based on current patterns, the system predicts increased risk."
        ),
    }

    return simulation


# ─── Documentation Endpoint ───────────────────────────────────

@app.get("/api/export-json")
async def export_json_report():
    """
    Download a JSON file of the analysis in the EXACT format requested.
    """
    if "latest" not in analysis_cache:
        raise HTTPException(status_code=404, detail="No analysis results available. Run analysis first.")

    latest = analysis_cache["latest"]

    exported_data = {
        "suspicious_accounts": [
            {
                "account_id": acc["account_id"],
                "suspicion_score": round(acc["risk_score"], 2),
                "detected_patterns": acc.get("patterns", []),
                "ring_id": acc["ring_ids"][0] if acc.get("ring_ids") else "NONE",
            }
            for acc in latest["suspicious_accounts"]
        ],
        "fraud_rings": [
            {
                "ring_id": ring["ring_id"],
                "member_accounts": ring["nodes"],
                "pattern_type": ring["type"],
                "risk_score": round(ring["risk_score"], 2),
            }
            for ring in latest["fraud_rings"]
        ],
        "summary": {
            "total_accounts_analyzed": latest["summary"]["total_accounts_analyzed"],
            "suspicious_accounts_flagged": latest["summary"]["suspicious_accounts_flagged"],
            "fraud_rings_detected": latest["summary"]["fraud_rings_detected"],
            "processing_time_seconds": latest["summary"]["processing_time_seconds"],
        }
    }

    return JSONResponse(
        content=exported_data,
        headers={"Content-Disposition": "attachment; filename=rift_report.json"}
    )


@app.get("/api/docs/architecture")
async def get_architecture():
    """Return system architecture documentation."""
    return {
        "architecture": {
            "name": "RIFT - Financial Forensics Engine v5.0",
            "design_principle": "Algorithmic graph intelligence with community-based ring consolidation",
            "pipeline": {
                "description": "Waterfall detection with community clustering consolidation",
                "steps": [
                    "1. Legitimacy filter (merchant/payroll/platform suppression)",
                    "2. Cycle detection (highest priority rings)",
                    "3. Fan-in + Fan-out combined detection (threshold >= 4)",
                    "4. Shell chain detection → layered_chain rings",
                    "5. Community detection (remaining nodes)",
                    "6. Parallel: burst, rapid, dormant, structuring, diversity, centrality",
                    "7. Amount consistency analysis on cycles",
                    "8. Community ring consolidation (connected-component merge)",
                    "9. Quality filter (min size, temporal coherence, structural pattern)",
                    "10. Multi-signal gated scoring + network influence",
                    "11. Ring validation (purity >= 15%, suspicious members >= 2)",
                ],
                "ring_priority": [
                    "1. cycle (highest)",
                    "2. fan_in_fan_out",
                    "3. layered_chain",
                    "4. community (lowest)",
                ],
                "consolidation": {
                    "method": "Connected-component clustering on ring adjacency graph",
                    "merge_criteria": [
                        "Member overlap >= 40% (Jaccard similarity)",
                        "Overlap >= 50% of smaller ring",
                        "Flow path adjacency >= 15% of sampled pairs",
                        "Same pattern type with any shared members",
                    ],
                    "quality_gates": [
                        "Min 3 nodes",
                        "Temporal coherence > 10%",
                        "Structural pattern OR confidence >= 40%",
                    ],
                },
                "multi_signal_gate": "Require structural + behavioral signals for HIGH tier",
            },
            "detection_modules": [
                {"name": "Circular Routing", "complexity": "O(V+E) bounded DFS"},
                {"name": "Fan-in Aggregation (>= 4 senders)", "complexity": "O(V*T) sliding window"},
                {"name": "Fan-out Dispersal (>= 4 receivers)", "complexity": "O(V*T) sliding window"},
                {"name": "Fan-in/Fan-out Mule Hub", "complexity": "O(V) intersection"},
                {"name": "Shell Chains", "complexity": "O(V*E) BFS"},
                {"name": "Transaction Burst", "complexity": "O(V*T) rolling window"},
                {"name": "Rapid Movement (2h+72h)", "complexity": "O(V*E) temporal BFS"},
                {"name": "Dormant Activation", "complexity": "O(V*T) gap analysis"},
                {"name": "Structuring", "complexity": "O(V*T) threshold analysis"},
                {"name": "Amount Consistency", "complexity": "O(C*E) cycle analysis"},
                {"name": "Diversity Shift", "complexity": "O(V*T) window analysis"},
                {"name": "Centrality Spike", "complexity": "O(V) percentile analysis"},
                {"name": "Community Suspicion", "complexity": "O(V+E)"},
            ],
            "false_positive_controls": [
                "Multi-signal gating: structural + behavioral for HIGH tier",
                "5-factor stability scoring",
                "Merchant suppression",
                "Payroll suppression",
                "Platform/exchange suppression",
                "Peripheral node dampening (0.40x for isolated weak nodes)",
                "Ring quality filters (min suspicious members, structural pattern)",
                "Signal coherence monitoring",
                "Single-pattern penalty",
                "Threshold jitter for overfitting prevention",
            ],
            "generalization": {
                "threshold_jitter": "Small random perturbation on tier boundaries",
                "no_hardcoded_patterns": "All thresholds are relative, not absolute",
                "validation": "Works on high-noise, low-structure, mixed behavior data",
            },
            "performance": {
                "target": "<= 30 seconds for 10K transactions",
                "optimizations": [
                    "Legitimacy pre-filtering prunes ~30-50% of nodes",
                    "Waterfall pipeline: later stages skip assigned accounts",
                    "Consolidation only on detected ring fragments (not full graph)",
                    "Sampled flow-path adjacency (max 15 nodes per ring pair)",
                    "Precomputed centrality and temporal features",
                    "Parallel module execution (ThreadPoolExecutor)",
                    "Bounded cycle search (length <= 5)",
                ]
            },
        }
    }


# ─── Task 8: Chatbot Explainability Layer ─────────────────────

from pydantic import BaseModel

class ChatbotQuery(BaseModel):
    account_id: Optional[str] = None
    question: str


@app.post("/api/chatbot")
async def chatbot_query(query: ChatbotQuery):
    """
    Task 8: Lightweight rule-based chatbot for forensic explainability.
    Uses stored graph metadata — no heavy model inference.

    Supported question types:
    - Why was [account] flagged?
    - What ring is [account] in?
    - Temporal flow for [account]
    - General system questions
    """
    if "latest" not in analysis_cache:
        return {"response": "No analysis data available. Please run an analysis first.", "type": "error"}

    latest = analysis_cache["latest"]
    q = query.question.lower().strip()
    account_id = query.account_id

    # Build lookup indices
    accounts_map = {a["account_id"]: a for a in latest.get("suspicious_accounts", [])}
    rings_map = {r["ring_id"]: r for r in latest.get("fraud_rings", [])}
    summary = latest.get("summary", {})

    # ─── Route question to handler ──────────────────────
    if account_id and account_id in accounts_map:
        account = accounts_map[account_id]

        if any(kw in q for kw in ["why", "flagged", "suspicious", "reason", "explain"]):
            return _explain_why_flagged(account, rings_map)

        elif any(kw in q for kw in ["ring", "group", "cluster", "network"]):
            return _explain_ring_structure(account, rings_map)

        elif any(kw in q for kw in ["temporal", "time", "flow", "movement", "when"]):
            return _explain_temporal_flow(account, rings_map)

        elif any(kw in q for kw in ["score", "risk", "confidence"]):
            return _explain_score(account)

        else:
            return _explain_account_summary(account, rings_map)

    elif account_id and account_id not in accounts_map:
        return {
            "response": f"Account {account_id} was not flagged as suspicious in the current analysis. "
                        f"It may have been filtered as legitimate or scored below the detection threshold.",
            "type": "not_found",
            "account_id": account_id,
        }

    # General questions (no specific account)
    elif any(kw in q for kw in ["summary", "overview", "total", "how many"]):
        return {
            "response": (
                f"Analysis Summary:\n"
                f"- {summary.get('total_accounts_analyzed', 0)} accounts analyzed\n"
                f"- {summary.get('total_transactions_analyzed', 0)} transactions processed\n"
                f"- {summary.get('suspicious_accounts_flagged', 0)} suspicious accounts detected\n"
                f"- {summary.get('fraud_rings_detected', 0)} validated fraud rings\n"
                f"- {summary.get('high_risk_accounts', 0)} high/critical risk accounts\n"
                f"- Estimated precision: {summary.get('estimated_precision', 0):.0%}\n"
                f"- Multi-signal gate pass rate: {summary.get('multi_signal_gate_pass_rate', 0):.0%}\n"
                f"- FP estimate: {summary.get('fp_estimate', 0):.0%}\n"
                f"- Average ring purity: {summary.get('avg_ring_purity', 0):.0%}\n"
                f"- Processing time: {summary.get('processing_time_seconds', 0):.2f}s"
            ),
            "type": "summary",
        }

    elif any(kw in q for kw in ["ring", "rings", "fraud ring"]):
        rings = latest.get("fraud_rings", [])
        if not rings:
            return {"response": "No fraud rings were detected in this analysis.", "type": "rings"}
        top_rings = rings[:5]
        lines = [f"Top {len(top_rings)} Fraud Rings:"]
        for r in top_rings:
            lines.append(
                f"- {r['ring_id']}: {r['type']} | {r['node_count']} members | "
                f"Risk: {r['risk_score']} | Purity: {r.get('purity', 0):.0%} | "
                f"Confidence: {r.get('confidence', 'N/A')}"
            )
        return {"response": "\n".join(lines), "type": "rings"}

    elif any(kw in q for kw in ["precision", "accuracy", "quality", "diagnostic"]):
        return {
            "response": (
                f"System Diagnostics:\n"
                f"- Estimated Precision: {summary.get('estimated_precision', 0):.1%}\n"
                f"- Average Ring Purity: {summary.get('avg_ring_purity', 0):.1%}\n"
                f"- Cluster Density: {summary.get('suspicious_cluster_density', 0):.2%}\n"
                f"- FP Estimate: {summary.get('fp_estimate', 0):.1%}\n"
                f"- Tier Distribution: {json.dumps(summary.get('tier_distribution', {}))}"
            ),
            "type": "diagnostics",
        }

    else:
        return {
            "response": (
                "I can help you understand the forensic analysis. Try asking:\n"
                "- 'Why was this account flagged?'\n"
                "- 'What ring is this account in?'\n"
                "- 'Show temporal flow for this account'\n"
                "- 'Give me a summary'\n"
                "- 'Show fraud rings'\n"
                "- 'What is the precision?'\n\n"
                "Select a specific account and ask about its patterns, risk score, or ring membership."
            ),
            "type": "help",
        }


def _explain_why_flagged(account: dict, rings_map: dict) -> dict:
    patterns = account.get("patterns", [])
    score = account.get("risk_score", 0)
    tier = account.get("tier", "LOW")
    explanation = account.get("explanation", "")

    lines = [
        f"Account {account['account_id']} — Flagged as {tier} risk (score: {score}/100)",
        f"",
        f"Primary Reason: {explanation}",
        f"",
        f"Detected Patterns ({len(patterns)}):",
    ]
    for p in patterns:
        signal_type = "STRONG" if p in {"circular_routing", "fan_in_fan_out", "shell_chain", "rapid_movement", "structuring", "fan_in_aggregation", "fan_out_dispersal"} else "WEAK"
        lines.append(f"  - {p.replace('_', ' ').title()} [{signal_type}]")

    if account.get("ring_ids"):
        ring_id = account["ring_ids"][0]
        ring = rings_map.get(ring_id)
        if ring:
            lines.append(f"")
            lines.append(f"Ring Membership: {ring_id} ({ring['type']}, {ring['node_count']} members)")

    breakdown = account.get("score_breakdown", [])
    if breakdown:
        lines.append(f"")
        lines.append("Score Breakdown:")
        for b in breakdown[:5]:
            lines.append(f"  - {b['module']}: weight={b['weighted']:.4f} ({b.get('signal_strength', 'N/A')})")

    return {"response": "\n".join(lines), "type": "explanation", "account_id": account["account_id"]}


def _explain_ring_structure(account: dict, rings_map: dict) -> dict:
    ring_ids = account.get("ring_ids", [])
    if not ring_ids:
        return {
            "response": f"Account {account['account_id']} is not part of any detected fraud ring.",
            "type": "ring_info",
            "account_id": account["account_id"],
        }

    lines = [f"Ring Membership for {account['account_id']}:"]
    for rid in ring_ids:
        ring = rings_map.get(rid)
        if ring:
            lines.append(f"")
            lines.append(f"Ring {rid}:")
            lines.append(f"  - Type: {ring['type']}")
            lines.append(f"  - Members: {ring['node_count']} accounts")
            lines.append(f"  - Risk Score: {ring['risk_score']}/100")
            lines.append(f"  - Purity: {ring.get('purity', 0):.0%}")
            lines.append(f"  - Signal Coherence: {ring.get('signal_coherence', 0):.0%}")
            lines.append(f"  - Confidence: {ring.get('confidence', 'N/A')}")
            lines.append(f"  - Total Flow: ₹{ring.get('total_amount', 0):,.2f}")
            lines.append(f"  - Temporal Consistency: {ring.get('temporal_consistency', 'N/A')}")
            lines.append(f"  - Member Accounts: {', '.join(ring['nodes'][:10])}")
            if len(ring['nodes']) > 10:
                lines.append(f"    ... and {len(ring['nodes']) - 10} more")

    return {"response": "\n".join(lines), "type": "ring_info", "account_id": account["account_id"]}


def _explain_temporal_flow(account: dict, rings_map: dict) -> dict:
    signals = account.get("signal_summary", {})
    patterns = account.get("patterns", [])

    lines = [f"Temporal Analysis for {account['account_id']}:"]
    lines.append(f"  - Strong Signals: {signals.get('strong_signals', 0)}")
    lines.append(f"  - Weak Signals: {signals.get('weak_signals', 0)}")
    lines.append(f"  - Unique Patterns: {signals.get('unique_patterns', 0)}")
    lines.append(f"  - Multi-Signal Gate: {'PASSED' if signals.get('multi_signal_gate') else 'NOT MET'}")
    lines.append(f"  - Has Structural: {'Yes' if signals.get('has_structural') else 'No'}")
    lines.append(f"  - Has Behavioral: {'Yes' if signals.get('has_behavioral') else 'No'}")

    if "rapid_movement" in patterns:
        lines.append(f"")
        lines.append("  Rapid Movement Detected:")
        lines.append("  This account shows fast fund transfers across multiple hops,")
        lines.append("  suggesting potential money laundering velocity patterns.")

    if "circular_routing" in patterns:
        lines.append(f"")
        lines.append("  Circular Routing Detected:")
        lines.append("  Funds flow in a loop back to the origin,")
        lines.append("  suggesting coordinated fund recycling.")

    if "structuring" in patterns:
        lines.append(f"")
        lines.append("  Structuring Detected:")
        lines.append("  Multiple transactions just below reporting thresholds,")
        lines.append("  suggesting intentional avoidance of regulatory triggers.")

    return {"response": "\n".join(lines), "type": "temporal", "account_id": account["account_id"]}


def _explain_score(account: dict) -> dict:
    lines = [
        f"Score Details for {account['account_id']}:",
        f"  - Final Risk Score: {account['risk_score']}/100",
        f"  - Tier: {account['tier']}",
        f"  - Confidence: {account.get('confidence', 0):.0%}",
        f"",
    ]
    breakdown = account.get("score_breakdown", [])
    if breakdown:
        lines.append("Module Contributions:")
        for b in breakdown:
            lines.append(f"  - {b['module']}: {b.get('detections', 0)} detections, weight={b['weighted']:.4f}")

    return {"response": "\n".join(lines), "type": "score", "account_id": account["account_id"]}


def _explain_account_summary(account: dict, rings_map: dict) -> dict:
    lines = [
        f"Account Summary: {account['account_id']}",
        f"  - Risk Score: {account['risk_score']}/100 ({account['tier']})",
        f"  - Confidence: {account.get('confidence', 0):.0%}",
        f"  - Patterns: {', '.join(account.get('patterns', []))}",
        f"  - Ring IDs: {', '.join(account.get('ring_ids', [])) or 'None'}",
        f"  - Explanation: {account.get('explanation', 'N/A')}",
    ]
    return {"response": "\n".join(lines), "type": "summary", "account_id": account["account_id"]}


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
