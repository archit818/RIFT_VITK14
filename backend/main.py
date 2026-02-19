"""
RIFT - Financial Forensics Engine
FastAPI Backend - Main Application

Provides REST API for CSV upload, analysis, and visualization.
"""

import os
import io
import time
import json
import asyncio
import traceback
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from validators import validate_csv
from graph_engine import TransactionGraph
from scoring import compute_scores
from visualization import generate_visualization
from synthetic import generate_synthetic_data, generate_edge_case_data

from detectors.circular_routing import detect_circular_routing
from detectors.fan_in import detect_fan_in
from detectors.fan_out import detect_fan_out
from detectors.shell_chains import detect_shell_chains
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
    version="1.0.0",
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
        "version": "1.0.0",
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
    Accepts CSV upload, runs all detection modules, returns structured results.
    """
    start_time = time.time()
    
    # --- Read and validate ---
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        # Try multiple encodings
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
        tg = TransactionGraph(df_clean)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph construction failed: {str(e)}"
        )
    
    # --- Run Detection Modules ---
    try:
        all_detections = await _run_detections(tg)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
    
    # --- Compute Scores ---
    try:
        suspicious_accounts, fraud_rings, summary = compute_scores(tg, all_detections)
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
    
    elapsed = round(time.time() - start_time, 2)
    
    # --- Build Response ---
    summary["processing_time_seconds"] = elapsed
    summary["validation_warnings"] = [
        e for e in validation_errors if e.get("error_type") not in
        ("MISSING_COLUMNS", "EMPTY_FILE", "ALL_ROWS_INVALID")
    ]
    
    response = {
        "suspicious_accounts": suspicious_accounts[:200],
        "fraud_rings": fraud_rings[:100],
        "summary": summary,
        "graph_url": "/static/graph.html" if graph_path else None,
    }
    
    # Cache the result
    analysis_cache["latest"] = response
    
    return response


async def _run_detections(tg: TransactionGraph) -> dict:
    """Run all detection modules, using thread pool for parallelism."""
    loop = asyncio.get_event_loop()
    
    # Run detection modules in parallel
    futures = {
        "circular_routing": loop.run_in_executor(executor, detect_circular_routing, tg),
        "fan_in_aggregation": loop.run_in_executor(executor, detect_fan_in, tg),
        "fan_out_dispersal": loop.run_in_executor(executor, detect_fan_out, tg),
        "shell_chain": loop.run_in_executor(executor, detect_shell_chains, tg),
        "transaction_burst": loop.run_in_executor(executor, detect_transaction_bursts, tg),
        "rapid_movement": loop.run_in_executor(executor, detect_rapid_movement, tg),
        "dormant_activation": loop.run_in_executor(executor, detect_dormant_activation, tg),
        "structuring": loop.run_in_executor(executor, detect_structuring, tg),
        "diversity_shift": loop.run_in_executor(executor, detect_diversity_shift, tg),
        "centrality_spike": loop.run_in_executor(executor, detect_centrality_spike, tg),
        "community_suspicion": loop.run_in_executor(executor, detect_community_suspicion, tg),
    }
    
    results = {}
    for key, future in futures.items():
        try:
            results[key] = await future
        except Exception as e:
            print(f"Detection module {key} failed: {e}")
            results[key] = []
    
    # Module 9 depends on cycle results
    cycles = results.get("circular_routing", [])
    try:
        results["amount_consistency_ring"] = detect_amount_consistency(tg, cycles)
    except Exception as e:
        print(f"Amount consistency module failed: {e}")
        results["amount_consistency_ring"] = []
    
    return results


# ─── Graph Visualization Endpoint ─────────────────────────────

@app.get("/api/graph")
async def get_graph():
    """Serve the latest graph visualization."""
    graph_path = os.path.join(STATIC_DIR, "graph.html")
    if os.path.exists(graph_path):
        return FileResponse(graph_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="No graph generated yet. Run analysis first.")


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
        
        # Save to static dir
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

@app.get("/api/docs/architecture")
async def get_architecture():
    """Return system architecture documentation."""
    return {
        "architecture": {
            "name": "RIFT - Financial Forensics Engine",
            "design_principle": "Algorithmic graph intelligence without ML models",
            "components": {
                "input_layer": {
                    "description": "CSV upload with strict validation",
                    "validations": [
                        "Schema validation (5 required columns)",
                        "Missing value handling",
                        "Duplicate transaction detection",
                        "Timestamp format validation",
                        "Amount validation (positive, numeric)",
                        "Self-transfer detection",
                        "Outlier flagging (IQR-based)"
                    ]
                },
                "graph_engine": {
                    "description": "Directed weighted temporal graph via NetworkX",
                    "precomputed_features": [
                        "In/out degree",
                        "Betweenness centrality",
                        "PageRank",
                        "Clustering coefficient",
                        "Temporal ordering",
                        "Edge temporal index"
                    ]
                },
                "detection_modules": [
                    {"name": "Circular Routing", "complexity": "O(V+E) with bounded DFS"},
                    {"name": "Fan-in Aggregation", "complexity": "O(V * T) sliding window"},
                    {"name": "Fan-out Dispersal", "complexity": "O(V * T) sliding window"},
                    {"name": "Shell Chains", "complexity": "O(V * E) BFS"},
                    {"name": "Transaction Burst", "complexity": "O(V * T) rolling window"},
                    {"name": "Rapid Movement", "complexity": "O(V * E) temporal BFS"},
                    {"name": "Dormant Activation", "complexity": "O(V * T) gap analysis"},
                    {"name": "Structuring", "complexity": "O(V * T) threshold analysis"},
                    {"name": "Amount Consistency", "complexity": "O(C * E) cycle analysis"},
                    {"name": "Diversity Shift", "complexity": "O(V * T) window analysis"},
                    {"name": "Centrality Spike", "complexity": "O(V) percentile analysis"},
                    {"name": "Community Suspicion", "complexity": "O(V + E) community detection"},
                ],
                "scoring_engine": {
                    "description": "Weighted multi-module scoring with FP reduction",
                    "normalization": "0-100 scale",
                    "false_positive_controls": [
                        "Merchant detection",
                        "Payroll detection",
                        "Behavior stability scoring",
                        "Long-term consistency analysis"
                    ]
                },
                "visualization": {
                    "description": "PyVis interactive graph with risk-based coloring",
                    "features": [
                        "Node color by risk level",
                        "Ring highlighting",
                        "Interactive tooltips",
                        "Dark theme",
                        "Physics-based layout"
                    ]
                }
            },
            "performance": {
                "target": "≤ 30 seconds for 10K transactions",
                "optimizations": [
                    "Vectorized Pandas operations",
                    "Cached graph features",
                    "Node degree filtering",
                    "Parallel module execution (ThreadPoolExecutor)",
                    "Early pruning and result limiting",
                    "Bounded cycle search (length ≤ 5)",
                    "Approximate centrality for large graphs"
                ]
            },
            "limitations": [
                "No ML-based detection (by design)",
                "Cycle detection limited to length 5",
                "Approximate centrality for >500 nodes",
                "Single-file processing (no streaming)",
                "In-memory graph (limited by RAM)"
            ],
            "threshold_tuning": {
                "fan_in_senders": "Default: 10, reduce for higher recall",
                "fan_out_receivers": "Default: 10, reduce for higher recall",
                "dormancy_days": "Default: 30, increase for strict detection",
                "burst_multiplier": "Default: 3x, reduce for sensitivity",
                "structuring_count": "Default: 3, increase for precision",
                "cycle_max_length": "Default: 5, increase for thoroughness"
            }
        }
    }


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
