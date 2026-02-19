# RIFT - Financial Forensics Engine

## Architecture

```
RIFT/
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # API entry point
│   ├── validators.py           # CSV input validation
│   ├── graph_engine.py         # NetworkX graph construction
│   ├── scoring.py              # Suspicion scoring engine
│   ├── visualization.py        # PyVis graph generation
│   ├── synthetic.py            # Synthetic data generator
│   ├── requirements.txt        # Python dependencies
│   └── detectors/              # 12 detection modules
│       ├── circular_routing.py # Module 1: Cycle detection
│       ├── fan_in.py           # Module 2: Fan-in aggregation
│       ├── fan_out.py          # Module 3: Fan-out dispersal
│       ├── shell_chains.py     # Module 4: Shell chain detection
│       ├── burst.py            # Module 5: Transaction burst
│       ├── rapid_movement.py   # Module 6: Rapid fund movement
│       ├── dormant.py          # Module 7: Dormant activation
│       ├── structuring.py      # Module 8: Threshold avoidance
│       └── advanced.py         # Modules 9-12
└── frontend/                   # React Vite frontend
    └── src/
        ├── App.jsx
        ├── index.css           # Design system
        └── components/
            ├── Header.jsx
            ├── UploadZone.jsx
            ├── Dashboard.jsx
            ├── StatsGrid.jsx
            ├── AccountsTable.jsx
            ├── FraudRings.jsx
            ├── GraphView.jsx
            ├── PatternBreakdown.jsx
            ├── LoadingOverlay.jsx
            └── ErrorAlert.jsx
```

## Detection Modules

| # | Module | Algorithm | Complexity |
|---|--------|-----------|------------|
| 1 | Circular Routing | Johnson's algorithm, bounded DFS | O(V+E) |
| 2 | Fan-in Aggregation | Sliding window + amount clustering | O(V×T) |
| 3 | Fan-out Dispersal | Sliding window + velocity analysis | O(V×T) |
| 4 | Shell Chains | BFS through low-degree nodes | O(V×E) |
| 5 | Transaction Burst | Rolling window spike detection | O(V×T) |
| 6 | Rapid Movement | Temporal BFS | O(V×E) |
| 7 | Dormant Activation | Gap analysis on timestamp series | O(V×T) |
| 8 | Structuring | Threshold proximity analysis | O(V×T) |
| 9 | Amount Consistency | Cycle amount variance analysis | O(C×E) |
| 10 | Diversity Shift | Windowed counterparty counting | O(V×T) |
| 11 | Centrality Spike | Percentile-based anomaly detection | O(V) |
| 12 | Community Suspicion | Greedy modularity + risk propagation | O(V+E) |

## False Positive Controls

- **Merchant Detection**: High diversity + stable behavior + mostly incoming
- **Payroll Detection**: Regular timing + fixed amounts + mostly outgoing
- **Behavior Stability**: Long-term consistent accounts get score reduction
- **Temporal Regularity**: Regular patterns score lower

## Threshold Tuning

| Parameter | Default | Adjust For |
|-----------|---------|------------|
| `fan_in_senders` | 10 | Lower = higher recall |
| `fan_out_receivers` | 10 | Lower = higher recall |
| `dormancy_days` | 30 | Higher = stricter |
| `burst_multiplier` | 3x | Lower = more sensitive |
| `structuring_count` | 3 | Higher = more precise |
| `cycle_max_length` | 5 | Higher = thorough |

## Running Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Deployment (Render)

Both services are configured for Render deployment.
- Backend: Python web service
- Frontend: Static site (build with `npm run build`)
