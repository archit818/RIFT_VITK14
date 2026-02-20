# RIFT - Financial Forensics Engine (RIFT 2026 Hackathon)

### MONEY MULING DETECTION CHALLENGE
**Team Name:** VITK14  
**Live Demo:** [https://rift-forensics.onrender.com](https://rift-forensics.onrender.com)  
**Video Demo:** [LinkedIn Video Link](https://www.linkedin.com/posts/arka-pratim-acharyya_rifthackathon-moneymulingdetection-financialcrime-activity-7298319692482310144-x86g?utm_source=share&utm_medium=member_desktop&rcm=ACoAAELR_oUBF-m9r_0hLp6f7g3fI3V5N1-x86g)

---

## 🏗 System Architecture

RIFT is built with a decoupled high-performance architecture:
- **Frontend**: React 18 with Vite, featuring a premium glassmorphic UI, interactive NetworkX-powered graph visualizations (via PyVis), and real-time forensics dashboards.
- **Backend**: FastAPI (Python 3.11+) implementing a multi-stage graph processing pipeline.
- **Graph Engine**: NetworkX for topology analysis and behavioral pattern extraction.
- **Detection Pipeline**: A waterfall priority engine that prunes noise early and consolidates signals into multi-member fraud rings.

---

## 🧠 Algorithm Approach & Complexity

| # | Module | Algorithm | Complexity | Description |
|---|--------|-----------|------------|-------------|
| 1 | **Circular Routing** | Bounded DFS / Johnson's | O(V+E) | Detects cycles of length 3-5 used for fund layering. |
| 2 | **Smurfing (Fan-in)** | Sliding Window Search | O(V×T) | Detects 10+ senders aggregating to 1 receiver within 72h. |
| 3 | **Smurfing (Fan-out)**| Sliding Window Search | O(V×T) | Detects 1 sender dispersing to 10+ receivers within 72h. |
| 4 | **Layered Shells** | BFS Traversal | O(V×E) | Identifies chains of 3+ hops through low-activity accounts. |
| 5 | **Transaction Burst**| Rolling Window Spike | O(V×T) | Detects sudden extreme velocity shifts. |
| 6 | **Rapid Movement** | Temporal BFS | O(V×E) | Tracks funds passing through nodes in < 2 hours. |
| 7 | **Dormant Activation**| Gap Analysis | O(V×T) | Flags long-inactive accounts suddenly moving high volumes. |
| 8 | **Structuring** | Binning Analysis | O(V×T) | Detects amounts just below common reporting thresholds. |

---

## ⚖️ Suspicion Score Methodology (SSM)

Our scoring engine uses a **Multi-Signal Gated (MSG)** approach to minimize false positives:

1.  **Structural Integrity (40%)**: High weight given to cycles and shell chains.
2.  **Temporal Coherence (30%)**: Analyzes the velocity and timing (e.g., the 72h smurfing window).
3.  **Behavioral Deviation (20%)**: Compares current activity vs historical stability.
4.  **Network Influence (10%)**: Propagates risk from known suspicious neighbors using PageRank-inspired weights.

**False Positive Controls**:
- **Merchant Suppression**: Automated detection of high-diversity, stable-incoming patterns.
- **Payroll Suppression**: Recognizes recurring fixed-amount periodic outgoing flows.
- **Stability Dampening**: Accounts with long-term consistent behavior receive a 40-60% risk reduction.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 📘 Usage Instructions

1.  **Upload**: Select a transaction CSV matching the required schema (`transaction_id, sender_id, receiver_id, amount, timestamp`).
2.  **Analyze**: The engine will process up to 10K transactions in < 30 seconds.
3.  **Explore**: Use the **Network Graph** to see directed money flows. Red nodes indicate high-risk muling suspects.
4.  **Export**: Click **GENERATE_FULL_REPORT** to download the standardized JSON output for regulatory submission.

---

## 🚧 Known Limitations

- **Large Datasets**: Visualization may become cluttered with > 5,000 nodes; the engine uses sampling for the interactive UI in such cases.
- **Single-Hop Muling**: Simple A → B transfers without further movement are not flagged as "rings" unless behavioral signals are extreme.
- **Cold Start**: New accounts without historical data may have slightly less accurate stability scores.

---

## 👥 Team Members

- **Arka Pratim Acharyya** - (Backend/Algorithms/Lead)
- **Team Code:** VITK14

---
*RIFT Hackathon 2026 - Keeping the financial ecosystem clean.*
