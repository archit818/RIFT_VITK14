import asyncio
import pandas as pd
from graph_engine import TransactionGraph
from main import _run_detection_pipeline, validate_csv
from scoring import compute_scores
import json

async def debug_csv(file_path):
    print("Loading", file_path)
    df = pd.read_csv(file_path)
    df_clean, errors = validate_csv(df)
    tg = TransactionGraph(df_clean)
    all_detections, ring_mgr, legit_set, legit_scores = await _run_detection_pipeline(tg)
    print(f"Legit set size: {len(legit_set)} / {len(legit_scores)}")
    if legit_scores:
        print(f"Max legit score: {max(legit_scores.values())}")
        print(f"Mean legit score: {sum(legit_scores.values())/len(legit_scores)}")
    
    suspicious_accounts, fraud_rings, summary = compute_scores(
        tg, all_detections, ring_mgr,
        legitimate_accounts=legit_set,
        legitimacy_scores=legit_scores
    )
    print(f"Suspicious accounts: {len(suspicious_accounts)}")
    print(f"Fraud rings: {len(fraud_rings)}")
    if suspicious_accounts:
        scores = [a["risk_score"] for a in suspicious_accounts]
        print(f"Max score: {max(scores)}")
        print(f"Top 5 scores: {sorted(scores, reverse=True)[:5]}")

    out = {
        "num_suspicious": len(suspicious_accounts),
        "num_rings": len(fraud_rings),
        "suspicious_accounts": suspicious_accounts[:100], # Limit for output size
        "rings": fraud_rings,
        "ring_mgr_rings": ring_mgr.get_rings()
    }
    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    with open("test_out.json", "w") as f:
        json.dump(out, f, indent=2, cls=SetEncoder)
    print("FINISHED")

if __name__ == '__main__':
    asyncio.run(debug_csv('../money-mulling.csv'))
