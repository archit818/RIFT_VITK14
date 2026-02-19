"""Compact test, writes to output.txt (utf-8)."""
import requests
import sys

r = requests.post(
    'http://localhost:8000/api/analyze',
    files={'file': open('backend/static/test_data.csv', 'rb')}
)
data = r.json()
s = data.get('summary', {})

lines = []
lines.append(f"STATUS: {r.status_code}")
lines.append(f"Accounts: {s.get('total_accounts_analyzed')}")
lines.append(f"Transactions: {s.get('total_transactions_analyzed')}")
lines.append(f"Legitimate excluded: {s.get('legitimate_accounts_excluded', 0)}")
lines.append(f"Suspicious accounts: {s.get('suspicious_accounts_found')}")
lines.append(f"Fraud rings: {s.get('fraud_rings_detected')}")
lines.append(f"High risk: {s.get('high_risk_accounts')}")
lines.append(f"Medium risk: {s.get('medium_risk_accounts')}")
lines.append(f"Low risk: {s.get('low_risk_accounts')}")
lines.append(f"Processing time: {s.get('processing_time_seconds')}s")
lines.append(f"Modules: {s.get('detection_modules_triggered')}")
lines.append("")

lines.append("=== FRAUD RINGS ===")
for ring in data.get('fraud_rings', []):
    lines.append(f"  {ring['ring_id']}  type={ring['type']}  score={ring['risk_score']}  nodes={ring['node_count']}  amount={ring['total_amount']}")
lines.append("")

types = {}
for ring in data.get('fraud_rings', []):
    t = ring['type']
    types[t] = types.get(t, 0) + 1
lines.append("=== RING TYPE COUNTS ===")
for t, c in sorted(types.items()):
    lines.append(f"  {t}: {c}")
lines.append("")

lines.append("=== TOP 10 SUSPICIOUS ===")
for a in data.get('suspicious_accounts', [])[:10]:
    lines.append(f"  {a['account_id']}  score={a['risk_score']}  legit={a.get('legitimacy_score',0):.2f}  patterns={a['triggered_patterns']}  rings={a.get('ring_ids',[])}")
lines.append("")

lines.append("=== SCORE DISTRIBUTION ===")
scores = [a['risk_score'] for a in data.get('suspicious_accounts', [])]
if scores:
    lines.append(f"  Min: {min(scores):.1f}")
    lines.append(f"  Max: {max(scores):.1f}")
    import statistics
    lines.append(f"  Mean: {statistics.mean(scores):.1f}")
    lines.append(f"  Median: {statistics.median(scores):.1f}")

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Written to output.txt")
