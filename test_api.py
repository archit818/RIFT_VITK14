"""Compact test of the refactored pipeline."""
import requests
import json

r = requests.post(
    'http://localhost:8000/api/analyze',
    files={'file': open('backend/static/test_data.csv', 'rb')}
)
data = r.json()
s = data.get('summary', {})

# Summary
print("STATUS:", r.status_code)
print(f"Accounts: {s.get('total_accounts_analyzed')}, "
      f"Txns: {s.get('total_transactions_analyzed')}, "
      f"Legit excluded: {s.get('legitimate_accounts_excluded', 0)}")
print(f"Suspicious: {s.get('suspicious_accounts_found')}, "
      f"Rings: {s.get('fraud_rings_detected')}")
print(f"HighRisk: {s.get('high_risk_accounts')}, "
      f"MedRisk: {s.get('medium_risk_accounts')}, "
      f"LowRisk: {s.get('low_risk_accounts')}")
print(f"Time: {s.get('processing_time_seconds')}s")
print()

# Rings
print("--- RINGS ---")
for ring in data.get('fraud_rings', []):
    print(f"  {ring['ring_id']} type={ring['type']} "
          f"score={ring['risk_score']} nodes={ring['node_count']} "
          f"amt=${ring['total_amount']:,.0f}")
print()

# Ring types
types = {}
for r2 in data.get('fraud_rings', []):
    t = r2['type']
    types[t] = types.get(t, 0) + 1
print("--- RING TYPES ---")
for t, c in sorted(types.items()):
    print(f"  {t}: {c}")
print()

# Top 5 accounts
print("--- TOP 5 ACCOUNTS ---")
for a in data.get('suspicious_accounts', [])[:5]:
    print(f"  {a['account_id']} score={a['risk_score']} "
          f"legit={a.get('legitimacy_score', 0):.2f} "
          f"patterns={a['triggered_patterns']}")
