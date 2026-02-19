import requests
import json

r = requests.post(
    'http://localhost:8000/api/analyze',
    files={'file': open('backend/static/test_data.csv', 'rb')}
)

data = r.json()
s = data.get('summary', {})

print(f"Status: {r.status_code}")
print(f"Accounts analyzed: {s.get('total_accounts_analyzed')}")
print(f"Transactions: {s.get('total_transactions_analyzed')}")
print(f"Suspicious accounts: {s.get('suspicious_accounts_found')}")
print(f"Fraud rings: {s.get('fraud_rings_detected')}")
print(f"High risk: {s.get('high_risk_accounts')}")
print(f"Medium risk: {s.get('medium_risk_accounts')}")
print(f"Low risk: {s.get('low_risk_accounts')}")
print(f"Processing time: {s.get('processing_time_seconds')}s")
print(f"Modules triggered: {s.get('detection_modules_triggered')}")
print(f"Graph URL: {data.get('graph_url')}")
print(f"Top 5 suspicious accounts:")
for acc in data.get('suspicious_accounts', [])[:5]:
    print(f"  {acc['account_id']}: {acc['risk_score']:.1f} - {acc['triggered_patterns']}")
print(f"Top 5 fraud rings:")
for ring in data.get('fraud_rings', [])[:5]:
    print(f"  {ring['ring_id']}: {ring['risk_score']:.1f} - {ring['type']} - {len(ring['nodes'])} nodes")
