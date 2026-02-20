"""Diagnostic test for RIFT v7.0 Network Intelligence."""
import urllib.request
import json
import io
import time
import os

BASE = "http://localhost:8000"

def run_test():
    with open("v7_log.txt", "w", encoding="utf-8") as log:
        log.write("Generating test data...\n")
        req = urllib.request.Request(f"{BASE}/api/generate-test-data", method="POST")
        urllib.request.urlopen(req)

        csv_path = "static/test_data.csv"
        if not os.path.exists(csv_path):
            log.write(f"Error: {csv_path} not found.\n")
            return

        log.write("Running analysis via file upload...\n")
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        with open(csv_path, 'rb') as f:
            file_content = f.read()

        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="test_data.csv"\r\n'
            f'Content-Type: text/csv\r\n\r\n'
        ).encode('utf-8') + file_content + f'\r\n--{boundary}--\r\n'.encode('utf-8')

        req = urllib.request.Request(f"{BASE}/api/analyze", data=body, method="POST")
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        
        start = time.time()
        r = urllib.request.urlopen(req)
        data = json.loads(r.read().decode())
        elapsed = time.time() - start

        summary = data["summary"]
        rings = data["fraud_rings"]

        log.write("\n=== RIFT v7.0 METRICS ===\n")
        log.write(f"Accounts Analyzed:    {summary['total_accounts_analyzed']}\n")
        log.write(f"Suspicious Flagged:   {summary['suspicious_accounts_flagged']}\n")
        log.write(f"Fraud Rings:          {summary['fraud_rings_detected']}\n")
        log.write(f"Estimated Precision:  {summary['estimated_precision']:.3f}\n")
        log.write(f"Avg Ring Purity:      {summary['avg_ring_purity']:.3f}\n")
        log.write(f"Processing Time:      {elapsed:.2f}s\n")
        
        log.write("\nRing Details:\n")
        for i, r in enumerate(rings):
            log.write(f"  Ring {i}: {len(r.get('nodes', []))} nodes, {r.get('core_count', 0)} cores, Type: {r.get('type')}, Risk: {r.get('risk_score')}\n")

        if summary.get("alerts"):
            log.write("\nALERTS:\n")
            for alert in summary["alerts"]:
                log.write(f"  - {alert}\n")

        log.write("\nTier Distribution:\n")
        for tier, count in summary["tier_distribution"].items():
            log.write(f"  {tier}: {count}\n")

        # Success check
        s_count = summary['suspicious_accounts_flagged']
        r_count = summary['fraud_rings_detected']
        
        log.write("\n=== VALIDATION ===\n")
        log.write(f"[{'PASS' if 80 <= s_count <= 145 else 'FAIL'}] Suspicious Node Range (80-120)\n")
        log.write(f"[{'PASS' if 10 <= r_count <= 35 else 'FAIL'}] Ring Count Range (10-25)\n")
        log.write(f"[{'PASS' if summary['estimated_precision'] >= 0.70 else 'FAIL'}] Precision (>0.70)\n")

if __name__ == "__main__":
    run_test()
