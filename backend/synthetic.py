"""
Synthetic data generator for testing the Financial Forensics Engine.
Generates realistic transaction datasets with embedded fraud patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string


def generate_synthetic_data(
    num_accounts: int = 500,
    num_transactions: int = 5000,
    fraud_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with embedded fraud patterns.
    
    Fraud patterns included:
    - Circular routing rings
    - Fan-in aggregation
    - Fan-out dispersal
    - Shell chains
    - Structuring
    - Dormant accounts
    - Transaction bursts
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate account IDs
    accounts = [f"ACC-{i:05d}" for i in range(num_accounts)]
    
    transactions = []
    tx_id = 0
    
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    time_span = timedelta(days=180)
    
    # ─── Legitimate Transactions ──────────────────────────────────
    num_legit = int(num_transactions * (1 - fraud_ratio))
    
    for _ in range(num_legit):
        tx_id += 1
        sender = random.choice(accounts[:int(num_accounts * 0.8)])
        receiver = random.choice(accounts[:int(num_accounts * 0.8)])
        while receiver == sender:
            receiver = random.choice(accounts[:int(num_accounts * 0.8)])
        
        # Scale amounts for Rupees (roughly 80x USD, but using lognormal parameter shift)
        amount = round(np.random.lognormal(mean=10.5, sigma=1.5), 2)
        amount = max(100, min(amount, 50000000))
        
        ts = base_time + timedelta(
            seconds=random.randint(0, int(time_span.total_seconds()))
        )
        
        transactions.append({
            "transaction_id": f"TX-{tx_id:08d}",
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    # ─── Fraud Pattern 1: Circular Routing ────────────────────────
    num_rings = max(2, int(num_accounts * 0.01))
    ring_accounts = accounts[int(num_accounts * 0.8):int(num_accounts * 0.85)]
    
    for ring_idx in range(num_rings):
        ring_size = random.choice([3, 4, 5])
        ring = random.sample(ring_accounts, min(ring_size, len(ring_accounts)))
        
        ring_amount = round(random.uniform(400000, 4000000), 2)
        ring_time = base_time + timedelta(days=random.randint(30, 150))
        
        for i in range(len(ring)):
            tx_id += 1
            sender = ring[i]
            receiver = ring[(i + 1) % len(ring)]
            
            amount = ring_amount + round(random.uniform(-100, 100), 2)
            ts = ring_time + timedelta(hours=i * random.randint(1, 6))
            
            transactions.append({
                "transaction_id": f"TX-{tx_id:08d}",
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # ─── Fraud Pattern 2: Fan-in Aggregation ──────────────────────
    aggregator_accounts = accounts[int(num_accounts * 0.85):int(num_accounts * 0.88)]
    
    for agg in aggregator_accounts[:3]:
        senders = random.sample(accounts[:int(num_accounts * 0.5)], min(15, int(num_accounts * 0.5)))
        fan_in_time = base_time + timedelta(days=random.randint(50, 160))
        
        for sender in senders:
            tx_id += 1
            amount = round(random.uniform(80000, 760000), 2)
            ts = fan_in_time + timedelta(hours=random.randint(0, 48))
            
            transactions.append({
                "transaction_id": f"TX-{tx_id:08d}",
                "sender_id": sender,
                "receiver_id": agg,
                "amount": amount,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # ─── Fraud Pattern 3: Fan-out Dispersal ───────────────────────
    disperser_accounts = accounts[int(num_accounts * 0.88):int(num_accounts * 0.90)]
    
    for disperser in disperser_accounts[:2]:
        receivers = random.sample(accounts[:int(num_accounts * 0.5)], min(12, int(num_accounts * 0.5)))
        fan_out_time = base_time + timedelta(days=random.randint(50, 160))
        
        for receiver in receivers:
            tx_id += 1
            amount = round(random.uniform(40000, 400000), 2)
            ts = fan_out_time + timedelta(hours=random.randint(0, 12))
            
            transactions.append({
                "transaction_id": f"TX-{tx_id:08d}",
                "sender_id": disperser,
                "receiver_id": receiver,
                "amount": amount,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # ─── Fraud Pattern 4: Structuring ─────────────────────────────
    structuring_accounts = accounts[int(num_accounts * 0.90):int(num_accounts * 0.93)]
    
    for structurer in structuring_accounts[:3]:
        receivers = random.sample(accounts[:int(num_accounts * 0.3)], 5)
        struct_time = base_time + timedelta(days=random.randint(30, 150))
        
        for i in range(8):
            tx_id += 1
            amount = round(random.uniform(45000, 49500), 2)  # Just below ₹50,000 PAN threshold
            ts = struct_time + timedelta(days=i * random.randint(1, 3))
            
            transactions.append({
                "transaction_id": f"TX-{tx_id:08d}",
                "sender_id": structurer,
                "receiver_id": random.choice(receivers),
                "amount": amount,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # ─── Fraud Pattern 5: Shell Chains ────────────────────────────
    shell_accounts = accounts[int(num_accounts * 0.93):int(num_accounts * 0.97)]
    
    for _ in range(2):
        chain_length = random.randint(4, 6)
        chain = random.sample(shell_accounts, min(chain_length, len(shell_accounts)))
        
        chain_time = base_time + timedelta(days=random.randint(40, 140))
        chain_amount = round(random.uniform(800000, 6400000), 2)
        
        for i in range(len(chain) - 1):
            tx_id += 1
            ts = chain_time + timedelta(hours=i * 2)
            
            transactions.append({
                "transaction_id": f"TX-{tx_id:08d}",
                "sender_id": chain[i],
                "receiver_id": chain[i + 1],
                "amount": chain_amount + round(random.uniform(-500, 500), 2),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # ─── Fraud Pattern 6: Dormant + Burst ─────────────────────────
    dormant_accounts = accounts[int(num_accounts * 0.97):]
    
    for dormant in dormant_accounts[:3]:
        # One transaction long ago
        tx_id += 1
        early_time = base_time + timedelta(days=random.randint(0, 10))
        transactions.append({
            "transaction_id": f"TX-{tx_id:08d}",
            "sender_id": random.choice(accounts[:50]),
            "receiver_id": dormant,
            "amount": round(random.uniform(100, 1000), 2),
            "timestamp": early_time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Sudden burst after 60+ days
        burst_time = early_time + timedelta(days=random.randint(60, 120))
        for _ in range(10):
            tx_id += 1
            ts = burst_time + timedelta(hours=random.randint(0, 24))
            transactions.append({
                "transaction_id": f"TX-{tx_id:08d}",
                "sender_id": dormant,
                "receiver_id": random.choice(accounts[:100]),
                "amount": round(random.uniform(160000, 1200000), 2),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # Build DataFrame
    df = pd.DataFrame(transactions)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def generate_edge_case_data() -> pd.DataFrame:
    """Generate edge case test data to stress-test the system."""
    transactions = []
    tx_id = 0
    base_time = datetime(2024, 6, 1, 12, 0, 0)
    
    # Edge case 1: Very large amounts
    tx_id += 1
    transactions.append({
        "transaction_id": f"TX-EDGE-{tx_id:08d}",
        "sender_id": "EDGE-001",
        "receiver_id": "EDGE-002",
        "amount": 99999999.99,
        "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Edge case 2: Very small amounts
    tx_id += 1
    transactions.append({
        "transaction_id": f"TX-EDGE-{tx_id:08d}",
        "sender_id": "EDGE-003",
        "receiver_id": "EDGE-004",
        "amount": 0.01,
        "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Edge case 3: Same timestamp transactions
    for i in range(20):
        tx_id += 1
        transactions.append({
            "transaction_id": f"TX-EDGE-{tx_id:08d}",
            "sender_id": f"EDGE-{100+i:03d}",
            "receiver_id": "EDGE-HUB",
            "amount": round(random.uniform(100, 5000), 2),
            "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    # Edge case 4: Rapid-fire from single account
    for i in range(30):
        tx_id += 1
        transactions.append({
            "transaction_id": f"TX-EDGE-{tx_id:08d}",
            "sender_id": "EDGE-RAPID",
            "receiver_id": f"EDGE-R{i:03d}",
            "amount": 9999.99,
            "timestamp": (base_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    return pd.DataFrame(transactions)


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("test_data.csv", index=False)
    print(f"Generated {len(df)} transactions with {df['sender_id'].nunique() + df['receiver_id'].nunique()} unique accounts")
