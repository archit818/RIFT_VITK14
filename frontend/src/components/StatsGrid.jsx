export default function StatsGrid({ summary }) {
    const stats = [
        {
            value: summary.total_accounts_analyzed?.toLocaleString() || '0',
            label: 'Accounts Analyzed',
        },
        {
            value: summary.total_transactions_analyzed?.toLocaleString() || '0',
            label: 'Transactions',
        },
        {
            value: summary.suspicious_accounts_found || 0,
            label: 'Deceptive Nodes',
        },
        {
            value: summary.fraud_rings_detected || 0,
            label: 'Validated Rings',
        },
        {
            value: summary.high_risk_accounts || 0,
            label: 'Critical Threats',
        },
        {
            value: `${((summary.estimated_precision || 0) * 100).toFixed(1)}%`,
            label: 'Est. Precision',
        },
        {
            value: `${((summary.avg_ring_purity || 0) * 100).toFixed(0)}%`,
            label: 'Ring Purity',
        },
        {
            value: `${((summary.multi_signal_gate_pass_rate || 0) * 100).toFixed(0)}%`,
            label: 'Multi-Signal Gate',
        },
        {
            value: `${((summary.fp_estimate || 0) * 100).toFixed(1)}%`,
            label: 'FP Estimate',
        },
        {
            value: summary.processing_time_seconds || 0,
            label: 'Inference Time',
            unit: 's'
        }
    ]

    return (
        <div className="stats-grid">
            {stats.map((stat, i) => (
                <div key={i} className="stat-card fade-up" style={{ animationDelay: `${i * 0.1}s` }}>
                    <div className="stat-value">
                        {stat.value}{stat.unit}
                    </div>
                    <div className="stat-label">{stat.label}</div>
                </div>
            ))}
        </div>
    )
}
