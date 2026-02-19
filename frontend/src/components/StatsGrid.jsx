export default function StatsGrid({ summary }) {
    const stats = [
        {
            value: summary.total_accounts_analyzed?.toLocaleString() || '0',
            label: 'Accounts Analyzed',
            icon: '👥',
            color: 'info',
            cardClass: 'info',
        },
        {
            value: summary.total_transactions_analyzed?.toLocaleString() || '0',
            label: 'Transactions',
            icon: '📄',
            color: 'accent',
            cardClass: 'accent',
        },
        {
            value: summary.suspicious_accounts_found || 0,
            label: 'Suspicious Accounts',
            icon: '🚨',
            color: 'danger',
            cardClass: 'high-risk',
        },
        {
            value: summary.fraud_rings_detected || 0,
            label: 'Fraud Rings',
            icon: '🔗',
            color: 'warning',
            cardClass: 'medium-risk',
        },
        {
            value: summary.high_risk_accounts || 0,
            label: 'High Risk',
            icon: '🔴',
            color: 'danger',
            cardClass: 'high-risk',
        },
        {
            value: summary.medium_risk_accounts || 0,
            label: 'Medium Risk',
            icon: '🟠',
            color: 'warning',
            cardClass: 'medium-risk',
        },
        {
            value: summary.low_risk_accounts || 0,
            label: 'Low Risk',
            icon: '🟢',
            color: 'success',
            cardClass: 'low-risk',
        },
        {
            value: summary.legitimate_accounts_excluded || 0,
            label: 'Legitimate Excluded',
            icon: '✅',
            color: 'success',
            cardClass: 'low-risk',
        },
        {
            value: `${summary.processing_time_seconds || 0}s`,
            label: 'Processing Time',
            icon: '⚡',
            color: 'accent',
            cardClass: 'accent',
        },
    ]

    return (
        <div className="stats-grid">
            {stats.map((stat, i) => (
                <div key={i} className={`stat-card ${stat.cardClass} fade-up fade-up-delay-${Math.min(i + 1, 4)}`}>
                    <span className="stat-icon">{stat.icon}</span>
                    <div className={`stat-value ${stat.color}`}>{stat.value}</div>
                    <div className="stat-label">{stat.label}</div>
                </div>
            ))}
        </div>
    )
}
