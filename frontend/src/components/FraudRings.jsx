function getRiskLevel(score) {
    if (score >= 70) return 'high'
    if (score >= 40) return 'medium'
    return 'low'
}

export default function FraudRings({ rings, compact = false }) {
    const displayRings = compact ? rings.slice(0, 5) : rings

    if (!rings.length) {
        return (
            <div className="empty-state">
                <h3>No Fraud Rings Detected</h3>
                <p>No coordinated fraud ring patterns were identified.</p>
            </div>
        )
    }

    if (compact) {
        return (
            <div>
                {displayRings.map((ring) => {
                    const risk = getRiskLevel(ring.risk_score)
                    return (
                        <div key={ring.ring_id} className="ring-card" id={`ring-${ring.ring_id}`}>
                            <div className="ring-header">
                                <div>
                                    <div className="ring-id">{ring.ring_id}</div>
                                    <div className="pattern-tags">
                                        <span className="pattern-tag">{ring.type?.replace(/_/g, ' ')}</span>
                                    </div>
                                </div>
                                <div style={{ textAlign: 'right' }}>
                                    <span className={`risk-badge ${risk}`}>{ring.risk_score?.toFixed(1)}</span>
                                    <div className="ring-meta">{ring.node_count} nodes</div>
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        )
    }

    return (
        <div className="table-container fade-up">
            <table className="data-table">
                <thead>
                    <tr>
                        <th style={{ width: '100px' }}>Ring ID</th>
                        <th style={{ width: '150px' }}>Pattern Type</th>
                        <th style={{ width: '100px' }}>Member Count</th>
                        <th style={{ width: '100px' }}>Risk Score</th>
                        <th>Member Account IDs</th>
                    </tr>
                </thead>
                <tbody>
                    {displayRings.map((ring) => {
                        const risk = getRiskLevel(ring.risk_score)
                        return (
                            <tr key={ring.ring_id} className="table-row">
                                <td className="font-mono" style={{ fontWeight: 'bold' }}>{ring.ring_id}</td>
                                <td>
                                    <span className="pattern-tag">
                                        {ring.type?.replace(/_/g, ' ')}
                                    </span>
                                </td>
                                <td style={{ textAlign: 'center' }}>
                                    <div className="member-count-badge">
                                        {ring.node_count}
                                    </div>
                                </td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <div className="risk-dot" style={{ backgroundColor: `var(--${risk})` }}></div>
                                        <span style={{ fontWeight: 600, color: `var(--${risk})` }}>
                                            {ring.risk_score?.toFixed(1)}
                                        </span>
                                    </div>
                                </td>
                                <td>
                                    <div className="account-list-scroll">
                                        {(ring.nodes || []).join(', ')}
                                    </div>
                                </td>
                            </tr>
                        )
                    })}
                </tbody>
            </table>
        </div>
    )
}
