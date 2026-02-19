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
                <div className="icon">🔗</div>
                <h3>No Fraud Rings Detected</h3>
                <p>No coordinated fraud ring patterns were identified.</p>
            </div>
        )
    }

    return (
        <div>
            {displayRings.map((ring, i) => {
                const risk = getRiskLevel(ring.risk_score)
                return (
                    <div key={ring.ring_id} className="ring-card" id={`ring-${ring.ring_id}`}>
                        <div className="ring-header">
                            <div>
                                <div className="ring-id">{ring.ring_id}</div>
                                <div style={{ display: 'flex', gap: '6px', marginTop: '4px', flexWrap: 'wrap' }}>
                                    <span className="pattern-tag">{ring.type?.replace(/_/g, ' ')}</span>
                                    {(ring.patterns || []).map((p, j) => (
                                        <span key={j} className="pattern-tag">{p.replace(/_/g, ' ')}</span>
                                    ))}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <span className={`risk-badge ${risk}`}>
                                    {ring.risk_score?.toFixed(1)}
                                </span>
                                <div style={{
                                    marginTop: '4px',
                                    fontSize: '0.75rem',
                                    fontFamily: 'var(--font-mono)',
                                    color: 'var(--text-secondary)',
                                }}>
                                    {ring.node_count} nodes · ${ring.total_amount?.toLocaleString()}
                                </div>
                            </div>
                        </div>

                        <div className="risk-bar" style={{ marginBottom: '8px' }}>
                            <div
                                className={`risk-bar-fill ${risk}`}
                                style={{ width: `${ring.risk_score}%` }}
                            />
                        </div>

                        {!compact && (ring.explanations || []).length > 0 && (
                            <div className="explanation">
                                {ring.explanations[0]}
                            </div>
                        )}

                        <div className="ring-nodes">
                            {(ring.nodes || []).slice(0, compact ? 6 : 20).map((node, j) => (
                                <span key={j} className="ring-node">{node}</span>
                            ))}
                            {(ring.nodes || []).length > (compact ? 6 : 20) && (
                                <span className="ring-node" style={{ color: 'var(--text-muted)' }}>
                                    +{ring.nodes.length - (compact ? 6 : 20)} more
                                </span>
                            )}
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
