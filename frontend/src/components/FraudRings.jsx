import { Fragment } from 'react'

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
                <p>No coordinated patterns identified in this dataset.</p>
            </div>
        )
    }

    if (compact) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {displayRings.map((ring) => {
                    const risk = getRiskLevel(ring.risk_score)
                    return (
                        <div key={ring.ring_id} className="card" style={{ padding: '16px 24px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div className="font-mono" style={{ fontSize: '0.8rem', fontWeight: 600 }}>{ring.ring_id}</div>
                                    <div style={{ marginTop: '4px' }}>
                                        <span className="pattern-tag">{ring.type?.replace(/_/g, ' ')}</span>
                                    </div>
                                </div>
                                <div style={{ textAlign: 'right' }}>
                                    <span className={`risk-badge ${risk}`}>{ring.risk_score?.toFixed(0)}</span>
                                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
                                        {ring.node_count} nodes
                                    </div>
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
            <table className="table">
                <thead>
                    <tr>
                        <th style={{ width: '120px' }}>Ring ID</th>
                        <th>Pattern Type</th>
                        <th style={{ width: '100px' }}>Members</th>
                        <th style={{ width: '100px' }}>Risk</th>
                        <th>Associated Nodes</th>
                    </tr>
                </thead>
                <tbody>
                    {displayRings.map((ring) => {
                        const risk = getRiskLevel(ring.risk_score)
                        return (
                            <tr key={ring.ring_id}>
                                <td className="font-mono" style={{ color: 'var(--text-primary)' }}>{ring.ring_id}</td>
                                <td>
                                    <span className="pattern-tag">
                                        {ring.type?.replace(/_/g, ' ')}
                                    </span>
                                </td>
                                <td style={{ textAlign: 'center' }}>
                                    <div style={{
                                        display: 'inline-flex',
                                        padding: '2px 8px',
                                        background: 'rgba(255,255,255,0.03)',
                                        borderRadius: '4px',
                                        fontSize: '0.75rem',
                                        fontFamily: 'var(--font-mono)'
                                    }}>
                                        {ring.node_count}
                                    </div>
                                </td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <span className={`risk-badge ${risk}`}>
                                            {ring.risk_score?.toFixed(0)}
                                        </span>
                                    </div>
                                </td>
                                <td>
                                    <div style={{
                                        maxWidth: '300px',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap',
                                        fontSize: '0.75rem',
                                        opacity: 0.6,
                                        fontFamily: 'var(--font-mono)'
                                    }}>
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
