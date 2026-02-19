import { useState } from 'react'

function getRiskLevel(score) {
    if (score >= 70) return 'high'
    if (score >= 40) return 'medium'
    return 'low'
}

function getRiskLabel(score) {
    if (score >= 70) return 'HIGH'
    if (score >= 40) return 'MEDIUM'
    return 'LOW'
}

export default function AccountsTable({ accounts, compact = false }) {
    const [expandedRow, setExpandedRow] = useState(null)
    const [page, setPage] = useState(0)
    const pageSize = compact ? 5 : 20
    const totalPages = Math.ceil(accounts.length / pageSize)
    const displayAccounts = accounts.slice(page * pageSize, (page + 1) * pageSize)

    if (!accounts.length) {
        return (
            <div className="empty-state">
                <div className="icon">✅</div>
                <h3>No Suspicious Accounts</h3>
                <p>No accounts were flagged by the detection modules.</p>
            </div>
        )
    }

    return (
        <div>
            <div className="table-container" style={{ maxHeight: compact ? '400px' : '600px', overflowY: 'auto' }}>
                <table className="table" id="accounts-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Account ID</th>
                            <th>Risk Score</th>
                            <th>Patterns</th>
                            <th>Rings</th>
                            {!compact && <th>Details</th>}
                        </tr>
                    </thead>
                    <tbody>
                        {displayAccounts.map((account, i) => {
                            const risk = getRiskLevel(account.risk_score)
                            const isExpanded = expandedRow === account.account_id

                            return (
                                <>
                                    <tr
                                        key={account.account_id}
                                        onClick={() => setExpandedRow(isExpanded ? null : account.account_id)}
                                        style={{ cursor: 'pointer' }}
                                        id={`account-row-${account.account_id}`}
                                    >
                                        <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                                            {page * pageSize + i + 1}
                                        </td>
                                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                                            {account.account_id}
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <span className={`risk-badge ${risk}`}>
                                                    {getRiskLabel(account.risk_score)} {account.risk_score.toFixed(1)}
                                                </span>
                                                <div className="risk-bar" style={{ width: '60px' }}>
                                                    <div
                                                        className={`risk-bar-fill ${risk}`}
                                                        style={{ width: `${account.risk_score}%` }}
                                                    />
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
                                                {(account.triggered_patterns || []).slice(0, compact ? 2 : 5).map((p, j) => (
                                                    <span key={j} className="pattern-tag">{p.replace(/_/g, ' ')}</span>
                                                ))}
                                                {(account.triggered_patterns || []).length > (compact ? 2 : 5) && (
                                                    <span className="pattern-tag">+{account.triggered_patterns.length - (compact ? 2 : 5)}</span>
                                                )}
                                            </div>
                                        </td>
                                        <td>
                                            {(account.ring_ids || []).length > 0 ? (
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent-secondary)' }}>
                                                    {account.ring_ids.slice(0, 2).join(', ')}
                                                </span>
                                            ) : (
                                                <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>—</span>
                                            )}
                                        </td>
                                        {!compact && (
                                            <td>
                                                <button
                                                    className="btn btn-secondary"
                                                    style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                                                    onClick={(e) => {
                                                        e.stopPropagation()
                                                        setExpandedRow(isExpanded ? null : account.account_id)
                                                    }}
                                                >
                                                    {isExpanded ? 'Hide' : 'View'}
                                                </button>
                                            </td>
                                        )}
                                    </tr>
                                    {isExpanded && (
                                        <tr key={`${account.account_id}-detail`}>
                                            <td colSpan={compact ? 5 : 6} style={{ padding: '0 16px 16px' }}>
                                                <div style={{ background: 'var(--bg-tertiary)', borderRadius: '10px', padding: '16px' }}>
                                                    <div style={{ fontWeight: 700, marginBottom: '8px', fontSize: '0.9rem' }}>
                                                        🔍 Explanation
                                                    </div>
                                                    {(account.explanations || []).map((exp, j) => (
                                                        <div key={j} className="explanation" style={{ marginBottom: '4px' }}>
                                                            {exp}
                                                        </div>
                                                    ))}
                                                    {account.score_breakdown && (
                                                        <div style={{ marginTop: '12px' }}>
                                                            <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '6px' }}>
                                                                Score Breakdown
                                                            </div>
                                                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '4px' }}>
                                                                {account.score_breakdown.map((sb, j) => (
                                                                    <div key={j} style={{
                                                                        fontSize: '0.75rem',
                                                                        fontFamily: 'var(--font-mono)',
                                                                        color: 'var(--text-secondary)',
                                                                        padding: '4px 8px',
                                                                        background: 'var(--bg-card)',
                                                                        borderRadius: '4px',
                                                                    }}>
                                                                        {sb.module}: {(sb.weighted * 100).toFixed(1)}%
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </>
                            )
                        })}
                    </tbody>
                </table>
            </div>

            {!compact && totalPages > 1 && (
                <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '16px' }}>
                    <button
                        className="btn btn-secondary"
                        onClick={() => setPage(Math.max(0, page - 1))}
                        disabled={page === 0}
                        style={{ padding: '4px 12px', fontSize: '0.8rem' }}
                    >
                        ← Previous
                    </button>
                    <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.85rem',
                        color: 'var(--text-secondary)',
                        padding: '4px 12px',
                        display: 'flex',
                        alignItems: 'center',
                    }}>
                        {page + 1} / {totalPages}
                    </span>
                    <button
                        className="btn btn-secondary"
                        onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                        disabled={page >= totalPages - 1}
                        style={{ padding: '4px 12px', fontSize: '0.8rem' }}
                    >
                        Next →
                    </button>
                </div>
            )}
        </div>
    )
}
