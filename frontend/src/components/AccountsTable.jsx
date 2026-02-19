import { useState, Fragment } from 'react'

function getRiskLevel(score) {
    if (score >= 80) return 'high'
    if (score >= 55) return 'medium'
    return 'low'
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
                <h3>No Suspicious Accounts</h3>
                <p>No accounts were flagged by the detection modules.</p>
            </div>
        )
    }

    return (
        <div className="fade-up">
            <div className="table-container">
                <table className="table" id="accounts-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Account ID</th>
                            <th>Risk Score</th>
                            <th>Tier</th>
                            <th>Patterns</th>
                            <th>Rings</th>
                            {!compact && <th>Action</th>}
                        </tr>
                    </thead>
                    <tbody>
                        {displayAccounts.map((account, i) => {
                            const risk = getRiskLevel(account.risk_score)
                            const isExpanded = expandedRow === account.account_id

                            return (
                                <Fragment key={account.account_id}>
                                    <tr
                                        onClick={() => setExpandedRow(isExpanded ? null : account.account_id)}
                                        style={{ cursor: 'pointer' }}
                                        id={`account-row-${account.account_id}`}
                                    >
                                        <td className="font-mono" style={{ opacity: 0.5 }}>
                                            {page * pageSize + i + 1}
                                        </td>
                                        <td className="font-mono" style={{ color: 'var(--text-primary)' }}>
                                            {account.account_id}
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                                <span className={`risk-badge ${risk}`}>
                                                    {account.risk_score.toFixed(0)}
                                                </span>
                                                <div className="risk-bar" style={{ width: '40px', height: '2px', background: 'rgba(255,255,255,0.05)' }}>
                                                    <div
                                                        className={`risk-bar-fill ${risk}`}
                                                        style={{ width: `${account.risk_score}%`, height: '100%' }}
                                                    />
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`tier-badge tier-${(account.tier || 'low').toLowerCase()}`}>
                                                {account.tier || 'LOW'}
                                            </span>
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                                {(account.patterns || []).slice(0, compact ? 2 : 5).map((p, j) => (
                                                    <span key={j} className="pattern-tag">{p.replace(/_/g, ' ')}</span>
                                                ))}
                                                {(account.patterns || []).length > (compact ? 2 : 5) && (
                                                    <span className="pattern-tag">+{account.patterns.length - (compact ? 2 : 5)}</span>
                                                )}
                                            </div>
                                        </td>
                                        <td className="font-mono">
                                            {(account.ring_ids || []).length > 0 ? (
                                                <span style={{ opacity: 0.8 }}>
                                                    {account.ring_ids.slice(0, 1).join(', ')}
                                                </span>
                                            ) : (
                                                <span style={{ opacity: 0.3 }}>—</span>
                                            )}
                                        </td>
                                        {!compact && (
                                            <td>
                                                <button className="btn" style={{ padding: '4px 12px', fontSize: '0.7rem' }}>
                                                    {isExpanded ? 'Hide' : 'Details'}
                                                </button>
                                            </td>
                                        )}
                                    </tr>
                                    {isExpanded && (
                                        <tr key={`${account.account_id}-detail`}>
                                            <td colSpan={compact ? 6 : 7} style={{ padding: '0 24px 24px' }}>
                                                <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', padding: '24px', border: '1px solid var(--border)' }}>
                                                    <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '16px' }}>
                                                        Forensic Explanation
                                                    </div>
                                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                            {account.explanation}
                                                        </div>
                                                    </div>

                                                    {account.signal_summary && (
                                                        <div style={{ display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap' }}>
                                                            {account.signal_summary.multi_signal_gate && (
                                                                <span className="tier-badge tier-high" style={{ fontSize: '0.55rem' }}>MULTI-SIGNAL CONFIRMED</span>
                                                            )}
                                                            {account.signal_summary.has_structural && (
                                                                <span className="pattern-tag" style={{ borderColor: 'var(--accent)' }}>STRUCTURAL</span>
                                                            )}
                                                            {account.signal_summary.has_behavioral && (
                                                                <span className="pattern-tag" style={{ borderColor: 'var(--warning)' }}>BEHAVIORAL</span>
                                                            )}
                                                        </div>
                                                    )}

                                                    {account.score_breakdown && (
                                                        <div style={{ marginTop: '24px' }}>
                                                            <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>
                                                                Detection Metrics
                                                            </div>
                                                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '8px' }}>
                                                                {account.score_breakdown.map((sb, j) => (
                                                                    <div key={j} style={{
                                                                        fontSize: '0.7rem',
                                                                        fontFamily: 'var(--font-mono)',
                                                                        color: 'var(--text-secondary)',
                                                                        padding: '8px 12px',
                                                                        background: 'rgba(255,255,255,0.02)',
                                                                        borderRadius: '4px',
                                                                        border: '1px solid var(--border)',
                                                                        display: 'flex',
                                                                        justifyContent: 'space-between',
                                                                        alignItems: 'center',
                                                                    }}>
                                                                        <span>{sb.module.replace(/_/g, ' ')}</span>
                                                                        <span style={{
                                                                            color: sb.signal_strength === 'STRONG' ? 'var(--danger)' : 'var(--text-muted)',
                                                                            fontSize: '0.6rem',
                                                                        }}>
                                                                            {sb.signal_class || sb.signal_strength} · {(sb.weighted * 100).toFixed(0)}%
                                                                        </span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </Fragment>
                            )
                        })}
                    </tbody>
                </table>
            </div>

            {!compact && totalPages > 1 && (
                <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '32px' }}>
                    <button
                        className="btn"
                        onClick={() => setPage(Math.max(0, page - 1))}
                        disabled={page === 0}
                        style={{ padding: '4px 12px', fontSize: '0.8rem' }}
                    >
                        Prev
                    </button>
                    <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                        padding: '4px 12px',
                        display: 'flex',
                        alignItems: 'center',
                    }}>
                        {page + 1} of {totalPages}
                    </span>
                    <button
                        className="btn"
                        onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                        disabled={page >= totalPages - 1}
                        style={{ padding: '4px 12px', fontSize: '0.8rem' }}
                    >
                        Next
                    </button>
                </div>
            )}
        </div>
    )
}
