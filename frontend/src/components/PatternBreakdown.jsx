export default function PatternBreakdown({ summary, accounts }) {
    const topPatterns = summary.top_patterns || []
    const detectionModules = summary.detection_modules_triggered || []

    // Calculate pattern frequency across all accounts
    const patternFreq = {}
        ; (accounts || []).forEach(acc => {
            (acc.triggered_patterns || []).forEach(p => {
                patternFreq[p] = (patternFreq[p] || 0) + 1
            })
        })

    const sortedPatterns = Object.entries(patternFreq)
        .sort((a, b) => b[1] - a[1])

    const maxCount = sortedPatterns.length > 0 ? sortedPatterns[0][1] : 1

    const patternDescriptions = {
        circular_routing: 'Circular fund flows through 3-5 account cycles',
        fan_in_aggregation: 'Multiple senders funneling into single account',
        fan_out_dispersal: 'Rapid dispersal to many receivers',
        shell_chain: 'Fund layering through low-activity intermediaries',
        transaction_burst: 'Sudden spikes in transaction frequency',
        rapid_movement: 'Multi-hop transfers within hours',
        dormant_activation: 'Sudden activation after prolonged inactivity',
        structuring: 'Repeated near-threshold amounts (smurfing)',
        amount_consistency_ring: 'Equal transfer amounts in detected cycles',
        diversity_shift: 'Sudden increase in unique counterparties',
        centrality_spike: 'Node becoming a sudden network hub',
        community_suspicion: 'Cluster-level risk propagation',
    }

    const patternIcons = {
        circular_routing: '🔄',
        fan_in_aggregation: '📥',
        fan_out_dispersal: '📤',
        shell_chain: '🐚',
        transaction_burst: '💥',
        rapid_movement: '⚡',
        dormant_activation: '😴',
        structuring: '📐',
        amount_consistency_ring: '💰',
        diversity_shift: '🔀',
        centrality_spike: '📍',
        community_suspicion: '👥',
    }

    return (
        <div>
            <h2 className="section-title">
                🧩 Detection Module Analysis
                <span className="count">{detectionModules.length} active</span>
            </h2>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
                gap: '16px',
                marginBottom: '32px',
            }}>
                {sortedPatterns.map(([pattern, count], i) => (
                    <div key={pattern} className={`card fade-up fade-up-delay-${Math.min(i + 1, 4)}`}>
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                            <div style={{ fontSize: '2rem' }}>
                                {patternIcons[pattern] || '🔍'}
                            </div>
                            <div style={{ flex: 1 }}>
                                <div style={{
                                    fontWeight: 700,
                                    fontSize: '0.95rem',
                                    marginBottom: '4px',
                                    textTransform: 'capitalize',
                                }}>
                                    {pattern.replace(/_/g, ' ')}
                                </div>
                                <div style={{
                                    fontSize: '0.8rem',
                                    color: 'var(--text-secondary)',
                                    marginBottom: '8px',
                                }}>
                                    {patternDescriptions[pattern] || 'Pattern detection module'}
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <div className="risk-bar" style={{ flex: 1 }}>
                                        <div
                                            className="risk-bar-fill medium"
                                            style={{ width: `${(count / maxCount) * 100}%` }}
                                        />
                                    </div>
                                    <span style={{
                                        fontFamily: 'var(--font-mono)',
                                        fontSize: '0.85rem',
                                        fontWeight: 700,
                                        color: 'var(--accent-secondary)',
                                        minWidth: '40px',
                                        textAlign: 'right',
                                    }}>
                                        {count}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Module stats from backend */}
            {topPatterns.length > 0 && (
                <>
                    <h2 className="section-title">📈 Module Performance</h2>
                    <div className="table-container">
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Module</th>
                                    <th>Detections</th>
                                    <th>Avg Risk</th>
                                    <th>Impact</th>
                                </tr>
                            </thead>
                            <tbody>
                                {topPatterns.map((tp, i) => (
                                    <tr key={i}>
                                        <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>
                                            {patternIcons[tp.pattern] || '🔍'} {tp.pattern?.replace(/_/g, ' ')}
                                        </td>
                                        <td>
                                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                                                {tp.count}
                                            </span>
                                        </td>
                                        <td>
                                            <span style={{
                                                fontFamily: 'var(--font-mono)',
                                                color: tp.avg_risk >= 0.7 ? 'var(--danger)' :
                                                    tp.avg_risk >= 0.4 ? 'var(--warning)' : 'var(--success)',
                                            }}>
                                                {(tp.avg_risk * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td>
                                            <div className="risk-bar" style={{ width: '80px' }}>
                                                <div
                                                    className={`risk-bar-fill ${tp.avg_risk >= 0.7 ? 'high' : tp.avg_risk >= 0.4 ? 'medium' : 'low'}`}
                                                    style={{ width: `${tp.avg_risk * 100}%` }}
                                                />
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}

            {/* Suspicious Amount Summary */}
            {summary.total_suspicious_amount > 0 && (
                <div className="card" style={{ marginTop: '24px', textAlign: 'center' }}>
                    <div style={{
                        fontSize: '0.85rem',
                        color: 'var(--text-secondary)',
                        textTransform: 'uppercase',
                        letterSpacing: '1px',
                        marginBottom: '8px',
                    }}>
                        Total Suspicious Flow
                    </div>
                    <div style={{
                        fontSize: '2.5rem',
                        fontWeight: 800,
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--danger)',
                    }}>
                        ${summary.total_suspicious_amount?.toLocaleString()}
                    </div>
                </div>
            )}
        </div>
    )
}
