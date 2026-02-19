export default function PatternBreakdown({ summary, accounts }) {
    const topPatterns = summary.top_patterns || []

    // Calculate pattern frequency across all accounts
    const patternFreq = {}
        ; (accounts || []).forEach(acc => {
            (acc.patterns || acc.triggered_patterns || []).forEach(p => {
                patternFreq[p] = (patternFreq[p] || 0) + 1
            })
        })

    const sortedPatterns = Object.entries(patternFreq)
        .sort((a, b) => b[1] - a[1])

    const maxCount = sortedPatterns.length > 0 ? sortedPatterns[0][1] : 1

    const patternDescriptions = {
        circular_routing: 'Cyclic flows through multi-intermediary nodes',
        fan_in_aggregation: 'Convergent volume funneling into terminal accounts',
        fan_out_dispersal: 'Divergent dispersal from central liquidity source',
        shell_chain: 'Layering through established low-activity proxies',
        transaction_burst: 'Uncharacteristic temporal volume distribution',
        rapid_movement: 'Velocity-based hop analysis between counterparties',
        dormant_activation: 'Systematic activation of legacy/aged accounts',
        structuring: 'Sub-threshold systematic volume layering',
        diversity_shift: 'Counterparty breadth deviation from baseline',
        centrality_spike: 'Structural hub formation in network topology',
    }

    return (
        <div className="fade-up">
            <h2 className="section-title">Heuristic Correlation Analysis</h2>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '12px',
                marginBottom: '40px',
            }}>
                {sortedPatterns.map(([pattern, count], i) => (
                    <div key={pattern} className="card" style={{ padding: '20px', animationDelay: `${i * 0.05}s` }}>
                        <div style={{ fontWeight: 600, fontSize: '0.7rem', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>
                            {pattern.replace(/_/g, ' ')}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', opacity: 0.6, marginBottom: '16px' }}>
                            {patternDescriptions[pattern] || 'System detection module'}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div className="risk-bar" style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.05)' }}>
                                <div
                                    className="risk-bar-fill medium"
                                    style={{ width: `${(count / maxCount) * 100}%`, height: '100%', background: 'var(--text-primary)' }}
                                />
                            </div>
                            <span className="font-mono" style={{ fontSize: '0.75rem', opacity: 0.8 }}>
                                {count}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {topPatterns.length > 0 && (
                <>
                    <h2 className="section-title">Engine Performance Metrics</h2>
                    <div className="table-container">
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Heuristic</th>
                                    <th>Instances</th>
                                    <th>Confidence</th>
                                    <th>Topology Influence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {topPatterns.map((tp, i) => (
                                    <tr key={i}>
                                        <td className="font-mono" style={{ textTransform: 'uppercase', fontSize: '0.75rem' }}>
                                            {tp.pattern?.replace(/_/g, ' ')}
                                        </td>
                                        <td className="font-mono">{tp.count}</td>
                                        <td>
                                            <span className="font-mono" style={{ color: 'var(--text-primary)', opacity: 0.8 }}>
                                                {(tp.avg_risk * 100).toFixed(0)}%
                                            </span>
                                        </td>
                                        <td>
                                            <div className="risk-bar" style={{ width: '60px', height: '1px' }}>
                                                <div
                                                    className={`risk-bar-fill ${tp.avg_risk >= 0.7 ? 'high' : tp.avg_risk >= 0.4 ? 'medium' : 'low'}`}
                                                    style={{ width: `${tp.avg_risk * 100}%`, height: '100%' }}
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

            {summary.total_suspicious_amount > 0 && (
                <div className="card" style={{ marginTop: '40px', padding: '40px', textAlign: 'center', borderStyle: 'dashed' }}>
                    <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.2em', color: 'var(--text-muted)', marginBottom: '16px' }}>
                        AGGREGATED_EXPOSURE_VALUATION
                    </div>
                    <div className="font-mono" style={{ fontSize: '3rem', fontWeight: 200, color: 'var(--text-primary)' }}>
                        ${summary.total_suspicious_amount?.toLocaleString()}
                    </div>
                </div>
            )}
        </div>
    )
}
