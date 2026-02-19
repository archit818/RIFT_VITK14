import { useState } from 'react'
import StatsGrid from './StatsGrid'
import AccountsTable from './AccountsTable'
import FraudRings from './FraudRings'
import GraphView from './GraphView'
import PatternBreakdown from './PatternBreakdown'

const TABS = [
    { id: 'overview', label: '📊 Overview', icon: '📊' },
    { id: 'accounts', label: '👤 Suspicious Accounts', icon: '👤' },
    { id: 'rings', label: '🔗 Fraud Rings', icon: '🔗' },
    { id: 'graph', label: '🌐 Network Graph', icon: '🌐' },
    { id: 'patterns', label: '🧩 Patterns', icon: '🧩' },
]

export default function Dashboard({ results, apiBase, onReset }) {
    const [activeTab, setActiveTab] = useState('overview')
    const { suspicious_accounts = [], fraud_rings = [], summary = {} } = results || {}

    const handleDownloadJson = () => {
        window.location.href = `${apiBase}/api/export-json`;
    };

    return (
        <div className="dashboard">
            {/* Summary success alert */}
            <div className="alert alert-success" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span className="alert-icon">✅</span>
                    <div className="alert-message">
                        Analysis complete in <strong>{summary.processing_time_seconds}s</strong>.
                        Found <strong>{summary.suspicious_accounts_found}</strong> suspicious accounts
                        and <strong>{summary.fraud_rings_detected}</strong> fraud rings.
                    </div>
                </div>
                <button
                    className="btn btn-primary"
                    onClick={handleDownloadJson}
                    style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', color: 'white' }}
                >
                    📥 Download JSON Report
                </button>
            </div>

            {/* Tabs */}
            <div className="tabs" role="tablist">
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                        role="tab"
                        aria-selected={activeTab === tab.id}
                        id={`tab-${tab.id}`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'overview' && (
                <div className="fade-up">
                    <StatsGrid summary={summary} />

                    <div className="two-col" style={{ marginTop: '24px' }}>
                        <div>
                            <h2 className="section-title">
                                🔥 Top Threats
                                <span className="count">{Math.min(5, suspicious_accounts.length)}</span>
                            </h2>
                            <AccountsTable
                                accounts={suspicious_accounts.slice(0, 5)}
                                compact
                            />
                        </div>
                        <div>
                            <h2 className="section-title">
                                🔗 Active Rings
                                <span className="count">{Math.min(5, fraud_rings.length)}</span>
                            </h2>
                            <FraudRings rings={fraud_rings.slice(0, 5)} compact />
                        </div>
                    </div>

                    {results.graph_url && (
                        <div style={{ marginTop: '24px' }}>
                            <h2 className="section-title">🌐 Network Overview</h2>
                            <GraphView url={`${apiBase}${results.graph_url}`} />
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'accounts' && (
                <div className="fade-up">
                    <h2 className="section-title">
                        👤 Suspicious Accounts
                        <span className="count">{suspicious_accounts.length}</span>
                    </h2>
                    <AccountsTable accounts={suspicious_accounts} />
                </div>
            )}

            {activeTab === 'rings' && (
                <div className="fade-up">
                    <h2 className="section-title">
                        🔗 Fraud Rings
                        <span className="count">{fraud_rings.length}</span>
                    </h2>
                    <FraudRings rings={fraud_rings} />
                </div>
            )}

            {activeTab === 'graph' && (
                <div className="fade-up">
                    <h2 className="section-title">🌐 Interactive Network Graph</h2>
                    {results.graph_url ? (
                        <GraphView url={`${apiBase}${results.graph_url}`} fullHeight />
                    ) : (
                        <div className="graph-placeholder">
                            <div className="icon">🌐</div>
                            <h3>No Graph Available</h3>
                            <p>Graph visualization was not generated for this analysis.</p>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'patterns' && (
                <div className="fade-up">
                    <PatternBreakdown summary={summary} accounts={suspicious_accounts} />
                </div>
            )}

            {/* Footer Actions */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', marginTop: '40px', paddingBottom: '48px' }}>
                <button className="btn btn-secondary" onClick={onReset} id="new-analysis-btn">
                    ← Start New Analysis
                </button>
                <button
                    className="btn btn-primary"
                    onClick={handleDownloadJson}
                    id="download-json-btn"
                >
                    📥 Download JSON Report
                </button>
            </div>
        </div>
    );
}

