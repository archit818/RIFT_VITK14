import { useState } from 'react'
import StatsGrid from './StatsGrid'
import AccountsTable from './AccountsTable'
import FraudRings from './FraudRings'
import GraphView from './GraphView'
import PatternBreakdown from './PatternBreakdown'
import Chatbot from './Chatbot'

const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'accounts', label: 'Suspicious Accounts' },
    { id: 'rings', label: 'Fraud Rings' },
    { id: 'graph', label: 'Network Graph' },
    { id: 'patterns', label: 'Patterns' },
]

export default function Dashboard({ results, apiBase, onReset }) {
    const [activeTab, setActiveTab] = useState('overview')
    const { suspicious_accounts = [], fraud_rings = [], summary = {} } = results || {}

    const handleDownloadJson = () => {
        window.location.href = `${apiBase}/api/export-json`;
    };

    const handleDownloadGraph = () => {
        window.location.href = `${apiBase}/api/download-graph`;
    };

    return (
        <div className="dashboard">
            {/* Summary success alert */}
            <div style={{
                background: 'rgba(0, 255, 153, 0.05)',
                border: '1px solid rgba(0, 255, 153, 0.1)',
                borderRadius: '8px',
                padding: '20px 32px',
                marginBottom: '48px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ width: '8px', height: '8px', background: 'var(--success)', borderRadius: '50%', boxShadow: '0 0 10px var(--success)' }} />
                    <div className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--success)', letterSpacing: '0.05em' }}>
                        ANALYSIS_COMPLETE: SUCCESSFUL_DETECTION (Nodes: {summary.total_accounts_analyzed})
                    </div>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={handleDownloadJson}
                    style={{ fontSize: '0.65rem', padding: '6px 16px' }}
                >
                    EXPORT_FORENSIC_REPORT
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
                                Top Threats
                                <span className="count">{Math.min(5, suspicious_accounts.length)}</span>
                            </h2>
                            <AccountsTable
                                accounts={suspicious_accounts.slice(0, 5)}
                                compact
                            />
                        </div>
                        <div>
                            <h2 className="section-title">
                                Active Rings
                                <span className="count">{Math.min(5, fraud_rings.length)}</span>
                            </h2>
                            <FraudRings rings={fraud_rings.slice(0, 5)} compact />
                        </div>
                    </div>

                    {results.graph_url && (
                        <div style={{ marginTop: '24px' }}>
                            <h2 className="section-title">Network Overview</h2>
                            <GraphView url={`${apiBase}${results.graph_url}`} />
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'accounts' && (
                <div className="fade-up">
                    <h2 className="section-title">
                        Suspicious Accounts
                        <span className="count">{suspicious_accounts.length}</span>
                    </h2>
                    <AccountsTable accounts={suspicious_accounts} />
                </div>
            )}

            {activeTab === 'rings' && (
                <div className="fade-up">
                    <h2 className="section-title">
                        Fraud Rings
                        <span className="count">{fraud_rings.length}</span>
                    </h2>
                    <FraudRings rings={fraud_rings} />
                </div>
            )}

            {activeTab === 'graph' && (
                <div className="fade-up">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                        <h2 className="section-title" style={{ margin: 0 }}>Interactive Network Graph</h2>
                        {results.graph_url && (
                            <button className="btn btn-secondary" onClick={handleDownloadGraph} style={{ fontSize: '0.65rem' }}>
                                DOWNLOAD_TOPOLOGY_MAP
                            </button>
                        )}
                    </div>
                    {results.graph_url ? (
                        <GraphView url={`${apiBase}${results.graph_url}`} fullHeight />
                    ) : (
                        <div className="graph-placeholder">
                            <h3>No Graph Available</h3>
                            <p>Visualization was not generated for this analysis.</p>
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
            <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', marginTop: '64px', paddingBottom: '64px' }}>
                <button className="btn btn-secondary" onClick={onReset} id="new-analysis-btn">
                    TERMINATE_SESSION
                </button>
                <button className="btn btn-secondary" onClick={handleDownloadGraph} id="download-graph-btn">
                    SAVE_TOPOLOGY_MAP
                </button>
                <button
                    className="btn btn-primary"
                    onClick={handleDownloadJson}
                    id="download-json-btn"
                >
                    GENERATE_FULL_REPORT
                </button>
            </div>

            {/* Task 8: Chatbot Explainability Layer */}
            <Chatbot apiBase={apiBase} accounts={suspicious_accounts} />
        </div>
    );
}

