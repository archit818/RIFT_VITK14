export default function Header({ onReset, hasResults }) {
    return (
        <header className="header">
            <div className="header-brand">
                <div className="header-logo">R</div>
                <div>
                    <div className="header-title">RIFT</div>
                    <div className="header-subtitle">Intelligence Engine</div>
                </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
                <div className="header-status">
                    <span className="dot"></span>
                    SECURE_NODE_ONLINE
                </div>
                {hasResults && (
                    <button className="btn btn-secondary" onClick={onReset} style={{ fontSize: '0.75rem', padding: '6px 16px' }}>
                        TERMINATE_ANALYSIS
                    </button>
                )}
            </div>
        </header>
    )
}
