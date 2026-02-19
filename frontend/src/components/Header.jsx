export default function Header({ onReset, hasResults }) {
    return (
        <header className="header">
            <div className="header-brand">
                <div className="header-logo">R</div>
                <div>
                    <div className="header-title">RIFT</div>
                    <div className="header-subtitle">Financial Forensics Engine</div>
                </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                {hasResults && (
                    <button className="btn btn-secondary" onClick={onReset}>
                        New Analysis
                    </button>
                )}
                <div className="header-status">
                    <span className="dot"></span>
                    SYSTEM ONLINE
                </div>
            </div>
        </header>
    )
}
