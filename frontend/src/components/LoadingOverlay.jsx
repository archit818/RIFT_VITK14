export default function LoadingOverlay({ message }) {
    return (
        <div className="loading-overlay">
            <div className="loading-content">
                <div className="loading-spinner" />
                <div className="loading-text">Analyzing Transactions</div>
                <div className="loading-subtext">{message || 'Processing...'}</div>
                <div className="progress-bar">
                    <div className="progress-fill" />
                </div>
            </div>
        </div>
    )
}
