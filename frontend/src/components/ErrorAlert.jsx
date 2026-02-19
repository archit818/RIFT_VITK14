export default function ErrorAlert({ message, onDismiss }) {
    return (
        <div className="alert alert-error" role="alert">
            <span className="alert-icon">⚠️</span>
            <div style={{ flex: 1 }}>
                <div className="alert-message">{message}</div>
            </div>
            {onDismiss && (
                <button
                    onClick={onDismiss}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--danger)',
                        cursor: 'pointer',
                        fontSize: '1.2rem',
                        padding: '4px',
                    }}
                    aria-label="Dismiss"
                >
                    ✕
                </button>
            )}
        </div>
    )
}
