export default function LoadingOverlay({ message }) {
    return (
        <div className="loading-overlay">
            <div className="loading-content" style={{ textAlign: 'center' }}>
                <div className="loading-spinner" style={{ margin: '0 auto' }} />
                <div className="loading-text" style={{ marginTop: '24px', letterSpacing: '0.2em' }}>
                    EXE_PROCESSING_DATA
                </div>
                <div className="font-mono" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '8px', textTransform: 'uppercase' }}>
                    {message || 'Awaiting system response...'}
                </div>
            </div>
        </div>
    )
}
