export default function ErrorAlert({ message, onDismiss }) {
    return (
        <div className="fade-up" style={{
            background: 'rgba(255, 77, 77, 0.05)',
            border: '1px solid rgba(255, 77, 77, 0.2)',
            borderRadius: '8px',
            padding: '16px 24px',
            marginBottom: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '16px'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ width: '6px', height: '6px', background: 'var(--danger)', borderRadius: '50%' }} />
                <div className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--danger)', letterSpacing: '0.05em' }}>
                    EXE_SYSTEM_FAULT: {message}
                </div>
            </div>
            {onDismiss && (
                <button
                    onClick={onDismiss}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--danger)',
                        cursor: 'pointer',
                        fontSize: '1rem',
                        opacity: 0.6,
                    }}
                >
                    ✕
                </button>
            )}
        </div>
    )
}
