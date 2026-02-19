export default function GraphView({ url, fullHeight = false }) {
    const handleDownload = () => {
        // Find the base URL (e.g., http://localhost:8000)
        const baseUrl = url.split('/static')[0];
        window.location.href = `${baseUrl}/api/download-graph`;
    };

    // Note: The graph-container CSS already defines the horizontal centralized rectangle shape.
    return (
        <div className="fade-up">
            <div className="graph-container" id="graph-view" style={{ position: 'relative' }}>
                <iframe
                    src={url}
                    title="Transaction Network Graph"
                    sandbox="allow-scripts allow-same-origin"
                />
                <button
                    onClick={handleDownload}
                    className="btn btn-secondary"
                    style={{
                        position: 'absolute',
                        bottom: '20px',
                        right: '20px',
                        fontSize: '0.6rem',
                        padding: '6px 12px',
                        zIndex: 10,
                        background: 'rgba(10, 10, 26, 0.9)',
                        backdropFilter: 'blur(4px)'
                    }}
                >
                    DOWNLOAD_TOPOLOGY_REPORT
                </button>
            </div>
            <div style={{
                marginTop: '12px',
                fontSize: '0.6rem',
                color: 'var(--text-muted)',
                textAlign: 'center',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                letterSpacing: '0.1em'
            }}>
                Topology_Visualization_Mode: Active
            </div>
        </div>
    )
}
