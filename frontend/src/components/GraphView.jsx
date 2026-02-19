export default function GraphView({ url, fullHeight = false }) {
    // Note: The graph-container CSS already defines the horizontal centralized rectangle shape.
    return (
        <div className="fade-up">
            <div className="graph-container" id="graph-view">
                <iframe
                    src={url}
                    title="Transaction Network Graph"
                    sandbox="allow-scripts allow-same-origin"
                />
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
