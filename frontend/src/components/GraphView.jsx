export default function GraphView({ url, fullHeight = false }) {
    return (
        <div className="graph-container" id="graph-view">
            <iframe
                src={url}
                title="Transaction Network Graph"
                style={{ height: fullHeight ? '750px' : '600px' }}
                sandbox="allow-scripts allow-same-origin"
            />
        </div>
    )
}
