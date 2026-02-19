import { useState, useRef, useCallback } from 'react'

export default function UploadZone({ onUpload, onGenerateTestData }) {
    const [isDragOver, setIsDragOver] = useState(false)
    const fileInputRef = useRef(null)

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setIsDragOver(true)
    }, [])

    const handleDragLeave = useCallback((e) => {
        e.preventDefault()
        setIsDragOver(false)
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setIsDragOver(false)
        const file = e.dataTransfer.files[0]
        if (file && file.name.endsWith('.csv')) {
            onUpload(file)
        }
    }, [onUpload])

    const handleFileSelect = useCallback((e) => {
        const file = e.target.files[0]
        if (file) {
            onUpload(file)
        }
    }, [onUpload])

    return (
        <div style={{ maxWidth: '800px', margin: '0 auto' }} className="fade-up">
            <div
                className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                id="upload-zone"
            >
                <div className="upload-title">INITIALIZE_DATA_INGESTION</div>
                <div className="upload-subtitle">
                    Drop forensic CSV dataset or select from directory
                </div>
                <div className="upload-format">
                    EXPECTED: id, sender, receiver, volume, timestamp
                </div>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    id="file-input"
                />
            </div>

            <div style={{
                display: 'flex',
                justifyContent: 'center',
                marginTop: '32px',
                gap: '16px'
            }}>
                <button
                    className="btn btn-primary"
                    onClick={(e) => {
                        e.stopPropagation()
                        onGenerateTestData()
                    }}
                    id="generate-test-data-btn"
                >
                    GENERATE_SYNTHETIC_DATA
                </button>
            </div>

            <div style={{
                marginTop: '80px',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                gap: '24px',
            }}>
                {[
                    { title: 'Cyclic_Analysis', desc: 'Recursive flow detection' },
                    { title: 'Modular_Detection', desc: 'Plug-and-play heuristics' },
                    { title: 'Forensic_Depth', desc: 'Multi-layer correlation' },
                    { title: 'Optimized_Engine', desc: 'Linear time complexity' },
                ].map((feature, i) => (
                    <div key={i} className="card" style={{ padding: '24px', textAlign: 'left', animationDelay: `${(i + 1) * 0.1}s` }}>
                        <div style={{ fontWeight: 600, fontSize: '0.7rem', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>
                            {feature.title}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4, opacity: 0.6 }}>
                            {feature.desc}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
