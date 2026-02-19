import { useState, useRef, useCallback } from 'react'

export default function UploadZone({ onUpload, onGenerateTestData }) {
    const [isDragOver, setIsDragOver] = useState(false)
    const [selectedFile, setSelectedFile] = useState(null)
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
            setSelectedFile(file)
            onUpload(file)
        }
    }, [onUpload])

    const handleFileSelect = useCallback((e) => {
        const file = e.target.files[0]
        if (file) {
            setSelectedFile(file)
            onUpload(file)
        }
    }, [onUpload])

    return (
        <div style={{ maxWidth: '700px', margin: '0 auto' }}>
            <div
                className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                id="upload-zone"
            >
                <span className="upload-icon">🔍</span>
                <div className="upload-title">Upload Transaction Data</div>
                <div className="upload-subtitle">
                    Drag & drop your CSV file here, or click to browse
                </div>
                <div className="upload-format">
                    Required: transaction_id, sender_id, receiver_id, amount, timestamp
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
                marginTop: '24px',
                gap: '12px',
                flexWrap: 'wrap'
            }}>
                <button
                    className="btn btn-secondary"
                    onClick={(e) => {
                        e.stopPropagation()
                        onGenerateTestData()
                    }}
                    id="generate-test-data-btn"
                >
                    🧪 Generate & Analyze Test Data
                </button>
            </div>

            <div style={{
                marginTop: '48px',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px',
                textAlign: 'center'
            }}>
                {[
                    { icon: '🔄', title: 'Cycle Detection', desc: 'Find circular money flows' },
                    { icon: '📊', title: '12 Modules', desc: 'Comprehensive pattern analysis' },
                    { icon: '🎯', title: 'Precision Tuned', desc: 'Low false positive rate' },
                    { icon: '⚡', title: 'Fast Analysis', desc: '≤ 30s for 10K transactions' },
                ].map((feature, i) => (
                    <div key={i} className={`card fade-up fade-up-delay-${i + 1}`} style={{ textAlign: 'center', padding: '20px' }}>
                        <div style={{ fontSize: '2rem', marginBottom: '8px' }}>{feature.icon}</div>
                        <div style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '4px' }}>{feature.title}</div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{feature.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}
