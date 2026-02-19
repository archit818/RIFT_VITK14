import { useState, useCallback } from 'react'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import LoadingOverlay from './components/LoadingOverlay'
import Dashboard from './components/Dashboard'
import ErrorAlert from './components/ErrorAlert'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [loadingMessage, setLoadingMessage] = useState('')

  const handleUpload = useCallback(async (file) => {
    setLoading(true)
    setError(null)
    setLoadingMessage('Uploading and validating CSV...')

    try {
      const formData = new FormData()
      formData.append('file', file)

      setLoadingMessage('Running detection modules...')

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => null)
        const msg = errData?.detail || errData?.error || `Server error: ${response.status}`
        throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
      }

      const data = await response.json()
      setResults(data)
      setLoadingMessage('')
    } catch (err) {
      setError(err.message)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleGenerateTestData = useCallback(async () => {
    setLoading(true)
    setError(null)
    setLoadingMessage('Generating synthetic fraud data...')

    try {
      const response = await fetch(`${API_BASE}/api/generate-test-data`, {
        method: 'POST',
      })

      if (!response.ok) throw new Error('Failed to generate test data')

      const data = await response.json()

      // Download the generated data
      const csvResponse = await fetch(`${API_BASE}${data.download_url}`)
      const csvBlob = await csvResponse.blob()
      const csvFile = new File([csvBlob], 'test_data.csv', { type: 'text/csv' })

      // Now analyze it
      await handleUpload(csvFile)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [handleUpload])

  const handleReset = useCallback(() => {
    setResults(null)
    setError(null)
    setLoadingMessage('')
  }, [])

  return (
    <>
      <div className="app-bg" />
      <div className="app-container">
        <Header onReset={handleReset} hasResults={!!results} />

        {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

        {loading && <LoadingOverlay message={loadingMessage} />}

        {!results && !loading && (
          <UploadZone
            onUpload={handleUpload}
            onGenerateTestData={handleGenerateTestData}
          />
        )}

        {results && (
          <Dashboard
            results={results}
            apiBase={API_BASE}
            onReset={handleReset}
          />
        )}
      </div>
    </>
  )
}

export default App
