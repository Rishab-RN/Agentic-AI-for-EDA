import { useState, useCallback } from 'react'
import { Routes, Route, useNavigate } from 'react-router-dom'
import axios from 'axios'
import CorrelationPage from './CorrelationPage'

// =====================================================
// API Service
// =====================================================

const API_BASE = '/api'

const api = {
    upload: (file) => {
        const formData = new FormData()
        formData.append('file', file)
        return axios.post(`${API_BASE}/upload`, formData)
    },
    analyze: (jobId) => axios.get(`${API_BASE}/analyze/${jobId}`),
    startPipeline: (jobId) => axios.post(`${API_BASE}/pipeline/start`, {
        job_id: jobId
    }, { timeout: 600000 }),
    getResults: (jobId) => axios.get(`${API_BASE}/pipeline/results/${jobId}`)
}

// =====================================================
// Main App Component (Router)
// =====================================================

function App() {
    return (
        <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/correlation" element={<CorrelationPage />} />
        </Routes>
    )
}

// =====================================================
// Main Page Component (Phase 1: Cleaning & Visualization)
// =====================================================

function MainPage() {
    const navigate = useNavigate()

    // Core state
    const [jobId, setJobId] = useState(null)
    const [status, setStatus] = useState('idle') // idle, uploading, analyzed, running, ready_for_correlation, error
    const [analysis, setAnalysis] = useState(null)
    const [results, setResults] = useState(null)
    const [stages, setStages] = useState({})
    const [error, setError] = useState(null)

    // Drag & drop
    const [dragging, setDragging] = useState(false)

    // ===== FILE UPLOAD =====
    const handleFileUpload = useCallback(async (file) => {
        if (!file || !file.name.endsWith('.csv')) {
            setError('Please upload a CSV file')
            return
        }

        try {
            setStatus('uploading')
            setError(null)

            // Upload file
            const uploadRes = await api.upload(file)
            const newJobId = uploadRes.data.job_id
            setJobId(newJobId)

            // Analyze file
            const analysisRes = await api.analyze(newJobId)
            setAnalysis(analysisRes.data.analysis)
            setStatus('analyzed')

        } catch (err) {
            setError(err.response?.data?.error || 'Upload failed')
            setStatus('error')
        }
    }, [])

    // ===== PIPELINE EXECUTION (Phase 1 only) =====
    const startPipeline = async () => {
        if (!jobId) return

        try {
            setStatus('running')
            setError(null)
            setStages({
                missing: { status: 'running' },
                outlier: { status: 'pending' },
                visualize: { status: 'pending' },
                correlation: { status: 'pending' }
            })

            // Execute pipeline Phase 1
            await api.startPipeline(jobId)

            // Fetch results
            const resultsRes = await api.getResults(jobId)
            const data = resultsRes.data

            // Update state with results
            setResults(data)

            // Update stages from results
            if (data.stages) {
                setStages(data.stages)
            }

            setStatus(data.status) // 'ready_for_correlation'

        } catch (err) {
            console.error('Pipeline error:', err)
            setError(err.response?.data?.error || 'Pipeline failed')
            setStatus('error')
        }
    }

    // ===== NAVIGATE TO CORRELATION PAGE =====
    const goToCorrelation = () => {
        navigate('/correlation', {
            state: {
                jobId,
                preCorrelationUrl: `/api/download/${jobId}/pre_correlation_cleaned.csv`,
                stages,
                analysis
            }
        })
    }

    // ===== RESET =====
    const resetPipeline = () => {
        setJobId(null)
        setStatus('idle')
        setAnalysis(null)
        setResults(null)
        setStages({})
        setError(null)
    }

    // ===== DRAG HANDLERS =====
    const handleDragOver = (e) => { e.preventDefault(); setDragging(true) }
    const handleDragLeave = () => setDragging(false)
    const handleDrop = (e) => {
        e.preventDefault()
        setDragging(false)
        handleFileUpload(e.dataTransfer.files[0])
    }

    // ===== RENDER =====
    return (
        <div className="app-container">
            {/* Hero Section - Only shown when idle */}
            {status === 'idle' && (
                <div className="hero-section">
                    <div className="hero-content">
                        <header className="header">
                            <h1>Agentic AI System<br />for EDA</h1>
                            <p>Intelligent Data Cleaning, Visualization & Analysis powered by Multi-Agent AI</p>
                        </header>
                    </div>
                </div>
            )}

            {/* Simple Header - Shown when not idle */}
            {status !== 'idle' && (
                <header className="header header-centered">
                    <h1>Agentic AI System for EDA</h1>
                    <p>Intelligent Data Cleaning & Visualization</p>
                </header>
            )}

            {/* Error Display */}
            {error && (
                <div className="card" style={{ borderColor: 'var(--accent-error)' }}>
                    <p style={{ color: 'var(--accent-error)' }}>⚠️ {error}</p>
                </div>
            )}

            {/* Upload Section */}
            {status === 'idle' && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-icon" style={{ background: 'var(--gradient-primary)' }}>📁</div>
                        <h2>Upload Dataset</h2>
                    </div>
                    <div
                        className={`dropzone ${dragging ? 'dragging' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="dropzone-content">
                            <div className="dropzone-icon">📊</div>
                            <h3>Drop your CSV file here</h3>
                            <p>or <label className="file-label" style={{ color: 'var(--accent-primary)', cursor: 'pointer', textDecoration: 'underline' }}>
                                click to browse
                                <input
                                    type="file"
                                    accept=".csv"
                                    onChange={(e) => handleFileUpload(e.target.files[0])}
                                    style={{ display: 'none' }}
                                />
                            </label></p>
                        </div>
                    </div>
                </div>
            )}

            {/* Loading State */}
            {status === 'uploading' && (
                <div className="card">
                    <div className="loading-spinner"></div>
                    <p style={{ textAlign: 'center', marginTop: '1rem' }}>Analyzing dataset...</p>
                </div>
            )}

            {/* Analysis Results & Pipeline Trigger */}
            {(status === 'analyzed' || status === 'running' || status === 'ready_for_correlation') && analysis && (
                <>
                    {/* Analysis Summary */}
                    <div className="card">
                        <div className="card-header">
                            <div className="card-icon" style={{ background: 'var(--gradient-primary)' }}>🔬</div>
                            <h2>Master Agent Analysis</h2>
                        </div>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-value">{analysis.total_rows?.toLocaleString()}</span>
                                <span className="stat-label">Rows</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-value">{analysis.total_columns}</span>
                                <span className="stat-label">Columns</span>
                            </div>
                            <div className="stat-item" style={{ color: analysis.missing_analysis?.total_missing > 0 ? 'var(--accent-secondary)' : 'inherit' }}>
                                <span className="stat-value">{analysis.missing_analysis?.total_missing?.toLocaleString() || 0}</span>
                                <span className="stat-label">Missing Values</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-value">{analysis.outlier_analysis?.columns_with_outliers || 0}</span>
                                <span className="stat-label">Outlier Columns</span>
                            </div>
                        </div>
                    </div>

                    {/* Agent Decisions */}
                    <div className="card">
                        <div className="card-header">
                            <div className="card-icon" style={{ background: 'var(--gradient-success)' }}>🤖</div>
                            <h2>Agent Decisions</h2>
                        </div>
                        {analysis.decision_trace?.map((decision, idx) => (
                            <div key={idx} className="decision-item" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderRadius: '8px', background: 'rgba(255,255,255,0.05)' }}>
                                <span className={`badge ${decision.decision === 'RUN' ? 'badge-success' : 'badge-skip'}`}>
                                    {decision.decision}
                                </span>
                                <strong style={{ marginLeft: '0.5rem' }}>{decision.agent.replace(/_/g, ' ').replace('agent', 'Agent')}</strong>
                                <p style={{ margin: '0.25rem 0 0 2.5rem', opacity: 0.8, fontSize: '0.9rem' }}>{decision.reason}</p>
                            </div>
                        ))}
                    </div>

                    {/* Pipeline Start - Phase 1 */}
                    {status === 'analyzed' && (
                        <div className="card">
                            <div className="card-header">
                                <div className="card-icon" style={{ background: 'var(--gradient-primary)' }}>🚀</div>
                                <h2>Start Data Cleaning</h2>
                            </div>
                            <p style={{ marginBottom: '1rem', color: '#ccc' }}>
                                This will run Missing Value Imputation, Outlier Detection, and Visualization.
                                Correlation analysis is on a separate page.
                            </p>
                            <button className="btn btn-primary" onClick={startPipeline}>
                                Start Pipeline (Phase 1)
                            </button>
                        </div>
                    )}

                    {/* Pipeline Progress */}
                    {(status === 'running' || status === 'ready_for_correlation') && (
                        <div className="card">
                            <div className="card-header">
                                <div className="card-icon" style={{ background: 'var(--gradient-success)' }}>⚡</div>
                                <h2>Pipeline Progress</h2>
                            </div>
                            <div className="pipeline-stages">
                                <StageIndicator label="Missing Values" icon="🔍" status={stages.missing?.status} />
                                <StageIndicator label="Outliers" icon="📊" status={stages.outlier?.status} />
                                <StageIndicator label="Visualization" icon="📈" status={stages.visualize?.status} />
                                <StageIndicator label="Correlation" icon="🔗" status={stages.correlation?.status} />
                            </div>
                        </div>
                    )}

                    {/* Phase 1 Complete - Navigate to Correlation */}
                    {status === 'ready_for_correlation' && results && (
                        <>
                            {/* Success Message & Navigation */}
                            <div className="card">
                                <div className="card-header">
                                    <div className="card-icon" style={{ background: 'var(--gradient-success)' }}>✅</div>
                                    <h2>Phase 1 Complete!</h2>
                                </div>
                                <p style={{ marginBottom: '1rem', color: '#ccc' }}>
                                    Data cleaning and visualization complete. You can now proceed to correlation analysis
                                    or download the cleaned data.
                                </p>
                                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                                    <button className="btn btn-primary" onClick={goToCorrelation} style={{ background: 'var(--accent-secondary)' }}>
                                        🔗 Go to Correlation Analysis →
                                    </button>
                                    <a href={`/api/download/${jobId}/pre_correlation_cleaned.csv`} className="btn btn-success" download>
                                        📥 Download Cleaned CSV
                                    </a>
                                    <a href={`/api/report/pdf/${jobId}`} className="btn btn-primary" download>
                                        📄 Download Report
                                    </a>
                                    <button className="btn btn-secondary" onClick={resetPipeline}>
                                        🔄 Process Another File
                                    </button>
                                </div>
                            </div>

                            {/* Plots Gallery */}
                            {results.plots && results.plots.length > 0 && (
                                <div className="card">
                                    <div className="card-header">
                                        <div className="card-icon" style={{ background: 'var(--accent-secondary)' }}>🎨</div>
                                        <h2>Generated Visualizations ({results.plots.length})</h2>
                                    </div>
                                    <div className="plots-grid">
                                        {results.plots.map((plotUrl, idx) => (
                                            <div key={idx} className="plot-card" onClick={() => window.open(plotUrl, '_blank')}>
                                                <img
                                                    src={plotUrl}
                                                    alt={`Plot ${idx + 1}`}
                                                />
                                                <div className="plot-card-footer">
                                                    {plotUrl.split('/').pop().replace('.png', '').replace(/_/g, ' ')}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </>
            )}
        </div>
    )
}

// =====================================================
// Stage Indicator Component
// =====================================================

function StageIndicator({ label, icon, status }) {
    const statusConfig = {
        pending: { label: 'Pending', className: 'pending' },
        running: { label: 'Running...', className: 'running' },
        done: { label: 'Complete', className: 'done' },
        skipped: { label: 'Skipped', className: 'skipped' },
        error: { label: 'Error', className: 'error' }
    }

    const config = statusConfig[status] || statusConfig.pending
    const displayIcon = status === 'done' ? '✓' : status === 'skipped' ? '−' : status === 'error' ? '✗' : icon

    return (
        <div className={`stage ${config.className}`}>
            <div className="stage-icon">{displayIcon}</div>
            <div className="stage-label">{label}</div>
            <div className="stage-status">{config.label}</div>
        </div>
    )
}

export default App
