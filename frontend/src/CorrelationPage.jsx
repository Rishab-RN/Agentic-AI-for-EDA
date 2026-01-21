import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import axios from 'axios'

const API_BASE = '/api'

const api = {
    runCorrelation: (jobId, modelType, targetColumn) => axios.post(`${API_BASE}/pipeline/correlation`, {
        job_id: jobId,
        model_type: modelType,
        target_column: targetColumn
    }, { timeout: 600000 }),
    getResults: (jobId) => axios.get(`${API_BASE}/pipeline/results/${jobId}`)
}

function CorrelationPage() {
    const navigate = useNavigate()
    const location = useLocation()

    // Get data passed from main page
    const { jobId, preCorrelationUrl, stages: initialStages, analysis } = location.state || {}

    // State
    const [modelType, setModelType] = useState('')
    const [targetColumn, setTargetColumn] = useState('')
    const [status, setStatus] = useState('ready') // ready, running, completed, error
    const [stages, setStages] = useState(initialStages || {})
    const [results, setResults] = useState(null)
    const [error, setError] = useState(null)

    // Redirect if no job data
    useEffect(() => {
        if (!jobId) {
            navigate('/')
        }
    }, [jobId, navigate])

    // Run correlation
    const runCorrelation = async () => {
        if (!jobId) return

        try {
            setStatus('running')
            setError(null)
            setStages(prev => ({
                ...prev,
                correlation: { status: 'running' }
            }))

            await api.runCorrelation(jobId, modelType, targetColumn)

            // Fetch final results
            const resultsRes = await api.getResults(jobId)
            const data = resultsRes.data

            setResults(data)
            if (data.stages) {
                setStages(data.stages)
            }
            setStatus('completed')

        } catch (err) {
            console.error('Correlation error:', err)
            setError(err.response?.data?.error || 'Correlation failed')
            setStages(prev => ({
                ...prev,
                correlation: { status: 'error' }
            }))
            setStatus('error')
        }
    }

    // Back to main page
    const goBack = () => {
        navigate('/')
    }

    if (!jobId) {
        return null // Will redirect
    }

    return (
        <div className="app-container">
            {/* Header */}
            <header className="header">
                <h1>Correlation Analysis</h1>
                <p>Feature Selection & Multicollinearity Detection</p>
            </header>

            {/* Navigation */}
            <div className="card" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <button className="btn btn-secondary" onClick={goBack}>
                    ← Back to Pipeline
                </button>
                <span style={{ color: '#888' }}>Job ID: {jobId}</span>
            </div>

            {/* Error Display */}
            {error && (
                <div className="card" style={{ borderColor: 'var(--accent-error)' }}>
                    <p style={{ color: 'var(--accent-error)' }}>⚠️ {error}</p>
                </div>
            )}

            {/* CSV Downloads */}
            <div className="card">
                <div className="card-header">
                    <div className="card-icon" style={{ background: 'var(--gradient-success)' }}>📥</div>
                    <h2>Download Datasets</h2>
                </div>
                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <a
                        href={preCorrelationUrl || `/api/download/${jobId}/pre_correlation_cleaned.csv`}
                        className="btn btn-success"
                        download
                    >
                        📊 Pre-Correlation CSV (Cleaned)
                    </a>
                    {status === 'completed' && results && (
                        <a
                            href={results.download_url || `/api/download/${jobId}/final_cleaned.csv`}
                            className="btn btn-primary"
                            download
                        >
                            🎯 Post-Correlation CSV (Final)
                        </a>
                    )}
                </div>
                <p style={{ marginTop: '1rem', color: '#888', fontSize: '0.9rem' }}>
                    Pre-Correlation: Data after Missing Value and Outlier treatment.<br />
                    Post-Correlation: Data after removing redundant/multicollinear features.
                </p>
            </div>

            {/* Correlation Settings */}
            {status !== 'completed' && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-icon" style={{ background: 'var(--accent-secondary)' }}>🔗</div>
                        <h2>Correlation Analysis Settings</h2>
                    </div>
                    <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                        <div style={{ flex: '1', minWidth: '200px' }}>
                            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>ML Model Type</label>
                            <select
                                value={modelType}
                                onChange={(e) => setModelType(e.target.value)}
                                className="form-select"
                                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
                                disabled={status === 'running'}
                            >
                                <option value="">-- Select Model --</option>
                                <option value="linear">Linear Regression</option>
                                <option value="logistic">Logistic Regression</option>
                                <option value="tree">Decision Tree</option>
                                <option value="forest">Random Forest</option>
                                <option value="xgboost">XGBoost</option>
                            </select>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                VIF-based multicollinearity removal only for linear/logistic models.
                            </p>
                        </div>
                        <div style={{ flex: '1', minWidth: '200px' }}>
                            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Target Column (Optional)</label>
                            <input
                                type="text"
                                value={targetColumn}
                                onChange={(e) => setTargetColumn(e.target.value)}
                                placeholder="e.g., SalePrice"
                                className="form-input"
                                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
                                disabled={status === 'running'}
                            />
                        </div>
                    </div>

                    <button
                        className="btn btn-primary"
                        onClick={runCorrelation}
                        style={{ marginTop: '1.5rem', background: 'var(--accent-secondary)' }}
                        disabled={status === 'running'}
                    >
                        {status === 'running' ? '⏳ Running...' : '🚀 Run Correlation Analysis'}
                    </button>
                </div>
            )}

            {/* Stage Indicator */}
            <div className="card">
                <div className="card-header">
                    <div className="card-icon" style={{ background: 'var(--gradient-success)' }}>⚡</div>
                    <h2>Correlation Status</h2>
                </div>
                <div className="pipeline-stages" style={{ justifyContent: 'center' }}>
                    <StageIndicator
                        label="Correlation Analysis"
                        icon="🔗"
                        status={stages.correlation?.status || 'pending'}
                    />
                </div>
            </div>

            {/* Results */}
            {status === 'completed' && results && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-icon" style={{ background: 'var(--gradient-success)' }}>✅</div>
                        <h2>Analysis Complete!</h2>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                        <div className="stat-item">
                            <span className="stat-value">{results.stages?.correlation?.result?.columns_removed || 0}</span>
                            <span className="stat-label">Columns Removed</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value">{results.stages?.correlation?.result?.redundant_pairs_found || 0}</span>
                            <span className="stat-label">Redundant Pairs</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value">{results.stages?.correlation?.result?.high_vif_features || 0}</span>
                            <span className="stat-label">High VIF Features</span>
                        </div>
                    </div>

                    {/* LLM Summary */}
                    {results.stages?.correlation?.result?.llm_summary && (
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                            <h3 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>📝 Analysis Summary</h3>
                            <p style={{ whiteSpace: 'pre-wrap', color: '#ccc', fontSize: '0.9rem' }}>
                                {results.stages.correlation.result.llm_summary}
                            </p>
                        </div>
                    )}

                    {/* Download Correlation Report */}
                    <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                        <a
                            href={`/api/report/correlation-pdf/${jobId}`}
                            className="btn btn-primary"
                            style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
                            download
                        >
                            📄 Download Correlation Report (PDF)
                        </a>
                    </div>
                </div>
            )}
        </div>
    )
}

// Stage Indicator Component
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

export default CorrelationPage
