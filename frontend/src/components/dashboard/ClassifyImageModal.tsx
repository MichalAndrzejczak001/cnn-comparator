import { useState, useRef, useCallback } from 'react'
import { classifyImage } from '../../api/client'
import type { ClassifyResponse } from '../../types/api'
import { MODEL_LABELS } from '../../types/api'

interface Props {
  experimentId: number
  model: string
  dataset: string
  onClose: () => void
}

export default function ClassifyImageModal({ experimentId, model, dataset, onClose }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<ClassifyResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(f: File) {
    setFile(f)
    setResult(null)
    setError('')
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) handleFile(f)
  }, [])

  async function handleClassify() {
    if (!file) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await classifyImage(experimentId, file)
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd klasyfikacji')
    } finally {
      setLoading(false)
    }
  }

  const maxConf = result ? Math.max(...result.confidences.map(c => c.confidence)) : 1

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: 520, width: '100%' }}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-header">
          <span className="modal-title">
            Klasyfikacja obrazu — {MODEL_LABELS[model] || model} / {dataset.toUpperCase()}
          </span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div
          style={{
            border: `2px dashed ${dragging ? '#4f86f7' : 'rgba(79,134,247,0.35)'}`,
            borderRadius: 10,
            padding: '1.5rem',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'border-color 0.18s',
            marginBottom: '1rem',
            background: dragging ? 'rgba(79,134,247,0.06)' : 'transparent',
          }}
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleInputChange}
          />
          {preview ? (
            <img
              src={preview}
              alt="Podgląd"
              style={{ maxHeight: 180, maxWidth: '100%', borderRadius: 8, display: 'block', margin: '0 auto 0.5rem' }}
            />
          ) : (
            <span style={{ color: '#64748b', fontSize: '0.9rem' }}>
              Przeciągnij obraz lub kliknij, aby wybrać plik
            </span>
          )}
          {file && (
            <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{file.name}</span>
          )}
        </div>

        {error && (
          <div className="form-error" style={{ marginBottom: '0.75rem' }}>{error}</div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', marginBottom: result ? '1.25rem' : 0 }}>
          <button className="btn-outline" onClick={onClose}>Zamknij</button>
          <button
            className="btn-primary"
            disabled={!file || loading}
            onClick={handleClassify}
          >
            {loading ? 'Klasyfikowanie...' : 'Klasyfikuj'}
          </button>
        </div>

        {result && (
          <div>
            <div style={{
              background: 'rgba(79,134,247,0.1)',
              border: '1px solid rgba(79,134,247,0.3)',
              borderRadius: 8,
              padding: '0.75rem 1rem',
              marginBottom: '1rem',
              textAlign: 'center',
            }}>
              <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Predykcja</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0', marginTop: 2 }}>
                {result.predicted_class}
              </div>
              <div style={{ color: '#4f86f7', fontSize: '0.9rem' }}>
                {(result.confidences[result.predicted_index].confidence * 100).toFixed(2)}% pewności
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              {result.confidences
                .slice()
                .sort((a, b) => b.confidence - a.confidence)
                .map(c => (
                  <div key={c.label} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <span style={{
                      width: 80,
                      textAlign: 'right',
                      fontSize: '0.8rem',
                      color: c.label === result.predicted_class ? '#4f86f7' : '#94a3b8',
                      fontWeight: c.label === result.predicted_class ? 600 : 400,
                      flexShrink: 0,
                    }}>
                      {c.label}
                    </span>
                    <div style={{ flex: 1, background: 'rgba(255,255,255,0.07)', borderRadius: 4, height: 14, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${(c.confidence / maxConf) * 100}%`,
                        background: c.label === result.predicted_class
                          ? 'linear-gradient(90deg, #4f86f7, #7c3aed)'
                          : 'rgba(79,134,247,0.35)',
                        borderRadius: 4,
                        transition: 'width 0.4s ease',
                      }} />
                    </div>
                    <span style={{ width: 52, fontSize: '0.78rem', color: '#64748b', textAlign: 'right', flexShrink: 0 }}>
                      {(c.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
