import { useState, useRef, useEffect, useCallback } from 'react'
import { gradCamImage } from '../../api/client'
import type { GradCamResponse } from '../../types/api'
import { MODEL_LABELS } from '../../types/api'

interface Props {
  experimentId: number
  model: string
  onClose: () => void
}

const CANVAS_SIZE = 280

export default function DrawDigitModal({ experimentId, model, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [drawing, setDrawing] = useState(false)
  const [hasDrawing, setHasDrawing] = useState(false)
  const [snapshot, setSnapshot] = useState<string | null>(null)
  const [result, setResult] = useState<GradCamResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
  }, [])

  function getPos(e: React.MouseEvent | React.TouchEvent) {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = CANVAS_SIZE / rect.width
    const scaleY = CANVAS_SIZE / rect.height
    if ('touches' in e) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY,
      }
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    }
  }

  function startDraw(e: React.MouseEvent | React.TouchEvent) {
    e.preventDefault()
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    const { x, y } = getPos(e)
    ctx.beginPath()
    ctx.moveTo(x, y)
    setDrawing(true)
    setHasDrawing(true)
    setResult(null)
    setSnapshot(null)
    setError('')
  }

  const draw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    if (!drawing) return
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    const { x, y } = getPos(e)
    ctx.lineWidth = 22
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#ffffff'
    ctx.lineTo(x, y)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(x, y)
  }, [drawing])

  function stopDraw(e: React.MouseEvent | React.TouchEvent) {
    e.preventDefault()
    setDrawing(false)
    const ctx = canvasRef.current?.getContext('2d')
    if (ctx) ctx.beginPath()
  }

  function clearCanvas() {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    setHasDrawing(false)
    setResult(null)
    setSnapshot(null)
    setError('')
  }

  async function handleClassify() {
    const canvas = canvasRef.current
    if (!canvas) return
    setLoading(true)
    setError('')
    setResult(null)
    const dataUrl = canvas.toDataURL('image/png')
    setSnapshot(dataUrl)
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setError('Błąd eksportu canvas')
        setLoading(false)
        return
      }
      const file = new File([blob], 'drawing.png', { type: 'image/png' })
      try {
        const res = await gradCamImage(experimentId, file)
        setResult(res)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Błąd generowania Grad-CAM')
      } finally {
        setLoading(false)
      }
    }, 'image/png')
  }

  const maxConf = result ? Math.max(...result.confidences.map(c => c.confidence)) : 1

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: 580, width: '100%' }}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-header">
          <span className="modal-title">
            Rysuj cyfrę — {MODEL_LABELS[model] || model} / MNIST
          </span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '1rem' }}>
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            style={{
              borderRadius: 10,
              cursor: 'crosshair',
              touchAction: 'none',
              width: '100%',
              maxWidth: CANVAS_SIZE,
              border: '2px solid rgba(79,134,247,0.35)',
            }}
            onMouseDown={startDraw}
            onMouseMove={draw}
            onMouseUp={stopDraw}
            onMouseLeave={stopDraw}
            onTouchStart={startDraw}
            onTouchMove={draw}
            onTouchEnd={stopDraw}
          />
          <span style={{ color: '#64748b', fontSize: '0.78rem', marginTop: '0.4rem' }}>
            Narysuj cyfrę (0–9) białym kolorem na czarnym tle
          </span>
        </div>

        {error && <div className="form-error" style={{ marginBottom: '0.75rem' }}>{error}</div>}

        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', marginBottom: result ? '1.25rem' : 0 }}>
          <button className="btn-outline" onClick={onClose}>Zamknij</button>
          <button className="btn-outline" onClick={clearCanvas} disabled={!hasDrawing}>
            Wyczyść
          </button>
          <button
            className="btn-primary"
            disabled={!hasDrawing || loading}
            onClick={handleClassify}
          >
            {loading ? 'Generowanie...' : 'Klasyfikuj + Grad-CAM'}
          </button>
        </div>

        {result && snapshot && (
          <div>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem', alignItems: 'flex-start' }}>
              <div style={{ flex: 1, textAlign: 'center' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.4rem' }}>Rysunek</div>
                <img
                  src={snapshot}
                  alt="Rysunek"
                  style={{ width: '100%', maxWidth: 180, borderRadius: 8, imageRendering: 'pixelated' }}
                />
              </div>
              <div style={{ flex: 1, textAlign: 'center' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.4rem' }}>Mapa aktywacji (Grad-CAM)</div>
                <img
                  src={`data:image/png;base64,${result.gradcam_image}`}
                  alt="Grad-CAM"
                  style={{ width: '100%', maxWidth: 180, borderRadius: 8, imageRendering: 'pixelated' }}
                />
              </div>
            </div>

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
