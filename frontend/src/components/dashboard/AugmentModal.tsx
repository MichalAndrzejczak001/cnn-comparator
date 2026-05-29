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

const CANVAS_SIZE = 224

export default function AugmentModal({ experimentId, model, dataset, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const [hasImage, setHasImage] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [rotation, setRotation] = useState(0)
  const [brightness, setBrightness] = useState(100)
  const [noise, setNoise] = useState(0)
  const [result, setResult] = useState<ClassifyResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function renderCanvas(rot: number, bright: number, noiseLevel: number) {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    ctx.save()
    ctx.filter = `brightness(${bright / 100})`
    ctx.translate(CANVAS_SIZE / 2, CANVAS_SIZE / 2)
    ctx.rotate((rot * Math.PI) / 180)
    ctx.drawImage(img, -CANVAS_SIZE / 2, -CANVAS_SIZE / 2, CANVAS_SIZE, CANVAS_SIZE)
    ctx.restore()

    if (noiseLevel > 0) {
      const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE)
      const d = imageData.data
      const amp = (noiseLevel / 100) * 128
      for (let i = 0; i < d.length; i += 4) {
        const n = (Math.random() - 0.5) * amp * 2
        d[i]     = Math.max(0, Math.min(255, d[i]     + n))
        d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n))
        d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n))
      }
      ctx.putImageData(imageData, 0, 0)
    }
  }

  function doClassify() {
    const canvas = canvasRef.current
    if (!canvas || !imgRef.current) return
    setLoading(true)
    setError('')
    canvas.toBlob(async (blob) => {
      if (!blob) { setLoading(false); return }
      const f = new File([blob], 'augmented.png', { type: 'image/png' })
      try {
        const res = await classifyImage(experimentId, f)
        setResult(res)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Błąd klasyfikacji')
      } finally {
        setLoading(false)
      }
    }, 'image/png')
  }

  function scheduleClassify(rot: number, bright: number, noiseLevel: number) {
    renderCanvas(rot, bright, noiseLevel)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(doClassify, 500)
  }

  function loadImage(f: File) {
    setResult(null)
    setError('')
    setRotation(0)
    setBrightness(100)
    setNoise(0)
    const url = URL.createObjectURL(f)
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setHasImage(true)
      renderCanvas(0, 100, 0)
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(doClassify, 300)
    }
    img.src = url
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) loadImage(f)
  }, [])

  const maxConf = result ? Math.max(...result.confidences.map(c => c.confidence)) : 1

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: 600, width: '100%' }}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-header">
          <span className="modal-title">
            Augmentacja — {MODEL_LABELS[model] || model} / {dataset.toUpperCase()}
          </span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {!hasImage ? (
          <div
            style={{
              border: `2px dashed ${dragging ? '#4f86f7' : 'rgba(79,134,247,0.35)'}`,
              borderRadius: 10,
              padding: '2.5rem',
              textAlign: 'center',
              cursor: 'pointer',
              marginBottom: '1rem',
              background: dragging ? 'rgba(79,134,247,0.06)' : 'transparent',
              transition: 'border-color 0.18s',
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
              onChange={e => { const f = e.target.files?.[0]; if (f) loadImage(f) }}
            />
            <span style={{ color: '#64748b', fontSize: '0.9rem' }}>
              Przeciągnij obraz lub kliknij, aby wybrać plik
            </span>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '1.25rem', marginBottom: '1rem', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                style={{
                  borderRadius: 8,
                  border: '2px solid rgba(79,134,247,0.35)',
                  imageRendering: 'pixelated',
                  width: CANVAS_SIZE,
                  height: CANVAS_SIZE,
                }}
              />
              <button
                className="btn-sm btn-outline"
                onClick={() => { setHasImage(false); imgRef.current = null; setResult(null) }}
              >
                Zmień obraz
              </button>
            </div>

            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.9rem', minWidth: 0 }}>
              <SliderRow
                label="Obrót"
                value={rotation} min={-180} max={180} step={1}
                display={`${rotation}°`}
                onChange={v => { setRotation(v); scheduleClassify(v, brightness, noise) }}
                onReset={() => { setRotation(0); scheduleClassify(0, brightness, noise) }}
              />
              <SliderRow
                label="Jasność"
                value={brightness} min={50} max={200} step={1}
                display={`${brightness}%`}
                onChange={v => { setBrightness(v); scheduleClassify(rotation, v, noise) }}
                onReset={() => { setBrightness(100); scheduleClassify(rotation, 100, noise) }}
              />
              <SliderRow
                label="Szum"
                value={noise} min={0} max={50} step={1}
                display={`${noise}%`}
                onChange={v => { setNoise(v); scheduleClassify(rotation, brightness, v) }}
                onReset={() => { setNoise(0); scheduleClassify(rotation, brightness, 0) }}
              />

              {error && <div className="form-error">{error}</div>}

              {result && (
                <div>
                  <div style={{
                    background: 'rgba(79,134,247,0.1)',
                    border: '1px solid rgba(79,134,247,0.3)',
                    borderRadius: 8,
                    padding: '0.6rem 0.75rem',
                    textAlign: 'center',
                    marginBottom: '0.65rem',
                  }}>
                    <div style={{ color: '#94a3b8', fontSize: '0.73rem' }}>
                      {loading ? 'Aktualizowanie...' : 'Predykcja'}
                    </div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#e2e8f0' }}>
                      {result.predicted_class}
                    </div>
                    <div style={{ color: '#4f86f7', fontSize: '0.82rem' }}>
                      {(result.confidences[result.predicted_index].confidence * 100).toFixed(2)}% pewności
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                    {result.confidences
                      .slice()
                      .sort((a, b) => b.confidence - a.confidence)
                      .map(c => (
                        <div key={c.label} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                          <span style={{
                            width: 72,
                            textAlign: 'right',
                            fontSize: '0.75rem',
                            color: c.label === result.predicted_class ? '#4f86f7' : '#94a3b8',
                            fontWeight: c.label === result.predicted_class ? 600 : 400,
                            flexShrink: 0,
                          }}>
                            {c.label}
                          </span>
                          <div style={{ flex: 1, background: 'rgba(255,255,255,0.07)', borderRadius: 4, height: 11, overflow: 'hidden' }}>
                            <div style={{
                              height: '100%',
                              width: `${(c.confidence / maxConf) * 100}%`,
                              background: c.label === result.predicted_class
                                ? 'linear-gradient(90deg, #4f86f7, #7c3aed)'
                                : 'rgba(79,134,247,0.35)',
                              borderRadius: 4,
                              transition: 'width 0.3s ease',
                            }} />
                          </div>
                          <span style={{ width: 42, fontSize: '0.72rem', color: '#64748b', textAlign: 'right', flexShrink: 0 }}>
                            {(c.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn-outline" onClick={onClose}>Zamknij</button>
        </div>
      </div>
    </div>
  )
}

interface SliderRowProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  display: string
  onChange: (v: number) => void
  onReset: () => void
}

function SliderRow({ label, value, min, max, step, display, onChange, onReset }: SliderRowProps) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
        <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>{label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <span style={{ fontSize: '0.78rem', color: '#e2e8f0', minWidth: 40, textAlign: 'right' }}>{display}</span>
          <button
            onClick={onReset}
            title="Resetuj"
            style={{
              fontSize: '0.85rem',
              color: '#64748b',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '0 2px',
              lineHeight: 1,
            }}
          >
            ↺
          </button>
        </div>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        style={{ width: '100%', accentColor: '#4f86f7', cursor: 'pointer' }}
        onChange={e => onChange(Number(e.target.value))}
      />
    </div>
  )
}
