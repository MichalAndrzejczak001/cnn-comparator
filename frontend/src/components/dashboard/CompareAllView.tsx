import { useState } from 'react'
import { runCompare } from '../../api/client'
import { DATASETS, MODEL_COLORS, MODEL_LABELS } from '../../types/api'
import type { CompareResult } from '../../types/api'
import LossChart from './LossChart'

export default function CompareAllView() {
  const [dataset, setDataset] = useState('mnist')
  const [epochs, setEpochs] = useState(5)
  const [batchSize, setBatchSize] = useState(32)
  const [lr, setLr] = useState(0.001)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<CompareResult | null>(null)
  const [error, setError] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const r = await runCompare(dataset, {
        epochs,
        batch_size: batchSize,
        learning_rate: lr,
      })
      setResult(r)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd połączenia z serwerem')
    } finally {
      setLoading(false)
    }
  }

  const sortedResults = result
    ? [...result.results].sort((a, b) => b.test_accuracy - a.test_accuracy)
    : []

  return (
    <div className="view">
      <h2 className="view-title">Porównaj wszystkie modele</h2>
      <p className="view-desc">
        Trenuje Simple CNN, LeNet-5, VGG-11 i ResNet-18 na tym samym zbiorze danych i porównuje wyniki.
      </p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Zbiór danych</label>
              <select
                className="form-input"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
              >
                {DATASETS.map((d) => (
                  <option key={d} value={d}>
                    {d.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Epoki</label>
              <input
                className="form-input"
                type="number"
                min={1}
                max={200}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Batch size</label>
              <input
                className="form-input"
                type="number"
                min={1}
                max={512}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Learning rate</label>
              <input
                className="form-input"
                type="number"
                step="0.0001"
                min={0.0001}
                max={1}
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
              />
            </div>
          </div>

          {error && <div className="form-error">{error}</div>}

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Trening wszystkich modeli...' : 'Uruchom porównanie'}
          </button>
        </form>
      </div>

      {loading && (
        <div className="loading-card">
          <div className="spinner" />
          <span>
            Trening 4 modeli — może potrwać kilka minut w zależności od sprzętu...
          </span>
        </div>
      )}

      {result && (
        <>
          <div className="card">
            <h3 className="result-title">
              Wyniki — {result.dataset.toUpperCase()}, {result.epochs}{' '}
              {result.epochs === 1 ? 'epoka' : 'epok'}
            </h3>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Model</th>
                  <th>Dokł. testowa</th>
                  <th>Strata testowa</th>
                  <th>Ost. strata trenin.</th>
                </tr>
              </thead>
              <tbody>
                {sortedResults.map((r, idx) => {
                  const lastLoss =
                    r.train_loss_per_epoch[r.train_loss_per_epoch.length - 1]
                  const color = MODEL_COLORS[r.model] || '#888'
                  return (
                    <tr key={r.model} className={idx === 0 ? 'row-selected' : ''}>
                      <td className="text-muted">{idx + 1}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span
                            style={{
                              width: 10,
                              height: 10,
                              borderRadius: '50%',
                              background: color,
                              flexShrink: 0,
                            }}
                          />
                          <span className="badge badge-model">
                            {MODEL_LABELS[r.model] || r.model}
                          </span>
                          {idx === 0 && (
                            <span
                              className="badge"
                              style={{
                                color: '#4ade80',
                                borderColor: 'rgba(74,222,128,.35)',
                                background: 'rgba(74,222,128,.07)',
                                fontSize: '0.7rem',
                              }}
                            >
                              najlepszy
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="text-green">
                        {(r.test_accuracy * 100).toFixed(2)}%
                      </td>
                      <td>{r.test_loss.toFixed(4)}</td>
                      <td>{lastLoss != null ? lastLoss.toFixed(4) : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="card">
            <LossChart
              title="Strata treningowa per epoka — porównanie modeli"
              series={result.results.map((r) => ({
                label: MODEL_LABELS[r.model] || r.model,
                data: r.train_loss_per_epoch,
                color: MODEL_COLORS[r.model] || '#888',
              }))}
            />
          </div>
        </>
      )}
    </div>
  )
}
