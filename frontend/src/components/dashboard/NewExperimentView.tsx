import { useState } from 'react'
import { runExperiment } from '../../api/client'
import { MODELS, DATASETS, MODEL_COLORS, MODEL_LABELS } from '../../types/api'
import type { ExperimentResponse } from '../../types/api'
import LossChart from './LossChart'
import ConfusionMatrix from './ConfusionMatrix'
import ClassMetrics from './ClassMetrics'

export default function NewExperimentView() {
  const [model, setModel] = useState('simple_cnn')
  const [dataset, setDataset] = useState('mnist')
  const [epochs, setEpochs] = useState(5)
  const [batchSize, setBatchSize] = useState(32)
  const [lr, setLr] = useState(0.001)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ExperimentResponse | null>(null)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const r = await runExperiment({
        model,
        dataset,
        training: { epochs, batch_size: batchSize, learning_rate: lr },
      })
      setResult(r)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd połączenia z serwerem')
    } finally {
      setLoading(false)
    }
  }

  const color = MODEL_COLORS[model] || '#4f86f7'

  return (
    <div className="view">
      <h2 className="view-title">Nowy eksperyment</h2>
      <p className="view-desc">Skonfiguruj i uruchom trening wybranego modelu CNN.</p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Model</label>
              <select
                className="form-input"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {MODELS.map((m) => (
                  <option key={m} value={m}>
                    {MODEL_LABELS[m]}
                  </option>
                ))}
              </select>
            </div>
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
          </div>

          <div className="form-row">
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
            {loading ? 'Trening w toku...' : 'Uruchom trening'}
          </button>
        </form>
      </div>

      {loading && (
        <div className="loading-card">
          <div className="spinner" />
          <span>Trening modelu może potrwać kilkadziesiąt sekund...</span>
        </div>
      )}

      {result && (
        <div className="card">
          <h3 className="result-title">
            Wyniki — {MODEL_LABELS[result.model]} na {result.dataset.toUpperCase()}
          </h3>
          <div className="stats-grid">
            <div className="stat-box">
              <div className="stat-value" style={{ color }}>
                {(result.test_accuracy * 100).toFixed(2)}%
              </div>
              <div className="stat-label">Dokładność testowa</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.test_loss.toFixed(4)}</div>
              <div className="stat-label">Strata testowa</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.epochs}</div>
              <div className="stat-label">Epoki</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.batch_size}</div>
              <div className="stat-label">Batch size</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.learning_rate}</div>
              <div className="stat-label">Learning rate</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.training_time_seconds}s</div>
              <div className="stat-label">Czas treningu</div>
            </div>
          </div>
          <LossChart
            title="Krzywa uczenia"
            series={[
              {
                label: `${MODEL_LABELS[result.model]} — trening`,
                data: result.train_loss_per_epoch,
                color,
              },
              ...(result.test_loss_per_epoch ? [{
                label: `${MODEL_LABELS[result.model]} — test`,
                data: result.test_loss_per_epoch,
                color,
                dashed: true,
              }] : []),
            ]}
          />
          {result.confusion_matrix && (
            <>
              <ConfusionMatrix matrix={result.confusion_matrix} dataset={result.dataset} />
              <ClassMetrics matrix={result.confusion_matrix} dataset={result.dataset} />
            </>
          )}
        </div>
      )}
    </div>
  )
}
