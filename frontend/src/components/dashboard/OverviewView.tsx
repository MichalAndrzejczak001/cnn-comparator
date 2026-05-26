import { useEffect, useState } from 'react'
import { getExperiments } from '../../api/client'
import type { ExperimentResponse } from '../../types/api'
import { MODEL_LABELS } from '../../types/api'

function fmtAccuracy(n: number) {
  return (n * 100).toFixed(2) + '%'
}

function mostUsedModel(experiments: ExperimentResponse[]): string | null {
  if (!experiments.length) return null
  const counts: Record<string, number> = {}
  for (const e of experiments) counts[e.model] = (counts[e.model] ?? 0) + 1
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
}

export default function OverviewView() {
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getExperiments()
      .then(data => { setExperiments(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const best = experiments.reduce<ExperimentResponse | null>(
    (acc, e) => (!acc || e.test_accuracy > acc.test_accuracy ? e : acc),
    null
  )
  const topModel = mostUsedModel(experiments)
  const recent = experiments.slice(0, 5)

  return (
    <div className="view">
      <h2 className="view-title">Przegląd</h2>
      <p className="view-desc">Podsumowanie wszystkich Twoich eksperymentów.</p>

      {loading ? (
        <div className="loading-card">
          <div className="spinner" />
          Ładowanie danych...
        </div>
      ) : (
        <>
          <div className="overview-stats">
            <div className="overview-stat">
              <div className="overview-stat-value">{experiments.length}</div>
              <div className="overview-stat-label">Łącznie eksperymentów</div>
            </div>

            <div className="overview-stat">
              <div className="overview-stat-value">
                {best ? fmtAccuracy(best.test_accuracy) : '—'}
              </div>
              <div className="overview-stat-label">Najlepsza dokładność</div>
              {best && (
                <div className="overview-stat-sub">
                  {MODEL_LABELS[best.model] ?? best.model} · {best.dataset}
                </div>
              )}
            </div>

            <div className="overview-stat">
              <div className="overview-stat-value">
                {topModel ? (MODEL_LABELS[topModel] ?? topModel) : '—'}
              </div>
              <div className="overview-stat-label">Najczęściej używany model</div>
            </div>

            <div className="overview-stat">
              <div className="overview-stat-value">
                {experiments[0]
                  ? new Date(experiments[0].created_at).toLocaleDateString('pl-PL')
                  : '—'}
              </div>
              <div className="overview-stat-label">Ostatni eksperyment</div>
            </div>
          </div>

          {recent.length > 0 ? (
            <div className="card">
              <div className="result-title">Ostatnie eksperymenty</div>
              <div className="table-wrapper" style={{ marginBottom: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Model</th>
                      <th>Dataset</th>
                      <th>Dokładność</th>
                      <th>Strata</th>
                      <th>Data</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map(e => (
                      <tr key={e.id}>
                        <td className="text-muted">{e.id}</td>
                        <td>
                          <span className="badge badge-model">
                            {MODEL_LABELS[e.model] ?? e.model}
                          </span>
                        </td>
                        <td>
                          <span className="badge badge-dataset">{e.dataset}</span>
                        </td>
                        <td className="text-green">{fmtAccuracy(e.test_accuracy)}</td>
                        <td className="text-muted">{e.test_loss.toFixed(4)}</td>
                        <td className="text-muted">
                          {new Date(e.created_at).toLocaleString('pl-PL')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              Brak eksperymentów. Uruchom pierwszy eksperyment, aby zobaczyć statystyki.
            </div>
          )}
        </>
      )}
    </div>
  )
}
