import { useState } from 'react'
import type { ModelResult } from '../../types/api'
import { MODEL_COLORS, MODEL_LABELS } from '../../types/api'

interface CompareClassMetricsProps {
  results: ModelResult[]
  dataset: string
}

type Metric = 'f1' | 'precision' | 'recall'

const MNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const CIFAR_LABELS = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

function classMetrics(matrix: number[][]) {
  return matrix.map((row, i) => {
    const tp = matrix[i][i]
    const fp = matrix.reduce((s, r, ri) => ri !== i ? s + r[i] : s, 0)
    const fn = row.reduce((s, v, j) => j !== i ? s + v : s, 0)
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
    return { precision, recall, f1 }
  })
}

function f1Color(v: number) {
  if (v >= 0.9) return '#10b981'
  if (v >= 0.7) return '#4f86f7'
  if (v >= 0.5) return '#f59e0b'
  return '#f43f5e'
}

function pct(v: number) {
  return (v * 100).toFixed(1) + '%'
}

export default function CompareClassMetrics({ results, dataset }: CompareClassMetricsProps) {
  const [metric, setMetric] = useState<Metric>('f1')

  const valid = results.filter(r => r.confusion_matrix)
  if (!valid.length) return null

  const labels = dataset === 'cifar10' ? CIFAR_LABELS : MNIST_LABELS
  const perModel = valid.map(r => classMetrics(r.confusion_matrix!))
  const n = labels.length

  function getValue(modelIdx: number, classIdx: number) {
    return perModel[modelIdx][classIdx][metric]
  }

  function macroAvg(modelIdx: number) {
    const vals = perModel[modelIdx].map(m => m[metric])
    return vals.reduce((s, v) => s + v, 0) / vals.length
  }

  return (
    <div style={{ marginTop: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <div className="chart-title" style={{ margin: 0 }}>Metryki per klasa — porównanie modeli</div>
        <div style={{ display: 'flex', gap: '0.35rem' }}>
          {(['f1', 'precision', 'recall'] as Metric[]).map(m => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={metric === m ? 'btn-sm btn-primary' : 'btn-sm btn-outline'}
              style={{ fontSize: '0.72rem', padding: '0.2rem 0.6rem' }}
            >
              {m === 'f1' ? 'F1' : m === 'precision' ? 'Precision' : 'Recall'}
            </button>
          ))}
        </div>
      </div>

      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Klasa</th>
              {valid.map(r => (
                <th key={r.model}>
                  <span style={{ color: MODEL_COLORS[r.model] }}>
                    {MODEL_LABELS[r.model] || r.model}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((label, ci) => {
              const rowVals = valid.map((_, mi) => getValue(mi, ci))
              const best = Math.max(...rowVals)
              return (
                <tr key={label}>
                  <td style={{ fontWeight: 600, color: '#e2e8f0' }}>{label}</td>
                  {rowVals.map((val, mi) => (
                    <td
                      key={mi}
                      style={{
                        fontWeight: val === best ? 700 : 400,
                        color: f1Color(val),
                        background: val === best ? 'rgba(79,134,247,0.06)' : undefined,
                      }}
                    >
                      {pct(val)}
                    </td>
                  ))}
                </tr>
              )
            })}
            <tr style={{ borderTop: '1px solid rgba(79,134,247,0.2)' }}>
              <td style={{ fontWeight: 700, color: '#e2e8f0' }}>Macro avg</td>
              {valid.map((_, mi) => {
                const val = macroAvg(mi)
                const allAvg = valid.map((__, mj) => macroAvg(mj))
                const best = Math.max(...allAvg)
                return (
                  <td
                    key={mi}
                    style={{
                      fontWeight: val === best ? 700 : 600,
                      color: f1Color(val),
                      background: val === best ? 'rgba(79,134,247,0.06)' : undefined,
                    }}
                  >
                    {pct(val)}
                  </td>
                )
              })}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
