interface ClassMetricsProps {
  matrix: number[][]
  dataset: string
}

const MNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const CIFAR_LABELS = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

function computeMetrics(matrix: number[][], dataset: string) {
  const labels = dataset === 'cifar10' ? CIFAR_LABELS : MNIST_LABELS
  return labels.map((label, i) => {
    const tp = matrix[i][i]
    const fp = matrix.reduce((sum, row, r) => r !== i ? sum + row[i] : sum, 0)
    const fn = matrix[i].reduce((sum, v, j) => j !== i ? sum + v : sum, 0)
    const support = matrix[i].reduce((sum, v) => sum + v, 0)
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
    return { label, precision, recall, f1, support }
  })
}

function f1Color(f1: number): string {
  if (f1 >= 0.9) return '#10b981'
  if (f1 >= 0.7) return '#4f86f7'
  if (f1 >= 0.5) return '#f59e0b'
  return '#f43f5e'
}

function pct(v: number) {
  return (v * 100).toFixed(1) + '%'
}

export default function ClassMetrics({ matrix, dataset }: ClassMetricsProps) {
  const metrics = computeMetrics(matrix, dataset)
  const n = metrics.length

  const macroP = metrics.reduce((s, m) => s + m.precision, 0) / n
  const macroR = metrics.reduce((s, m) => s + m.recall, 0) / n
  const macroF1 = metrics.reduce((s, m) => s + m.f1, 0) / n
  const totalSupport = metrics.reduce((s, m) => s + m.support, 0)

  return (
    <div style={{ marginTop: '1.5rem' }}>
      <div className="chart-title" style={{ marginBottom: '0.75rem' }}>
        Metryki per klasa
      </div>
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Klasa</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(({ label, precision, recall, f1, support }) => (
              <tr key={label}>
                <td style={{ fontWeight: 600, color: '#e2e8f0' }}>{label}</td>
                <td>{pct(precision)}</td>
                <td>{pct(recall)}</td>
                <td style={{ fontWeight: 700, color: f1Color(f1) }}>{pct(f1)}</td>
                <td style={{ color: '#64748b' }}>{support}</td>
              </tr>
            ))}
            <tr style={{ borderTop: '1px solid rgba(79,134,247,0.2)' }}>
              <td style={{ fontWeight: 700, color: '#e2e8f0' }}>Macro avg</td>
              <td style={{ fontWeight: 600 }}>{pct(macroP)}</td>
              <td style={{ fontWeight: 600 }}>{pct(macroR)}</td>
              <td style={{ fontWeight: 700, color: f1Color(macroF1) }}>{pct(macroF1)}</td>
              <td style={{ color: '#64748b' }}>{totalSupport}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
