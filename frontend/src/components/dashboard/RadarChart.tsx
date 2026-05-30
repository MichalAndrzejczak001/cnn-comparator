import type { ModelResult } from '../../types/api'
import { MODEL_COLORS, MODEL_LABELS } from '../../types/api'

interface RadarChartProps {
  results: ModelResult[]
}

const AXES = ['Dokładność', 'Precyzja', 'Czułość', 'F1', 'Szybkość']
const N = AXES.length
const CX = 200, CY = 210, R = 140
const W = 400, H = 420

function angleOf(i: number) {
  return -Math.PI / 2 + (2 * Math.PI * i) / N
}

function polarPoint(i: number, score: number) {
  const a = angleOf(i)
  return { x: CX + score * R * Math.cos(a), y: CY + score * R * Math.sin(a) }
}

function macroMetrics(matrix: number[][]) {
  const n = matrix.length
  let sumP = 0, sumR = 0, sumF1 = 0
  for (let i = 0; i < n; i++) {
    const tp = matrix[i][i]
    const fp = matrix.reduce((s, row, r) => r !== i ? s + row[i] : s, 0)
    const fn = matrix[i].reduce((s, v, j) => j !== i ? s + v : s, 0)
    const p = tp + fp > 0 ? tp / (tp + fp) : 0
    const r = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = p + r > 0 ? 2 * p * r / (p + r) : 0
    sumP += p; sumR += r; sumF1 += f1
  }
  return { precision: sumP / n, recall: sumR / n, f1: sumF1 / n }
}

export default function RadarChart({ results }: RadarChartProps) {
  const valid = results.filter(r => r.confusion_matrix)
  if (!valid.length) return null

  const minTime = Math.min(...valid.map(r => r.training_time_seconds))

  const modelScores = valid.map(r => {
    const { precision, recall, f1 } = macroMetrics(r.confusion_matrix!)
    const speed = minTime / r.training_time_seconds
    return {
      model: r.model,
      scores: [r.test_accuracy, precision, recall, f1, speed] as number[],
    }
  })

  function gridPolygon(frac: number) {
    return Array.from({ length: N }, (_, i) => {
      const { x, y } = polarPoint(i, frac)
      return `${x},${y}`
    }).join(' ')
  }

  return (
    <div className="chart-container" style={{ marginTop: '1.5rem' }}>
      <div className="chart-title">Porównanie wielokryterialne</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 440 }}>

        {[0.25, 0.5, 0.75, 1.0].map(frac => (
          <polygon
            key={frac}
            points={gridPolygon(frac)}
            fill="none"
            stroke={frac === 1.0 ? '#253550' : '#1a2740'}
            strokeWidth={frac === 1.0 ? 1 : 0.7}
          />
        ))}

        {[0.25, 0.5, 0.75].map(frac => {
          const { x, y } = polarPoint(0, frac)
          return (
            <text key={frac} x={x + 4} y={y - 3} fill="#3d5070" fontSize="8">
              {Math.round(frac * 100)}%
            </text>
          )
        })}

        {Array.from({ length: N }, (_, i) => {
          const end = polarPoint(i, 1)
          return (
            <line key={i} x1={CX} y1={CY} x2={end.x} y2={end.y}
              stroke="#1a2740" strokeWidth="1" />
          )
        })}

        {modelScores.map(({ model, scores }) => {
          const color = MODEL_COLORS[model] || '#888'
          const pts = scores.map((s, i) => {
            const { x, y } = polarPoint(i, s)
            return `${x},${y}`
          }).join(' ')
          return (
            <polygon key={model} points={pts}
              fill={color} fillOpacity="0.12"
              stroke={color} strokeWidth="2" strokeLinejoin="round" />
          )
        })}

        {modelScores.map(({ model, scores }) => {
          const color = MODEL_COLORS[model] || '#888'
          return scores.map((s, i) => {
            const { x, y } = polarPoint(i, s)
            return <circle key={`${model}-${i}`} cx={x} cy={y} r="3.5" fill={color} />
          })
        })}

        {AXES.map((label, i) => {
          const { x, y } = polarPoint(i, 1.22)
          const anchor = x < CX - 5 ? 'end' : x > CX + 5 ? 'start' : 'middle'
          return (
            <text key={i} x={x} y={y + 4} textAnchor={anchor}
              fill="#8899b0" fontSize="11" fontWeight="600">
              {label}
            </text>
          )
        })}

        {modelScores.map(({ model }, i) => {
          const color = MODEL_COLORS[model] || '#888'
          const lx = 20 + i * 95
          return (
            <g key={model} transform={`translate(${lx}, ${H - 18})`}>
              <circle cx="6" cy="0" r="5" fill={color} fillOpacity="0.3"
                stroke={color} strokeWidth="1.5" />
              <text x="14" y="4" fill="#8899b0" fontSize="10">{MODEL_LABELS[model] || model}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
