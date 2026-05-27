import type { ModelResult } from '../../types/api'
import { MODEL_COLORS, MODEL_LABELS } from '../../types/api'

interface AccuracyTimeChartProps {
  results: ModelResult[]
}

const W = 500, H = 220
const PL = 52, PR = 20, PT = 18, PB = 44

export default function AccuracyTimeChart({ results }: AccuracyTimeChartProps) {
  if (!results.length) return null

  const IW = W - PL - PR
  const IH = H - PT - PB

  const minT = 0
  const maxT = Math.max(...results.map(r => r.training_time_seconds)) * 1.15
  const minA = Math.max(0, Math.min(...results.map(r => r.test_accuracy)) - 0.05)
  const maxA = Math.min(1, Math.max(...results.map(r => r.test_accuracy)) + 0.03)

  function toX(t: number) { return PL + ((t - minT) / (maxT - minT)) * IW }
  function toY(a: number) { return PT + (1 - (a - minA) / (maxA - minA)) * IH }

  const gridY = 4
  const gridX = 4

  return (
    <div className="chart-container">
      <div className="chart-title">Dokładność vs czas treningu</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 240 }}>
        {/* Y grid */}
        {Array.from({ length: gridY + 1 }, (_, i) => {
          const v = minA + (i / gridY) * (maxA - minA)
          return (
            <g key={i}>
              <line x1={PL} y1={toY(v)} x2={W - PR} y2={toY(v)}
                stroke="#1a2740" strokeWidth="0.8" strokeDasharray="4,3" />
              <text x={PL - 5} y={toY(v) + 4} textAnchor="end" fill="#3d5070" fontSize="9.5">
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          )
        })}

        {/* X grid */}
        {Array.from({ length: gridX + 1 }, (_, i) => {
          const v = minT + (i / gridX) * (maxT - minT)
          return (
            <g key={i}>
              <line x1={toX(v)} y1={PT} x2={toX(v)} y2={PT + IH}
                stroke="#1a2740" strokeWidth="0.8" strokeDasharray="4,3" />
              <text x={toX(v)} y={H - PB + 14} textAnchor="middle" fill="#3d5070" fontSize="9.5">
                {v.toFixed(0)}s
              </text>
            </g>
          )
        })}

        {/* Axes */}
        <line x1={PL} y1={PT + IH} x2={W - PR} y2={PT + IH} stroke="#253550" strokeWidth="1" />
        <line x1={PL} y1={PT} x2={PL} y2={PT + IH} stroke="#253550" strokeWidth="1" />

        {/* Axis labels */}
        <text x={PL + IW / 2} y={H - 2} textAnchor="middle" fill="#3d5070" fontSize="11">
          Czas treningu (s)
        </text>
        <text textAnchor="middle" fill="#3d5070" fontSize="11"
          transform={`translate(12,${PT + IH / 2}) rotate(-90)`}>
          Dokładność
        </text>

        {/* Points */}
        {results.map(r => {
          const cx = toX(r.training_time_seconds)
          const cy = toY(r.test_accuracy)
          const color = MODEL_COLORS[r.model] || '#888'
          const label = MODEL_LABELS[r.model] || r.model
          const labelRight = cx > W - PL - 80
          return (
            <g key={r.model}>
              <circle cx={cx} cy={cy} r="7" fill={color} fillOpacity="0.2" stroke={color} strokeWidth="2" />
              <circle cx={cx} cy={cy} r="3.5" fill={color} />
              <text
                x={labelRight ? cx - 11 : cx + 11}
                y={cy - 10}
                textAnchor={labelRight ? 'end' : 'start'}
                fill={color}
                fontSize="10"
                fontWeight="600"
              >
                {label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
