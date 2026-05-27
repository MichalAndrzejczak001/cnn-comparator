interface Series {
  label: string
  data: number[]
  color: string
  dashed?: boolean
}

interface LossChartProps {
  series: Series[]
  title?: string
}

const W = 500, H = 200
const PL = 52, PR = 12, PT = 18, PB = 38
const INNER_W = W - PL - PR
const INNER_H = H - PT - PB

export default function LossChart({ series, title }: LossChartProps) {
  const validSeries = series.filter(s => s.data.length > 0)
  if (!validSeries.length) return null

  const allData = validSeries.flatMap(s => s.data)
  const maxV = Math.max(...allData)
  const minV = Math.min(...allData)
  const range = maxV - minV || 1

  const maxEpochs = Math.max(...validSeries.map(s => s.data.length))

  function toX(i: number, n: number) {
    return PL + (i / Math.max(n - 1, 1)) * INNER_W
  }
  function toY(v: number) {
    return PT + (1 - (v - minV) / range) * INNER_H
  }

  const gridYCount = 4
  const gridVals = Array.from({ length: gridYCount + 1 }, (_, i) =>
    minV + (i / gridYCount) * range
  )

  const xLabelCount = Math.min(maxEpochs, 8)
  const xLabels = Array.from({ length: xLabelCount }, (_, i) => {
    const epoch = Math.round(1 + (i / Math.max(xLabelCount - 1, 1)) * (maxEpochs - 1))
    return { epoch, x: toX(epoch - 1, maxEpochs) }
  })

  return (
    <div className="chart-container">
      {title && <div className="chart-title">{title}</div>}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 220 }}>
        {/* Grid lines */}
        {gridVals.map((v, i) => (
          <g key={i}>
            <line
              x1={PL} y1={toY(v)} x2={W - PR} y2={toY(v)}
              stroke="#1a2740" strokeWidth="0.8" strokeDasharray="4,3"
            />
            <text x={PL - 5} y={toY(v) + 4} textAnchor="end" fill="#3d5070" fontSize="9.5">
              {v.toFixed(3)}
            </text>
          </g>
        ))}

        {/* Axes */}
        <line x1={PL} y1={PT + INNER_H} x2={W - PR} y2={PT + INNER_H} stroke="#253550" strokeWidth="1" />
        <line x1={PL} y1={PT} x2={PL} y2={PT + INNER_H} stroke="#253550" strokeWidth="1" />

        {/* X labels */}
        {xLabels.map(({ epoch, x }) => (
          <text key={epoch} x={x} y={H - PB + 16} textAnchor="middle" fill="#3d5070" fontSize="9.5">
            {epoch}
          </text>
        ))}

        {/* Axis label — Epoch */}
        <text x={PL + INNER_W / 2} y={H - 2} textAnchor="middle" fill="#3d5070" fontSize="11">
          Epoka
        </text>
        {/* Axis label — Loss (rotated) */}
        <text
          textAnchor="middle" fill="#3d5070" fontSize="11"
          transform={`translate(12,${PT + INNER_H / 2}) rotate(-90)`}
        >
          Strata
        </text>

        {/* Series */}
        {validSeries.map(s => {
          const pts = s.data.map((v, i) => [toX(i, s.data.length), toY(v)] as [number, number])
          const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')
          const areaD =
            pathD +
            ` L${pts[pts.length - 1][0].toFixed(1)},${(PT + INNER_H).toFixed(1)} L${pts[0][0].toFixed(1)},${(PT + INNER_H).toFixed(1)} Z`

          return (
            <g key={s.label}>
              {!s.dashed && <path d={areaD} fill={s.color} fillOpacity="0.07" />}
              <path
                d={pathD}
                fill="none"
                stroke={s.color}
                strokeWidth="2.2"
                strokeDasharray={s.dashed ? '6,3' : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {s.data.length <= 30 &&
                pts.map((p, i) => (
                  <circle key={i} cx={p[0]} cy={p[1]} r="2.8" fill={s.color} />
                ))}
            </g>
          )
        })}
      </svg>

      {validSeries.length > 1 && (
        <div className="chart-legend">
          {validSeries.map(s => (
            <div key={s.label} className="legend-item">
              <span className="legend-dot" style={{ background: s.color }} />
              {s.label}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
