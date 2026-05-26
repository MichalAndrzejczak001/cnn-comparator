const LAYERS = [3, 5, 5, 4, 2]
const LABELS = ['Input', 'Conv', 'Pool', 'FC', 'Output']
const WIDTH = 440
const HEIGHT = 310
const SPACING = 52
const MARGIN_TOP = 25
const MARGIN_BOTTOM = 28
const EFFECTIVE_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
const X_POSITIONS = [45, 135, 225, 315, 395]
const NODE_R = 11
const GLOW_R = 26
const COLORS = ['#4f86f7', '#5a93f8', '#8b5cf6', '#7c3aed', '#a78bfa']

function getLayerNodes(li: number, n: number) {
  const totalH = (n - 1) * SPACING
  const startY = MARGIN_TOP + (EFFECTIVE_H - totalH) / 2
  return Array.from({ length: n }, (_, i) => ({
    x: X_POSITIONS[li],
    y: startY + i * SPACING,
  }))
}

export default function NeuralNetSVG() {
  const allNodes = LAYERS.map((n, li) => getLayerNodes(li, n))

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width="100%"
      style={{ maxWidth: 440, filter: 'drop-shadow(0 0 40px rgba(79,134,247,0.12))' }}
    >
      <defs>
        <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#4f86f7" stopOpacity="0.2" />
          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.2" />
        </linearGradient>
        {COLORS.map((color, li) => (
          <radialGradient key={li} id={`glow${li}`} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={color} stopOpacity="0.35" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </radialGradient>
        ))}
      </defs>

      {/* Connections */}
      {allNodes.slice(0, -1).map((layer, li) =>
        layer.map((from, fi) =>
          allNodes[li + 1].map((to, ti) => (
            <line
              key={`${li}-${fi}-${ti}`}
              x1={from.x} y1={from.y}
              x2={to.x} y2={to.y}
              stroke="url(#lineGrad)"
              strokeWidth="0.7"
            />
          ))
        )
      )}

      {/* Nodes */}
      {allNodes.map((layer, li) =>
        layer.map((node, ni) => {
          const delay = `${((ni * 0.25 + li * 0.4) % 1.8).toFixed(2)}s`
          const dur = `${(2.0 + ((ni + li) % 3) * 0.4).toFixed(1)}s`
          return (
            <g key={`${li}-${ni}`}>
              <circle
                cx={node.x} cy={node.y} r={GLOW_R}
                fill={`url(#glow${li})`}
                className="nn-glow"
                style={{ animationDelay: delay, animationDuration: dur }}
              />
              <circle
                cx={node.x} cy={node.y} r={NODE_R}
                fill={COLORS[li]}
                fillOpacity="0.12"
                stroke={COLORS[li]}
                strokeWidth="1.5"
                className="nn-node"
                style={{ animationDelay: delay, animationDuration: dur }}
              />
              <circle
                cx={node.x} cy={node.y} r={3}
                fill={COLORS[li]}
                fillOpacity="0.9"
              />
            </g>
          )
        })
      )}

      {/* Labels */}
      {LABELS.map((label, li) => (
        <text
          key={li}
          x={X_POSITIONS[li]} y={HEIGHT - 6}
          textAnchor="middle"
          fill={COLORS[li]}
          fontSize="10.5"
          fillOpacity="0.65"
          fontFamily="system-ui, sans-serif"
        >
          {label}
        </text>
      ))}
    </svg>
  )
}
