interface ConfusionMatrixProps {
  matrix: number[][]
  dataset: string
}

const MNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const CIFAR_LABELS = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

const CELL = 38
const LABEL_LEFT = 48
const LABEL_BOTTOM = 52
const PAD_TOP = 20
const PAD_RIGHT = 10

export default function ConfusionMatrix({ matrix, dataset }: ConfusionMatrixProps) {
  const n = matrix.length
  const labels = dataset === 'cifar10' ? CIFAR_LABELS : MNIST_LABELS

  const maxVal = Math.max(...matrix.flatMap(row => row))

  const W = LABEL_LEFT + n * CELL + PAD_RIGHT
  const H = PAD_TOP + n * CELL + LABEL_BOTTOM

  function cellColor(val: number) {
    const t = maxVal > 0 ? val / maxVal : 0
    const r = Math.round(13 + (79 - 13) * t)
    const g = Math.round(22 + (134 - 22) * t)
    const b = Math.round(39 + (247 - 39) * t)
    return `rgb(${r},${g},${b})`
  }

  function textColor(val: number) {
    const t = maxVal > 0 ? val / maxVal : 0
    return t > 0.55 ? '#0a0e1a' : '#e2e8f0'
  }

  return (
    <div className="chart-container">
      <div className="chart-title">Macierz pomyłek</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 480 }}>

        {/* Y-axis label */}
        <text
          textAnchor="middle" fill="#3d5070" fontSize="10"
          transform={`translate(8,${PAD_TOP + (n * CELL) / 2}) rotate(-90)`}
        >
          Prawdziwa klasa
        </text>

        {/* X-axis label */}
        <text
          textAnchor="middle" fill="#3d5070" fontSize="10"
          x={LABEL_LEFT + (n * CELL) / 2}
          y={H - 2}
        >
          Przewidziana klasa
        </text>

        {/* Row labels (left) */}
        {labels.map((label, i) => (
          <text
            key={`row-${i}`}
            x={LABEL_LEFT - 5}
            y={PAD_TOP + i * CELL + CELL / 2 + 4}
            textAnchor="end"
            fill="#8899b0"
            fontSize="9"
          >
            {label}
          </text>
        ))}

        {/* Column labels (bottom, rotated 45°) */}
        {labels.map((label, j) => (
          <text
            key={`col-${j}`}
            textAnchor="end"
            fill="#8899b0"
            fontSize="9"
            transform={`translate(${LABEL_LEFT + j * CELL + CELL / 2 + 4},${PAD_TOP + n * CELL + 6}) rotate(45)`}
          >
            {label}
          </text>
        ))}

        {/* Cells */}
        {matrix.map((row, i) =>
          row.map((val, j) => {
            const x = LABEL_LEFT + j * CELL
            const y = PAD_TOP + i * CELL
            const isDiagonal = i === j
            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={x} y={y}
                  width={CELL} height={CELL}
                  fill={cellColor(val)}
                  stroke={isDiagonal ? '#4f86f7' : '#0d1627'}
                  strokeWidth={isDiagonal ? 1.5 : 0.5}
                />
                {val > 0 && (
                  <text
                    x={x + CELL / 2}
                    y={y + CELL / 2 + 4}
                    textAnchor="middle"
                    fill={textColor(val)}
                    fontSize="9.5"
                    fontWeight={isDiagonal ? 'bold' : 'normal'}
                  >
                    {val}
                  </text>
                )}
              </g>
            )
          })
        )}
      </svg>
    </div>
  )
}
