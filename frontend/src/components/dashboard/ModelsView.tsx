import { MODEL_COLORS } from '../../types/api'

type LayerType = 'conv' | 'pool' | 'fc' | 'flatten' | 'resblock'

interface LayerBlock {
  label: string
  type: LayerType
  note?: string
}

interface ModelData {
  key: string
  label: string
  accent: string
  year: string
  description: string
  layers: LayerBlock[]
  specs: { label: string; value: string }[]
}

const LAYER_COLORS: Record<LayerType, string> = {
  conv: '#4f86f7',
  pool: '#0ea5e9',
  fc: '#7c3aed',
  flatten: '#475569',
  resblock: '#10b981',
}

const LAYER_LABELS: Record<LayerType, string> = {
  conv: 'Konwolucja',
  pool: 'Pooling',
  fc: 'Liniowa (FC)',
  flatten: 'Flatten',
  resblock: 'Blok resztkowy',
}

const MODELS_DATA: ModelData[] = [
  {
    key: 'simple_cnn',
    label: 'Simple CNN',
    accent: MODEL_COLORS['simple_cnn'],
    year: '—',
    description:
      'Bazowa architektura splotowa z dwoma warstwami konwolucyjnymi i dwiema w pełni połączonymi. Szybka w trenowaniu i prosta do analizy — dobry punkt odniesienia dla porównań.',
    layers: [
      { label: 'Conv 32', type: 'conv', note: '3×3 + ReLU' },
      { label: 'MaxPool', type: 'pool', note: '2×2' },
      { label: 'Conv 64', type: 'conv', note: '3×3 + ReLU' },
      { label: 'MaxPool', type: 'pool', note: '2×2' },
      { label: 'Flatten', type: 'flatten' },
      { label: 'FC 128', type: 'fc', note: 'ReLU' },
      { label: 'FC 10', type: 'fc', note: 'wyjście' },
    ],
    specs: [
      { label: 'Parametry', value: '~200 K' },
      { label: 'Warstwy', value: '4' },
      { label: 'Filtry', value: '32 / 64' },
      { label: 'Aktywacja', value: 'ReLU' },
      { label: 'Pooling', value: 'Max 2×2' },
      { label: 'Rok', value: '—' },
    ],
  },
  {
    key: 'lenet5',
    label: 'LeNet-5',
    accent: MODEL_COLORS['lenet5'],
    year: '1998',
    description:
      'Klasyczna architektura Yann LeCun\'a do rozpoznawania odręcznych cyfr. Pionierska sieć CNN z uśredniającym poolingiem i aktywacją Tanh — opublikowana w 1998 r.',
    layers: [
      { label: 'Conv 6', type: 'conv', note: '5×5 + Tanh' },
      { label: 'AvgPool', type: 'pool', note: '2×2' },
      { label: 'Conv 16', type: 'conv', note: '5×5 + Tanh' },
      { label: 'AvgPool', type: 'pool', note: '2×2' },
      { label: 'Flatten', type: 'flatten' },
      { label: 'FC 120', type: 'fc', note: 'Tanh' },
      { label: 'FC 84', type: 'fc', note: 'Tanh' },
      { label: 'FC 10', type: 'fc', note: 'wyjście' },
    ],
    specs: [
      { label: 'Parametry', value: '~60 K' },
      { label: 'Warstwy', value: '5' },
      { label: 'Filtry', value: '6 / 16' },
      { label: 'Aktywacja', value: 'Tanh' },
      { label: 'Pooling', value: 'Avg 2×2' },
      { label: 'Rok', value: '1998' },
    ],
  },
  {
    key: 'vgg11',
    label: 'VGG-11',
    accent: MODEL_COLORS['vgg11'],
    year: '2014',
    description:
      'Architektura Oxford VGG Group z jedenastoma warstwami konwolucyjnymi 3×3. Prosta, regularna struktura: bloki o rosnącej liczbie kanałów (64→512) oddzielone MaxPoolingiem.',
    layers: [
      { label: 'Conv 64', type: 'conv', note: '3×3' },
      { label: 'MaxPool', type: 'pool' },
      { label: 'Conv 128', type: 'conv', note: '3×3' },
      { label: 'MaxPool', type: 'pool' },
      { label: 'Conv 256×2', type: 'conv', note: '3×3' },
      { label: 'MaxPool', type: 'pool' },
      { label: 'Conv 512×4', type: 'conv', note: '3×3' },
      { label: 'MaxPool×2', type: 'pool' },
      { label: 'Flatten', type: 'flatten' },
      { label: 'FC 4096', type: 'fc', note: 'ReLU' },
      { label: 'FC 4096', type: 'fc', note: 'ReLU' },
      { label: 'FC 10', type: 'fc', note: 'wyjście' },
    ],
    specs: [
      { label: 'Parametry', value: '~130 M' },
      { label: 'Warstwy', value: '11' },
      { label: 'Kanały', value: '64–512' },
      { label: 'Aktywacja', value: 'ReLU' },
      { label: 'Pooling', value: 'Max 2×2' },
      { label: 'Rok', value: '2014' },
    ],
  },
  {
    key: 'resnet18',
    label: 'ResNet-18',
    accent: MODEL_COLORS['resnet18'],
    year: '2015',
    description:
      'Residual Network z 18 warstwami — He et al., CVPR 2015. Połączenia resztkowe (skip connections) rozwiązują problem zanikającego gradientu w głębokich sieciach.',
    layers: [
      { label: 'Conv 64', type: 'conv', note: '7×7, s=2' },
      { label: 'MaxPool', type: 'pool', note: '3×3' },
      { label: 'ResBlock×2', type: 'resblock', note: '64 ch' },
      { label: 'ResBlock×2', type: 'resblock', note: '128 ch' },
      { label: 'ResBlock×2', type: 'resblock', note: '256 ch' },
      { label: 'ResBlock×2', type: 'resblock', note: '512 ch' },
      { label: 'AvgPool', type: 'pool' },
      { label: 'FC 10', type: 'fc', note: 'wyjście' },
    ],
    specs: [
      { label: 'Parametry', value: '~11 M' },
      { label: 'Warstwy', value: '18' },
      { label: 'Kanały', value: '64–512' },
      { label: 'Aktywacja', value: 'ReLU + BN' },
      { label: 'Skip conn.', value: 'Tak' },
      { label: 'Rok', value: '2015' },
    ],
  },
]

function LayerDiagram({ layers }: { layers: LayerBlock[] }) {
  return (
    <div className="layer-diagram">
      {layers.map((l, i) => (
        <div key={i} className="layer-diagram-item">
          <div
            className="layer-block"
            style={{
              background: LAYER_COLORS[l.type] + '1a',
              borderColor: LAYER_COLORS[l.type] + '55',
              color: LAYER_COLORS[l.type],
            }}
          >
            <span className="layer-block-label">{l.label}</span>
            {l.note && <span className="layer-block-note">{l.note}</span>}
          </div>
          {i < layers.length - 1 && <div className="layer-arrow">›</div>}
        </div>
      ))}
    </div>
  )
}

export default function ModelsView() {
  return (
    <div className="view">
      <h2 className="view-title">O modelach</h2>
      <p className="view-desc">
        Architektura, parametry i schemat warstw każdej sieci dostępnej w aplikacji.
      </p>

      <div className="models-grid">
        {MODELS_DATA.map(m => (
          <div key={m.key} className="model-card">
            <div className="model-card-header" style={{ borderLeftColor: m.accent }}>
              <div className="model-card-title">{m.label}</div>
              {m.year !== '—' && (
                <div className="model-card-year">Opublikowano: {m.year}</div>
              )}
            </div>

            <p className="model-card-desc">{m.description}</p>

            <div className="model-specs-grid">
              {m.specs.map(s => (
                <div key={s.label} className="model-spec">
                  <div className="model-spec-value" style={{ color: m.accent }}>
                    {s.value}
                  </div>
                  <div className="model-spec-label">{s.label}</div>
                </div>
              ))}
            </div>

            <div className="model-section-label">Schemat warstw</div>
            <LayerDiagram layers={m.layers} />
          </div>
        ))}
      </div>

      <div className="model-legend">
        <span className="model-legend-title">Legenda:</span>
        {(Object.entries(LAYER_LABELS) as [LayerType, string][]).map(([type, name]) => (
          <span key={type} className="legend-item">
            <span className="legend-dot" style={{ background: LAYER_COLORS[type] }} />
            {name}
          </span>
        ))}
      </div>
    </div>
  )
}
