const CELL = 6

// 10×10 pixel art digits — each cell = 6px → 60×60 SVG
const DIGIT_1: number[][] = [
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,1,1,1,0,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,1,1,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
]

const DIGIT_7: number[][] = [
  [0,1,1,1,1,1,1,0,0,0],
  [0,1,1,1,1,1,1,0,0,0],
  [0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,0,1,1,0,0,0,0,0],
  [0,0,1,1,0,0,0,0,0,0],
  [0,0,1,1,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
]

const DIGIT_3: number[][] = [
  [0,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0],
  [0,0,1,1,1,0,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0],
  [0,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
]

function MnistSample({ pixels, label }: { pixels: number[][], label: string }) {
  const cols = pixels[0]?.length ?? 0
  const rows = pixels.length
  return (
    <div className="dataset-sample">
      <svg width={cols * CELL} height={rows * CELL} style={{ display: 'block', borderRadius: 4 }}>
        <rect width={cols * CELL} height={rows * CELL} fill="#080808" />
        {pixels.flatMap((row, r) =>
          row.map((v, c) =>
            v > 0 ? (
              <rect
                key={`${r}-${c}`}
                x={c * CELL}
                y={r * CELL}
                width={CELL}
                height={CELL}
                fill="#e8e8e8"
              />
            ) : null
          )
        )}
      </svg>
      <div className="dataset-sample-label">{label}</div>
    </div>
  )
}

function CifarAirplane() {
  return (
    <svg viewBox="0 0 60 60" width="60" height="60" style={{ display: 'block', borderRadius: 4 }}>
      <rect width="60" height="60" fill="#5a8fc0" />
      <ellipse cx="46" cy="12" rx="9" ry="4" fill="rgba(255,255,255,0.55)" />
      <ellipse cx="12" cy="20" rx="6" ry="3" fill="rgba(255,255,255,0.4)" />
      <ellipse cx="30" cy="33" rx="20" ry="5" fill="#e0e0e0" />
      <ellipse cx="50" cy="33" rx="4" ry="4" fill="#d0d0d0" />
      <polygon points="24,33 38,33 34,22 22,22" fill="#c8c8c8" />
      <polygon points="10,31 17,31 17,23" fill="#c8c8c8" />
      <polygon points="10,33 17,33 17,37 10,37" fill="#c8c8c8" />
    </svg>
  )
}

function CifarCar() {
  return (
    <svg viewBox="0 0 60 60" width="60" height="60" style={{ display: 'block', borderRadius: 4 }}>
      <rect width="60" height="60" fill="#3a3a3a" />
      <rect x="0" y="44" width="60" height="16" fill="#2a2a2a" />
      <rect x="14" y="48" width="8" height="2" fill="#555" />
      <rect x="28" y="48" width="8" height="2" fill="#555" />
      <rect x="5" y="28" width="50" height="16" rx="3" fill="#cc2222" />
      <rect x="14" y="18" width="28" height="14" rx="3" fill="#aa1111" />
      <rect x="16" y="20" width="11" height="10" rx="1" fill="#88ccff" opacity="0.8" />
      <rect x="29" y="20" width="11" height="10" rx="1" fill="#88ccff" opacity="0.8" />
      <circle cx="16" cy="44" r="6" fill="#111" />
      <circle cx="44" cy="44" r="6" fill="#111" />
      <circle cx="16" cy="44" r="2.5" fill="#555" />
      <circle cx="44" cy="44" r="2.5" fill="#555" />
    </svg>
  )
}

function CifarFrog() {
  return (
    <svg viewBox="0 0 60 60" width="60" height="60" style={{ display: 'block', borderRadius: 4 }}>
      <rect width="60" height="60" fill="#1a3d14" />
      <ellipse cx="30" cy="38" rx="18" ry="14" fill="#2ecc40" />
      <ellipse cx="30" cy="26" rx="14" ry="11" fill="#2ecc40" />
      <circle cx="20" cy="19" r="6" fill="#2ecc40" />
      <circle cx="40" cy="19" r="6" fill="#2ecc40" />
      <circle cx="20" cy="18" r="4" fill="#fff" />
      <circle cx="40" cy="18" r="4" fill="#fff" />
      <circle cx="20" cy="18" r="2" fill="#111" />
      <circle cx="40" cy="18" r="2" fill="#111" />
      <path d="M 22 32 Q 30 37 38 32" stroke="#1a8a28" strokeWidth="1.5" fill="none" />
    </svg>
  )
}

const MNIST_CLASSES = [
  '0 – zero', '1 – jeden', '2 – dwa', '3 – trzy', '4 – cztery',
  '5 – pięć', '6 – sześć', '7 – siedem', '8 – osiem', '9 – dziewięć',
]

const CIFAR_CLASSES = [
  'Samolot', 'Samochód', 'Ptak', 'Kot', 'Jeleń',
  'Pies', 'Żaba', 'Koń', 'Statek', 'Ciężarówka',
]

const COMPARE_ROWS: [string, string, string][] = [
  ['Liczba klas', '10 (cyfry 0–9)', '10 (obiekty)'],
  ['Rozmiar obrazu', '28×28 px', '32×32 px'],
  ['Kanały', '1 (szarość)', '3 (RGB)'],
  ['Zbiór treningowy', '60 000', '50 000'],
  ['Zbiór testowy', '10 000', '10 000'],
  ['Łącznie', '70 000', '60 000'],
  ['Trudność', 'Łatwy', 'Średni / trudny'],
  ['Simple CNN – typowa dokładność', '~98–99%', '~65–75%'],
]

export default function DatasetsView() {
  return (
    <div className="view">
      <h2 className="view-title">O zbiorach danych</h2>
      <p className="view-desc">
        Charakterystyka zbiorów MNIST i CIFAR-10 dostępnych w aplikacji.
      </p>

      <div className="datasets-grid">
        {/* MNIST */}
        <div className="dataset-card">
          <div className="dataset-card-header" style={{ borderLeftColor: '#4f86f7' }}>
            <div className="dataset-card-title">MNIST</div>
            <div className="dataset-card-subtitle">Modified National Institute of Standards and Technology</div>
          </div>

          <p className="dataset-desc">
            Zbiór 70 000 odręcznych cyfr arabskich (0–9) w skali szarości.
            Powszechny punkt odniesienia dla sieci klasyfikujących obrazy — prosty i dobrze zbadany.
          </p>

          <div className="dataset-stats">
            {[
              ['70 000', 'Obrazów'],
              ['10', 'Klas'],
              ['28×28', 'Rozmiar'],
              ['1', 'Kanał'],
              ['60 000', 'Trening'],
              ['10 000', 'Test'],
            ].map(([v, l]) => (
              <div key={l} className="dataset-stat">
                <div className="dataset-stat-value" style={{ color: '#4f86f7' }}>{v}</div>
                <div className="dataset-stat-label">{l}</div>
              </div>
            ))}
          </div>

          <div className="dataset-classes">
            <div className="dataset-classes-title">Klasy</div>
            <div className="dataset-class-list">
              {MNIST_CLASSES.map(c => (
                <span key={c} className="dataset-class-tag">{c}</span>
              ))}
            </div>
          </div>

          <div className="dataset-samples-title">Przykładowe obrazy</div>
          <div className="dataset-samples">
            <MnistSample pixels={DIGIT_1} label="cyfra 1" />
            <MnistSample pixels={DIGIT_7} label="cyfra 7" />
            <MnistSample pixels={DIGIT_3} label="cyfra 3" />
          </div>
        </div>

        {/* CIFAR-10 */}
        <div className="dataset-card">
          <div className="dataset-card-header" style={{ borderLeftColor: '#a78bfa' }}>
            <div className="dataset-card-title">CIFAR-10</div>
            <div className="dataset-card-subtitle">Canadian Institute for Advanced Research</div>
          </div>

          <p className="dataset-desc">
            Zbiór 60 000 kolorowych zdjęć 10 kategorii obiektów.
            Znacznie trudniejszy od MNIST — wymaga głębszych sieci i dłuższego trenowania.
          </p>

          <div className="dataset-stats">
            {[
              ['60 000', 'Obrazów'],
              ['10', 'Klas'],
              ['32×32', 'Rozmiar'],
              ['3', 'Kanały (RGB)'],
              ['50 000', 'Trening'],
              ['10 000', 'Test'],
            ].map(([v, l]) => (
              <div key={l} className="dataset-stat">
                <div className="dataset-stat-value" style={{ color: '#a78bfa' }}>{v}</div>
                <div className="dataset-stat-label">{l}</div>
              </div>
            ))}
          </div>

          <div className="dataset-classes">
            <div className="dataset-classes-title">Klasy</div>
            <div className="dataset-class-list">
              {CIFAR_CLASSES.map(c => (
                <span key={c} className="dataset-class-tag">{c}</span>
              ))}
            </div>
          </div>

          <div className="dataset-samples-title">Przykładowe obrazy</div>
          <div className="dataset-samples">
            <div className="dataset-sample">
              <CifarAirplane />
              <div className="dataset-sample-label">samolot</div>
            </div>
            <div className="dataset-sample">
              <CifarCar />
              <div className="dataset-sample-label">samochód</div>
            </div>
            <div className="dataset-sample">
              <CifarFrog />
              <div className="dataset-sample-label">żaba</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="result-title">Porównanie zbiorów</div>
        <div className="table-wrapper" style={{ marginBottom: 0 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Cecha</th>
                <th>MNIST</th>
                <th>CIFAR-10</th>
              </tr>
            </thead>
            <tbody>
              {COMPARE_ROWS.map(([f, m, c]) => (
                <tr key={f}>
                  <td className="text-muted" style={{ fontWeight: 500 }}>{f}</td>
                  <td>{m}</td>
                  <td>{c}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
