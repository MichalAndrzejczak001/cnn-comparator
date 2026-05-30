import type { SampleGradCam } from '../../types/api'

interface Props {
  samples: SampleGradCam[]
}

export default function GradCamGallery({ samples }: Props) {
  if (!samples || samples.length === 0) return null

  return (
    <div>
      <h3 className="result-title">Przykładowe Grad-CAM ze zbioru testowego</h3>
      <p className="view-desc" style={{ marginBottom: '1rem' }}>
        Po jednym przykładzie na klasę — obszary zaznaczone ciepłymi kolorami wskazują, na czym skupił się model.
      </p>
      <div className="gradcam-gallery">
        {samples.map((s, i) => {
          const correct = s.true_label === s.predicted_label
          return (
            <div
              key={i}
              className="gradcam-card"
              style={{ borderColor: correct ? 'rgba(74,222,128,.4)' : 'rgba(248,113,113,.4)' }}
            >
              <img
                src={`data:image/png;base64,${s.gradcam_image}`}
                alt={`Grad-CAM: ${s.true_label}`}
                className="gradcam-img"
              />
              <div className="gradcam-labels">
                <div className="gradcam-label-row">
                  <span className="text-muted" style={{ fontSize: '0.7rem' }}>prawdziwa</span>
                  <span className="badge badge-dataset">{s.true_label}</span>
                </div>
                <div className="gradcam-label-row">
                  <span className="text-muted" style={{ fontSize: '0.7rem' }}>predykcja</span>
                  <span
                    className="badge"
                    style={{
                      color: correct ? '#4ade80' : '#f87171',
                      borderColor: correct ? 'rgba(74,222,128,.35)' : 'rgba(248,113,113,.35)',
                      background: correct ? 'rgba(74,222,128,.07)' : 'rgba(248,113,113,.07)',
                    }}
                  >
                    {s.predicted_label}
                  </span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem', textAlign: 'center' }}>
                  {(s.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
