import type { ExperimentResponse } from '../../types/api'
import { MODEL_COLORS, MODEL_LABELS } from '../../types/api'
import LossChart from './LossChart'

interface CompareSelectedViewProps {
  experiments: ExperimentResponse[]
}

export default function CompareSelectedView({ experiments }: CompareSelectedViewProps) {
  const best = [...experiments].sort((a, b) => b.test_accuracy - a.test_accuracy)[0]

  return (
    <div className="view">
      <h2 className="view-title">Porównanie wybranych eksperymentów</h2>
      <p className="view-desc">
        Zestawienie {experiments.length} wybranych eksperymentów z historii.
      </p>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Dataset</th>
              <th>Epoki</th>
              <th>Batch</th>
              <th>LR</th>
              <th>Dokł. testowa</th>
              <th>Strata testowa</th>
              <th>Notatka</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.id} className={exp.id === best.id ? 'row-selected' : ''}>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: MODEL_COLORS[exp.model] || '#888',
                        flexShrink: 0,
                      }}
                    />
                    <span className="badge badge-model">{MODEL_LABELS[exp.model] || exp.model}</span>
                    {exp.id === best.id && (
                      <span className="badge" style={{ color: '#4ade80', borderColor: 'rgba(74,222,128,.35)', background: 'rgba(74,222,128,.07)', fontSize: '0.7rem' }}>
                        najlepsza
                      </span>
                    )}
                  </div>
                </td>
                <td>
                  <span className="badge badge-dataset">{exp.dataset.toUpperCase()}</span>
                </td>
                <td>{exp.epochs}</td>
                <td>{exp.batch_size}</td>
                <td>{exp.learning_rate}</td>
                <td className="text-green">{(exp.test_accuracy * 100).toFixed(2)}%</td>
                <td>{exp.test_loss.toFixed(4)}</td>
                <td className="note-cell text-muted">{exp.note || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <LossChart
          title="Strata treningowa per epoka"
          series={experiments.map((exp) => ({
            label: `${MODEL_LABELS[exp.model] || exp.model} #${exp.id}`,
            data: exp.train_loss_per_epoch,
            color: MODEL_COLORS[exp.model] || '#4f86f7',
          }))}
        />
      </div>
    </div>
  )
}
