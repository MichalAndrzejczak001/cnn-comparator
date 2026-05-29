import { useState, useEffect } from 'react'
import { getExperiments, rerunExperiment, updateNote } from '../../api/client'
import type { ExperimentResponse } from '../../types/api'
import { MODEL_LABELS, DATASET_LABELS } from '../../types/api'
import { downloadCsv, csvDate } from '../../utils/csv'
import ClassifyImageModal from './ClassifyImageModal'
import GradCamModal from './GradCamModal'
import DrawDigitModal from './DrawDigitModal'
import AugmentModal from './AugmentModal'

interface HistoryViewProps {
  onCompareSelected: (ids: number[]) => void
}

interface NoteModal {
  id: number
  text: string
}

interface ClassifyModal {
  id: number
  model: string
  dataset: string
}

function formatDate(dt: string): string {
  return new Date(dt).toLocaleString('pl-PL', { dateStyle: 'short', timeStyle: 'short' })
}

export default function HistoryView({ onCompareSelected }: HistoryViewProps) {
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [noteModal, setNoteModal] = useState<NoteModal | null>(null)
  const [classifyModal, setClassifyModal] = useState<ClassifyModal | null>(null)
  const [gradCamModal, setGradCamModal] = useState<ClassifyModal | null>(null)
  const [drawModal, setDrawModal] = useState<ClassifyModal | null>(null)
  const [augmentModal, setAugmentModal] = useState<ClassifyModal | null>(null)
  const [rerunning, setRerunning] = useState<number | null>(null)
  const [savingNote, setSavingNote] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchHistory()
  }, [])

  async function fetchHistory() {
    setLoading(true)
    setError('')
    try {
      const data = await getExperiments()
      setExperiments(data)
    } catch {
      setError('Błąd ładowania historii')
    } finally {
      setLoading(false)
    }
  }

  function toggleSelect(id: number) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function toggleAll() {
    if (selected.size === experiments.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(experiments.map((e) => e.id)))
    }
  }

  async function handleRerun(id: number) {
    setRerunning(id)
    setError('')
    try {
      const result = await rerunExperiment(id)
      setExperiments((prev) => [result, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd ponownego uruchomienia')
    } finally {
      setRerunning(null)
    }
  }

  async function handleSaveNote() {
    if (!noteModal) return
    setSavingNote(true)
    setError('')
    try {
      const updated = await updateNote(noteModal.id, noteModal.text)
      setExperiments((prev) => prev.map((e) => (e.id === updated.id ? updated : e)))
      setNoteModal(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd zapisu notatki')
    } finally {
      setSavingNote(false)
    }
  }

  function exportCsv() {
    const header = ['ID', 'Model', 'Zbiór danych', 'Epoki', 'Batch size', 'Learning rate',
                    'Dokładność test.', 'Strata test.', 'Notatka', 'Data']
    const body = experiments.map(e => [
      e.id,
      MODEL_LABELS[e.model] ?? e.model,
      e.dataset,
      e.epochs,
      e.batch_size,
      e.learning_rate,
      (e.test_accuracy * 100).toFixed(4),
      e.test_loss.toFixed(6),
      e.note ?? '',
      new Date(e.created_at).toLocaleString('pl-PL'),
    ])
    downloadCsv(`historia_${csvDate()}.csv`, [header, ...body])
  }

  const selectedArr = Array.from(selected)
  const allSelected = experiments.length > 0 && selected.size === experiments.length

  return (
    <div className="view">
      <div className="view-header">
        <div>
          <h2 className="view-title">Historia eksperymentów</h2>
          <p className="view-desc" style={{ margin: 0 }}>
            {loading ? 'Ładowanie...' : `${experiments.length} eksperymentów`}
          </p>
        </div>
        <div className="view-actions">
          {selected.size >= 2 && (
            <button
              className="btn-primary"
              onClick={() => onCompareSelected(selectedArr)}
            >
              Porównaj wybrane ({selected.size})
            </button>
          )}
          {experiments.length > 0 && (
            <button className="btn-outline" onClick={exportCsv}>
              Eksportuj CSV
            </button>
          )}
          <button className="btn-outline" onClick={fetchHistory}>
            Odśwież
          </button>
        </div>
      </div>

      {error && <div className="form-error" style={{ marginBottom: '1rem' }}>{error}</div>}

      {!loading && experiments.length === 0 && (
        <div className="empty-state">
          Brak eksperymentów. Uruchom pierwszy trening w zakładce "Nowy eksperyment"!
        </div>
      )}

      {experiments.length > 0 && (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 36 }}>
                  <input
                    type="checkbox"
                    checked={allSelected}
                    onChange={toggleAll}
                    title="Zaznacz wszystkie"
                  />
                </th>
                <th>ID</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Epoki</th>
                <th>Dokł. test.</th>
                <th>Strata test.</th>
                <th>Data</th>
                <th>Notatka</th>
                <th>Akcje</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr key={exp.id} className={selected.has(exp.id) ? 'row-selected' : ''}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.has(exp.id)}
                      onChange={() => toggleSelect(exp.id)}
                    />
                  </td>
                  <td className="text-muted">#{exp.id}</td>
                  <td>
                    <span className="badge badge-model">{MODEL_LABELS[exp.model] || exp.model}</span>
                  </td>
                  <td>
                    <span className="badge badge-dataset">{DATASET_LABELS[exp.dataset] ?? exp.dataset.toUpperCase()}</span>
                  </td>
                  <td>{exp.epochs}</td>
                  <td className="text-green">{(exp.test_accuracy * 100).toFixed(2)}%</td>
                  <td>{exp.test_loss.toFixed(4)}</td>
                  <td className="text-muted">{formatDate(exp.created_at)}</td>
                  <td className="note-cell">
                    {exp.note ? (
                      <span title={exp.note}>{exp.note}</span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    <div className="row-actions">
                      <button
                        className="btn-sm btn-outline"
                        onClick={() => setNoteModal({ id: exp.id, text: exp.note || '' })}
                      >
                        Notatka
                      </button>
                      <button
                        className="btn-sm btn-outline"
                        disabled={rerunning === exp.id}
                        onClick={() => handleRerun(exp.id)}
                      >
                        {rerunning === exp.id ? '...' : 'Wznów'}
                      </button>
                      {exp.model_id && (
                        <button
                          className="btn-sm btn-primary"
                          onClick={() => setClassifyModal({ id: exp.id, model: exp.model, dataset: exp.dataset })}
                        >
                          Klasyfikuj
                        </button>
                      )}
                      {exp.model_id && (
                        <button
                          className="btn-sm btn-outline"
                          onClick={() => setGradCamModal({ id: exp.id, model: exp.model, dataset: exp.dataset })}
                        >
                          Grad-CAM
                        </button>
                      )}
                      {exp.model_id && exp.dataset === 'mnist' && (
                        <button
                          className="btn-sm btn-outline"
                          onClick={() => setDrawModal({ id: exp.id, model: exp.model, dataset: exp.dataset })}
                        >
                          Rysuj
                        </button>
                      )}
                      {exp.model_id && (
                        <button
                          className="btn-sm btn-outline"
                          onClick={() => setAugmentModal({ id: exp.id, model: exp.model, dataset: exp.dataset })}
                        >
                          Augmentuj
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {classifyModal && (
        <ClassifyImageModal
          experimentId={classifyModal.id}
          model={classifyModal.model}
          dataset={classifyModal.dataset}
          onClose={() => setClassifyModal(null)}
        />
      )}

      {gradCamModal && (
        <GradCamModal
          experimentId={gradCamModal.id}
          model={gradCamModal.model}
          dataset={gradCamModal.dataset}
          onClose={() => setGradCamModal(null)}
        />
      )}

      {drawModal && (
        <DrawDigitModal
          experimentId={drawModal.id}
          model={drawModal.model}
          onClose={() => setDrawModal(null)}
        />
      )}

      {augmentModal && (
        <AugmentModal
          experimentId={augmentModal.id}
          model={augmentModal.model}
          dataset={augmentModal.dataset}
          onClose={() => setAugmentModal(null)}
        />
      )}

      {noteModal && (
        <div className="modal-backdrop" onClick={() => setNoteModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">Notatka — eksperyment #{noteModal.id}</span>
              <button className="modal-close" onClick={() => setNoteModal(null)}>
                ✕
              </button>
            </div>
            <div className="form-group">
              <label className="form-label">Treść notatki</label>
              <textarea
                className="form-input"
                rows={4}
                placeholder="Dodaj notatkę do eksperymentu..."
                value={noteModal.text}
                onChange={(e) => setNoteModal({ ...noteModal, text: e.target.value })}
                style={{ resize: 'vertical' }}
              />
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
              <button className="btn-outline" onClick={() => setNoteModal(null)}>
                Anuluj
              </button>
              <button
                className="btn-primary"
                onClick={handleSaveNote}
                disabled={savingNote}
              >
                {savingNote ? 'Zapisywanie...' : 'Zapisz'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
