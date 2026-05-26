import { useState } from 'react'
import { compareExperiments } from '../../api/client'
import type { ExperimentResponse } from '../../types/api'
import OverviewView from './OverviewView'
import NewExperimentView from './NewExperimentView'
import HistoryView from './HistoryView'
import CompareSelectedView from './CompareSelectedView'
import CompareAllView from './CompareAllView'
import ModelsView from './ModelsView'
import DatasetsView from './DatasetsView'

type View = 'overview' | 'new' | 'history' | 'compare-selected' | 'compare-all' | 'models' | 'datasets'

interface DashboardProps {
  username: string
  onLogout: () => void
}

export default function Dashboard({ username, onLogout }: DashboardProps) {
  const [view, setView] = useState<View>('overview')
  const [compareData, setCompareData] = useState<ExperimentResponse[] | null>(null)
  const [compareError, setCompareError] = useState('')

  const handleCompareSelected = async (ids: number[]) => {
    setCompareError('')
    try {
      const data = await compareExperiments(ids)
      setCompareData(data)
      setView('compare-selected')
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : 'Błąd pobierania eksperymentów')
    }
  }

  return (
    <div className="dash">
      <header className="dash-header">
        <div className="nav-brand">
          <span className="nav-brand-accent">CNN</span> Comparator
        </div>
        <div className="dash-user">
          <span className="dash-username">{username}</span>
          <button className="btn-outline" onClick={onLogout}>
            Wyloguj
          </button>
        </div>
      </header>

      <div className="dash-body">
        <nav className="dash-nav">
          <button
            className={`dash-nav-item${view === 'overview' ? ' active' : ''}`}
            onClick={() => setView('overview')}
          >
            Przegląd
          </button>
          <button
            className={`dash-nav-item${view === 'new' ? ' active' : ''}`}
            onClick={() => setView('new')}
          >
            Nowy eksperyment
          </button>
          <button
            className={`dash-nav-item${view === 'history' ? ' active' : ''}`}
            onClick={() => setView('history')}
          >
            Historia
          </button>
          <button
            className={`dash-nav-item${view === 'compare-selected' ? ' active' : ''}${!compareData ? ' disabled' : ''}`}
            onClick={() => compareData && setView('compare-selected')}
            disabled={!compareData}
            title={!compareData ? 'Wybierz eksperymentu w historii' : undefined}
          >
            Porównaj wybrane
          </button>
          <button
            className={`dash-nav-item${view === 'compare-all' ? ' active' : ''}`}
            onClick={() => setView('compare-all')}
          >
            Porównaj wszystkie
          </button>
          <div className="dash-nav-divider" />
          <button
            className={`dash-nav-item${view === 'models' ? ' active' : ''}`}
            onClick={() => setView('models')}
          >
            O modelach
          </button>
          <button
            className={`dash-nav-item${view === 'datasets' ? ' active' : ''}`}
            onClick={() => setView('datasets')}
          >
            O zbiorach danych
          </button>
        </nav>

        <main className="dash-content">
          {compareError && (
            <div className="form-error" style={{ marginBottom: '1rem' }}>
              {compareError}
            </div>
          )}
          {view === 'overview' && <OverviewView />}
          {view === 'new' && <NewExperimentView />}
          {view === 'history' && (
            <HistoryView onCompareSelected={handleCompareSelected} />
          )}
          {view === 'compare-selected' && compareData && (
            <CompareSelectedView experiments={compareData} />
          )}
          {view === 'compare-all' && <CompareAllView />}
          {view === 'models' && <ModelsView />}
          {view === 'datasets' && <DatasetsView />}
        </main>
      </div>
    </div>
  )
}
