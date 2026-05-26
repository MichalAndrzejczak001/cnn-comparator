import { useState } from 'react'
import AuthModal from './AuthModal'
import NeuralNetSVG from './NeuralNetSVG'

interface LandingPageProps {
  onAuth: (token: string) => void
}

export default function LandingPage({ onAuth }: LandingPageProps) {
  const [modalMode, setModalMode] = useState<'login' | 'register' | null>(null)

  return (
    <div className="landing">
      <nav className="nav">
        <div className="nav-brand">
          <span className="nav-brand-accent">CNN</span> Comparator
        </div>
        <div className="nav-actions">
          <button className="btn-outline" onClick={() => setModalMode('login')}>
            Zaloguj się
          </button>
          <button className="btn-primary" onClick={() => setModalMode('register')}>
            Zarejestruj się
          </button>
        </div>
      </nav>

      <main className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            Porównuj modele CNN<br />
            <span className="hero-accent">w jednym miejscu</span>
          </h1>
          <p className="hero-desc">
            Platforma do trenowania, porównywania i analizowania konwolucyjnych
            sieci neuronowych. Śledź historię eksperymentów, dodawaj notatki
            i zestawiaj wyniki wielu modeli jednocześnie.
          </p>
          <ul className="hero-features">
            <li className="feature">
              <span className="feature-dot" />
              Trenowanie i konfiguracja modeli CNN
            </li>
            <li className="feature">
              <span className="feature-dot" />
              Porównywanie wielu eksperymentów
            </li>
            <li className="feature">
              <span className="feature-dot" />
              Historia eksperymentów i notatki
            </li>
            <li className="feature">
              <span className="feature-dot" />
              Analiza wyników przy pomocy AI
            </li>
          </ul>
          <button
            className="btn-primary btn-lg"
            onClick={() => setModalMode('register')}
          >
            Zacznij teraz
          </button>
        </div>

        <div className="hero-visual">
          <div className="visual-card">
            <NeuralNetSVG />
            <p className="visual-caption">Architektura konwolucyjnej sieci neuronowej</p>
          </div>
        </div>
      </main>

      {modalMode && (
        <AuthModal
          initialMode={modalMode}
          onClose={() => setModalMode(null)}
          onAuth={onAuth}
        />
      )}
    </div>
  )
}
