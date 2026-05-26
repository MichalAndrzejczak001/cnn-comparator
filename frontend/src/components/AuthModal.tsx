import { useState } from 'react'

interface AuthModalProps {
  initialMode: 'login' | 'register'
  onClose: () => void
  onAuth: (token: string) => void
}

export default function AuthModal({ initialMode, onClose, onAuth }: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'register'>(initialMode)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)

  const switchMode = (next: 'login' | 'register') => {
    setMode(next)
    setError('')
    setSuccess('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSuccess('')
    setLoading(true)

    try {
      const endpoint = mode === 'login' ? '/auth/login' : '/auth/register'
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })

      const text = await res.text()

      if (!res.ok) {
        setError(text || (mode === 'login' ? 'Nieprawidłowy login lub hasło' : 'Błąd rejestracji'))
        return
      }

      if (mode === 'register') {
        setSuccess('Konto zostało utworzone. Możesz się teraz zalogować.')
        switchMode('login')
        setPassword('')
      } else {
        onAuth(text)
      }
    } catch {
      setError('Brak połączenia z serwerem')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">
            {mode === 'login' ? 'Zaloguj się' : 'Utwórz konto'}
          </span>
          <button className="modal-close" onClick={onClose} aria-label="Zamknij">
            ✕
          </button>
        </div>

        <div className="modal-tabs">
          <button
            className={`tab-btn${mode === 'login' ? ' active' : ''}`}
            onClick={() => switchMode('login')}
          >
            Logowanie
          </button>
          <button
            className={`tab-btn${mode === 'register' ? ' active' : ''}`}
            onClick={() => switchMode('register')}
          >
            Rejestracja
          </button>
        </div>

        {success && <div className="form-success">{success}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Nazwa użytkownika</label>
            <input
              className="form-input"
              type="text"
              placeholder="np. jan_kowalski"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
            />
          </div>
          <div className="form-group">
            <label className="form-label">Hasło</label>
            <input
              className="form-input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
          </div>

          {error && <div className="form-error">{error}</div>}

          <button
            type="submit"
            className="btn-primary btn-block"
            disabled={loading}
          >
            {loading
              ? 'Łączenie...'
              : mode === 'login'
              ? 'Zaloguj się'
              : 'Utwórz konto'}
          </button>
        </form>

        <div className="modal-footer">
          {mode === 'login' ? (
            <>
              Nie masz konta?{' '}
              <button onClick={() => switchMode('register')}>Zarejestruj się</button>
            </>
          ) : (
            <>
              Masz już konto?{' '}
              <button onClick={() => switchMode('login')}>Zaloguj się</button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
