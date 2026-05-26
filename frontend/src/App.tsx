import { useState } from 'react'
import LandingPage from './components/LandingPage'
import './App.css'

function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))

  const handleAuth = (newToken: string) => {
    localStorage.setItem('token', newToken)
    setToken(newToken)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    setToken(null)
  }

  if (!token) {
    return <LandingPage onAuth={handleAuth} />
  }

  return (
    <div className="dashboard-placeholder">
      <h1>CNN Comparator</h1>
      <p>Zalogowano pomyślnie.</p>
      <button className="btn-outline" onClick={handleLogout}>
        Wyloguj się
      </button>
    </div>
  )
}

export default App
