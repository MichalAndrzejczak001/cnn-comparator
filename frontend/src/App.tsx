import { useState } from 'react'
import LandingPage from './components/LandingPage'
import Dashboard from './components/dashboard/Dashboard'
import './App.css'

function decodeJwtSub(token: string): string {
  try {
    const payload = JSON.parse(
      atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/'))
    )
    return (payload.sub as string) || 'user'
  } catch {
    return 'user'
  }
}

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

  return <Dashboard username={decodeJwtSub(token)} onLogout={handleLogout} />
}

export default App
