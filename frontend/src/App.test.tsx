import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('./components/LandingPage', () => ({
  default: () => <div data-testid="landing-page" />,
}))

vi.mock('./components/dashboard/Dashboard', () => ({
  default: ({ username }: { username: string }) => (
    <div data-testid="dashboard">{username}</div>
  ),
}))

function makeJwt(payload: Record<string, unknown>): string {
  const encode = (obj: unknown) =>
    btoa(JSON.stringify(obj))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '')
  return `${encode({ alg: 'HS256' })}.${encode(payload)}.signature`
}

describe('App', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('renders LandingPage when no token in localStorage', () => {
    render(<App />)
    expect(screen.getByTestId('landing-page')).toBeInTheDocument()
  })

  it('renders Dashboard when token exists in localStorage', () => {
    localStorage.setItem('token', makeJwt({ sub: 'alice' }))
    render(<App />)
    expect(screen.getByTestId('dashboard')).toBeInTheDocument()
  })

  it('passes decoded username from JWT sub claim to Dashboard', () => {
    localStorage.setItem('token', makeJwt({ sub: 'alice' }))
    render(<App />)
    expect(screen.getByTestId('dashboard')).toHaveTextContent('alice')
  })

  it('falls back to "user" when token is not a valid JWT', () => {
    localStorage.setItem('token', 'not-a-jwt')
    render(<App />)
    expect(screen.getByTestId('dashboard')).toHaveTextContent('user')
  })

  it('falls back to "user" when JWT payload has no sub claim', () => {
    localStorage.setItem('token', makeJwt({ name: 'alice' }))
    render(<App />)
    expect(screen.getByTestId('dashboard')).toHaveTextContent('user')
  })
})
