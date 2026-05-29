import { TEST_JWT } from '../support/helpers'

// --- Landing page ---

describe('Landing page', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('displays app branding', () => {
    cy.get('.nav-brand').should('contain.text', 'CNN Comparator')
  })

  it('shows Zaloguj się and Zarejestruj się buttons', () => {
    cy.contains('button', 'Zaloguj się').should('be.visible')
    cy.contains('button', 'Zarejestruj się').should('be.visible')
  })

  it('shows hero Zacznij teraz button', () => {
    cy.contains('button', 'Zacznij teraz').should('be.visible')
  })

  it('no modal visible on initial load', () => {
    cy.get('.modal-backdrop').should('not.exist')
  })
})

// --- Auth modal opening ---

describe('Auth modal', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('clicking Zaloguj się opens modal in login mode', () => {
    cy.contains('button', 'Zaloguj się').click()
    cy.get('.modal').should('be.visible')
    cy.get('.modal-title').should('contain.text', 'Zaloguj się')
  })

  it('clicking Zarejestruj się opens modal in register mode', () => {
    cy.contains('button', 'Zarejestruj się').first().click()
    cy.get('.modal').should('be.visible')
    cy.get('.modal-title').should('contain.text', 'Utwórz konto')
  })

  it('modal shows username and password inputs', () => {
    cy.contains('button', 'Zaloguj się').click()
    cy.get('[placeholder="np. jan_kowalski"]').should('be.visible')
    cy.get('[placeholder="••••••••"]').should('be.visible')
  })

  it('clicking backdrop closes the modal', () => {
    cy.contains('button', 'Zaloguj się').click()
    cy.get('.modal').should('be.visible')
    cy.get('.modal-backdrop').click(5, 5)
    cy.get('.modal-backdrop').should('not.exist')
  })

  it('✕ button closes the modal', () => {
    cy.contains('button', 'Zaloguj się').click()
    cy.get('[aria-label="Zamknij"]').click()
    cy.get('.modal-backdrop').should('not.exist')
  })

  it('switching to Rejestracja tab changes title', () => {
    cy.contains('button', 'Zaloguj się').click()
    cy.contains('button', 'Rejestracja').click()
    cy.get('.modal-title').should('contain.text', 'Utwórz konto')
  })

  it('switching back to Logowanie tab changes title back', () => {
    cy.contains('button', 'Zarejestruj się').first().click()
    cy.contains('button', 'Logowanie').click()
    cy.get('.modal-title').should('contain.text', 'Zaloguj się')
  })
})

// --- Login flow ---

describe('Login flow', () => {
  beforeEach(() => {
    cy.intercept('POST', '**/auth/login', {
      statusCode: 200,
      body: TEST_JWT,
      headers: { 'content-type': 'text/plain' },
    }).as('login')
    cy.intercept('GET', '**/experiments', { body: [] }).as('getExperiments')
    cy.visit('/')
    cy.contains('button', 'Zaloguj się').click()
  })

  it('successful login navigates to dashboard', () => {
    cy.get('[placeholder="np. jan_kowalski"]').type('testuser')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Zaloguj się').click()
    cy.get('.dash').should('be.visible')
  })

  it('dashboard displays decoded username after login', () => {
    cy.get('[placeholder="np. jan_kowalski"]').type('testuser')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Zaloguj się').click()
    cy.get('.dash-username').should('contain.text', 'testuser')
  })

  it('failed login shows error message', () => {
    cy.intercept('POST', '**/auth/login', {
      statusCode: 401,
      body: 'Nieprawidłowy login lub hasło',
    }).as('loginFail')
    cy.get('[placeholder="np. jan_kowalski"]').type('testuser')
    cy.get('[placeholder="••••••••"]').type('wrongpassword')
    cy.get('.modal').contains('button', 'Zaloguj się').click()
    cy.get('.form-error').should('be.visible')
  })

  it('submit button shows Łączenie... while loading', () => {
    cy.intercept('POST', '**/auth/login', {
      delay: 500,
      statusCode: 200,
      body: TEST_JWT,
      headers: { 'content-type': 'text/plain' },
    }).as('loginDelayed')
    cy.get('[placeholder="np. jan_kowalski"]').type('testuser')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Zaloguj się').click()
    cy.contains('Łączenie...').should('be.visible')
  })

  it('network failure shows connection error', () => {
    cy.intercept('POST', '**/auth/login', { forceNetworkError: true }).as('loginNetworkError')
    cy.get('[placeholder="np. jan_kowalski"]').type('testuser')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Zaloguj się').click()
    cy.get('.form-error').should('contain.text', 'Brak połączenia')
  })
})

// --- Register flow ---

describe('Register flow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.contains('button', 'Zarejestruj się').first().click()
  })

  it('successful register clears the password field', () => {
    cy.intercept('POST', '**/auth/register', {
      statusCode: 200,
      body: TEST_JWT,
      headers: { 'content-type': 'text/plain' },
    }).as('register')
    cy.get('[placeholder="np. jan_kowalski"]').type('newuser')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Utwórz konto').click()
    cy.get('[placeholder="••••••••"]').should('have.value', '')
  })

  it('after register, modal switches to login tab', () => {
    cy.intercept('POST', '**/auth/register', {
      statusCode: 200,
      body: TEST_JWT,
      headers: { 'content-type': 'text/plain' },
    }).as('register')
    cy.get('[placeholder="np. jan_kowalski"]').type('newuser')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Utwórz konto').click()
    cy.get('.modal-title').should('contain.text', 'Zaloguj się')
  })

  it('failed register shows error', () => {
    cy.intercept('POST', '**/auth/register', {
      statusCode: 409,
      body: 'Użytkownik już istnieje',
    }).as('registerFail')
    cy.get('[placeholder="np. jan_kowalski"]').type('existing')
    cy.get('[placeholder="••••••••"]').type('password123')
    cy.get('.modal').contains('button', 'Utwórz konto').click()
    cy.get('.form-error').should('be.visible')
  })
})
