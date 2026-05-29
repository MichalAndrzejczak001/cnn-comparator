import { TEST_JWT, MOCK_EXPERIMENT, MOCK_EXPERIMENT_2 } from '../support/helpers'

beforeEach(() => {
  cy.intercept('GET', '**/experiments', { body: [] }).as('getExperiments')
  cy.intercept('GET', '**/experiments/**', { body: MOCK_EXPERIMENT }).as('getExperiment')
  cy.visit('/', {
    onBeforeLoad(win) {
      win.localStorage.setItem('token', TEST_JWT)
    },
  })
})

// --- Navigation ---

describe('Dashboard navigation', () => {
  it('shows all nav items', () => {
    cy.contains('button', 'Przegląd').should('be.visible')
    cy.contains('button', 'Nowy eksperyment').should('be.visible')
    cy.contains('button', 'Historia').should('be.visible')
    cy.contains('button', 'Porównaj wybrane').should('be.visible')
    cy.contains('button', 'Porównaj wszystkie').should('be.visible')
    cy.contains('button', 'O modelach').should('be.visible')
    cy.contains('button', 'O zbiorach danych').should('be.visible')
  })

  it('"Porównaj wybrane" is disabled by default', () => {
    cy.contains('button', 'Porównaj wybrane').should('be.disabled')
  })

  it('clicking Nowy eksperyment shows the experiment form', () => {
    cy.contains('button', 'Nowy eksperyment').click()
    cy.get('h1,h2,h3,h4,h5,h6').contains('Nowy eksperyment').should('be.visible')
    cy.contains('button', 'Uruchom trening').should('be.visible')
  })

  it('clicking Historia shows the history view', () => {
    cy.contains('button', 'Historia').click()
    cy.contains('Historia eksperymentów').should('be.visible')
  })

  it('clicking O modelach shows models info', () => {
    cy.contains('button', 'O modelach').click()
    cy.get('.view').should('be.visible')
  })
})

// --- Logout ---

describe('Logout', () => {
  it('clicking Wyloguj returns to landing page', () => {
    cy.contains('button', 'Wyloguj').click()
    cy.get('.landing').should('be.visible')
    cy.contains('button', 'Zaloguj się').should('be.visible')
  })

  it('after logout localStorage token is cleared', () => {
    cy.contains('button', 'Wyloguj').click()
    cy.window().then((win) => {
      expect(win.localStorage.getItem('token')).to.be.null
    })
  })
})

// --- History view ---

describe('History view', () => {
  beforeEach(() => {
    cy.contains('button', 'Historia').click()
  })

  it('shows empty state when no experiments', () => {
    cy.contains(/Brak eksperymentów/).should('be.visible')
  })

  it('shows experiment rows when data is loaded', () => {
    cy.intercept('GET', '**/experiments', {
      body: [MOCK_EXPERIMENT, MOCK_EXPERIMENT_2],
    }).as('getExperimentsWithData')
    cy.visit('/', {
      onBeforeLoad(win) {
        win.localStorage.setItem('token', TEST_JWT)
      },
    })
    cy.contains('button', 'Historia').click()
    cy.contains('LeNet-5').should('be.visible')
    cy.contains('ResNet-18').should('be.visible')
  })

  it('shows correct experiment count in header', () => {
    cy.contains('0 eksperymentów').should('be.visible')
  })

  it('Odśwież button is visible', () => {
    cy.contains('button', 'Odśwież').should('be.visible')
  })
})

// --- New Experiment view ---

describe('New Experiment view', () => {
  beforeEach(() => {
    cy.contains('button', 'Nowy eksperyment').click()
  })

  it('shows model select with all 4 options', () => {
    cy.get('select').first().within(() => {
      cy.contains('option', 'Simple CNN').should('exist')
      cy.contains('option', 'LeNet-5').should('exist')
      cy.contains('option', 'VGG-11').should('exist')
      cy.contains('option', 'ResNet-18').should('exist')
    })
  })

  it('shows dataset select with mnist and cifar10', () => {
    cy.get('select').eq(1).within(() => {
      cy.contains('option', 'MNIST').should('exist')
      cy.contains('option', 'CIFAR10').should('exist')
    })
  })

  it('successful experiment run displays results', () => {
    cy.intercept('POST', '**/experiments', { body: MOCK_EXPERIMENT }).as('runExperiment')
    cy.contains('button', 'Uruchom trening').click()
    cy.contains('95.00%').should('be.visible')
    cy.contains('Dokładność testowa').should('be.visible')
  })

  it('shows loading state while training', () => {
    cy.intercept('POST', '**/experiments', {
      delay: 500,
      body: MOCK_EXPERIMENT,
    }).as('runExperimentDelayed')
    cy.contains('button', 'Uruchom trening').click()
    cy.contains('Trening w toku...').should('be.visible')
  })

  it('API error shows error message', () => {
    cy.intercept('POST', '**/experiments', {
      statusCode: 500,
      body: 'Internal Server Error',
    }).as('runExperimentError')
    cy.contains('button', 'Uruchom trening').click()
    cy.get('.form-error').should('be.visible')
  })
})
