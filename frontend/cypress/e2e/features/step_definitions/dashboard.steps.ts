import { Given, When, Then } from '@badeball/cypress-cucumber-preprocessor'
import { TEST_JWT, MOCK_EXPERIMENT } from '../../../support/helpers'

Given('użytkownik jest zalogowany', () => {
  cy.intercept('GET', '**/experiments', { body: [] }).as('getExperiments')
  cy.visit('/', {
    onBeforeLoad(win) {
      win.localStorage.setItem('token', TEST_JWT)
    },
  })
})

Given('serwer treningu zwraca wynik z dokładnością 0.95', () => {
  cy.intercept('POST', '**/experiments', { body: MOCK_EXPERIMENT }).as('runExperiment')
})

Given('serwer treningu zwraca błąd 500', () => {
  cy.intercept('POST', '**/experiments', {
    statusCode: 500,
    body: 'Internal Server Error',
  }).as('runExperimentError')
})

When('klika przycisk {string}', (name: string) => {
  cy.contains('button', name).click()
})

When('przechodzi do widoku {string}', (view: string) => {
  cy.contains('button', view).click()
})

Then('widzi stronę główną', () => {
  cy.get('.landing').should('be.visible')
})

Then('token zostaje usunięty z localStorage', () => {
  cy.window().then((win) => {
    expect(win.localStorage.getItem('token')).to.be.null
  })
})

Then('widzi komunikat o braku eksperymentów', () => {
  cy.contains(/Brak eksperymentów/).should('be.visible')
})

Then('widzi liczbę eksperymentów {string}', (count: string) => {
  cy.contains(`${count} eksperymentów`).should('be.visible')
})

Then('formularz zawiera opcję modelu {string}', (model: string) => {
  cy.get('select').first().contains('option', model).should('exist')
})

Then('formularz zawiera opcję datasetu {string}', (dataset: string) => {
  cy.get('select').eq(1).contains('option', dataset).should('exist')
})

Then('widzi dokładność testową {string}', (accuracy: string) => {
  cy.contains(accuracy).should('be.visible')
})

Then('widzi etykietę {string}', (label: string) => {
  cy.contains(label).should('be.visible')
})

Then('przycisk {string} jest nieaktywny', (name: string) => {
  cy.contains('button', name).should('be.disabled')
})
