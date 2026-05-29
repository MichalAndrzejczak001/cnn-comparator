import { Given, When, Then } from '@badeball/cypress-cucumber-preprocessor'
import { TEST_JWT, MOCK_EXPERIMENT } from '../../../support/helpers'

Given('użytkownik otwiera stronę główną', () => {
  cy.visit('/')
})

When('klika przycisk nawigacji {string}', (name: string) => {
  cy.contains('button', name).first().click()
})

When('wpisuje login {string} i hasło {string}', (login: string, password: string) => {
  cy.get('[placeholder="np. jan_kowalski"]').type(login)
  cy.get('[placeholder="••••••••"]').type(password)
})

When('zatwierdza formularz logowania', () => {
  cy.get('.modal').contains('button', 'Zaloguj się').click()
})

When('zatwierdza formularz rejestracji', () => {
  cy.get('.modal').contains('button', 'Utwórz konto').click()
})

When('klika przycisk zamknięcia modala', () => {
  cy.get('[aria-label="Zamknij"]').click()
})

Given('serwer logowania zwraca token JWT', () => {
  cy.intercept('POST', '**/auth/login', {
    statusCode: 200,
    body: TEST_JWT,
    headers: { 'content-type': 'text/plain' },
  }).as('login')
  cy.intercept('GET', '**/experiments', { body: [] }).as('getExperiments')
})

Given('serwer logowania zwraca błąd 401', () => {
  cy.intercept('POST', '**/auth/login', {
    statusCode: 401,
    body: 'Nieprawidłowy login lub hasło',
  }).as('loginFail')
})

Given('serwer logowania jest niedostępny', () => {
  cy.intercept('POST', '**/auth/login', { forceNetworkError: true }).as('loginNetworkError')
})

Given('serwer rejestracji zwraca token JWT', () => {
  cy.intercept('POST', '**/auth/register', {
    statusCode: 200,
    body: TEST_JWT,
    headers: { 'content-type': 'text/plain' },
  }).as('register')
})

Given('serwer rejestracji zwraca błąd 409', () => {
  cy.intercept('POST', '**/auth/register', {
    statusCode: 409,
    body: 'Użytkownik już istnieje',
  }).as('registerFail')
})

Then('widzi napis {string} w nagłówku', (text: string) => {
  cy.get('.nav-brand').should('contain.text', text)
})

Then('widzi modal z tytułem {string}', (title: string) => {
  cy.get('.modal-title').should('contain.text', title)
})

Then('modal jest zamknięty', () => {
  cy.get('.modal-backdrop').should('not.exist')
})

Then('widzi dashboard z nazwą użytkownika {string}', (username: string) => {
  cy.get('.dash').should('be.visible')
  cy.get('.dash-username').should('contain.text', username)
})

Then('komunikat o błędzie zawiera {string}', (text: string) => {
  cy.get('.form-error').should('contain.text', text)
})

Then('pole hasła jest wyczyszczone', () => {
  cy.get('[placeholder="••••••••"]').should('have.value', '')
})
