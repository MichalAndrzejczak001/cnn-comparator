import { Then } from '@badeball/cypress-cucumber-preprocessor'

Then('widzi komunikat o błędzie', () => {
  cy.get('.form-error').should('be.visible')
})

Then('widzi przycisk {string}', (name: string) => {
  cy.contains('button', name).should('be.visible')
})
