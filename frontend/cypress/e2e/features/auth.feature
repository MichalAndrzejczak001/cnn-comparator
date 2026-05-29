Feature: Uwierzytelnianie użytkownika

  Background:
    Given użytkownik otwiera stronę główną

  Scenario: Strona główna wyświetla branding aplikacji
    Then widzi napis "CNN Comparator" w nagłówku
    And widzi przycisk "Zaloguj się"
    And widzi przycisk "Zarejestruj się"

  Scenario: Otwarcie modala logowania
    When klika przycisk nawigacji "Zaloguj się"
    Then widzi modal z tytułem "Zaloguj się"

  Scenario: Otwarcie modala rejestracji
    When klika przycisk nawigacji "Zarejestruj się"
    Then widzi modal z tytułem "Utwórz konto"

  Scenario: Zamknięcie modala przyciskiem X
    When klika przycisk nawigacji "Zaloguj się"
    And klika przycisk zamknięcia modala
    Then modal jest zamknięty

  Scenario: Pomyślne logowanie przekierowuje do dashboardu
    Given serwer logowania zwraca token JWT
    When klika przycisk nawigacji "Zaloguj się"
    And wpisuje login "testuser" i hasło "password123"
    And zatwierdza formularz logowania
    Then widzi dashboard z nazwą użytkownika "testuser"

  Scenario: Błędne hasło pokazuje komunikat o błędzie
    Given serwer logowania zwraca błąd 401
    When klika przycisk nawigacji "Zaloguj się"
    And wpisuje login "testuser" i hasło "złe_hasło"
    And zatwierdza formularz logowania
    Then widzi komunikat o błędzie

  Scenario: Brak sieci pokazuje komunikat o braku połączenia
    Given serwer logowania jest niedostępny
    When klika przycisk nawigacji "Zaloguj się"
    And wpisuje login "testuser" i hasło "password123"
    And zatwierdza formularz logowania
    Then komunikat o błędzie zawiera "Brak połączenia"

  Scenario: Pomyślna rejestracja przełącza na zakładkę logowania
    Given serwer rejestracji zwraca token JWT
    When klika przycisk nawigacji "Zarejestruj się"
    And wpisuje login "newuser" i hasło "password123"
    And zatwierdza formularz rejestracji
    Then pole hasła jest wyczyszczone
    And widzi modal z tytułem "Zaloguj się"

  Scenario: Rejestracja zajętego loginu pokazuje błąd
    Given serwer rejestracji zwraca błąd 409
    When klika przycisk nawigacji "Zarejestruj się"
    And wpisuje login "existing" i hasło "password123"
    And zatwierdza formularz rejestracji
    Then widzi komunikat o błędzie
