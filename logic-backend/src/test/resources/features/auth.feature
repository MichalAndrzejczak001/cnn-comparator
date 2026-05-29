Feature: Rejestracja i logowanie

  Scenario: Pomyślna rejestracja nowego użytkownika
    Given baza danych jest pusta
    When rejestruję użytkownika "alice" z hasłem "haslo123"
    Then status odpowiedzi wynosi 200
    And odpowiedź zawiera niepusty token JWT

  Scenario: Rejestracja zajętego loginu zwraca błąd konfliktu
    Given istnieje użytkownik "alice" z hasłem "haslo123"
    When rejestruję użytkownika "alice" z hasłem "haslo123"
    Then status odpowiedzi wynosi 409

  Scenario: Pomyślne logowanie zwraca token JWT
    Given istnieje użytkownik "alice" z hasłem "haslo123"
    When loguję się jako "alice" z hasłem "haslo123"
    Then status odpowiedzi wynosi 200
    And odpowiedź zawiera niepusty token JWT

  Scenario: Logowanie z błędnym hasłem zwraca 401
    Given istnieje użytkownik "alice" z hasłem "haslo123"
    When loguję się jako "alice" z hasłem "złe_hasło"
    Then status odpowiedzi wynosi 401

  Scenario: Logowanie nieistniejącego użytkownika zwraca 401
    Given baza danych jest pusta
    When loguję się jako "nobody" z hasłem "pass"
    Then status odpowiedzi wynosi 401

  Scenario: Dostęp do chronionego zasobu bez tokenu zwraca 401
    When pobieram listę eksperymentów bez uwierzytelnienia
    Then status odpowiedzi wynosi 401
