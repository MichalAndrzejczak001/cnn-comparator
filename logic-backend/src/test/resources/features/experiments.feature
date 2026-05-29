Feature: Zarządzanie eksperymentami

  Scenario: Nowy użytkownik ma pustą listę eksperymentów
    Given zalogowany użytkownik "alice"
    When pobieram listę eksperymentów jako "alice"
    Then status odpowiedzi wynosi 200
    And odpowiedź zawiera pustą listę

  Scenario: Użytkownik widzi tylko swoje eksperymenty
    Given zalogowany użytkownik "alice"
    And zalogowany użytkownik "bob"
    And "alice" ma 2 eksperymenty w bazie
    And "bob" ma 1 eksperyment w bazie
    When pobieram listę eksperymentów jako "alice"
    Then status odpowiedzi wynosi 200
    And odpowiedź zawiera 2 elementy

  Scenario: Dostęp do listy eksperymentów bez tokenu zwraca 401
    When pobieram listę eksperymentów bez uwierzytelnienia
    Then status odpowiedzi wynosi 401

  Scenario: Porównanie własnych eksperymentów
    Given zalogowany użytkownik "alice"
    And "alice" ma 2 eksperymenty w bazie
    When porównuję eksperymenty "alice" z jej tokenem
    Then status odpowiedzi wynosi 200
    And odpowiedź zawiera 2 elementy

  Scenario: Próba porównania cudzego eksperymentu zwraca 403
    Given zalogowany użytkownik "alice"
    And zalogowany użytkownik "bob"
    And "bob" ma 1 eksperyment w bazie
    When porównuję eksperyment "bob" jako "alice"
    Then status odpowiedzi wynosi 403

  Scenario: Aktualizacja notatki do własnego eksperymentu
    Given zalogowany użytkownik "alice"
    And "alice" ma 1 eksperyment w bazie
    When aktualizuję notatkę własnego eksperymentu na "świetny wynik" z tokenem "alice"
    Then status odpowiedzi wynosi 200
    And odpowiedź zawiera pole "note" o wartości "świetny wynik"

  Scenario: Aktualizacja notatki do cudzego eksperymentu zwraca 403
    Given zalogowany użytkownik "alice"
    And zalogowany użytkownik "bob"
    And "bob" ma 1 eksperyment w bazie
    When aktualizuję notatkę eksperymentu "bob" na "próba" z tokenem "alice"
    Then status odpowiedzi wynosi 403
