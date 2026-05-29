Feature: Dashboard użytkownika

  Background:
    Given użytkownik jest zalogowany

  Scenario: Nawigacja zawiera wszystkie pozycje menu
    Then widzi przycisk "Przegląd"
    And widzi przycisk "Nowy eksperyment"
    And widzi przycisk "Historia"
    And przycisk "Porównaj wybrane" jest nieaktywny

  Scenario: Wylogowanie wraca do strony głównej
    When klika przycisk "Wyloguj"
    Then widzi stronę główną
    And token zostaje usunięty z localStorage

  Scenario: Pusta historia eksperymentów
    When przechodzi do widoku "Historia"
    Then widzi komunikat o braku eksperymentów
    And widzi liczbę eksperymentów "0"

  Scenario: Lista modeli w formularzu nowego eksperymentu
    When przechodzi do widoku "Nowy eksperyment"
    Then formularz zawiera opcję modelu "LeNet-5"
    And formularz zawiera opcję modelu "ResNet-18"
    And formularz zawiera opcję datasetu "MNIST"
    And formularz zawiera opcję datasetu "CIFAR10"

  Scenario: Pomyślne uruchomienie treningu wyświetla wyniki
    Given serwer treningu zwraca wynik z dokładnością 0.95
    When przechodzi do widoku "Nowy eksperyment"
    And klika przycisk "Uruchom trening"
    Then widzi dokładność testową "95.00%"
    And widzi etykietę "Dokładność testowa"

  Scenario: Błąd serwera podczas treningu
    Given serwer treningu zwraca błąd 500
    When przechodzi do widoku "Nowy eksperyment"
    And klika przycisk "Uruchom trening"
    Then widzi komunikat o błędzie
