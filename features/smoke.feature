Feature: Smoke E2E for Handwritten Digit Recognition (Streamlit)

Scenario: TC-SM-01 Upload valid image shows preview
    Given the app is running
    When I open the app page
    And I upload file "features/tests_data/valid/digit_7.png"
    Then I should see the uploaded image preview

Scenario: TC-SM-02 Recognition shows digit, confidence and top-3
    Given the app is running
    When I open the app page
    And I upload file "features/tests_data/valid/digit_7.png"
    Then I should see prediction block
    And I should see confidence value
    And I should see top-3 predictions

Scenario: TC-SM-03 Upload too large image shows error
    Given the app is running
    When I open the app page
    And I upload file "features/tests_data/invalid/photo_2.jpg"
    Then I should see an error message

Scenario Outline: TC-SM-04 Upload valid image shows result (parameterized)
    Given the app is running
    When I open the app page
    And I upload file "<file>"
    Then I should see prediction block
    And I should see confidence value
    And I should see top-3 predictions