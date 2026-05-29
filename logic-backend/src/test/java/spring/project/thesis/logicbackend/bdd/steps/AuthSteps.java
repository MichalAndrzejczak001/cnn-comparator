package spring.project.thesis.logicbackend.bdd.steps;

import io.cucumber.java.Before;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;
import io.restassured.RestAssured;
import io.restassured.http.ContentType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import spring.project.thesis.logicbackend.bdd.ScenarioContext;
import spring.project.thesis.logicbackend.enums.Role;
import spring.project.thesis.logicbackend.experiment.ExperimentRepository;
import spring.project.thesis.logicbackend.user.User;
import spring.project.thesis.logicbackend.user.UserRepository;

import java.util.Map;

import static org.hamcrest.Matchers.emptyOrNullString;
import static org.hamcrest.Matchers.not;

public class AuthSteps {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private ExperimentRepository experimentRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private ScenarioContext ctx;

    @Before
    public void cleanDatabase() {
        experimentRepository.deleteAll();
        userRepository.deleteAll();
    }

    @Given("baza danych jest pusta")
    public void databaseIsEmpty() {
        experimentRepository.deleteAll();
        userRepository.deleteAll();
    }

    @Given("istnieje użytkownik {string} z hasłem {string}")
    public void userExists(String username, String password) {
        userRepository.save(User.builder()
                .username(username)
                .password(passwordEncoder.encode(password))
                .role(Role.USER)
                .build());
    }

    @When("rejestruję użytkownika {string} z hasłem {string}")
    public void postRegister(String username, String password) {
        ctx.response = RestAssured
                .given()
                .contentType(ContentType.JSON)
                .body(Map.of("username", username, "password", password))
                .when()
                .post("/auth/register")
                .then();
    }

    @When("loguję się jako {string} z hasłem {string}")
    public void postLogin(String username, String password) {
        ctx.response = RestAssured
                .given()
                .contentType(ContentType.JSON)
                .body(Map.of("username", username, "password", password))
                .when()
                .post("/auth/login")
                .then();
    }

    @Then("status odpowiedzi wynosi {int}")
    public void statusIs(int status) {
        ctx.response.statusCode(status);
    }

    @Then("odpowiedź zawiera niepusty token JWT")
    public void responseContainsToken() {
        ctx.response.body(not(emptyOrNullString()));
    }
}
