package spring.project.thesis.logicbackend.bdd.steps;

import io.cucumber.java.Before;
import io.cucumber.java.en.And;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;
import io.restassured.RestAssured;
import io.restassured.http.ContentType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import spring.project.thesis.logicbackend.bdd.ScenarioContext;
import spring.project.thesis.logicbackend.enums.Role;
import spring.project.thesis.logicbackend.experiment.Experiment;
import spring.project.thesis.logicbackend.experiment.ExperimentRepository;
import spring.project.thesis.logicbackend.security.JwtService;
import spring.project.thesis.logicbackend.user.User;
import spring.project.thesis.logicbackend.user.UserRepository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;

public class ExperimentSteps {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private ExperimentRepository experimentRepository;

    @Autowired
    private JwtService jwtService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private ScenarioContext ctx;

    private final Map<String, User> users = new HashMap<>();
    private final Map<String, List<Experiment>> experiments = new HashMap<>();

    @Before
    public void cleanState() {
        experimentRepository.deleteAll();
        userRepository.deleteAll();
        users.clear();
        experiments.clear();
    }

    @Given("zalogowany użytkownik {string}")
    public void loggedInUser(String username) {
        User user = userRepository.save(User.builder()
                .username(username)
                .password(passwordEncoder.encode("pass"))
                .role(Role.USER)
                .build());
        users.put(username, user);
    }

    @And("{string} ma {int} eksperymenty w bazie")
    public void userHasTwoExperiments(String username, int count) {
        saveExperimentsForUser(username, count);
    }

    @And("{string} ma {int} eksperyment w bazie")
    public void userHasOneExperiment(String username, int count) {
        saveExperimentsForUser(username, count);
    }

    private void saveExperimentsForUser(String username, int count) {
        User user = users.get(username);
        List<Experiment> saved = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            saved.add(experimentRepository.save(Experiment.builder()
                    .user(user)
                    .model("lenet5")
                    .dataset("mnist")
                    .epochs(5)
                    .batchSize(32)
                    .learningRate(0.001)
                    .trainLossPerEpoch(List.of(0.5, 0.3))
                    .testLossPerEpoch(List.of(0.45, 0.28))
                    .testLoss(0.28)
                    .testAccuracy(0.92)
                    .trainingTimeSeconds(12.5)
                    .createdAt(LocalDateTime.now())
                    .build()));
        }
        experiments.put(username, saved);
    }

    @When("pobieram listę eksperymentów jako {string}")
    public void getExperimentsWithToken(String username) {
        ctx.response = RestAssured
                .given()
                .header("Authorization", "Bearer " + jwtService.generateToken(username))
                .when()
                .get("/experiments")
                .then();
    }

    @When("pobieram listę eksperymentów bez uwierzytelnienia")
    public void getExperimentsWithoutToken() {
        ctx.response = RestAssured
                .given()
                .when()
                .get("/experiments")
                .then();
    }

    @When("porównuję eksperymenty {string} z jej tokenem")
    public void compareOwnExperiments(String username) {
        List<Long> ids = experiments.get(username).stream().map(Experiment::getId).toList();
        ctx.response = RestAssured
                .given()
                .header("Authorization", "Bearer " + jwtService.generateToken(username))
                .contentType(ContentType.JSON)
                .body(Map.of("ids", ids))
                .when()
                .post("/experiments/compare")
                .then();
    }

    @When("porównuję eksperyment {string} jako {string}")
    public void compareOtherUsersExperiment(String owner, String requester) {
        List<Long> ids = experiments.get(owner).stream().map(Experiment::getId).toList();
        ctx.response = RestAssured
                .given()
                .header("Authorization", "Bearer " + jwtService.generateToken(requester))
                .contentType(ContentType.JSON)
                .body(Map.of("ids", ids))
                .when()
                .post("/experiments/compare")
                .then();
    }

    @When("aktualizuję notatkę własnego eksperymentu na {string} z tokenem {string}")
    public void updateOwnNote(String note, String username) {
        Long id = experiments.get(username).get(0).getId();
        ctx.response = RestAssured
                .given()
                .header("Authorization", "Bearer " + jwtService.generateToken(username))
                .contentType(ContentType.JSON)
                .body(Map.of("note", note))
                .when()
                .patch("/experiments/" + id + "/note")
                .then();
    }

    @When("aktualizuję notatkę eksperymentu {string} na {string} z tokenem {string}")
    public void updateOtherUsersNote(String owner, String note, String requester) {
        Long id = experiments.get(owner).get(0).getId();
        ctx.response = RestAssured
                .given()
                .header("Authorization", "Bearer " + jwtService.generateToken(requester))
                .contentType(ContentType.JSON)
                .body(Map.of("note", note))
                .when()
                .patch("/experiments/" + id + "/note")
                .then();
    }

    @Then("odpowiedź zawiera pustą listę")
    public void responseIsEmptyList() {
        ctx.response.body(hasSize(0));
    }

    @Then("odpowiedź zawiera {int} elementy")
    public void responseHasSize(int size) {
        ctx.response.body(hasSize(size));
    }

    @Then("odpowiedź zawiera pole {string} o wartości {string}")
    public void responseHasField(String field, String value) {
        ctx.response.body(field, equalTo(value));
    }
}
