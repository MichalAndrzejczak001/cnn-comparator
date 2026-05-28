package spring.project.thesis.logicbackend.experiment;

import tools.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.client.RestTemplate;
import spring.project.thesis.logicbackend.dto.CompareExperimentsRequest;
import spring.project.thesis.logicbackend.dto.NoteRequest;
import spring.project.thesis.logicbackend.dto.TrainingConfig;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.enums.Role;
import spring.project.thesis.logicbackend.security.JwtService;
import spring.project.thesis.logicbackend.user.User;
import spring.project.thesis.logicbackend.user.UserRepository;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ExperimentControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private ExperimentRepository experimentRepository;

    @Autowired
    private JwtService jwtService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private RestTemplate restTemplate;

    private MockRestServiceServer mockAiServer;
    private String aliceToken;
    private String bobToken;
    private User alice;
    private User bob;

    private static final String AI_EXPERIMENT_RESPONSE = """
            {
              "status": "done",
              "train_loss_per_epoch": [0.5, 0.3],
              "test_loss_per_epoch": [0.45, 0.28],
              "test_loss": 0.28,
              "test_accuracy": 0.92,
              "confusion_matrix": [[9, 1], [0, 10]],
              "training_time_seconds": 12.5
            }
            """;

    @BeforeEach
    void setUp() {
        experimentRepository.deleteAll();
        userRepository.deleteAll();

        alice = userRepository.save(User.builder()
                .username("alice")
                .password(passwordEncoder.encode("pass"))
                .role(Role.USER)
                .build());

        bob = userRepository.save(User.builder()
                .username("bob")
                .password(passwordEncoder.encode("pass"))
                .role(Role.USER)
                .build());

        aliceToken = jwtService.generateToken("alice");
        bobToken = jwtService.generateToken("bob");

        mockAiServer = MockRestServiceServer.createServer(restTemplate);
    }

    // --- GET /experiments ---

    @Test
    void getHistory_returns200WithEmptyList() throws Exception {
        mockMvc.perform(get("/experiments")
                        .header("Authorization", "Bearer " + aliceToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(0)));
    }

    @Test
    void getHistory_returnsOnlyCurrentUsersExperiments() throws Exception {
        saveExperiment(alice, "resnet18", "mnist");
        saveExperiment(alice, "lenet5", "cifar10");
        saveExperiment(bob, "vgg11", "mnist");

        mockMvc.perform(get("/experiments")
                        .header("Authorization", "Bearer " + aliceToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)));
    }

    @Test
    void getHistory_returns4xxWithoutToken() throws Exception {
        mockMvc.perform(get("/experiments"))
                .andExpect(status().is4xxClientError());
    }

    // --- POST /experiments ---

    @Test
    void runExperiment_returns200AndSavesResult() throws Exception {
        mockAiServer.expect(requestTo("http://ai-backend-test/experiments"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess(AI_EXPERIMENT_RESPONSE, MediaType.APPLICATION_JSON));

        ExperimentRequest request = buildExperimentRequest("resnet18", "mnist");

        mockMvc.perform(post("/experiments")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("resnet18"))
                .andExpect(jsonPath("$.dataset").value("mnist"))
                .andExpect(jsonPath("$.test_accuracy").value(0.92));

        assertThat(experimentRepository.findByUserOrderByCreatedAtDesc(alice)).hasSize(1);

        mockAiServer.verify();
    }

    @Test
    void runExperiment_returns4xxWithoutToken() throws Exception {
        ExperimentRequest request = buildExperimentRequest("resnet18", "mnist");

        mockMvc.perform(post("/experiments")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().is4xxClientError());
    }

    // --- POST /experiments/compare ---

    @Test
    void compareExperiments_returns200ForOwnExperiments() throws Exception {
        Experiment e1 = saveExperiment(alice, "resnet18", "mnist");
        Experiment e2 = saveExperiment(alice, "lenet5", "mnist");

        CompareExperimentsRequest request = new CompareExperimentsRequest();
        request.setIds(List.of(e1.getId(), e2.getId()));

        mockMvc.perform(post("/experiments/compare")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)));
    }

    @Test
    void compareExperiments_returns403ForOtherUsersExperiment() throws Exception {
        Experiment bobsExperiment = saveExperiment(bob, "vgg11", "cifar10");

        CompareExperimentsRequest request = new CompareExperimentsRequest();
        request.setIds(List.of(bobsExperiment.getId()));

        mockMvc.perform(post("/experiments/compare")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isForbidden());
    }

    @Test
    void compareExperiments_returns404ForNonExistentId() throws Exception {
        CompareExperimentsRequest request = new CompareExperimentsRequest();
        request.setIds(List.of(99999L));

        mockMvc.perform(post("/experiments/compare")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isNotFound());
    }

    // --- POST /experiments/{id}/rerun ---

    @Test
    void rerunExperiment_returns200AndCreatesNewEntry() throws Exception {
        Experiment original = saveExperiment(alice, "lenet5", "mnist");

        mockAiServer.expect(requestTo("http://ai-backend-test/experiments"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess(AI_EXPERIMENT_RESPONSE, MediaType.APPLICATION_JSON));

        mockMvc.perform(post("/experiments/" + original.getId() + "/rerun")
                        .header("Authorization", "Bearer " + aliceToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("lenet5"));

        assertThat(experimentRepository.findByUserOrderByCreatedAtDesc(alice)).hasSize(2);

        mockAiServer.verify();
    }

    @Test
    void rerunExperiment_returns403ForOtherUsersExperiment() throws Exception {
        Experiment bobsExperiment = saveExperiment(bob, "vgg11", "cifar10");

        mockMvc.perform(post("/experiments/" + bobsExperiment.getId() + "/rerun")
                        .header("Authorization", "Bearer " + aliceToken))
                .andExpect(status().isForbidden());
    }

    @Test
    void rerunExperiment_returns404ForNonExistentId() throws Exception {
        mockMvc.perform(post("/experiments/99999/rerun")
                        .header("Authorization", "Bearer " + aliceToken))
                .andExpect(status().isNotFound());
    }

    // --- PATCH /experiments/{id}/note ---

    @Test
    void updateNote_returns200AndPersistsNote() throws Exception {
        Experiment e = saveExperiment(alice, "resnet18", "mnist");

        NoteRequest noteRequest = new NoteRequest();
        noteRequest.setNote("very good result");

        mockMvc.perform(patch("/experiments/" + e.getId() + "/note")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(noteRequest)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.note").value("very good result"));

        assertThat(experimentRepository.findById(e.getId()).get().getNote())
                .isEqualTo("very good result");
    }

    @Test
    void updateNote_returns403ForOtherUsersExperiment() throws Exception {
        Experiment bobsExperiment = saveExperiment(bob, "vgg11", "cifar10");

        NoteRequest noteRequest = new NoteRequest();
        noteRequest.setNote("trying to modify");

        mockMvc.perform(patch("/experiments/" + bobsExperiment.getId() + "/note")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(noteRequest)))
                .andExpect(status().isForbidden());
    }

    @Test
    void updateNote_returns404ForNonExistentId() throws Exception {
        NoteRequest noteRequest = new NoteRequest();
        noteRequest.setNote("note");

        mockMvc.perform(patch("/experiments/99999/note")
                        .header("Authorization", "Bearer " + aliceToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(noteRequest)))
                .andExpect(status().isNotFound());
    }

    // --- helpers ---

    private Experiment saveExperiment(User user, String model, String dataset) {
        return experimentRepository.save(Experiment.builder()
                .user(user)
                .model(model)
                .dataset(dataset)
                .epochs(5)
                .batchSize(32)
                .learningRate(0.001)
                .trainLossPerEpoch(List.of(0.5, 0.3))
                .testLossPerEpoch(List.of(0.45, 0.28))
                .testLoss(0.28)
                .testAccuracy(0.92)
                .trainingTimeSeconds(12.5)
                .createdAt(LocalDateTime.now())
                .build());
    }

    private ExperimentRequest buildExperimentRequest(String model, String dataset) {
        ExperimentRequest request = new ExperimentRequest();
        request.setModel(model);
        request.setDataset(dataset);
        TrainingConfig config = new TrainingConfig();
        config.setEpochs(5);
        config.setBatchSize(32);
        config.setLearningRate(0.001);
        request.setTraining(config);
        return request;
    }
}
