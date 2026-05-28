package spring.project.thesis.logicbackend.experiment;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.server.ResponseStatusException;
import spring.project.thesis.logicbackend.dto.CompareExperimentsRequest;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.dto.ExperimentResponse;
import spring.project.thesis.logicbackend.dto.ExperimentResult;
import spring.project.thesis.logicbackend.dto.NoteRequest;
import spring.project.thesis.logicbackend.dto.TrainingConfig;
import spring.project.thesis.logicbackend.user.User;
import spring.project.thesis.logicbackend.user.UserRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ExperimentServiceTest {

    @Mock
    private RestTemplate restTemplate;

    @Mock
    private ExperimentRepository experimentRepository;

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private ExperimentService experimentService;

    private User currentUser;
    private User otherUser;

    @BeforeEach
    void setUp() {
        ReflectionTestUtils.setField(experimentService, "aiBackendUrl", "http://ai-backend:8000");

        currentUser = User.builder().id(1L).username("alice").password("hashed").build();
        otherUser = User.builder().id(2L).username("bob").password("hashed").build();

        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken("alice", null, List.of())
        );

        lenient().when(userRepository.findByUsername("alice")).thenReturn(Optional.of(currentUser));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    // --- getHistory ---

    @Test
    void getHistory_returnsExperimentsForCurrentUser() {
        Experiment e1 = buildExperiment(1L, currentUser, "resnet18", "mnist");
        Experiment e2 = buildExperiment(2L, currentUser, "lenet5", "cifar10");

        when(experimentRepository.findByUserOrderByCreatedAtDesc(currentUser))
                .thenReturn(List.of(e1, e2));

        List<ExperimentResponse> result = experimentService.getHistory();

        assertThat(result).hasSize(2);
        assertThat(result.get(0).getModel()).isEqualTo("resnet18");
        assertThat(result.get(1).getModel()).isEqualTo("lenet5");
    }

    @Test
    void getHistory_returnsEmptyListWhenNoExperiments() {
        when(experimentRepository.findByUserOrderByCreatedAtDesc(currentUser))
                .thenReturn(List.of());

        assertThat(experimentService.getHistory()).isEmpty();
    }

    // --- compareExperiments ---

    @Test
    void compareExperiments_returnsResponsesForValidIds() {
        Experiment e1 = buildExperiment(1L, currentUser, "resnet18", "mnist");
        Experiment e2 = buildExperiment(2L, currentUser, "lenet5", "cifar10");

        when(experimentRepository.findById(1L)).thenReturn(Optional.of(e1));
        when(experimentRepository.findById(2L)).thenReturn(Optional.of(e2));

        CompareExperimentsRequest request = new CompareExperimentsRequest();
        request.setIds(List.of(1L, 2L));

        List<ExperimentResponse> result = experimentService.compareExperiments(request);

        assertThat(result).hasSize(2);
    }

    @Test
    void compareExperiments_throwsNotFoundForMissingId() {
        when(experimentRepository.findById(99L)).thenReturn(Optional.empty());

        CompareExperimentsRequest request = new CompareExperimentsRequest();
        request.setIds(List.of(99L));

        assertThatThrownBy(() -> experimentService.compareExperiments(request))
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(e -> assertThat(((ResponseStatusException) e).getStatusCode())
                        .isEqualTo(HttpStatus.NOT_FOUND));
    }

    @Test
    void compareExperiments_throwsForbiddenForOtherUsersExperiment() {
        Experiment e = buildExperiment(5L, otherUser, "vgg11", "mnist");
        when(experimentRepository.findById(5L)).thenReturn(Optional.of(e));

        CompareExperimentsRequest request = new CompareExperimentsRequest();
        request.setIds(List.of(5L));

        assertThatThrownBy(() -> experimentService.compareExperiments(request))
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(ex -> assertThat(((ResponseStatusException) ex).getStatusCode())
                        .isEqualTo(HttpStatus.FORBIDDEN));
    }

    // --- rerunExperiment ---

    @Test
    void rerunExperiment_throwsNotFoundForMissingId() {
        when(experimentRepository.findById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> experimentService.rerunExperiment(99L))
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(e -> assertThat(((ResponseStatusException) e).getStatusCode())
                        .isEqualTo(HttpStatus.NOT_FOUND));
    }

    @Test
    void rerunExperiment_throwsForbiddenForOtherUsersExperiment() {
        Experiment e = buildExperiment(5L, otherUser, "vgg11", "mnist");
        when(experimentRepository.findById(5L)).thenReturn(Optional.of(e));

        assertThatThrownBy(() -> experimentService.rerunExperiment(5L))
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(ex -> assertThat(((ResponseStatusException) ex).getStatusCode())
                        .isEqualTo(HttpStatus.FORBIDDEN));
    }

    @Test
    void rerunExperiment_callsAiBackendWithOriginalParams() {
        Experiment original = buildExperiment(1L, currentUser, "resnet18", "cifar10");
        original.setEpochs(10);
        original.setBatchSize(64);
        original.setLearningRate(0.01);

        when(experimentRepository.findById(1L)).thenReturn(Optional.of(original));

        ExperimentResult aiResult = new ExperimentResult();
        aiResult.setTrainLossPerEpoch(List.of(0.5, 0.3));
        aiResult.setTestLoss(0.2);
        aiResult.setTestAccuracy(0.95);
        aiResult.setTrainingTimeSeconds(10.0);

        when(restTemplate.postForObject(anyString(), any(ExperimentRequest.class), eq(ExperimentResult.class)))
                .thenReturn(aiResult);
        when(experimentRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        experimentService.rerunExperiment(1L);

        verify(restTemplate).postForObject(
                eq("http://ai-backend:8000/experiments"),
                any(ExperimentRequest.class),
                eq(ExperimentResult.class)
        );
    }

    // --- updateNote ---

    @Test
    void updateNote_updatesNoteAndSaves() {
        Experiment e = buildExperiment(1L, currentUser, "resnet18", "mnist");
        when(experimentRepository.findById(1L)).thenReturn(Optional.of(e));
        when(experimentRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        NoteRequest request = new NoteRequest();
        request.setNote("good result");

        ExperimentResponse result = experimentService.updateNote(1L, request);

        assertThat(result.getNote()).isEqualTo("good result");
        verify(experimentRepository).save(e);
    }

    @Test
    void updateNote_throwsNotFoundForMissingId() {
        when(experimentRepository.findById(99L)).thenReturn(Optional.empty());

        NoteRequest request = new NoteRequest();
        request.setNote("note");

        assertThatThrownBy(() -> experimentService.updateNote(99L, request))
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(e -> assertThat(((ResponseStatusException) e).getStatusCode())
                        .isEqualTo(HttpStatus.NOT_FOUND));
    }

    @Test
    void updateNote_throwsForbiddenForOtherUsersExperiment() {
        Experiment e = buildExperiment(5L, otherUser, "vgg11", "mnist");
        when(experimentRepository.findById(5L)).thenReturn(Optional.of(e));

        NoteRequest request = new NoteRequest();
        request.setNote("note");

        assertThatThrownBy(() -> experimentService.updateNote(5L, request))
                .isInstanceOf(ResponseStatusException.class)
                .satisfies(ex -> assertThat(((ResponseStatusException) ex).getStatusCode())
                        .isEqualTo(HttpStatus.FORBIDDEN));
    }

    // --- runExperiment ---

    @Test
    void runExperiment_savesExperimentWithCorrectFields() {
        ExperimentRequest request = new ExperimentRequest();
        request.setModel("lenet5");
        request.setDataset("mnist");
        TrainingConfig config = new TrainingConfig();
        config.setEpochs(3);
        config.setBatchSize(32);
        config.setLearningRate(0.001);
        request.setTraining(config);

        ExperimentResult aiResult = new ExperimentResult();
        aiResult.setTrainLossPerEpoch(List.of(0.8, 0.5, 0.3));
        aiResult.setTestLoss(0.25);
        aiResult.setTestAccuracy(0.92);
        aiResult.setTrainingTimeSeconds(15.0);

        when(restTemplate.postForObject(anyString(), any(), eq(ExperimentResult.class)))
                .thenReturn(aiResult);
        when(experimentRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        ExperimentResponse response = experimentService.runExperiment(request);

        assertThat(response.getModel()).isEqualTo("lenet5");
        assertThat(response.getDataset()).isEqualTo("mnist");
        assertThat(response.getTestAccuracy()).isEqualTo(0.92);
        assertThat(response.getTrainLossPerEpoch()).hasSize(3);
    }

    // --- helper ---

    private Experiment buildExperiment(Long id, User user, String model, String dataset) {
        return Experiment.builder()
                .id(id)
                .user(user)
                .model(model)
                .dataset(dataset)
                .epochs(5)
                .batchSize(32)
                .learningRate(0.001)
                .trainLossPerEpoch(List.of(0.5, 0.3))
                .testLoss(0.2)
                .testAccuracy(0.9)
                .trainingTimeSeconds(10.0)
                .createdAt(LocalDateTime.now())
                .build();
    }
}
