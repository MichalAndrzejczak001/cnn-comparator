package spring.project.thesis.logicbackend.experiment;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import spring.project.thesis.logicbackend.dto.CompareExperimentsRequest;
import spring.project.thesis.logicbackend.dto.CompareRequest;
import spring.project.thesis.logicbackend.dto.CompareResult;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.dto.ExperimentResponse;
import spring.project.thesis.logicbackend.dto.ExperimentResult;
import spring.project.thesis.logicbackend.dto.NoteRequest;
import spring.project.thesis.logicbackend.dto.TrainingConfig;
import spring.project.thesis.logicbackend.user.User;
import spring.project.thesis.logicbackend.user.UserRepository;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ExperimentService {

    private final RestTemplate restTemplate;
    private final ExperimentRepository experimentRepository;
    private final UserRepository userRepository;

    @Value("${ai-backend.url}")
    private String aiBackendUrl;

    public ExperimentResponse runExperiment(ExperimentRequest request) {
        ExperimentResult result = restTemplate.postForObject(
                aiBackendUrl + "/experiments",
                request,
                ExperimentResult.class
        );

        Experiment saved = experimentRepository.save(Experiment.builder()
                .user(currentUser())
                .model(request.getModel())
                .dataset(request.getDataset())
                .epochs(request.getTraining().getEpochs())
                .batchSize(request.getTraining().getBatchSize())
                .learningRate(request.getTraining().getLearningRate())
                .trainLossPerEpoch(result.getTrainLossPerEpoch())
                .testLoss(result.getTestLoss())
                .testAccuracy(result.getTestAccuracy())
                .createdAt(LocalDateTime.now())
                .build());

        return ExperimentResponse.from(saved);
    }

    public List<ExperimentResponse> getHistory() {
        return experimentRepository.findByUserOrderByCreatedAtDesc(currentUser())
                .stream()
                .map(ExperimentResponse::from)
                .toList();
    }

    public List<ExperimentResponse> compareExperiments(CompareExperimentsRequest request) {
        User user = currentUser();
        return request.getIds().stream()
                .map(id -> experimentRepository.findById(id)
                        .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND)))
                .peek(experiment -> {
                    if (!experiment.getUser().getId().equals(user.getId())) {
                        throw new ResponseStatusException(HttpStatus.FORBIDDEN);
                    }
                })
                .map(ExperimentResponse::from)
                .toList();
    }

    public ExperimentResponse rerunExperiment(Long experimentId) {
        Experiment original = experimentRepository.findById(experimentId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        if (!original.getUser().getId().equals(currentUser().getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN);
        }

        ExperimentRequest request = new ExperimentRequest();
        request.setModel(original.getModel());
        request.setDataset(original.getDataset());

        TrainingConfig training = new TrainingConfig();
        training.setEpochs(original.getEpochs());
        training.setBatchSize(original.getBatchSize());
        training.setLearningRate(original.getLearningRate());
        request.setTraining(training);

        return runExperiment(request);
    }

    public ExperimentResponse updateNote(Long experimentId, NoteRequest request) {
        Experiment experiment = experimentRepository.findById(experimentId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        if (!experiment.getUser().getId().equals(currentUser().getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN);
        }

        experiment.setNote(request.getNote());
        return ExperimentResponse.from(experimentRepository.save(experiment));
    }

    public CompareResult runCompare(CompareRequest request) {
        return restTemplate.postForObject(
                aiBackendUrl + "/compare",
                request,
                CompareResult.class
        );
    }

    private User currentUser() {
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        return userRepository.findByUsername(username).orElseThrow();
    }
}
