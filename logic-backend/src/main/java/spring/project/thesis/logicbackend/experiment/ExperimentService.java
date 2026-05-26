package spring.project.thesis.logicbackend.experiment;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import spring.project.thesis.logicbackend.dto.CompareRequest;
import spring.project.thesis.logicbackend.dto.CompareResult;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.dto.ExperimentResponse;
import spring.project.thesis.logicbackend.dto.ExperimentResult;
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

    public ExperimentResult runExperiment(ExperimentRequest request) {
        ExperimentResult result = restTemplate.postForObject(
                aiBackendUrl + "/experiments",
                request,
                ExperimentResult.class
        );

        experimentRepository.save(Experiment.builder()
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

        return result;
    }

    public List<ExperimentResponse> getHistory() {
        return experimentRepository.findByUserOrderByCreatedAtDesc(currentUser())
                .stream()
                .map(ExperimentResponse::from)
                .toList();
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
