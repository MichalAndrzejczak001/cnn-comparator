package spring.project.thesis.logicbackend.experiment;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import spring.project.thesis.logicbackend.dto.CompareRequest;
import spring.project.thesis.logicbackend.dto.CompareResult;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.dto.ExperimentResult;

@Service
@RequiredArgsConstructor
public class ExperimentService {

    private final RestTemplate restTemplate;

    @Value("${ai-backend.url}")
    private String aiBackendUrl;

    public ExperimentResult runExperiment(ExperimentRequest request) {
        return restTemplate.postForObject(
                aiBackendUrl + "/experiments",
                request,
                ExperimentResult.class
        );
    }

    public CompareResult runCompare(CompareRequest request) {
        return restTemplate.postForObject(
                aiBackendUrl + "/compare",
                request,
                CompareResult.class
        );
    }
}
