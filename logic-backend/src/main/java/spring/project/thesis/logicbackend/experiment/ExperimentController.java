package spring.project.thesis.logicbackend.experiment;

import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import spring.project.thesis.logicbackend.dto.CompareRequest;
import spring.project.thesis.logicbackend.dto.CompareResult;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.dto.ExperimentResult;

@RestController
@RequestMapping
@RequiredArgsConstructor
public class ExperimentController {

    private final ExperimentService experimentService;

    @PostMapping("/experiments")
    public ExperimentResult runExperiment(@RequestBody ExperimentRequest request) {
        return experimentService.runExperiment(request);
    }

    @PostMapping("/compare")
    public CompareResult runCompare(@RequestBody CompareRequest request) {
        return experimentService.runCompare(request);
    }
}
