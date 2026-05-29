package spring.project.thesis.logicbackend.experiment;

import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import spring.project.thesis.logicbackend.dto.ClassifyResponse;
import spring.project.thesis.logicbackend.dto.NoteRequest;
import spring.project.thesis.logicbackend.dto.CompareExperimentsRequest;
import spring.project.thesis.logicbackend.dto.CompareRequest;
import spring.project.thesis.logicbackend.dto.CompareResult;
import spring.project.thesis.logicbackend.dto.ExperimentRequest;
import spring.project.thesis.logicbackend.dto.ExperimentResponse;

import java.util.List;

@RestController
@RequestMapping
@RequiredArgsConstructor
public class ExperimentController {

    private final ExperimentService experimentService;

    @PostMapping("/experiments")
    public ExperimentResponse runExperiment(@RequestBody ExperimentRequest request) {
        return experimentService.runExperiment(request);
    }

    @GetMapping("/experiments")
    public List<ExperimentResponse> getHistory() {
        return experimentService.getHistory();
    }

    @PostMapping("/experiments/{id}/rerun")
    public ExperimentResponse rerunExperiment(@PathVariable Long id) {
        return experimentService.rerunExperiment(id);
    }

    @PostMapping("/experiments/compare")
    public List<ExperimentResponse> compareExperiments(@RequestBody CompareExperimentsRequest request) {
        return experimentService.compareExperiments(request);
    }

    @PatchMapping("/experiments/{id}/note")
    public ExperimentResponse updateNote(@PathVariable Long id, @RequestBody NoteRequest request) {
        return experimentService.updateNote(id, request);
    }

    @PostMapping("/experiments/{id}/classify")
    public ClassifyResponse classifyImage(
            @PathVariable Long id,
            @RequestParam("file") MultipartFile file
    ) {
        return experimentService.classifyImage(id, file);
    }

    @PostMapping("/compare")
    public CompareResult runCompare(@RequestBody CompareRequest request) {
        return experimentService.runCompare(request);
    }
}
