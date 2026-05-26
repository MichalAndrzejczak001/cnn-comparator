package spring.project.thesis.logicbackend.dto;

import lombok.Data;

@Data
public class ExperimentRequest {
    private String model;
    private String dataset;
    private TrainingConfig training = new TrainingConfig();
}
