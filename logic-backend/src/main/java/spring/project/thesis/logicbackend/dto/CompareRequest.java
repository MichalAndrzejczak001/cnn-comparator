package spring.project.thesis.logicbackend.dto;

import lombok.Data;

@Data
public class CompareRequest {
    private String dataset;
    private TrainingConfig training = new TrainingConfig();
}
