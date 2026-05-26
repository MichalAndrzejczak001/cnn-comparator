package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class TrainingConfig {
    private int epochs = 5;

    @JsonProperty("batch_size")
    private int batchSize = 32;

    @JsonProperty("learning_rate")
    private double learningRate = 0.001;
}
