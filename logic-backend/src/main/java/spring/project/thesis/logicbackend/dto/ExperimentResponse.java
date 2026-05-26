package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;
import spring.project.thesis.logicbackend.experiment.Experiment;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
public class ExperimentResponse {

    private Long id;
    private String model;
    private String dataset;
    private int epochs;

    @JsonProperty("batch_size")
    private int batchSize;

    @JsonProperty("learning_rate")
    private double learningRate;

    @JsonProperty("train_loss_per_epoch")
    private List<Double> trainLossPerEpoch;

    @JsonProperty("test_loss")
    private double testLoss;

    @JsonProperty("test_accuracy")
    private double testAccuracy;

    @JsonProperty("created_at")
    private LocalDateTime createdAt;

    public static ExperimentResponse from(Experiment experiment) {
        return ExperimentResponse.builder()
                .id(experiment.getId())
                .model(experiment.getModel())
                .dataset(experiment.getDataset())
                .epochs(experiment.getEpochs())
                .batchSize(experiment.getBatchSize())
                .learningRate(experiment.getLearningRate())
                .trainLossPerEpoch(experiment.getTrainLossPerEpoch())
                .testLoss(experiment.getTestLoss())
                .testAccuracy(experiment.getTestAccuracy())
                .createdAt(experiment.getCreatedAt())
                .build();
    }
}
