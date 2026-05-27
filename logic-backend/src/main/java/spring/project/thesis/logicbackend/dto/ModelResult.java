package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class ModelResult {
    private String model;

    @JsonProperty("train_loss_per_epoch")
    private List<Double> trainLossPerEpoch;

    @JsonProperty("test_loss_per_epoch")
    private List<Double> testLossPerEpoch;

    @JsonProperty("test_loss")
    private double testLoss;

    @JsonProperty("test_accuracy")
    private double testAccuracy;

    @JsonProperty("training_time_seconds")
    private double trainingTimeSeconds;

    @JsonProperty("confusion_matrix")
    private List<List<Integer>> confusionMatrix;
}
