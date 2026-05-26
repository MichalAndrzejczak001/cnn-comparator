package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class ExperimentResult {
    private String status;

    @JsonProperty("train_loss_per_epoch")
    private List<Double> trainLossPerEpoch;

    @JsonProperty("test_loss")
    private double testLoss;

    @JsonProperty("test_accuracy")
    private double testAccuracy;
}
