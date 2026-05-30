package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class ExperimentResult {
    private String status;

    @JsonProperty("model_id")
    private String modelId;

    @JsonProperty("sample_gradcams")
    private List<SampleGradCam> sampleGradcams;

    @JsonProperty("train_loss_per_epoch")
    private List<Double> trainLossPerEpoch;

    @JsonProperty("test_loss_per_epoch")
    private List<Double> testLossPerEpoch;

    @JsonProperty("test_loss")
    private double testLoss;

    @JsonProperty("test_accuracy")
    private double testAccuracy;

    @JsonProperty("confusion_matrix")
    private List<List<Integer>> confusionMatrix;

    @JsonProperty("training_time_seconds")
    private double trainingTimeSeconds;
}
