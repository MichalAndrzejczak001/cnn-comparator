package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SampleGradCam {

    @JsonProperty("true_label")
    private String trueLabel;

    @JsonProperty("predicted_label")
    private String predictedLabel;

    private double confidence;

    @JsonProperty("gradcam_image")
    private String gradcamImage;
}
