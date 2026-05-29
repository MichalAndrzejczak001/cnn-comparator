package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class GradCamResponse {

    @JsonProperty("predicted_class")
    private String predictedClass;

    @JsonProperty("predicted_index")
    private int predictedIndex;

    private List<ClassifyResponse.ClassConfidence> confidences;

    @JsonProperty("gradcam_image")
    private String gradcamImage;
}
