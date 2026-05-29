package spring.project.thesis.logicbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class ClassifyResponse {

    @JsonProperty("predicted_class")
    private String predictedClass;

    @JsonProperty("predicted_index")
    private int predictedIndex;

    private List<ClassConfidence> confidences;

    @Data
    public static class ClassConfidence {
        private String label;
        private double confidence;
    }
}
