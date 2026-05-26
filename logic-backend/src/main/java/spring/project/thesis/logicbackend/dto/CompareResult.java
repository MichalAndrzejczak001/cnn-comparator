package spring.project.thesis.logicbackend.dto;

import lombok.Data;
import java.util.List;

@Data
public class CompareResult {
    private String dataset;
    private int epochs;
    private List<ModelResult> results;
}
