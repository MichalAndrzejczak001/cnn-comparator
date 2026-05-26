package spring.project.thesis.logicbackend.dto;

import lombok.Data;

import java.util.List;

@Data
public class CompareExperimentsRequest {
    private List<Long> ids;
}
