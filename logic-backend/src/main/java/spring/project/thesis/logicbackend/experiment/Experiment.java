package spring.project.thesis.logicbackend.experiment;

import jakarta.persistence.*;
import lombok.*;
import spring.project.thesis.logicbackend.config.DoubleListConverter;
import spring.project.thesis.logicbackend.config.IntMatrixConverter;
import spring.project.thesis.logicbackend.config.SampleGradCamListConverter;
import spring.project.thesis.logicbackend.dto.SampleGradCam;
import spring.project.thesis.logicbackend.user.User;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "experiments")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Experiment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    private String model;
    private String dataset;
    private int epochs;
    private int batchSize;
    private double learningRate;

    @Convert(converter = DoubleListConverter.class)
    @Column(columnDefinition = "TEXT")
    private List<Double> trainLossPerEpoch;

    @Convert(converter = DoubleListConverter.class)
    @Column(columnDefinition = "TEXT")
    private List<Double> testLossPerEpoch;

    private double testLoss;
    private double testAccuracy;

    private double trainingTimeSeconds;

    @Convert(converter = IntMatrixConverter.class)
    @Column(columnDefinition = "TEXT")
    private List<List<Integer>> confusionMatrix;

    @Column(length = 1000)
    private String note;

    private String modelId;

    private LocalDateTime createdAt;

    @Convert(converter = SampleGradCamListConverter.class)
    @Column(columnDefinition = "MEDIUMTEXT")
    private List<SampleGradCam> sampleGradcams;
}
