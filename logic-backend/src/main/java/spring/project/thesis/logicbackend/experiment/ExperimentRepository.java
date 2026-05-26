package spring.project.thesis.logicbackend.experiment;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import spring.project.thesis.logicbackend.user.User;

import java.util.List;

@Repository
public interface ExperimentRepository extends JpaRepository<Experiment, Long> {

    List<Experiment> findByUserOrderByCreatedAtDesc(User user);
}
