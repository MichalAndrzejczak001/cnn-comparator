package spring.project.thesis.logicbackend.bdd;

import io.restassured.response.ValidatableResponse;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("cucumber-glue")
public class ScenarioContext {
    public ValidatableResponse response;
}
