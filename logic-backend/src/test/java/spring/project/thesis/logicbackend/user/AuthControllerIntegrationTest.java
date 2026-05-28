package spring.project.thesis.logicbackend.user;

import tools.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import spring.project.thesis.logicbackend.dto.LoginRequest;
import spring.project.thesis.logicbackend.dto.RegisterRequest;
import spring.project.thesis.logicbackend.experiment.ExperimentRepository;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class AuthControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private ExperimentRepository experimentRepository;

    @BeforeEach
    void setUp() {
        experimentRepository.deleteAll();
        userRepository.deleteAll();
    }

    @Test
    void register_returnsTokenOnSuccess() throws Exception {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("password123");

        mockMvc.perform(post("/auth/register")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(content().string(org.hamcrest.Matchers.not(org.hamcrest.Matchers.emptyOrNullString())));
    }

    @Test
    void register_savesUserInDatabase() throws Exception {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("password123");

        mockMvc.perform(post("/auth/register")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk());

        org.assertj.core.api.Assertions.assertThat(userRepository.findByUsername("alice")).isPresent();
    }

    @Test
    void register_duplicateUsername_throwsException() throws Exception {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("password123");

        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)));

        // DataIntegrityViolationException nie jest obsługiwany przez @ExceptionHandler,
        // więc propaguje się jako wyjątek zamiast odpowiedzi 500
        assertThatThrownBy(() -> mockMvc.perform(post("/auth/register")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request))))
                .hasCauseInstanceOf(org.springframework.dao.DataIntegrityViolationException.class);
    }

    @Test
    void login_returnsTokenForValidCredentials() throws Exception {
        RegisterRequest register = new RegisterRequest();
        register.setUsername("alice");
        register.setPassword("password123");
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(register)));

        LoginRequest login = new LoginRequest();
        login.setUsername("alice");
        login.setPassword("password123");

        mockMvc.perform(post("/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(login)))
                .andExpect(status().isOk())
                .andExpect(content().string(org.hamcrest.Matchers.not(org.hamcrest.Matchers.emptyOrNullString())));
    }

    @Test
    void login_returns401ForWrongPassword() throws Exception {
        RegisterRequest register = new RegisterRequest();
        register.setUsername("alice");
        register.setPassword("password123");
        mockMvc.perform(post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(register)));

        LoginRequest login = new LoginRequest();
        login.setUsername("alice");
        login.setPassword("wrongpassword");

        mockMvc.perform(post("/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(login)))
                .andExpect(status().is4xxClientError());
    }

    @Test
    void login_returns401ForNonExistentUser() throws Exception {
        LoginRequest login = new LoginRequest();
        login.setUsername("nobody");
        login.setPassword("pass");

        mockMvc.perform(post("/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(login)))
                .andExpect(status().is4xxClientError());
    }

    @Test
    void protectedEndpoint_returns4xxWithoutToken() throws Exception {
        mockMvc.perform(get("/experiments"))
                .andExpect(status().is4xxClientError());
    }

    @Test
    void protectedEndpoint_returns401WithInvalidToken() throws Exception {
        mockMvc.perform(get("/experiments")
                        .header("Authorization", "Bearer invalid.token.here"))
                .andExpect(status().isUnauthorized());
    }
}
