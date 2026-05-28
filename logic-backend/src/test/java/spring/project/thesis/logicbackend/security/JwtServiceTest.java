package spring.project.thesis.logicbackend.security;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class JwtServiceTest {

    private JwtService jwtService;

    @BeforeEach
    void setUp() {
        jwtService = new JwtService();
    }

    @Test
    void generateToken_returnsNonNullToken() {
        String token = jwtService.generateToken("user1");
        assertThat(token).isNotNull().isNotBlank();
    }

    @Test
    void generateToken_tokenHasThreeParts() {
        String token = jwtService.generateToken("user1");
        assertThat(token.split("\\.")).hasSize(3);
    }

    @Test
    void extractUsername_returnsCorrectUsername() {
        String token = jwtService.generateToken("testuser");
        assertThat(jwtService.extractUsername(token)).isEqualTo("testuser");
    }

    @Test
    void isTokenValid_returnsTrueForFreshToken() {
        String token = jwtService.generateToken("user1");
        assertThat(jwtService.isTokenValid(token)).isTrue();
    }

    @Test
    void generateToken_differentUsersProduceDifferentTokens() {
        String token1 = jwtService.generateToken("alice");
        String token2 = jwtService.generateToken("bob");
        assertThat(token1).isNotEqualTo(token2);
    }

    @Test
    void extractUsername_throwsOnTamperedToken() {
        String token = jwtService.generateToken("user1");
        String tampered = token.substring(0, token.length() - 5) + "XXXXX";
        assertThatThrownBy(() -> jwtService.extractUsername(tampered))
                .isInstanceOf(Exception.class);
    }
}
