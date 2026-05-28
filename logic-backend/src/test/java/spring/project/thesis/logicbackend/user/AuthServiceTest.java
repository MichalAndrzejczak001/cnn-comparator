package spring.project.thesis.logicbackend.user;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import spring.project.thesis.logicbackend.dto.LoginRequest;
import spring.project.thesis.logicbackend.dto.RegisterRequest;
import spring.project.thesis.logicbackend.enums.Role;
import spring.project.thesis.logicbackend.security.JwtService;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AuthServiceTest {

    @Mock
    private UserRepository repository;

    @Mock
    private PasswordEncoder encoder;

    @Mock
    private JwtService jwtService;

    @Mock
    private AuthenticationManager authManager;

    @InjectMocks
    private AuthService authService;

    @Test
    void register_savesUserToRepository() {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("password123");

        when(encoder.encode("password123")).thenReturn("hashed");
        when(repository.save(any())).thenAnswer(inv -> inv.getArgument(0));
        when(jwtService.generateToken(anyString())).thenReturn("token");

        authService.register(request);

        ArgumentCaptor<User> captor = ArgumentCaptor.forClass(User.class);
        verify(repository).save(captor.capture());
        assertThat(captor.getValue().getUsername()).isEqualTo("alice");
    }

    @Test
    void register_savesHashedPassword() {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("password123");

        when(encoder.encode("password123")).thenReturn("hashed");
        when(repository.save(any())).thenAnswer(inv -> inv.getArgument(0));
        when(jwtService.generateToken(anyString())).thenReturn("token");

        authService.register(request);

        ArgumentCaptor<User> captor = ArgumentCaptor.forClass(User.class);
        verify(repository).save(captor.capture());
        assertThat(captor.getValue().getPassword()).isEqualTo("hashed");
        assertThat(captor.getValue().getPassword()).isNotEqualTo("password123");
    }

    @Test
    void register_assignsUserRole() {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("pass");

        when(encoder.encode(anyString())).thenReturn("hashed");
        when(repository.save(any())).thenAnswer(inv -> inv.getArgument(0));
        when(jwtService.generateToken(anyString())).thenReturn("token");

        authService.register(request);

        ArgumentCaptor<User> captor = ArgumentCaptor.forClass(User.class);
        verify(repository).save(captor.capture());
        assertThat(captor.getValue().getRole()).isEqualTo(Role.USER);
    }

    @Test
    void register_returnsTokenFromJwtService() {
        RegisterRequest request = new RegisterRequest();
        request.setUsername("alice");
        request.setPassword("pass");

        when(encoder.encode(anyString())).thenReturn("hashed");
        when(repository.save(any())).thenAnswer(inv -> inv.getArgument(0));
        when(jwtService.generateToken("alice")).thenReturn("jwt-token-alice");

        String result = authService.register(request);

        assertThat(result).isEqualTo("jwt-token-alice");
    }

    @Test
    void login_callsAuthManagerWithCredentials() {
        LoginRequest request = new LoginRequest();
        request.setUsername("alice");
        request.setPassword("pass");

        when(jwtService.generateToken("alice")).thenReturn("token");

        authService.login(request);

        ArgumentCaptor<UsernamePasswordAuthenticationToken> captor =
                ArgumentCaptor.forClass(UsernamePasswordAuthenticationToken.class);
        verify(authManager).authenticate(captor.capture());
        assertThat(captor.getValue().getPrincipal()).isEqualTo("alice");
        assertThat(captor.getValue().getCredentials()).isEqualTo("pass");
    }

    @Test
    void login_returnsTokenFromJwtService() {
        LoginRequest request = new LoginRequest();
        request.setUsername("alice");
        request.setPassword("pass");

        when(jwtService.generateToken("alice")).thenReturn("jwt-token");

        String result = authService.login(request);

        assertThat(result).isEqualTo("jwt-token");
    }

    @Test
    void login_throwsWhenAuthManagerThrows() {
        LoginRequest request = new LoginRequest();
        request.setUsername("alice");
        request.setPassword("wrong");

        when(authManager.authenticate(any())).thenThrow(new BadCredentialsException("Bad credentials"));

        assertThatThrownBy(() -> authService.login(request))
                .isInstanceOf(BadCredentialsException.class);
    }
}
