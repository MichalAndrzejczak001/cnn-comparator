package spring.project.thesis.logicbackend.dto;

import lombok.Data;

@Data
public class RegisterRequest {
    private String username;
    private String password;
}