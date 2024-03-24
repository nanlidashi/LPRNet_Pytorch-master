package com.example.loginspring;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

import java.io.File;

@EnableCaching
@SpringBootApplication
public class LoginSpringApplication {

    public static void main(String[] args) {
        String projectPath = System.getProperty("user.dir");
        File uploadDir = new File(projectPath + "/temp");
        if (!uploadDir.exists()) {
            uploadDir.mkdir();
        }
        SpringApplication.run(LoginSpringApplication.class, args);
    }
}
