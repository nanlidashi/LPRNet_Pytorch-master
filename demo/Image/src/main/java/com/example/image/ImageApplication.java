package com.example.image;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.File;

@SpringBootApplication
public class ImageApplication {

    public static void main(String[] args) {
        // 确保 upload 文件夹存在
        String projectPath = System.getProperty("user.dir");
        File uploadDir = new File(projectPath + "/temp");
        if (!uploadDir.exists()) {
            uploadDir.mkdir();
        }
        SpringApplication.run(ImageApplication.class, args);
    }
}
