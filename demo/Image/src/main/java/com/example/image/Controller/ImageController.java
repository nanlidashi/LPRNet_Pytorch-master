package com.example.image.Controller;

import com.example.image.Service.ImageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@RestController
@RequestMapping("/api")
public class ImageController {

    @Autowired
    private ImageService imageService;

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("file") MultipartFile file) {
        try {
            String fileName = file.getOriginalFilename();
            String projectPath = System.getProperty("user.dir");
            String filePath = projectPath + "/temp/" + fileName;
            file.transferTo(new File(filePath));
            String result = imageService.processImage(filePath);

            // 删除临时文件
            File tempFile = new File(filePath);
            if (tempFile.exists()) {
                tempFile.delete();
            }

            return result;
        } catch (IOException e) {
            e.printStackTrace();
            return "Failed to upload and process image";
        }
    }
}
