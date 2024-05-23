package com.example.loginspring.Service;

import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

@Service
public class ImageService {

    public String processImage(String filePath) {
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "D:/Python/LPRNet_Pytorch-master/models/test_one_ceshi.py", filePath);
            String projectPath = System.getProperty("user.dir");
            pb.directory(new File(projectPath + "/src/temp"));// 设置工作目录
            pb.redirectErrorStream(true);

            // 设置字符集为 UTF-8
            pb.redirectOutput(ProcessBuilder.Redirect.PIPE);

            Process p = pb.start();

            // 从输出中读取处理结果
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
            StringBuilder result = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line).append("\n");
            }

            // 打印处理结果
//            System.out.println( result);

            return result.toString();
        } catch (IOException  e) {
            e.printStackTrace();
            return "Failed to process image";
        }
    }
    public String getBase64Image(String result) {
        String[] parts = result.split("Base64 Image: ");
        if (parts.length < 2) {
            return "Failed to extract Base64 image";
        }
        return parts[1].trim();
    }

    public String getPrediction(String result) {
        String[] parts = result.split("Base64 Image: ");
        if (parts.length < 1) {
            return "Failed to extract prediction";
        }
        return parts[0].trim();
    }
}
