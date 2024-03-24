package com.example.loginspring.Controller;

import com.example.loginspring.dao.PlateRepository;
import com.example.loginspring.entity.Plate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import com.example.loginspring.Service.ImageService;
import java.io.File;
import java.io.IOException;

import java.util.Optional;

@RestController
public class ImageController {

    @Autowired
    private ImageService imageService;

    @Autowired
    private PlateRepository plateRepository;

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("file") MultipartFile file) {
        try {
            if (file.isEmpty()) {
                return "上传失败，请选择文件";
            }
            String fileName = file.getOriginalFilename();
            String projectPath = System.getProperty("user.dir");
            String filePath = projectPath + "/temp/" + fileName;
            file.transferTo(new File(filePath));
            String result = imageService.processImage(filePath);

            // 解析结果并保存到数据库
            String[] parts = result.split("###");

            if (parts.length < 3) {
                deleteTempFile(filePath); // 删除临时文件
                return "处理结果格式错误";
            }

            String target = parts[0].trim().substring(9);// 获取 "target: 宁ASE106" 中的 "宁ASE106"
            String flag = parts[1].trim();// 获取 "flag:F" 中的 "F"
            String predict = parts[2].trim().substring(10);// 获取 "predict: 闽ASE10G" 中的 "闽ASE10G"

            //检查数据库中是否已经存在该车牌
            Optional<Plate> existingPlate = plateRepository.findByTargetAndFlagAndPredict(target, flag, predict);
            if (existingPlate.isPresent()) {
                deleteTempFile(filePath); // 删除临时文件
                return "该车牌已经存在";
            }

            Plate plate = new Plate();
            plate.setTarget(target);
            plate.setFlag(flag);
            plate.setPredict(predict);
            plateRepository.save(plate);

            // 删除临时文件
            deleteTempFile(filePath);

            return result;
        } catch (IOException e) {
            e.printStackTrace();
            return "Failed to upload and process image";
        }
    }
    public void deleteTempFile(String filePath) {
        File tempFile = new File(filePath);
        if (tempFile.exists()) {
            tempFile.delete();
        }
    }

}
