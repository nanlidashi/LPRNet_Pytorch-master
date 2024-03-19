package org.example.helloworld.controller;

//import jakarta.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@RestController
public class FileUploadController {
    //@PostMapping("/upload")
    @RequestMapping(value = "/upload",method = RequestMethod.POST)
    public String up(String nickname, MultipartFile photo, HttpServletRequest request)throws IOException {
        System.out.println(nickname);
        System.out.println(photo.getOriginalFilename());//获取文件名
        System.out.println(photo.getContentType());//获取文件类型
        //System.out.println(System.getProperty("user.dir"));//获取项目路径

        String path = request.getServletContext().getRealPath("/upload/");
        System.out.println(path);
        saveFile(photo, path);
        return "success";
    }

    public void saveFile(MultipartFile photo, String path) throws IOException{
        File dir = new File(path);
        if(!dir.exists()){
            dir.mkdir();
        }
        File file = new File(path + photo.getOriginalFilename());
        photo.transferTo(file);
    }

}
