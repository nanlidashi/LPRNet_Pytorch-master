package com.example.loginspring.Controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class IndexController {
    @GetMapping("/index")
    public String index() {
        return "欢迎使用车牌识别系统";
    }
}
