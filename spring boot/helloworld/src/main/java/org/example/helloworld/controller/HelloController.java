package org.example.helloworld.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
@RestController
public class HelloController {

    // http://localhost:8080/hello?nickname=zhangsan&phone=123
    //https://www.baidu.com//  --> 协议//域名//路径//参数
    //getmapping里面就路径-网站

    @RequestMapping (value = "/hello",method = RequestMethod.GET)
    //等价于@GetMapping("/hello")
    public String hello(String nickname,String phone) {
        System.out.println(phone);
        return "hello world 你好--" + nickname + "---" + phone;
    }
}
