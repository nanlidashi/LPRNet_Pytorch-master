package com.example.loginspring.Controller;

import com.example.loginspring.Service.UserService;
import com.example.loginspring.entity.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = "http://localhost:8080")  // 允许跨域访问
@RestController
public class RegisterController {

    @Autowired
    private UserService userService;

    @RequestMapping(value = "/register", method = RequestMethod.POST)
    public ResponseEntity<?> registerUser(@RequestBody User user) {
        try{
            if(userService.isUsernameExists(user.getUsername())){
                return ResponseEntity.badRequest().body("用户名已存在");
            }
            if (!userService.checkPasswordMatch(user.getPassword(), user.getConfirmPassword())) {
                return ResponseEntity.badRequest().body("密码不一致");
            }
            User newUser = userService.saveUser(user);
            return ResponseEntity.ok(newUser);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body("注册失败" + e.getMessage());
        }

    }

}
