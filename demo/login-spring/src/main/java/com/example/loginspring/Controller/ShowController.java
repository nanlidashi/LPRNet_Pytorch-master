package com.example.loginspring.Controller;

import com.example.loginspring.entity.Plate;
import com.example.loginspring.repository.PlateRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:8081") // 允许跨域请求的前端地址
public class ShowController {

    @Autowired
    private PlateRepository plateRepository;

    @GetMapping("/plates")
    public List<Plate> getAllPlates() {
        return plateRepository.findAll();
    }

    @GetMapping("/search")
    public List<Plate> searchPlates(@RequestParam String keyword) {
        return plateRepository.findByTargetContaining(keyword);
    }

}
