package com.example.loginspring.Controller;

import com.example.loginspring.Service.PlateService;
import com.example.loginspring.entity.Plate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "http://localhost:8080")  // 允许跨域访问
@RestController
@RequestMapping("/api")
public class CeShiController {

    @Autowired
    private PlateService plateService;

    @GetMapping("/plate")
    public Page<Plate> getAllPlates(@RequestParam(defaultValue = "0") Integer page,
                                    @RequestParam(defaultValue = "10") Integer size) {
        Pageable pageable = PageRequest.of(page, size);
        return plateService.getAllPlates(pageable);
    }
}
