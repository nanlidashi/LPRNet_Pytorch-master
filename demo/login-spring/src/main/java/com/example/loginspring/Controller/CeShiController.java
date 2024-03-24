package com.example.loginspring.Controller;

import com.example.loginspring.Service.PlateService;
import com.example.loginspring.entity.Plate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "http://localhost:8080")  // 允许跨域访问
@RestController
@RequestMapping("/api")
public class CeShiController {

    @Autowired
    private PlateService plateService;

    @GetMapping("/plate")
    public List<Plate> getAllPlates(){
        return plateService.getAllPlates();
    }

//    @PostMapping
//    public Plate savePlate(@RequestBody Plate plate) {
//        return plateService.savePlate(plate);
//    }
}
