package com.example.loginspring.Service;

import com.example.loginspring.repository.PlateRepository;
import com.example.loginspring.entity.Plate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PlateService {
    @Autowired
    private PlateRepository plateRepository;

    public Page<Plate> getAllPlates(Pageable pageable) {
        return plateRepository.findAll(pageable);
    }

    public Optional<Plate> getPlateByTargetAndFlagAndPredictAndBase64(String target, String flag, String predict, String base64) {
        return plateRepository.findByTargetAndFlagAndPredictAndBase64(target, flag, predict, base64);
    }

    public Plate savePlate(Plate plate) {
        return plateRepository.save(plate);
    }

    @Autowired
    public PlateService(PlateRepository plateRepository) {
        this.plateRepository = plateRepository;
    }

    // 根据关键字搜索 Plate 实体
    public List<Plate> searchPlatesByKeyword(String keyword) {
        return plateRepository.findByTargetContaining(keyword);
    }
}
