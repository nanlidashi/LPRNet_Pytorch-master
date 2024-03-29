package com.example.loginspring.Service;

import com.example.loginspring.repository.PlateRepository;
import com.example.loginspring.entity.Plate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class PlateService {
    @Autowired
    private PlateRepository plateRepository;

    public Page<Plate> getAllPlates(Pageable pageable) {
        return plateRepository.findAll(pageable);
    }

    public Optional<Plate> getPlateByTargetAndFlagAndPredict(String target, String flag, String predict) {
        return plateRepository.findByTargetAndFlagAndPredict(target, flag, predict);
    }

    public Plate savePlate(Plate plate) {
        return plateRepository.save(plate);
    }
}
