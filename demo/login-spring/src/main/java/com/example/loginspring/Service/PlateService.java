package com.example.loginspring.Service;

import com.example.loginspring.dao.PlateRepository;
import com.example.loginspring.entity.Plate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PlateService {
    @Autowired
    private PlateRepository plateRepository;

    public List<Plate> getAllPlates(){
        return plateRepository.findAll();
    }

    public Optional<Plate> getPlateByTargetAndFlagAndPredict(String target, String flag, String predict) {
        return plateRepository.findByTargetAndFlagAndPredict(target, flag, predict);
    }

    public Plate savePlate(Plate plate) {
        return plateRepository.save(plate);
    }
}
