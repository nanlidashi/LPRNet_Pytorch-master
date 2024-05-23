package com.example.loginspring.repository;

import com.example.loginspring.entity.Plate;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PlateRepository extends JpaRepository<Plate, Long> {
    Optional<Plate> findByTargetAndFlagAndPredictAndBase64(String target, String flag, String predict,String base64);

    List<Plate> findByTargetContaining(String keyword);
}