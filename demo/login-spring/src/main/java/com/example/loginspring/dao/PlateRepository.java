package com.example.loginspring.dao;

import com.example.loginspring.entity.Plate;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface PlateRepository extends JpaRepository<Plate, Long> {
    Optional<Plate> findByTargetAndFlagAndPredict(String target, String flag, String predict);
}