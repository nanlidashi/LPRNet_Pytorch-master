package com.example.loginspring.repository;

import com.example.loginspring.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
public interface UserRepository extends JpaRepository<User,Integer> {
    User findByUsername(String username);

    User getByUsernameAndPassword(String username,String password);

    boolean existsByUsername(String username);
}
