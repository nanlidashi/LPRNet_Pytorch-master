package com.example.loginspring.Service;

import com.example.loginspring.repository.UserRepository;
import com.example.loginspring.entity.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
@Service
public class UserService {
    @Autowired
    UserRepository userRepository;

    public boolean isExist(String username) {
        User user = getByName(username);
        return null!=user;
    }

    public User getByName(String username) {
        return userRepository.findByUsername(username);
    }

    public User get(String username, String password){
        return userRepository.getByUsernameAndPassword(username, password);
    }
    public User getemail(String email){
        return userRepository.getByEmail(email);
    }

    public void add(User user) {
        userRepository.save(user);
    }

    public User saveUser(User user) {
        return userRepository.save(user);
    }
    public boolean checkPasswordMatch(String password, String cPassword) {
        return password.equals(cPassword);
    }

    public boolean isUsernameExists(String username) {
        return userRepository.existsByUsername(username);
    }

    public boolean isEmailExists(String email) {
        return userRepository.existsByEmail(email);
    }
}