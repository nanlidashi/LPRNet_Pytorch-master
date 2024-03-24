package com.example.loginspring.entity;

import javax.persistence.*;

@Entity
@Table(name = "plate_info")
public class Plate {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "ID")
    private Long id;

    @Column(name = "Target", nullable = false)
    private String target;

    @Column(name = "Flag", nullable = false)
    private String flag;
    @Column(name = "Predict", nullable = false)
    private String predict;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTarget() {
        return target;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public String getFlag() {
        return flag;
    }

    public void setFlag(String flag) {
        this.flag = flag;
    }

    public String getPredict() {
        return predict;
    }

    public void setPredict(String predict) {
        this.predict = predict;
    }
}
