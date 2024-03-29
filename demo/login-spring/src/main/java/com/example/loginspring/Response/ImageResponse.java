package com.example.loginspring.Response;

public class ImageResponse {
    private String result;
    public byte[] image;

    public ImageResponse(String result, byte[] image) {
        this.result = result;
        this.image = image;
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public byte[] getImage() {
        return image;
    }

    public void setImage(byte[] image) {
        this.image = image;
    }
}
