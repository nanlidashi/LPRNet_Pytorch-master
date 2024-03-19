package com.example.services;

// src/main/java/com/example/demo/services/LPRNetService.java

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.beans.factory.annotation.Value;

import cv2.*;
import numpy as np;
import torch;
import torch.autograd.Variable;
import argparse;
import os;
import os.path as osp;
import sys;
from PIL import Image, ImageDraw, ImageFont;
from LPRNet import build_lprnet;
from load_data import CHARS,  LPRDataLoader;

@Service
public class LPRNetService {

    @Value("${lprnet.pretrained_model}")
    private String pretrainedModelPath;

    public String recognizePlate(MultipartFile imageFile) {
        try {
            // 将 MultipartFile 转换为 OpenCV 图像
            byte[] imageData = imageFile.getBytes();
            Mat mat = Imgcodecs.imdecode(new MatOfByte(imageData), Imgcodecs.IMREAD_COLOR);

            // 调用 LPRNet 进行车牌识别
            String plateNumber = testSingleImage(mat);

            return plateNumber;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private String testSingleImage(Mat image) {
        try {
            // 解析参数
            // args = get_parser()
            int imgHeight = 94;
            int imgWidth = 24;
            int lprMaxLen = 8;
            boolean phaseTrain = false;
            int numWorkers = 8;
            boolean useCuda = false;
            String testImgDirs = "data/one";

            // 创建 LPRNet 模型
            LPRNet lprnet = buildLPRNet(lprMaxLen, phaseTrain, CHARS.size(), 0);
            System.out.println("Successful to build network!");

            // 加载预训练模型
            lprnet.loadStateDict(torch.load(pretrainedModelPath, torch.device("cpu")));
            System.out.println("Load pretrained model successful!");

            // 构建数据集
            LPRDataLoader testDataset = new LPRDataLoader(testImgDirs.split(','), new int[]{imgHeight, imgWidth}, lprMaxLen);

            // 调用 Greedy_Decode_Eval 进行识别
            return greedyDecodeEval(lprnet, testDataset);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private String greedyDecodeEval(LPRNet net, LPRDataLoader datasets) {
        try {
            // 获取批次数据
            DataLoader loader = new DataLoader(datasets, true, numWorkers);
            Iterator batchIterator = loader.iterator();
            Map.Entry<String, String> next = (Map.Entry<String, String>) batchIterator.next();
            Mat images = next.getKey();
            String labels = next.getValue();
            int[] lengths = labels.length();

            // 进行推断
            MatOfByte mob = new MatOfByte();
            Imgcodecs.imencode(".jpg", images, mob);
            byte[] byteArray = mob.toArray();
            byte[] data = byteArray.clone();
            Mat image = Imgcodecs.imdecode(new MatOfByte(data), Imgcodecs.IMREAD_COLOR);
            image.convertTo(image, CvType.CV_32FC1);
            Mat imageTensor = new Mat();
            Core.divide(image, Scalar.all(255), imageTensor);
            Tensor tensor = Tensor.fromBlob(imageTensor, new long[]{1, imageTensor.size(0), imageTensor.size(1), imageTensor.size(2)});

            // 转换为 Variable
            Variable imageVar = Variable(tensor);

            // 将 Variable 移动到 GPU 上
            if (useCuda) {
                imageVar = imageVar.cuda();
            }

            // 进行预测
            Tensor prebs = net.forward(imageVar);
            float[][] prebsArray = prebs.getDataAsFloatArray();

            // 处理预测结果
            String plateNumber = "";
            for (int i = 0; i < prebsArray.length; i++) {
                float[] preb = prebsArray[i];
                for (int j = 0; j < preb.length; j++) {
                    int index = np.argmax(preb);
                    plateNumber += CHARS.get(index);
                }
            }
            return plateNumber;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
