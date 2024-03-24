import cv2
import numpy as np
import torch
from torch.autograd import Variable
from LPRNet import build_lprnet
from load_data import CHARS,  LPRDataLoader
from torch.utils.data import *
import argparse
import os
import cv2  
import base64
import urllib
import sys
import io

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_path', type=str, help='path to the image file')
  
    args = parser.parse_args()
    return args
def test_single_image(img_path):
    
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    device = torch.device("cuda:0" if False else "cpu")
    lprnet.to(device)
    pretrained_model = r'D:\\Python\\LPRNet_Pytorch-master\weights\\lprnet-pretrain.pth'
    
    lprnet.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
    
    test_img_dirs = os.path.dirname(os.path.abspath(img_path))
    test_dataset = LPRDataLoader(test_img_dirs.split(','), [94, 24], 8)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset)
    finally:
        cv2.destroyAllWindows()
def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
def Greedy_Decode_Eval(Net, datasets):
    batch_iterator = iter(DataLoader(datasets,  shuffle=True, num_workers=8, collate_fn=collate_fn))
    images, labels, lengths = next(batch_iterator)
    start = 0
    targets = []
    for length in lengths:
        label = labels[start:start+length]
        targets.append(label)
        start += length
    targets = np.array([el.numpy() for el in targets]) 
    
    images = Variable(images)
    prebs = Net(images)
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label: 
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    for i, label in enumerate(preb_labels):
        show(label, targets[i])
def show(label, target):
    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]
    flag = "F"
    if lb == tg:
        flag = "T"
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        # 读取图像文件的内容
        image_data = img_file.read()
        # 将图像内容转换为Base64编码
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return base64_data


if __name__ == "__main__":
    # args = get_parser()
    # 图像文件路径
    # image_path = urllib.parse.unquote(args.img_path)
    image_path = urllib.parse.unquote(sys.argv[1])
    # image_path="D:\Python\LPRNet_Pytorch-master\workspace\ccpd2019_base_val\皖AVN699.jpg"
    # 获取图像文件名（不包括文件扩展名）
    image_name = os.path.splitext(image_path)[0]

    # 将图像转换为Base64编码
    base64_string = image_to_base64(image_path)

    # 将Base64编码保存到文件，文件名为原始图像名加上后缀 '_base64.txt'
    output_file = image_name + ".txt"
    with open(output_file, "w") as txt_file:
        txt_file.write(base64_string)
    # 设置字符集为 UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    test_single_image(output_file)
