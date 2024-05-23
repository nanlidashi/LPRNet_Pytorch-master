from load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

def get_parser():
    '''
    这段代码定义了一个函数 get_parser()，其目的是使用 argparse 模块从命令行获取参数。argparse 是 Python 的标准库之一，用于编写用户友好的命令行接口。
    以下是参数的详细说明：
    --img_size: 默认值为 [94, 24]，表示输入图像的大小。
    --test_img_dirs: 默认值为 "./data/test"，表示测试图像的路径。
    --dropout_rate: 默认值为 0，表示 dropout 的比率。dropout 是一种防止神经网络过拟合的技术。
    --lpr_max_len: 默认值为 8，表示车牌号码的最大长度。
    --test_batch_size: 默认值为 100，表示测试时的批量大小。
    --phase_train: 默认值为 False，表示当前是训练阶段还是测试阶段。
    --num_workers: 默认值为 8，表示数据加载时使用的 worker 数量。
    --cuda: 默认值为 True，表示是否使用 CUDA（即 GPU）进行训练。
    --show: 默认值为 False，表示是否显示测试图像及其预测结果。
    --pretrained_model: 默认值为 './weights/Final_LPRNet_model.pth'，表示预训练模型的文件路径。
    '''
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default=r"workspace/ccpd_val", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default=r'weights\\lprnet-pretrain.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    """ 这段代码定义了一个名为collate_fn的函数，它用于处理输入的批量数据（batch）并将其整理为适合输入到LPRNet模型的形式。下面是对代码的逐行解释：
def collate_fn(batch):定义一个名为collate_fn的函数，它接受一个参数batch，代表一批数据。
imgs = []初始化一个空列表，用于存储图像数据。
labels = []初始化一个空列表，用于存储标签数据。
lengths = []初始化一个空列表，用于存储每个图像的长度数据。
for _, sample in enumerate(batch):
遍历输入的批量数据。enumerate函数返回每个数据的索引和值。这里使用下划线_表示我们不关心索引值。
img, label, length = sample将当前样本解包为三个部分：图像数据、标签和长度。
imgs.append(torch.from_numpy(img))
将图像数据从NumPy数组转换为PyTorch张量，并添加到imgs列表中。
labels.extend(label)
将标签数据添加到labels列表中。这里使用extend方法是因为每个图像可能有多个标签。
lengths.append(length)
将长度数据添加到lengths列表中。
labels = np.asarray(labels).flatten().astype(np.float32)
将标签列表转换为NumPy数组，然后将其扁平化（如果之前是二维数组），并转换为浮点数类型。
return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
返回整理后的数据：图像数据（已堆叠）、标签数据（从NumPy转换为PyTorch张量）和长度数据。这里使用torch.stack(imgs, 0)将多个图像张量堆叠成一个批量的图像张量，以便可以输入到模型中。
总结：这个函数的主要目的是将输入的批量数据整理为适合输入到LPRNet模型的形式，特别是处理图像和标签数据，并确保它们都是适当的PyTorch张量格式。 """
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

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        #lprnet.load_state_dict(torch.load(args.pretrained_model))
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
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
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            # show image and its predict label
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

def show(img, label, target):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    cv2.imshow("test", img)
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
    '''
    LPRNet__iteration_78500
    Successful to build network!
    load pretrained model successful!
    [Info] Test Accuracy: 0.9425694444444445 [13573:226:601:14400]
    [Info] Test Speed: 0.005979876438561726s 1/14421]
    lprnet-pretrain
    Successful to build network!
    load pretrained model successful!
    [Info] Test Accuracy: 0.9533333333333334 [13728:168:504:14400]
    [Info] Test Speed: 0.0056876137818569995s 1/14421]
    '''