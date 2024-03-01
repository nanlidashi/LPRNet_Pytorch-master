import cv2
import numpy as np
import torch
from torch.autograd import Variable
from LPRNet import build_lprnet
from load_data import CHARS,  LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import *
import argparse
import os

# Assuming you have a function to build the LPRNet model
def get_parser():
    # 创建一个解析器对象，并设置描述信息
    parser = argparse.ArgumentParser(description='parameters to train net')
    # 添加命令行参数--img_size，并设置默认值为[94, 24]，用于指定图像大小
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    # 添加命令行参数--dropout_rate，并设置默认值为0，用于指定dropout率
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    # 添加命令行参数--lpr_max_len，并设置默认值为8，用于指定车牌号码的最大长度
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    # 添加命令行参数--phase_train，并设置默认值为False，类型为bool，用于指定训练或测试阶段
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    # 添加命令行参数--num_workers，并设置默认值为8，类型为int，用于指定数据加载时使用的worker数量
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    # 添加命令行参数--cuda，并设置默认值为False，类型为bool，用于指定是否使用cuda进行模型训练
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    # 添加命令行参数--pretrained_model，并设置默认值为'weights\\lprnet-pretrain.pth'，用于指定预训练模型的路径
    parser.add_argument('--pretrained_model', default=r'weights\\lprnet-pretrain.pth', help='pretrained base model')
    # 添加命令行参数--test_img_dirs，并设置默认值为'data\\one'，用于指定测试图像的路径
    parser.add_argument('--test_img_dirs', default=r"data\\one", help='the test images path')
    # 解析命令行参数并返回结果
    args = parser.parse_args()
    return args

def test_single_image():
    # 加载预训练模型
    # Load the pretrained model
    args = get_parser()

    # 构建LPRNet模型
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    # 指定设备（使用GPU或CPU）
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # 将模型移至指定设备上
    lprnet.to(device)
    print("Successful to build network!")

    # 如果指定了预训练模型
    if args.pretrained_model:
        # 加载预训练模型参数
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    # 获取测试图片的路径列表
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    # 创建LPRDataLoader数据集对象
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    try:
        # 进行解码评估操作
        Greedy_Decode_Eval(lprnet, test_dataset , args)
    finally:
        cv2.destroyAllWindows()




def collate_fn(batch):
    # 图像列表
    imgs = []
    # 标签列表
    labels = []
    # 长度列表
    lengths = []
    for _, sample in enumerate(batch):
        # 获取图像、标签和长度
        img, label, length = sample
        # 将图像转换为PyTorch张量并添加到图像列表中
        imgs.append(torch.from_numpy(img))
        # 将标签添加到标签列表中
        labels.extend(label)
        # 将长度添加到长度列表中
        lengths.append(length)
    # 将标签列表转换为NumPy数组，并展平并转换为float32类型
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def Greedy_Decode_Eval(Net, datasets,  args):
    # 创建一个迭代器，用于从数据集中按批次获取数据
    # TestNet = Net.eval()
    batch_iterator = iter(DataLoader(datasets,  shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    # 获取第一批次的图像、标签和长度
    images, labels, lengths = next(batch_iterator)
    start = 0
    targets = []# 只是预测结果可以删除本行
    # 遍历每个长度，根据长度提取对应的标签，并添加到目标列表中
    for length in lengths:
        label = labels[start:start+length]
        targets.append(label)# 只是预测结果可以删除本行
        start += length
    # 将目标列表转换为NumPy数组
    targets = np.array([el.numpy() for el in targets]) # 只是预测结果可以删除本行
    # 将图像数据转换为NumPy数组并复制一份
    imgs = images.numpy().copy()
    # 根据是否使用CUDA进行判断，将图像数据转换为Variable类型
    if args.cuda:
        images = Variable(images.cuda())
    else:
        images = Variable(images)

    # 将图像输入到网络中，获取预测结果
    prebs = Net(images)
    # 将预测结果从GPU转移到CPU，并转换为NumPy数组
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    # 遍历每个预测结果，获取每个样本的预测标签列表
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        # 如果第一个字符不是空白字符，则将其添加到结果列表中
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        # 遍历每个字符，判断是否重复或为空白字符，如果不是则添加到结果列表中
        for c in preb_label: 
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    # 遍历每个预测标签列表，显示对应的图像、预测标签和真实标签
    for i, label in enumerate(preb_labels):
        show(imgs[i], label, targets[i])
        # show(imgs[i], label) 只是预测结果



def show(img, label, target):
# 只是预测结果可以改为def show(img, label):
    # 将图像从形状为 (高, 宽, 通道数) 转换为 (通道数, 高, 宽)
    img = np.transpose(img, (1, 2, 0))
    # 将图像每个像素值乘以 128
    img *= 128.
    # 将图像每个像素值加上 127.5，以将像素值缩放到 [127.5, 255] 的范围内
    img += 127.5
    # 将图像数据类型转换为 uint8
    img = img.astype(np.uint8)

    lb = ""
    # 将标签转换为对应的字符并拼接成字符串
    for i in label:
        lb += CHARS[i]
    
    tg = ""
    # 将目标转换为对应的字符并拼接成字符串
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    # 只是预测结果的话上述六行代码皆可删除
    # 在图像上添加标签字符并显示图像
    img = cv2ImgAddText(img, lb, (0, 0))
    cv2.imshow("test", img)
    # 打印目标、标记和预测结果
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    # 只是预测结果print("predict: ", lb)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    # 判断输入的图像是否为 OpenCV 格式
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        # 将 OpenCV 格式的图像转换为 PIL 格式
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建 PIL 的 Draw 对象，用于在图像上绘制文本
    draw = ImageDraw.Draw(img)
    # 定义字体样式，这里使用的是 NotoSansCJK-Regular.ttc 字体，大小为 textSize，编码方式为 utf-8
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    # 在指定位置 pos 上绘制文本，颜色为 textColor，字体为 fontText
    draw.text(pos, text, textColor, font=fontText)

    # 将 PIL 格式的图像转换回 OpenCV 格式，并返回
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


    
if __name__ == "__main__":
    test_single_image()
