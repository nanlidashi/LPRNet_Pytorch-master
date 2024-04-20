# -*- coding: utf-8 -*-
# /usr/bin/env/python3

from torch.utils.data import DataLoader
from LPRNet import LPRNet
from load_data import LPRDataLoader, CHARS_DICT,  CHARS

# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os
import datetime  
import pandas as pd  


def sparse_tuple_for_ctc(T_length, lengths):
    # 根据输入的长度列表和目标长度列表，生成相应的输入长度元组和目标长度元组。
    # 初始化输入长度列表和目标长度列表
    input_lengths = []
    target_lengths = []

    # 遍历长度列表
    for ch in lengths:
        # 将输入长度列表中的每个元素都设置为T_length
        input_lengths.append(T_length)
        # 将目标长度列表中的每个元素都设置为当前遍历到的长度ch
        target_lengths.append(ch)

    # 返回输入长度元组和目标长度元组
    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    设置学习率  
    """
    # 初始化学习率为0
    lr = 0
    # 遍历学习率调度列表
    for i, e in enumerate(lr_schedule):
        # 如果当前轮次小于学习率调度列表中的值，则计算学习率并跳出循环
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    # 如果学习率仍为0，则将学习率设置为基本学习率
    if lr == 0:
        lr = base_lr
    # 遍历优化器的参数组
    for param_group in optimizer.param_groups:
        # 将参数组的学习率设置为计算得到的学习率
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=200, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default=r"D:\\Python\\LPRNet_Pytorch-master\workspace\\base_train", help='the train images path')
    parser.add_argument('--test_img_dirs', default=r"D:\\Python\\LPRNet_Pytorch-master\workspace\\base_test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.1, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, help='training batch size.')
    parser.add_argument('--test_batch_size', default=128, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=100, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=500, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=500, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[100,120,140,160], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default=r'a\\',help='Location to save checkpoint models')
    parser.add_argument('--pretrained_model', default=r'D:\Python\LPRNet_Pytorch-master\weights\\LPRNet__iteration_78500.pth', help='no pretrain')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    # 存储图像的列表
    imgs = []
    # 存储标签的列表
    labels = []
    # 存储长度的列表
    lengths = []
    # 遍历批次中的每个样本
    for _, sample in enumerate(batch):
        # 获取样本中的图像、标签和长度
        img, label, length = sample
        # 将图像转换为PyTorch张量并添加到列表中
        imgs.append(torch.from_numpy(img))
        # 将标签添加到标签列表中
        labels.extend(label)
        # 将长度添加到长度列表中
        lengths.append(length)
    # 将标签列表转换为NumPy数组，并展平并转换为整数类型
    labels = np.asarray(labels).flatten().astype(int)
    # 将图像列表转换为PyTorch张量堆叠，并返回结果
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    output = []
    out_acc = []
    args = get_parser()

    T_length = 18  # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    # 构建网络
    lprnet = LPRNet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))# 加载预训练模型
        print("load pretrained model successful!")
    else:
        def xavier(param):
            # 使用Xavier初始化方法对参数进行初始化
            nn.init.xavier_uniform(param)

        def weights_init(m):
            # 遍历模型的state_dict中的所有键
            for key in m.state_dict():
                # 如果键的最后一个部分是'weight'
                if key.split('.')[-1] == 'weight':
                    # 如果键中包含'conv'，使用kaiming_normal_初始化方法对权重进行初始化
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    # 如果键中包含'bn'，使用xavier方法对权重进行初始化
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                # 如果键的最后一个部分是'bias'，将偏置初始化为0.01
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01
# 注意：xavier方法需要单独定义，这里只是作为示例使用，实际使用时需要确保xavier方法可用。

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
    #                      momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // args.train_batch_size  # 每个epoch的大小
    max_iter = args.max_epoch * epoch_size # 最大迭代次数

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')  # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size # 开始迭代次数
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1
            acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
            out_acc.append('Epoch:' + repr(epoch) + "||" + str(acc))

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        """ if (iteration + 1) % args.test_interval == 0: # 每隔多少次迭代测试一次
            Greedy_Decode_Eval(lprnet, test_dataset, args) """ 
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths) # 输入长度和标签长度
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward 前馈传播
        logits = lprnet(images) # 输出[bs, 68, 18]
        # 这一行代码将输入的images通过一个名为lprnet的模型，并将输出存储在logits中。从注释中，我们知道logits的形状是[bs, 68, 18]，其中"bs"代表批量大小。
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # permute函数用于改变张量的维度顺序。在这里，它将logits的维度从[bs, 68, 18]变为[18, bs, 68]。这种重排是为了适应CTC损失的计算。
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_softmax函数对张量在指定维度（这里是第2维，即18）进行softmax操作，然后取对数。这通常用于计算多分类问题的log-probabilities。
        # .requires_grad_()是一个PyTorch的特性，表示从这一步开始计算梯度。这是为了之后的反向传播做准备。

        # log_probs = log_probs.detach().requires_grad_()
        # 这一行代码将log_probs与计算图分离，这样它就不会在反向传播时更新梯度。使用.detach()通常是为了在某些情况下避免不必要的梯度计算。
        # print(log_probs.shape)


        # backprop 反向传播
        optimizer.zero_grad() # 梯度清零

        # log_probs: 预测结果 [18, bs, 68]  其中18为序列长度  68为字典数
        # labels: [93]
        # input_lengths:  tuple   example: 000=18  001=18...   每个序列长度
        # target_lengths: tuple   example: 000=7   001=8 ...   每个gt长度
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths) # 计算损失值
        if loss.item() == np.inf:
            continue
        loss.backward() #  反向传播
        optimizer.step() # 梯度更新
        loss_val += loss.item() # 损失值累加
        end_time = time.time()
        if iteration % 20 == 0:
            # 每个周期的迭代次数（epochiter）、总迭代次数（Totel iter）、损失值（Loss）、批次时间（Batch time）和LR（学习率）
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
            output.append('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr)) 
        
    df = pd.DataFrame(output, columns=['Epoch'])
    # 将DataFrame写入Excel文件  
    df.to_excel("output.xlsx", index=False)   
    df1 = pd.DataFrame(out_acc, columns = ['Epoch'])
    df1.to_excel('out_acc.xlsx', index=False)
    # final test
    print("Final test Accuracy:")
    # Greedy_Decode_Eval(lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'lprnet-pretrain.pth')

def Greedy_Decode_Eval(Net, datasets, args): # 贪婪搜索
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0  # 正确预测的数量
    Tn_1 = 0  # 预测长度与真实标签长度不匹配的数量
    Tn_2 = 0  # 预测标签与真实标签不匹配的数量0
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
        targets = np.array([el.numpy() for el in targets]) # 转成numpy数组

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        # images: [bs, 3, 24, 94]
        # prebs:  [bs, 68, 18]
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]  # 对每张图片 [68, 18]
            preb_label = list()
            for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1: # 记录重复字符
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label 去除重复字符和空白字符'-'
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label) # 得到最终的无重复字符和无空白字符的序列
        for i, label in enumerate(preb_labels):# 逐个样本计算准确率
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1  # 完全正确+1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))

    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


    # return " Accuracy: {} [{}:{}]".format(Acc, Tp, (Tp+Tn_1+Tn_2))

    

if __name__ == "__main__":
 
    train()
 