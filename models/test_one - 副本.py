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

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--pretrained_model', default=r'weights\\lprnet-pretrain.pth', help='pretrained base model')
    args = parser.parse_args()
    return args
def test_single_image(img_path):
    args = get_parser()
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
    else:
        #print("[Error] Can't found pretrained mode, please check!")
        return False
    test_img_dirs = os.path.dirname(os.path.abspath(img_path))
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset , args)
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
def Greedy_Decode_Eval(Net, datasets,  args):
    batch_iterator = iter(DataLoader(datasets,  shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
    images, labels, lengths = next(batch_iterator)
    start = 0
    targets = []
    for length in lengths:
        label = labels[start:start+length]
        targets.append(label)
        start += length
    targets = np.array([el.numpy() for el in targets]) 
    if args.cuda:
        images = Variable(images.cuda())
    else:
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
if __name__ == "__main__":
    file_path = "data\one\宁ASE106.txt"
    test_single_image(file_path)
