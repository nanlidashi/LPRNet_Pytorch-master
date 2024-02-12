from imutils import paths
import numpy as np
import random
import cv2
import os

from torch.utils.data import Dataset

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    '''
        初始化 (__init__ 方法):
        img_dir: 图像的目录路径。
        imgSize: 目标图像的大小。
        lpr_max_len: 车牌的最大长度（可能是指车牌中字符的最大数量）。
        PreprocFun: 图像预处理函数。如果提供了这个参数，它会被设置为预处理函数；否则，默认使用 transform 方法作为预处理函数。
        img_paths: 存储从给定目录中获取的所有图像路径。
        img_size: 目标图像的大小。
        lpr_max_len: 车牌的最大长度。
        '''
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        '''
        获取数据集长度 (__len__ 方法):
        返回数据集的长度，即图像的数量。
        '''
        return len(self.img_paths)

    def __getitem__(self, index):
        '''
        根据给定的索引 index，返回一个图像和对应的标签。
        filename: 图像文件的路径。
        Image: 使用 OpenCV 读取图像。
        如果图像的大小与目标大小不匹配，则调整图像大小。
        使用预处理函数对图像进行处理。
        basename: 文件的基本名（不含路径和扩展名）。
        imgname: 仅包含文件名（不包括扩展名）。
        从 imgname 中提取车牌字符作为标签。
        返回处理后的图像、标签和标签的长度。
        '''
        filename = self.img_paths[index]
        # Image = cv2.imread(filename)
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        """ if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!! """

        return Image, label, len(label)

    def transform(self, img):
        '''
        将图像转换为适合深度学习的格式。具体来说：
        将图像数据类型转换为 float32。
        从像素值中减去 127.5，然后乘以 0.0078125，这是为了将像素值缩放到 [-1,1] 的范围内。
        将图像从原来的形状 (高, 宽, 通道) 转换为 (通道, 高, 宽) 的形状，这是为了与深度学习框架期望的输入形状一致。
        '''
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    """ def check(self, label):
    能源电动车的车牌
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True """
