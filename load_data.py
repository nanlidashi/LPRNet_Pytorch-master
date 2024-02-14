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
    
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        # 初始化函数，接收四个参数：img_dir（图片目录）、imgSize（图片大小）、lpr_max_len（车牌号码最大长度）和PreprocFun（预处理函数）
        self.img_dir = img_dir
        # 初始化属性 img_dir，存储图片目录
        self.img_paths = []
        # 初始化属性 img_paths，存储图片路径列表
        for i in range(len(img_dir)):
            # 遍历 img_dir 列表中的每个元素
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # 将每个子目录下的图片路径添加到 img_paths 列表中
        random.shuffle(self.img_paths)
        # 将 img_paths 列表随机排序
        self.img_size = imgSize
        # 初始化属性 img_size，存储图片大小
        self.lpr_max_len = lpr_max_len
        # 初始化属性 lpr_max_len，存储车牌号码最大长度
        if PreprocFun is not None:
            # 如果 PreprocFun 不为空
            self.PreprocFun = PreprocFun
        else:
            # 如果 PreprocFun 为空，则将 self.transform 赋值给 PreprocFun
            self.PreprocFun = self.transform


    def __len__(self):
        '''
        获取数据集长度 (__len__ 方法):
        返回数据集的长度，即图像的数量。
        '''
        return len(self.img_paths)

    def __getitem__(self, index):
        # 获取指定索引的图片路径
        filename = self.img_paths[index]
        # 读取图片
        # Image = cv2.imread(filename)
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        # 将图片从RGB格式转换为BGR格式
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        # 获取图片的高度、宽度和通道数
        height, width, _ = Image.shape
        # 如果图片的高度或宽度与期望的尺寸不一致，则对图片进行缩放
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        # 对图片进行预处理
        Image = self.PreprocFun(Image)

        # 获取图片的文件名（不包含路径）
        basename = os.path.basename(filename)
        # 分离文件名和后缀
        imgname, suffix = os.path.splitext(basename)
        # 提取文件名中的特定部分，这里假设是"-"前和"_"前的部分
        imgname = imgname.split("-")[0].split("_")[0]
        # 初始化标签列表
        label = list()
        # 将文件名中的每个字符转换为对应的标签并添加到标签列表中
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
