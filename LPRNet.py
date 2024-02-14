import torch.nn as nn
import torch
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        # 调用父类的构造函数
        super(small_basic_block, self).__init__()
        # 定义一个序列模型，包含多个卷积层和激活函数
        self.block = nn.Sequential(
            # 第一层卷积，输入通道数为ch_in，输出通道数为ch_out的四分之一，卷积核大小为1x1
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            # 激活函数ReLU
            nn.ReLU(),
            # 第二层卷积，输入通道数为ch_out的四分之一，输出通道数仍为ch_out的四分之一，卷积核大小为3x1，填充为(1,0)
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            # 激活函数ReLU
            nn.ReLU(),
            # 第三层卷积，输入通道数为ch_out的四分之一，输出通道数仍为ch_out的四分之一，卷积核大小为1x3，填充为(0,1)
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            # 激活函数ReLU
            nn.ReLU(),
            # 第四层卷积，输入通道数为ch_out的四分之一，输出通道数为ch_out，卷积核大小为1x1
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        # 调用self.block函数，将输入x作为参数传入，并返回其结果
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        '''
        初始化 (__init__ 方法):
        lpr_max_len: 车牌的最大长度。
        phase: 模型的训练或测试阶段。
        class_num: 分类的类别数，可能是车牌中字符的数量。
        dropout_rate: dropout层的概率。
        '''
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        '''
        Backbone网络:主干网络
        该网络首先使用一个2D卷积层对输入图像进行特征提取。
        然后是一个批归一化层和ReLU激活函数。
        接下来是一个3D最大池化层，用于下采样特征图。
        之后是三个连续的小基本块，每个小基本块都包含多个卷积层和ReLU激活函数。这些小基本块用于进一步提取特征。
        之后是另一个批归一化层、ReLU激活函数和另一个3D最大池化层。
        然后是一个dropout层，用于防止过拟合。
        之后是一个1x4的卷积层，可能是为了进一步降维或提取特征。
        接着是另一个批归一化层、ReLU激活函数、dropout层和一个分类用的卷积层。
        BatchNorm2d--批归一化层，对输入进行归一化。
        BatchNorm2d还有一些重要的参数，如eps（稳定系数，防止分母出现0，默认为1e-5）、momentum（用于计算running mean
            和running var的动量，默认为0.1）、affine（当设为True时，会给定可以学习的系数矩阵gamma和beta）和track_running_stats
            （当设为True时，会计算并保存running mean和running var，用于测试或评估模式，默认为True）。
            总的来说，BatchNorm2d是一种有效的深度学习技术，可以提高模型的训练速度和稳定性，减少过拟合，并改善模型的性能。
        '''
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),    # 0  [bs,3,24,94] -> [bs,64,22,92]  1
            nn.BatchNorm2d(num_features=64),                                       # 1  -> [bs,64,22,92]               2
            nn.ReLU(),                                                             # 2  -> [bs,64,22,92]
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),                 # 3  -> [bs,64,20,90]
            small_basic_block(ch_in=64, ch_out=128),                               # 4  -> [bs,128,20,90]
            nn.BatchNorm2d(num_features=128),                                      # 5  -> [bs,128,20,90]               3
            nn.ReLU(),                                                             # 6  -> [bs,128,20,90]
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),                 # 7  -> [bs,64,18,44]
            small_basic_block(ch_in=64, ch_out=256),                               # 8  -> [bs,256,18,44]
            nn.BatchNorm2d(num_features=256),                                      # 9  -> [bs,256,18,44]               4
            nn.ReLU(),                                                             # 10 -> [bs,256,18,44]
            small_basic_block(ch_in=256, ch_out=256),                              # 11 -> [bs,256,18,44]
            nn.BatchNorm2d(num_features=256),                                      # 12 -> [bs,256,18,44]               5
            nn.ReLU(),                                                             # 13 -> [bs,256,18,44]
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),                 # 14 -> [bs,64,16,21]
            nn.Dropout(dropout_rate),  # 0.5 dropout rate                          # 15 -> [bs,64,16,21]                6
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),   # 16 -> [bs,256,16,18]         7
            nn.BatchNorm2d(num_features=256),                                            # 17 -> [bs,256,16,18]         8
            nn.ReLU(),                                                                   # 18 -> [bs,256,16,18]
            nn.Dropout(dropout_rate),  # 0.5 dropout rate                                  19 -> [bs,256,16,18]         9
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # class_num=68  20  -> [bs,68,4,18]         10
            nn.BatchNorm2d(num_features=class_num),                                             # 21 -> [bs,68,4,18]    11
            nn.ReLU(),                                                                          # 22 -> [bs,68,4,18]
        )
        '''
        nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)):
        这是一个二维卷积层。
        in_channels=448+self.class_num: 输入通道数是448加上分类数self.class_num。
        out_channels=self.class_num: 输出通道数与分类数相同。
        kernel_size=(1, 1): 卷积核的大小是1x1。
        stride=(1, 1): 步长也是1x1，这意味着卷积操作不会改变特征图的尺寸。
        注释掉的部分是其他两个层，分别是批归一化层(nn.BatchNorm2d)和ReLU激活函数。这些层被注释掉了，意味着在当前版本中它们不会被使用。
        总的来说，self.container这个模块似乎是为了对前面网络提取的特征进行进一步的分类或处理，但由于某些层被注释掉了，所以具体的功能不完全清晰。
        根据这个代码片段，这个模块将输入的特征图通过一个1x1的卷积层，然后将结果输出到与分类数相同数量的通道上。
        '''
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(num_features=self.class_num),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
        #     nn.ReLU(),
        )
        # self.connected = nn.Sequential(
        #    nn.Linear(class_num * 88, 128),
        #   nn.ReLU(),
        # )
        
    '''
        这段代码定义了一个神经网络模型的前向传播过程。下面是对这段代码的详细解释：
        定义forward函数:
            是模型的前向传播函数，它描述了数据通过网络时的计算过程。
        保存重要特征:
            keep_features列表用于保存网络中的重要特征。
            通过遍历self.backbone的子层，将数据x传递给每个子层，并在特定的层（索引为2, 6, 13, 22）处保存特征。
        计算全局上下文:
            global_context列表用于保存处理后的特征。
            对保存的特征进行进一步的计算以提取全局上下文信息。对于前两个特征（索引为0和1），使用5x5的平均池化层；对于第三个特征（索引为2），使用(4,10)的平均池化层。
            然后计算特征的平方，取平均值，再将该特征除以该平均值以进行归一化。这些处理后的特征被添加到global_context列表中。
        拼接特征:
            使用torch.cat(global_context, 1)将所有处理后的特征在第二个维度上拼接起来。
        通过最后的容器网络:
            将拼接后的特征传递给self.container网络。
        计算logits:
            logits是模型输出的原始结果，它是拼接后特征的均值沿第三个维度。
        返回结果:
            返回计算得到的logits。
        简而言之，这段代码定义了一个神经网络的前向传播过程，该网络首先通过一个名为self.backbone的骨干网络提取特征，
        然后在特定的层保存这些特征以进行全局上下文信息的提取，最后将提取到的特征拼接并传递给一个名为self.container的容器网络，并返回logits作为输出。
        '''
    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = list()
        # keep_features: [bs,64,22,92]  [bs,128,20,90] [bs,256,18,44] [bs,68,4,18]
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                # [bs,64,22,92] -> [bs,64,4,18]
                # [bs,128,20,90] -> [bs,128,4,18]
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                # [bs,256,18,44] -> [bs,256,4,18]
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)

            # 没看懂这是在干嘛？有上面的avg提取上下文信息不久可以了？
            f_pow = torch.pow(f, 2)     # [bs,64,4,18]  所有元素求平方
            f_mean = torch.mean(f_pow)  # 1 所有元素求平均
            f = torch.div(f, f_mean)    # [bs,64,4,18]  所有元素除以这个均值
            global_context.append(f)

        x = torch.cat(global_context, 1)  # [bs,516,4,18]
        x = self.container(x)  # -> [bs, 68, 4, 18]   head头
        logits = torch.mean(x, dim=2)  # -> [bs, 68, 18]  # 68 字符类别数   18字符序列长度
        #68代表序列中每个位置字符为相应类别的概率，车牌字符共有68类，18代表字符序列长度

        return logits

def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    '''
    函数定义:
    lpr_max_len: 定义了LPR（License Plate Recognition）的最大长度，默认值为8。
    phase: 用于指示模型应该处于训练模式还是评估模式，默认为False。
    class_num: 分类的数量，默认值为66。
    dropout_rate: dropout率，用于防止过拟合，默认值为0.5。
    创建LPRNet实例:
        使用给定的参数创建一个 LPRNet 类的实例，并将其赋值给变量 Net。
    判断phase参数:
        如果 phase 参数的值为 "train"，则返回训练模式下的 Net 实例。
        否则，返回评估模式下的 Net 实例。
    '''
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
"""     return Net
    
if __name__ == "__main__":
    from torchsummary import summary
    model = build_lprnet(68,0.5)
    summary(model, (3,24,94), device="cpu")

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 22, 92]           1,792
       BatchNorm2d-2           [-1, 64, 22, 92]             128
              ReLU-3           [-1, 64, 22, 92]               0
         MaxPool3d-4           [-1, 64, 20, 90]               0
            Conv2d-5           [-1, 32, 20, 90]           2,080
              ReLU-6           [-1, 32, 20, 90]               0
            Conv2d-7           [-1, 32, 20, 90]           3,104
              ReLU-8           [-1, 32, 20, 90]               0
            Conv2d-9           [-1, 32, 20, 90]           3,104
             ReLU-10           [-1, 32, 20, 90]               0
           Conv2d-11          [-1, 128, 20, 90]           4,224
small_basic_block-12          [-1, 128, 20, 90]               0
      BatchNorm2d-13          [-1, 128, 20, 90]             256
             ReLU-14          [-1, 128, 20, 90]               0
        MaxPool3d-15           [-1, 64, 18, 44]               0
           Conv2d-16           [-1, 64, 18, 44]           4,160
             ReLU-17           [-1, 64, 18, 44]               0
           Conv2d-18           [-1, 64, 18, 44]          12,352
             ReLU-19           [-1, 64, 18, 44]               0
           Conv2d-20           [-1, 64, 18, 44]          12,352
             ReLU-21           [-1, 64, 18, 44]               0
           Conv2d-22          [-1, 256, 18, 44]          16,640
small_basic_block-23          [-1, 256, 18, 44]               0
      BatchNorm2d-24          [-1, 256, 18, 44]             512
             ReLU-25          [-1, 256, 18, 44]               0
           Conv2d-26           [-1, 64, 18, 44]          16,448
             ReLU-27           [-1, 64, 18, 44]               0
           Conv2d-28           [-1, 64, 18, 44]          12,352
             ReLU-29           [-1, 64, 18, 44]               0
           Conv2d-30           [-1, 64, 18, 44]          12,352
             ReLU-31           [-1, 64, 18, 44]               0
           Conv2d-32          [-1, 256, 18, 44]          16,640
small_basic_block-33          [-1, 256, 18, 44]               0
      BatchNorm2d-34          [-1, 256, 18, 44]             512
             ReLU-35          [-1, 256, 18, 44]               0
        MaxPool3d-36           [-1, 64, 16, 21]               0
          Dropout-37           [-1, 64, 16, 21]               0
           Conv2d-38          [-1, 256, 16, 18]          65,792
      BatchNorm2d-39          [-1, 256, 16, 18]             512
             ReLU-40          [-1, 256, 16, 18]               0
          Dropout-41          [-1, 256, 16, 18]               0
           Conv2d-42            [-1, 66, 4, 18]         219,714
      BatchNorm2d-43            [-1, 66, 4, 18]             132
             ReLU-44            [-1, 66, 4, 18]               0
           Conv2d-45            [-1, 66, 4, 18]          33,990
================================================================
Total params: 439,148
Trainable params: 439,148
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 33.64
Params size (MB): 1.68
Estimated Total Size (MB): 35.34
---------------------------------------------------------------- """