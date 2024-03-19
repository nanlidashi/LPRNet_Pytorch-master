import torch.nn as nn
import torch
from thop import profile
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
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
       
        self.phase = phase
        self.lpr_max_len = lpr_max_len 
        self.class_num = class_num
       
        self.backbone = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),    # 0  [bs,3,24,94] -> [bs,64,22,92]  1
            # 归一化
            nn.BatchNorm2d(num_features=64),                                       # 1  -> [bs,64,22,92]               
            nn.ReLU(),                                                             # 2  -> [bs,64,22,92]
            # 池化层
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),                 # 3  -> [bs,64,20,90]                2
            small_basic_block(ch_in=64, ch_out=128),                               # 4  -> [bs,128,20,90]               3
            nn.BatchNorm2d(num_features=128),                                      # 5  -> [bs,128,20,90]     
            nn.ReLU(),                                                             # 6  -> [bs,128,20,90]
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),                 # 7  -> [bs,64,18,44]                4
            small_basic_block(ch_in=64, ch_out=256),                               # 8  -> [bs,256,18,44]               5
            nn.BatchNorm2d(num_features=256),                                      # 9  -> [bs,256,18,44]     
            nn.ReLU(),                                                             # 10 -> [bs,256,18,44]
            small_basic_block(ch_in=256, ch_out=256),                              # 11 -> [bs,256,18,44]               6
            nn.BatchNorm2d(num_features=256),                                      # 12 -> [bs,256,18,44]
            nn.ReLU(),                                                             # 13 -> [bs,256,18,44]
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),                 # 14 -> [bs,64,16,21]                7
            nn.Dropout(dropout_rate),  # 0.5 dropout rate                          # 15 -> [bs,64,16,21]                8
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),   # 16 -> [bs,256,16,18]         9
            nn.BatchNorm2d(num_features=256),                                            # 17 -> [bs,256,16,18]
            nn.ReLU(),                                                                   # 18 -> [bs,256,16,18]
            nn.Dropout(dropout_rate),  # 0.5 dropout rate                                  19 -> [bs,256,16,18]         10
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # class_num=68  20  -> [bs,68,4,18]         11  
            nn.BatchNorm2d(num_features=class_num),                                             # 21 -> [bs,68,4,18]   
            nn.ReLU(),                                                                          # 22 -> [bs,68,4,18]
        )
        
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),#   23 -> [bs,68,4,18]         12
        
        )
    
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
   
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
    
"""    return Net
    
if __name__ == "__main__":
    # from torchsummary import summary
    # model = build_lprnet(68,0.5)
    # summary(model, (3,24,94), device="cpu")
    # flops, params = profile(model.to(device="cpu"), inputs=(torch.randn(1, 3, 24, 94),))
    # print("参数量：", params)
    # print("FLOPS：", flops)

   

参数量：        439148.0   0.43M
FLOPS：        147223904.0 0.147GFlops
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
总参数： 439,148
可训练参数：439,148
不可训练参数：0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 33.64
Params size (MB): 1.68
Estimated Total Size (MB): 35.34
输入大小 （MB）： 0.03
前向/后向传递大小 （MB）：33.64
参数大小 （MB）： 1.68
预计总大小 （MB）： 35.34
---------------------------------------------------------------- """