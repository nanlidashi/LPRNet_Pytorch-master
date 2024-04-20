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
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),    # 0  [bs,3,24,94] -> [bs,64,22,92]  1
            nn.BatchNorm2d(num_features=64),                                       # 1  -> [bs,64,22,92]               
            nn.ReLU(),                                                             # 2  -> [bs,64,22,92]
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
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)   
            f_mean = torch.mean(f_pow)  
            f = torch.div(f, f_mean)    
            global_context.append(f)
        x = torch.cat(global_context, 1)
        x = self.container(x)  
        logits = torch.mean(x, dim=2)  
        return logits

def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
   
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
    
