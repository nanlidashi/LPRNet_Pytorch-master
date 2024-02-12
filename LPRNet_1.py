# An highlighted block
import torch.nn as nn
import torch

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
    def __init__(self,class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            small_basic_block(ch_in=128, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(1, 13), stride=1,padding=[0,6]), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.connected = nn.Sequential(
            nn.Linear(class_num*88,128),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=128+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):


        x = self.backbone(x)
        pattern = x.flatten(1,-1)
        pattern = self.connected(pattern)
        width = x.size()[-1]
        pattern = torch.reshape(pattern,[-1,128,1,1])
        pattern = pattern.repeat(1,1,1,width)
        x = torch.cat([x,pattern],dim=1)
        x = self.container(x)
        logits = x.squeeze(2)


        return logits

def build_lprnet(class_num=66, dropout_rate=0.5):

    Net = LPRNet(class_num, dropout_rate)

    # if phase == "train":
    #     return Net.train()
    # else:
    #     return Net.eval()
    return Net
if __name__ == "__main__":
    from torchsummary import summary
    model = build_lprnet(75,0.5)
    summary(model, (3,24,94), device="cpu")

""" ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 24, 94]           1,792
       BatchNorm2d-2           [-1, 64, 24, 94]             128
              ReLU-3           [-1, 64, 24, 94]               0
         MaxPool2d-4           [-1, 64, 22, 92]               0
            Conv2d-5           [-1, 32, 22, 92]           2,080
              ReLU-6           [-1, 32, 22, 92]               0
            Conv2d-7           [-1, 32, 22, 92]           3,104
              ReLU-8           [-1, 32, 22, 92]               0
            Conv2d-9           [-1, 32, 22, 92]           3,104
             ReLU-10           [-1, 32, 22, 92]               0
           Conv2d-11          [-1, 128, 22, 92]           4,224
small_basic_block-12          [-1, 128, 22, 92]               0
      BatchNorm2d-13          [-1, 128, 22, 92]             256
             ReLU-14          [-1, 128, 22, 92]               0
        MaxPool2d-15          [-1, 128, 10, 90]               0
           Conv2d-16           [-1, 64, 10, 90]           8,256
             ReLU-17           [-1, 64, 10, 90]               0
           Conv2d-18           [-1, 64, 10, 90]          12,352
             ReLU-19           [-1, 64, 10, 90]               0
           Conv2d-20           [-1, 64, 10, 90]          12,352
             ReLU-21           [-1, 64, 10, 90]               0
           Conv2d-22          [-1, 256, 10, 90]          16,640
small_basic_block-23          [-1, 256, 10, 90]               0
      BatchNorm2d-24          [-1, 256, 10, 90]             512
             ReLU-25          [-1, 256, 10, 90]               0
           Conv2d-26           [-1, 64, 10, 90]          16,448
             ReLU-27           [-1, 64, 10, 90]               0
           Conv2d-28           [-1, 64, 10, 90]          12,352
             ReLU-29           [-1, 64, 10, 90]               0
           Conv2d-30           [-1, 64, 10, 90]          12,352
             ReLU-31           [-1, 64, 10, 90]               0
           Conv2d-32          [-1, 256, 10, 90]          16,640
small_basic_block-33          [-1, 256, 10, 90]               0
      BatchNorm2d-34          [-1, 256, 10, 90]             512
             ReLU-35          [-1, 256, 10, 90]               0
        MaxPool2d-36           [-1, 256, 4, 88]               0
          Dropout-37           [-1, 256, 4, 88]               0
           Conv2d-38           [-1, 256, 1, 88]         262,400
      BatchNorm2d-39           [-1, 256, 1, 88]             512
             ReLU-40           [-1, 256, 1, 88]               0
          Dropout-41           [-1, 256, 1, 88]               0
           Conv2d-42            [-1, 75, 1, 88]         249,675
      BatchNorm2d-43            [-1, 75, 1, 88]             150
             ReLU-44            [-1, 75, 1, 88]               0
           Linear-45                  [-1, 128]         844,928
             ReLU-46                  [-1, 128]               0
           Conv2d-47            [-1, 75, 1, 88]          15,300
================================================================
Total params: 1,496,069
Trainable params: 1,496,069
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 37.64
Params size (MB): 5.71
Estimated Total Size (MB): 43.38 """
