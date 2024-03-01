import torch  
from torch.utils.tensorboard import SummaryWriter  
from LPRNet import LPRNet 
from load_data import CHARS
import argparse

import torchvision.models as models

# 实例化LPRNet模型  
def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    args = parser.parse_args()
    return args


args = get_parser()
model = LPRNet(lpr_max_len=args.lpr_max_len,phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)  

# 定义样例数据+网络
data = torch.randn(1, 3, 24, 94).to(device)

  

with SummaryWriter(log_dir='') as sw:  # 实例化 SummaryWriter ,可以自定义数据输出路径
    sw.add_graph(model,data)  # 输出网络结构图
    sw.close()  # 关闭  sw
# 命令 tensorboard --logdir D:\Python\LPRNet_Pytorch-master\runs
    
""" # 导出为onnx格式
torch.onnx.export(
    model,
    data,
    'model.onnx',
    export_params=True,
    opset_version=11,
)
 """
""" Installing collected packages: protobuf, onnx
  Attempting uninstall: protobuf
    Found existing installation: protobuf 3.20.0
    Uninstalling protobuf-3.20.0:
      Successfully uninstalled protobuf-3.20.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.3.1 requires numpy<1.19.0,>=1.16.0, but you have numpy 1.19.0 which is incompatible.
Successfully installed onnx-1.15.0 protobuf-4.25.3 """