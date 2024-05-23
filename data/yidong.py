import os  
import shutil  
#从源文件夹移动到目标文件夹下
# 定义源文件夹和目标文件夹  
src_dir = "D:\Python\Vehicle identification\CCPD2020\ccpd_green\ccpd_train"  
dst_dir = "D:\Python\LPRNet_Pytorch-master\workspace\ccpd2020_test"  
  
# 检查目标文件夹是否存在，不存在则创建  
if not os.path.exists(dst_dir):  
    os.makedirs(dst_dir)  
  
# 读取val.txt文件中的内容  
with open("D:\BaiduNetdiskDownload\CCPD2019\splits\\ccpd_blur.txt", "r") as f:  
    file_names = f.readlines()  
  
# 遍历文件名列表，将照片从ccpd-base移动到test文件夹  
for name in file_names:  
    src = os.path.join(src_dir, name.strip())  # 获取完整源文件路径  
    dst = os.path.join(dst_dir, name.strip())  # 获取完整目标文件路径  
      
    # 检查源文件是否存在，存在则移动到目标文件夹  
    if os.path.exists(src):  
        shutil.move(src, dst)  
        print(f"Moved {src} to {dst}")  
    else:  
        print(f"File {src} does not exist!")