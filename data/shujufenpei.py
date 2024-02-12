import os  
import random  
import shutil  
  
def split_dataset(base_folder, output_folder, split_ratio=(7, 2, 1)):  
    # 确保输出文件夹存在  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    # 获取base文件夹中的所有照片文件  
    photo_files = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f))]  
  
    # 计算每个数据集应包含的照片数量  
    total_photos = len(photo_files)  
    train_photos = int(total_photos * split_ratio[0] / sum(split_ratio))  
    test_photos = int(total_photos * split_ratio[1] / sum(split_ratio))  
    val_photos = total_photos - train_photos - test_photos  
  
    # 创建三个数据集文件夹  
    train_folder = os.path.join(output_folder, 'ccpd2019_train')  
    test_folder = os.path.join(output_folder, 'ccpd2019_test')  
    val_folder = os.path.join(output_folder, 'ccpd2019_val')  
    if not os.path.exists(train_folder):  
        os.makedirs(train_folder)  
    if not os.path.exists(test_folder):  
        os.makedirs(test_folder)  
    if not os.path.exists(val_folder):  
        os.makedirs(val_folder)  
  
    # 随机分配照片到各个数据集文件夹中  
    random.shuffle(photo_files)  # 先打乱照片文件列表  
    train_index = 0  
    test_index = 0  
    val_index = 0  
    for photo in photo_files:  
        if train_index < train_photos:  
            shutil.copy(os.path.join(base_folder, photo), train_folder)  # 复制照片到train文件夹中  
            train_index += 1  
        elif test_index < test_photos:  
            shutil.copy(os.path.join(base_folder, photo), test_folder)  # 复制照片到test文件夹中  
            test_index += 1  
        else:  
            shutil.copy(os.path.join(base_folder, photo), val_folder)  # 复制照片到val文件夹中  
            val_index += 1  
  
    print(f"Train set: {train_photos} photos")  
    print(f"Test set: {test_photos} photos")  
    print(f"Val set: {val_photos} photos")  
  
# 使用示例：将base文件夹中的照片随机分配到train、test和val三个数据集文件夹中，比例为7：2：1。  
split_dataset("workspace\\ccpd_blur", "workspace")