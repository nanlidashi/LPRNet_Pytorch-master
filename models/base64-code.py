""" import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        # 读取图像文件的内容
        img_data = img_file.read()
        # 使用 base64 编码
        base64_encoded = base64.b64encode(img_data).decode('utf-8')
    return base64_encoded

def save_base64_to_file(base64_str, output_file):
    with open(output_file, "w") as f:
        f.write(base64_str)

if __name__ == "__main__":
    # 图像文件路径
    image_path = "data\\two\沪C3A788.jpg"
    # 调用函数将图像转换为 base64 编码
    base64_str = image_to_base64(image_path)
    # 将 base64 编码保存到文件中
    output_file = "data/base64.txt"
    save_base64_to_file(base64_str, output_file)
    print("Base64 Encoded String saved to:", output_file)
 """

import base64
import os

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        # 读取图像文件的内容
        image_data = img_file.read()
        # 将图像内容转换为Base64编码
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return base64_data

# 图像文件路径
image_path = "data\\one\宁ASE106.jpg"
# 获取图像文件名（不包括文件扩展名）
image_name = os.path.splitext(image_path)[0]

# 将图像转换为Base64编码
base64_string = image_to_base64(image_path)

# 将Base64编码保存到文件，文件名为原始图像名加上后缀 '_base64.txt'
output_file = image_name + ".txt"
with open(output_file, "w") as txt_file:
    txt_file.write(base64_string)



