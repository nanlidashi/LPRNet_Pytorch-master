import base64
import os
import sys

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        # 读取图像文件的内容
        image_data = img_file.read()
        # 将图像内容转换为Base64编码
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return base64_data

# 图像文件路径
image_path = "data\one\宁ASE106.jpg"
# 获取图像文件名（不包括文件扩展名）
image_name = os.path.splitext(image_path)[0]

# 将图像转换为Base64编码
base64_string = image_to_base64(image_path)

# 将Base64编码保存到文件，
output_file = image_name + ".txt"
with open(output_file, "w") as txt_file:
    txt_file.write(base64_string)
print(output_file)



