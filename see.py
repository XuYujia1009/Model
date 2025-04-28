import os
import numpy as np
from PIL import Image

def visualize_labels(input_dir, output_dir, color_mapping):
    """
    将灰度标注文件转换为彩色标注文件并保存。
    
    :param input_dir: 输入标注文件夹路径
    :param output_dir: 输出文件夹路径
    :param color_mapping: 灰度值到RGB颜色的映射字典 {灰度值: (R, G, B)}
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入文件夹中的所有PNG文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.png'):
            # 读取标注文件
            label_path = os.path.join(input_dir, file_name)
            label_img = Image.open(label_path)
            label_array = np.array(label_img)

            # 创建一个彩色图像 (默认背景为黑色)
            color_img = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

            # 根据映射字典将灰度值转换为对应的颜色
            for gray_value, color in color_mapping.items():
                color_img[label_array == gray_value] = color

            # 保存为彩色PNG文件
            output_path = os.path.join(output_dir, file_name)
            Image.fromarray(color_img).save(output_path)
            print(f"已处理: {file_name}")

# 参数设置
input_dir = "result/TestPreLab"  # 原标注文件夹路径
output_dir = "result/TestPreLabCol"  # 处理后文件存放的新文件夹

# 灰度值到RGB颜色的映射: 0 (背景) -> 黑色, 1 -> 红色, 2 -> 绿色, 3 -> 蓝色
color_mapping = {
    0: (0, 0, 0),       # 背景 -> 黑色
    1: (255, 0, 0),     # 类别1 -> 红色
    2: (0, 255, 0),     # 类别2 -> 绿色
    3: (0, 0, 255)      # 类别3 -> 蓝色
}

# 运行函数进行处理
visualize_labels(input_dir, output_dir, color_mapping)
print("所有文件已处理并保存为彩色标注！")


