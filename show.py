import os
import cv2
import numpy as np

# 定义输入文件夹路径
# folders_row1 = [
#     "satellite", "BAM_satellite", "Trip_satellite"
# ]
# folders_row2 = [
#     "drone", "BAM_drone", "Trip_drone"
# ]
folders_row1 = [
    "scSE_satellite", "CBAM_satellite", "AMFA_satellite"
]
folders_row2 = [
    "scSE_drone", "CBAM_drone", "AMFA_drone"
]


input_root = ""  # 替换为实际路径
output_folder = "./show_scSE_CBAM_AMFA"
os.makedirs(output_folder, exist_ok=True)

# 空间设置
space = 30  # 行与行、列与列之间的像素空间
background_color = (255, 255, 255)  # 白色背景

# 提取所有文件夹中的文件名（不包括后缀）
file_lists = [
    {os.path.splitext(file)[0] for file in os.listdir(os.path.join(input_root, folder))}
    for folder in folders_row1 + folders_row2
]

# 获取文件名相同的图片集合（不包括后缀名）
common_files = set.intersection(*file_lists)

# 遍历所有相同文件名的图片并合成
for file_name in sorted(common_files):
    images_row1 = []
    images_row2 = []

    # 读取第一行的图片
    for folder in folders_row1:
        img_path = os.path.join(input_root, folder, file_name + ".jpeg")  # 假设后缀是 .jpeg，可根据实际情况调整
        if not os.path.exists(img_path):
            img_path = os.path.join(input_root, folder, file_name + ".jpg")  # 尝试其他后缀
        if not os.path.exists(img_path):
            img_path = os.path.join(input_root, folder, file_name + ".png")  # 尝试其他后缀
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image {img_path} not found or could not be read.")
        images_row1.append(img)

    # 读取第二行的图片
    for folder in folders_row2:
        img_path = os.path.join(input_root, folder, file_name + ".jpeg")  # 假设后缀是 .jpeg，可根据实际情况调整
        if not os.path.exists(img_path):
            img_path = os.path.join(input_root, folder, file_name + ".jpg")  # 尝试其他后缀
        if not os.path.exists(img_path):
            img_path = os.path.join(input_root, folder, file_name + ".png")  # 尝试其他后缀
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image {img_path} not found or could not be read.")
        images_row2.append(img)

    # 获取单张图片的高度和宽度
    img_height, img_width = images_row1[0].shape[:2]

    # 创建空白背景用于拼接
    total_width = len(images_row1) * img_width + (len(images_row1) - 1) * space
    total_height = 2 * img_height + space
    canvas = np.full((total_height, total_width, 3), background_color, dtype=np.uint8)

    # 放置第一行图片
    for i, img in enumerate(images_row1):
        start_x = i * (img_width + space)
        canvas[:img_height, start_x:start_x + img_width] = img

    # 放置第二行图片
    for i, img in enumerate(images_row2):
        start_x = i * (img_width + space)
        start_y = img_height + space
        canvas[start_y:start_y + img_height, start_x:start_x + img_width] = img

    # 保存合并后的图片
    output_path = os.path.join(output_folder, file_name + ".jpeg")
    cv2.imwrite(output_path, canvas)
    print(f"已完成类别：{file_name} 的展示")

print("All images have been merged and saved!")
