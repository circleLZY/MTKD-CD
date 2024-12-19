import os
import numpy as np
from PIL import Image
import itertools

# 定义输入文件夹路径的根目录和输出文件夹路径
root_folder = "/nas/datasets/lzy/RS-ChangeDetection/"
best_output_folder = os.path.join(root_folder, "Figures-KD/TEST/Average/Average-new")

# 定义需要处理的模型子文件夹名称
model_folders = [
    "Figures-KD/TEST/Average/Average-77.41",
    "Figures-KD/TEST/Average/Average-77.50",
    "Figures-KD/TEST/Average/Average-77.80",
    "Figures-KD/TEST/Average/Average-77.87",
    # "Figures/TEST/Extended-Selected/Single/Changer-mit-b1/vis_data/vis_image"
    # "Figures-KD/TEST/Three-Teachers/Changer-mit-b1/distill/image/vis_data/vis_image",
    # "Figures-KD/TEST/Three-Teachers/Changer-mit-b0/distill/vis_data/vis_image",
    
    "Figures-KD/TEST/Two-Teachers/Changer-mit-b0/distill/vis_data/vis_image",
    "Figures-KD/TEST/Two-Teachers/Changer-mit-b1/distill/vis_data/vis_image",
    
    "Figures/TEST/Extended/Single/Changer-mit-b0/vis_data/vis_image",
    "Figures/TEST/Extended/Single/Changer-mit-b1/vis_data/vis_image",
    
    # "Figures/TEST/Single/Changer-mit-b0/vis_data/vis_image",
    # "Figures/TEST/Single/Changer-mit-b1/vis_data/vis_image"
]

model_folders = [
    "Figures-KD/TEST/Average/Average-78.09",
    
    "Figures/TEST/Single/Changer-mit-b1/vis_data/vis_image",
    "Figures/TEST/Extended/Single/Changer-mit-b1/vis_data/vis_image",
    "Figures/TEST/Extended-Selected/Single/Changer-mit-b1/vis_data/vis_image",
    "Figures-KD/TEST/Three-Teachers/Changer-mit-b1/distill/image/vis_data/vis_image",
    "Figures-KD/TEST/Two-Teachers/Changer-mit-b1/distill/vis_data/vis_image",
]

# 定义每个文件夹中图片的数量
image_count = 1000  # 根据你的数据集的大小设置
gt_folder = "/nas/datasets/lzy/RS-ChangeDetection/CGWX-Original/test/label"

# 创建输出文件夹
if not os.path.exists(best_output_folder):
    os.makedirs(best_output_folder)

# 全局变量用于累计TP、FP、FN、TN
TP_cnt, FP_cnt, FN_cnt, TN_cnt = 0, 0, 0, 0

# 计算mIOU的函数
def calculate_mIOU(gt_image, result_image):
    global TP_cnt, FP_cnt, FN_cnt, TN_cnt
    gt_array = np.array(gt_image)
    result_array = np.array(result_image)
    
    # 将图片中的255转化为1，表示变化
    gt_array = np.where(gt_array == 255, 1, 0)
    result_array = np.where(result_array == 255, 1, 0)
    
    # 计算TP, FP, FN, TN
    TP = np.sum((gt_array == 1) & (result_array == 1))
    FP = np.sum((gt_array == 0) & (result_array == 1))
    FN = np.sum((gt_array == 1) & (result_array == 0))
    TN = np.sum((gt_array == 0) & (result_array == 0))
    
    TP_cnt += TP
    FP_cnt += FP
    FN_cnt += FN
    TN_cnt += TN
    
    # 防止分母为零的情况
    if TP + FP + FN == 0 or TN + FP + FN == 0:
        return 1
    
    mIOU = 0.5 * TP / (TP + FP + FN) + 0.5 * TN / (TN + FP + FN)
    return mIOU

# 计算所有图片的平均mIOU
def calculate_average_mIOU(gt_folder, result_folder):
    global TP_cnt, FP_cnt, FN_cnt, TN_cnt
    TP_cnt, FP_cnt, FN_cnt, TN_cnt = 0, 0, 0, 0  # 重置全局计数

    gt_images = sorted(os.listdir(gt_folder))
    result_images = sorted(os.listdir(result_folder))
    # 确保GT文件夹和结果文件夹中的文件数量一致
    assert len(gt_images) == len(result_images), "GT文件夹和结果文件夹中的图片数量不一致"

    mIOU_list = []
    
    for gt_image_name, result_image_name in zip(gt_images, result_images):
        gt_image_path = os.path.join(gt_folder, gt_image_name)
        result_image_path = os.path.join(result_folder, result_image_name)
        
        gt_image = Image.open(gt_image_path).convert('L')
        result_image = Image.open(result_image_path).convert('L')
        
        mIOU = calculate_mIOU(gt_image, result_image)
        mIOU_list.append(mIOU)
    
    average_mIOU = np.mean(mIOU_list)
    return average_mIOU

# 逐个图片进行处理
best_mIOU = 0
best_combination = []

# 遍历所有可能的组合
for r in range(1, len(model_folders) + 1):
    for combination in itertools.combinations(model_folders, r):
        # 创建临时文件夹用于保存当前组合的结果图片
        temp_output_folder = "/nas/datasets/lzy/RS-ChangeDetection/Figures/tmp/temp_result_images"
        if not os.path.exists(temp_output_folder):
            os.makedirs(temp_output_folder)

        # 逐个图片进行处理并保存到临时文件夹
        for i in range(1, image_count + 1):
            images = []

            # 遍历组合中的每个模型
            for model in combination:
                vis_image_folder = os.path.join(root_folder, model)
                image_path = os.path.join(vis_image_folder, f"image_{i}.png")
                # 检查图片是否存在
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('L')
                    image_array = np.array(image)
                    images.append(image_array)
                else:
                    print(f"Warning: {image_path} does not exist and will be skipped.")

            # 仅当组合中有图片时进行计算
            if images:
                # 平均组合中的图片
                images_stack = np.stack(images, axis=0)
                mean_image = np.mean(images_stack, axis=0)
                result_image = np.where(mean_image >= 127.5, 255, 0).astype(np.uint8)

                # 保存处理后的图片到临时文件夹
                output_image_path = os.path.join(temp_output_folder, f"image_{i}.png")
                result_image_pil = Image.fromarray(result_image)
                result_image_pil.save(output_image_path)

        # 计算该组合的平均mIOU
        print(combination)
        average_mIOU = calculate_average_mIOU(gt_folder, temp_output_folder)
        print(f"Combination {combination} has mIOU {average_mIOU}")

        # 如果当前组合的mIOU更高，则更新最佳组合
        if average_mIOU > best_mIOU:
            best_mIOU = average_mIOU
            best_combination = combination

        # 删除临时文件夹中的图片
        for f in os.listdir(temp_output_folder):
            os.remove(os.path.join(temp_output_folder, f))

# 保存最佳组合的图片
if best_combination:
    print(f"Best combination is {best_combination} with mIOU {best_mIOU}")
    
    # 逐个图片处理并保存最佳组合结果
    for i in range(1, image_count + 1):
        images = []

        for model in best_combination:
            vis_image_folder = os.path.join(root_folder, model)

            image_path = os.path.join(vis_image_folder, f"image_{i}.png")
            
            # 检查图片是否存在
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('L')
                image_array = np.array(image)
                images.append(image_array)
        
        # 平均图片并保存结果
        if images:
            images_stack = np.stack(images, axis=0)
            mean_image = np.mean(images_stack, axis=0)
            result_image = np.where(mean_image >= 127.5, 255, 0).astype(np.uint8)

            # 保存到最佳组合的文件夹中
            output_image_path = os.path.join(best_output_folder, f"image_{i}.png")
            result_image_pil = Image.fromarray(result_image)
            result_image_pil.save(output_image_path)

else:
    print("No valid combination found.")
