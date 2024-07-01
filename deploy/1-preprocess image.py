# %%
from PIL import Image
import os
from collections import Counter
import time

def get_image_info(directory):
    image_info = []
    width_distribution = Counter()
    height_distribution = Counter()
    resolution_distribution = Counter()

    max_resolution = 0
    min_resolution = float('inf')
    max_res_file = ""
    min_res_file = ""

    max_width = 0
    min_width = float('inf')
    max_width_file = ""
    min_width_file = ""

    max_height = 0
    min_height = float('inf')
    max_height_file = ""
    min_height_file = ""

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                width, height = img.size
                resolution = img.width * img.height

                # 更新宽度统计
                if width > max_width:
                    max_width = width
                    max_width_file = filename
                if width < min_width:
                    min_width = width
                    min_width_file = filename

                # 更新高度统计
                if height > max_height:
                    max_height = height
                    max_height_file = filename
                if height < min_height:
                    min_height = height
                    min_height_file = filename

                # 更新分辨率统计
                if resolution > max_resolution:
                    max_resolution = resolution
                    max_res_file = filename
                
                if resolution < min_resolution:
                    min_resolution = resolution
                    min_res_file = filename

                # 分布统计
                width_distribution[width // 100 * 100] += 1
                height_distribution[height // 100 * 100] += 1
                resolution_distribution[resolution // 50000 * 50000] += 1

                # 收集图片信息
                info = {
                    'filename': filename,
                    'width': width,
                    'height': height,
                    'resolution': resolution
                }
                image_info.append(info)

    # 打印统计数据
    print("Width Distribution (by 100 pixels):")
    for w, count in sorted(width_distribution.items()):
        print(f"{w} to {w+99} pixels: {count} images")
    print(f"Max Width: {max_width} pixels (File: {max_width_file}), Min Width: {min_width} pixels (File: {min_width_file})")

    print("Height Distribution (by 100 pixels):")
    for h, count in sorted(height_distribution.items()):
        print(f"{h} to {h+99} pixels: {count} images")
    print(f"Max Height: {max_height} pixels (File: {max_height_file}), Min Height: {min_height} pixels (File: {min_height_file})")

    print("Resolution Distribution (by 1 MP):")
    for r, count in sorted(resolution_distribution.items()):
        print(f"{r} to {r+49999} pixels: {count} images")
    print(f"Max Resolution: {max_resolution} pixels (File: {max_res_file}), Min Resolution: {min_resolution} pixels (File: {min_res_file})")
    
    return image_info




# 使用示例
directory = '../data/source/trainning set/png'
image_info = get_image_info(directory)
image_info

# %%
from PIL import Image, ImageOps
import os

def pad_images(directory, image_info, save_directory):
    idx = 0
    for info in image_info:
        path = os.path.join(directory, info['filename'])
        idx += 1
        if idx % 10 == 0:
            print(f"Padding images: {idx}/{len(image_info)} finished")
        with Image.open(path) as img:
            padding_width = int(img.width * 0.05)
            padding_height = int(img.height * 0.05)
            padded_img = ImageOps.expand(img, border=(padding_width, padding_height), fill='white')
            # 保存填充后的图片
            save_path = os.path.join(save_directory, f"padded_{info['filename']}")
            padded_img.save(save_path)
            
            
def resize_images(directory, target_height=256, save_directory='../data/resized'):
    file_list = [f for f in os.listdir(directory) if f.startswith('padded_')]
    idx = 0
    for filename in file_list:
        idx += 1
        if idx % 10 == 0:
            print(f"Resizing images: {idx}/{len(file_list)} finished")
        path = os.path.join(directory, filename)
        with Image.open(path) as img:
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            resized_img = img.resize((new_width, target_height), Image.ANTIALIAS)
            save_path = os.path.join(save_directory, f"{filename[len('padded_'):]}")
            resized_img.save(save_path)


# %%
import os
import cv2
import numpy as np

def add_gaussian_noise_and_save_batch(input_directory, output_directory, mean=0, var=0.01):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    total_images = len(image_files)
    
    idx = 0
    for image_file in image_files:
        path = os.path.join(input_directory, image_file)
        idx += 1
        if idx % 10 == 0 or idx == total_images:
            print(f"Processing images: {idx}/{total_images} finished")
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图像高度为128
        if image.shape[0] != 128:
            raise ValueError(f"Image {image_file} height must be 128 pixels")
        
        # 计算标准差
        sigma = var ** 0.5
        
        # 生成高斯噪声
        gauss = np.random.normal(mean, sigma, image.shape)
        
        # 添加噪声到图像
        noisy_image = image + gauss
        
        # 裁剪像素值到有效范围
        noisy_image = np.clip(noisy_image, 0, 255)
        
        # 转换为无符号8位整数类型
        noisy_image = noisy_image.astype(np.uint8)
        
        # 保存图像
        save_path = os.path.join(output_directory, f"{image_file}")
        cv2.imwrite(save_path, noisy_image)

# 示例使用
# add_gaussian_noise_and_save_batch('input_folder', 'output_folder')

# %%
import os
import cv2
import numpy as np

def sharpen_image_and_save_batch(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    total_images = len(image_files)
    
    idx = 0
    for image_file in image_files:
        path = os.path.join(input_directory, image_file)
        idx += 1
        if idx % 10 == 0 or idx == total_images:
            print(f"Processing images: {idx}/{total_images} finished")
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图像高度为128
        if image.shape[0] != 128:
            raise ValueError(f"Image {image_file} height must be 128 pixels")
        
        # 创建锐化内核
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
        
        # 应用锐化内核
        sharpened_image = cv2.filter2D(image, -1, kernel)
        
        # 保存图像
        save_path = os.path.join(output_directory, f"{image_file}")
        cv2.imwrite(save_path, sharpened_image)

# 示例使用
# sharpen_image_and_save_batch('input_folder', 'output_folder')

# %%
def main():
    start_time = time.time()
    source_directory = '../data/source/trainning set/png'
    padded_directory = '../data/padding'
    resized_directory = '../data/resized'
    # gaussian_noise_directory = '../data/gauss'
    sharpen_directory = '../data/sharpen'
    # sharpen_gauss_disrectory = '../data/gauss_sharpen'
    
    # 确保保存目录存在
    os.makedirs(padded_directory, exist_ok=True)
    os.makedirs(resized_directory, exist_ok=True)
    os.makedirs(sharpen_directory, exist_ok=True)
    # os.makedirs(gaussian_noise_directory, exist_ok=True)
    
    # 获取图片信息
    image_info = get_image_info(source_directory)
    print(f"Image information collection completed. Time elapsed: {time.time() - start_time} seconds")
    
    # 应用padding并保存
    pad_images(source_directory, image_info, padded_directory)
    
    print(f"Padding completed. Time elapsed: {time.time() - start_time} seconds")
    
    # 从保存的padding图片读取，调整大小并保存
    resize_images(padded_directory, 128, resized_directory)
    
    print(f"Resizing completed. Total time elapsed: {time.time() - start_time} seconds")
    
    # add_gaussian_noise_and_save_batch(resized_directory, gaussian_noise_directory)
    # add_gaussian_noise_and_save_batch(sharpen_directory, sharpen_gauss_disrectory)
    
    print(f"Gaussian noise addition completed. Total time elapsed: {time.time() - start_time} seconds")
    
    sharpen_image_and_save_batch(resized_directory, sharpen_directory)
    
    print("All processes are completed.")

main()


# %%
directory = '../data/resized'
image_info = get_image_info(directory)
image_info

# %%



