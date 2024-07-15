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



from PIL import Image, ImageOps
import os
import time

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


def main():
    start_time = time.time()
    source_directory = '../data/testBdata/Test_B set/'
    padded_directory = '../data/testBdata/padding'
    resized_directory = '../data/testBdata/resized'
    
    # 确保保存目录存在
    os.makedirs(padded_directory, exist_ok=True)
    os.makedirs(resized_directory, exist_ok=True)
    
    # 获取图片信息
    image_info = get_image_info(source_directory)
    print(f"Image information collection completed. Time elapsed: {time.time() - start_time} seconds")
    
    # 应用padding并保存
    pad_images(source_directory, image_info, padded_directory)
    
    print(f"Padding completed. Time elapsed: {time.time() - start_time} seconds")
    
    # 从保存的padding图片读取，调整大小并保存
    resize_images(padded_directory, 128, resized_directory)
    
    print(f"Resizing completed. Total time elapsed: {time.time() - start_time} seconds")

    print("All processes are completed.")

main()


import json
def make_test_caption(img_path,caption_path):
     # 检查路径是否存在
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path {img_path} does not exist.")
    
    captions = []
    img_files = [f.split(".")[0] for f  in os.listdir(img_path)]
    
    for img_file in img_files:
        # test no caption 
        captions.append([img_file, ". . . ."])

    
    print("Loading data count:",len(captions))
    with open(os.path.join(caption_path,'caption.txt'), 'w') as file:
        for row in captions:
            file.write(f"{row[0]}\t{row[1]}\n")


def make_gray(source_dir, gray_dir,isgray=True):

    

    # 获取所有文件的列表
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    gray_files = files
    
    # 创建目标目录
    os.makedirs(gray_dir, exist_ok=True)
    
    # resize image to small than 32e4 
    def convert_and_save(source_file, dest_file):
        with Image.open(source_file) as img:
            width, height = img.size
     
            # resize every image height to 128
            if True:
                new_width = int(width )
                new_height = int(height )
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            
            if isgray:
                gray_img = img.convert('L')
                gray_img.save(dest_file, format='BMP')
            else:
                img.save(dest_file, format='png')
  
    # 移动文件到相应的目录，并转换为灰度图和BMP格式
    for f in files:
        source_file = os.path.join(source_dir, f)
        if isgray:
            dest_file = os.path.join(gray_dir, f.replace('.png', '.bmp'))
        else:
            dest_file = os.path.join(gray_dir, f)
        convert_and_save(source_file, dest_file)

    
    print(f'Total files: {len(files)}')


make_gray('../data/testBdata/resized','../data/testBdata/gray/img')

make_test_caption('../data/testBdata/gray/img','../data/testBdata/gray')