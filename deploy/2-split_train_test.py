# %%
import os
import json

import shutil
import random
from PIL import Image

# %%
def make_caption(json_path,img_path,caption_path):
     # 检查路径是否存在
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON path {json_path} does not exist.")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path {img_path} does not exist.")
    
    captions = []
    files = os.listdir(json_path)
    img_files = [f.split(".")[0] for f  in os.listdir(img_path)]
    
    
    for f in files:
        with open(os.path.join(json_path,f), 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 获取文件名（不包含扩展名）
            filename = os.path.splitext(f)[0]
            # 确保文件名在图像列表中
            if filename in img_files:
                captions.append([filename, data['latex_styled']])
    # load data count
    
    print("Loading data count:",len(captions))
    with open(os.path.join(caption_path,'caption.txt'), 'w') as file:
        for row in captions:
            file.write(f"{row[0]}\t{row[1]}\n")


# %%
# make_caption("../data/source/json","../data/source/png","../data/source/")

# %% [markdown]
# Resolution Distribution (by 1 MP):
# 0 to 999999 pixels: 78 images
# 1000000 to 1999999 pixels: 1244 images
# 2000000 to 2999999 pixels: 2911 images
# 3000000 to 3999999 pixels: 3023 images
# 4000000 to 4999999 pixels: 2473 images
# 5000000 to 5999999 pixels: 1869 images
# 6000000 to 6999999 pixels: 1221 images
# 7000000 to 7999999 pixels: 823 images
# 8000000 to 8999999 pixels: 533 images
# 9000000 to 9999999 pixels: 318 images
# 10000000 to 10999999 pixels: 194 images
# 11000000 to 11999999 pixels: 145 images
# 12000000 to 12999999 pixels: 77 images
# 13000000 to 13999999 pixels: 50 images
# 14000000 to 14999999 pixels: 20 images
# 15000000 to 15999999 pixels: 10 images
# 16000000 to 16999999 pixels: 6 images
# 17000000 to 17999999 pixels: 2 images
# 18000000 to 18999999 pixels: 2 images
# 19000000 to 19999999 pixels: 1 images
# 
# 将高设置为1024，得到分辨率的统计。发现现有的480000的最大pixels，至少要把图片变成256的，极端图片变成128的格式

# %%
def split_train_test(source_dir, train_dir, test_dir, valid_dir,test_size=0.2, seed=7,isgray=True):
    if seed is not None:
        random.seed(seed)
    

    # 获取所有文件的列表
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # 随机化文件列表
    random.shuffle(files)
    
    # 计算测试集的大小
    test_count = int(len(files) * test_size)
    
    # 分配文件到训练集和测试集，和验证集
    validation_files = files[:test_count]
    test_files = files[:test_count]
    # train_files = files[test_count]
    train_files = files
    
    # 创建目标目录
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # resize image to small than 32e4 
    def convert_and_save(source_file, dest_file):
        with Image.open(source_file) as img:
            width, height = img.size
            # 将数据分成两类，由于最大
            # if width * height < 200000 :
            #     new_width = int(width )
            #     new_height = int(height )
            # elif width * height >= 200000 :
            
            # every image height is 128
            if True:
                new_width = int(width )
                new_height = int(height )
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            # # resize 图片大小小于160000
            # if width * height > 160000:
            #     scaling_factor = (160000 / (width * height)) ** 0.5
            #     new_width = int(width * scaling_factor)
            #     new_height = int(height * scaling_factor)
            #     img = img.resize((new_width, new_height), Image.ANTIALIAS)
            if isgray:
                gray_img = img.convert('L')
                gray_img.save(dest_file, format='BMP')
            else:
                img.save(dest_file, format='png')
  
    # 移动文件到相应的目录，并转换为灰度图和BMP格式
    for f in train_files:
        source_file = os.path.join(source_dir, f)
        if isgray:
            dest_file = os.path.join(train_dir, f.replace('.png', '.bmp'))
        else:
            dest_file = os.path.join(train_dir, f)
        convert_and_save(source_file, dest_file)
    
    for f in test_files:
        source_file = os.path.join(source_dir, f)
        if isgray:
            dest_file = os.path.join(test_dir, f.replace('.png', '.bmp'))
        else:
             dest_file = os.path.join(test_dir, f)
        convert_and_save(source_file, dest_file)
        
    for f in validation_files:
        source_file = os.path.join(source_dir, f)
        if isgray:
            dest_file = os.path.join(valid_dir, f.replace('.png', '.bmp'))
        else:
            dest_file = os.path.join(valid_dir, f)
        convert_and_save(source_file, dest_file)
    
    print(f'Total files: {len(files)}')
    print(f'Training files: {len(train_files)}')
    print(f'Testing files: {len(test_files)}')
    print(f'Validation files: {len(validation_files)}')
    
    # test every division is same

    last_train_sum ,last_test_sum,last_valid_sum = 739164792,182438981,60031392
    train_sum ,test_sum,valid_sum = 0,0,0
    for f in train_files:
        train_sum += int(f.split(".")[0])
    for f in test_files:
        test_sum += int(f.split(".")[0])
    for f in validation_files:
        valid_sum += int(f.split(".")[0])
    print(f'Training sum: {train_sum}')
    print(f'Testing sum: {test_sum}')
    print(f'Validation sum: {valid_sum}')
    print(f'is same? {train_sum == last_train_sum and test_sum == last_test_sum and last_valid_sum == valid_sum}')


# %%
split_train_test("../data/resized","../data/train/img","../data/test/img","../data/validation/img")
# split_train_test("../data/gauss","../data/train/img","../data/test/img","../data/validation/img")
# split_train_test("../data/gauss_sharpen","../data/train/img","../data/test/img","../data/validation/img")

# split_train_test("../data/resized","../data/train/png","../data/test/png","../data/validation/png",isgray=False)


# %%
make_caption("../data/source/trainning set/json","../data/train/img","../data/train/")
make_caption("../data/source/trainning set/json","../data/test/img","../data/test/")
make_caption("../data/source/trainning set/json","../data/validation/img","../data/validation/")
    

# %%



# %%
from PIL import Image
import os
from collections import Counter

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
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):
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
                resolution_distribution[resolution // 10000 * 10000] += 1

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
    print()
    print("Height Distribution (by 100 pixels):")
    for h, count in sorted(height_distribution.items()):
        print(f"{h} to {h+99} pixels: {count} images")
    print(f"Max Height: {max_height} pixels (File: {max_height_file}), Min Height: {min_height} pixels (File: {min_height_file})")
    print()
    print("Resolution Distribution (by 1 MP):")
    for r, count in sorted(resolution_distribution.items()):
        print(f"{r} to {r+9999} pixels: {count} images")
    print(f"Max Resolution: {max_resolution} pixels (File: {max_res_file}), Min Resolution: {min_resolution} pixels (File: {min_res_file})")
    
    return image_info




# %%
# 使用示例
directory = '../data/train/img'
image_info = get_image_info(directory)
image_info

# %%
directory = '../data/test/img'
image_info = get_image_info(directory)
image_info

# %% [markdown]
# 8bit bmp image to 1bit gray image


