import os

def count_jpg_images(directory_path):
    # 检查路径是否存在
    if not os.path.exists(directory_path):
        print(f"错误：路径不存在 -> {directory_path}")
        return

    jpg_count = 0
    png_count = 0
    
    print(f"正在统计路径下的图像: {directory_path}")
    
    # os.walk 会递归遍历所有子文件夹
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 检查后缀名，使用 .lower() 确保 .JPG 和 .jpg 都能被统计
            if file.lower().endswith('.jpg') and not file.lower().startswith('.'):
                jpg_count += 1
            if file.lower().endswith('.png') and not file.lower().startswith('.'):
                png_count += 1
            # if file.lower().startswith('.'):
            #     jpg_count -= 1
                
    print(f"---------------------------")
    print(f"统计完成！")
    print(f"总计找到 .jpg 图像: {jpg_count} 张")
    print(f"总计找到 .png 图像: {png_count} 张")


# target_path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1/original/valid'
# target_path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1/original/train'
target_path = '/remote-home/share/21-yuanruntian-21210240410/iccv25/challenge1/original/train'

# 执行统计
count_jpg_images(target_path)