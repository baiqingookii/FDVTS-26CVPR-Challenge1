import os
import cv2
import numpy as np
from PIL import Image
import sys, time
sys.stdout.reconfigure(line_buffering=True)
import re

def numeric_key(name: str):
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else name

t0 = time.time()
path = "/remote-home/share/25-jianfabai/cvpr2026/challenge1/"


# save_path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/train/covid'
# files = ['original/train/covid1','original/train/covid2']

# save_path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/valid/covid'
# save_path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1-iccv25/3d/test'
save_path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/test'
files = ['original/test']

for file in files:
    file_path = path + file + '/'
    scan_count = 0
    # for scan in sorted(os.listdir(file_path)):
    for scan in sorted(os.listdir(file_path), key=numeric_key):
    # for i in range(128,len(sorted(os.listdir(file_path)))):
        # scan = sorted(os.listdir(file_path))[i]
        print(scan)
        # img_len = len(sorted(os.listdir(file_path+patient+'/'+scan+'/')))
        count = 0
        for img in range(len(os.listdir(file_path+scan+'/'))):
            # print(scan,img)
            img_array = cv2.imread(os.path.join(file_path+scan+'/'+str(img)+'.jpg'))
            if img_array is None:
                continue
            img_array = img_array[:,:,0]
            if count == 0:
                img_array_npy = img_array[np.newaxis,:,:]
            else:
                # print(img_array_npy.shape,img_array.shape)
                if img_array.shape != img_array_npy.shape[1:3]:
                    ow = img_array_npy.shape[2]
                    oh = img_array_npy.shape[1]
                    img_array = cv2.resize(img_array, dsize=(ow,oh), interpolation=Image.BILINEAR)
                    # print(img_array.shape)
                img_array_npy = np.concatenate((img_array_npy,img_array[np.newaxis,:,:]),axis=0)
            count += 1
        # print(img_array_npy.shape)
        # save_dir = save_path + file + '/'
        save_dir = save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        np.save(os.path.join(save_dir, scan + '.npy'), img_array_npy)
        print(save_dir,scan,img_array_npy.shape)
        scan_count += 1
        print(scan_count)