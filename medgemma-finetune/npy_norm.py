import os
import cv2
import numpy as np
from PIL import Image
import sys, time
sys.stdout.reconfigure(line_buffering=True)
import re

t0 = time.time()

# def rescale_gao_xy(images_zyx, size=448, mode='image'):
#     """
#     images_zyx: (D,H,W)
#     对每一层 slice 做 2D resize，最后 stack 回 (D,size,size)
#     """
#     D, H, W = images_zyx.shape
#     if mode == 'image':
#         interpolation_mode = cv2.INTER_LINEAR if (H > 768 and W > 768) else cv2.INTER_AREA
#     else:
#         interpolation_mode = cv2.INTER_NEAREST

#     # cv2 对 float32 最稳
#     imgs = images_zyx.astype(np.float32, copy=False)

#     out = np.empty((D, size, size), dtype=imgs.dtype)
#     for z in range(D):
#         out[z] = cv2.resize(imgs[z], (size, size), interpolation=interpolation_mode)
#     return out


# def rescale_z(images_zyx, target_depth, is_mask_image=False, verbose=False):
#     """
#     images_zyx: (D,H,W)
#     只在 z 方向 resize：把 (D,H,W) 展平成 (D, H*W) 的 2D，再 resize 到 (target_depth, H*W)，最后 reshape 回来
#     """
#     D, H, W = images_zyx.shape
#     interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR

#     imgs = images_zyx.astype(np.float32, copy=False)
#     flat = imgs.reshape(D, -1)  # (D, H*W)

#     # dsize = (width, height) = (flat.shape[1], target_depth)
#     flat_rs = cv2.resize(flat, (flat.shape[1], target_depth), interpolation=interpolation)
#     out = flat_rs.reshape(target_depth, H, W)
#     return out

def rescale_gao_xy(images_zyx, size=448, mode='image'):
    # 原本
    interpolation_mode = cv2.INTER_LINEAR if mode == 'image' else cv2.INTER_NEAREST
    # interpolation_mode = cv2.INTER_AREA if mode == 'image' else cv2.INTER_NEAREST
    # images_zyx: (D,H,W)
    # D, H, W = images_zyx.shape
    # if mode == 'image':
    #     # 默认用 AREA；若 H/W 都 >1200，则改用 LINEAR
    #     interpolation_mode = cv2.INTER_LINEAR if (H > 768 and W > 768) else cv2.INTER_AREA
    # else:
    #     interpolation_mode = cv2.INTER_NEAREST
    res = images_zyx
    if res.shape[0] > 512:
        res1 = res[:size]
        res2 = res[size:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(size, size), interpolation=interpolation_mode)
        res2 = cv2.resize(res2, dsize=(size, size), interpolation=interpolation_mode)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = np.vstack([res1, res2])
        # res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res.transpose(1, 2, 0), dsize=(size,size), interpolation=interpolation_mode)
        if len(res.shape)==2:
            res = res[:,:,np.newaxis]
        res = res.transpose(2, 0, 1)
    # print("Shape after: ", res.shape)
    return res

# def rescale_z(images_zyx, target_depth, is_mask_image=False, verbose=False):
#     # print("Resizing dim z")
#     resize_x = 1.0
#     resize_y = target_depth/images_zyx.shape[0]
#     interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
#     res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
#     return res

def rescale_z(images_zyx, target_depth, is_mask_image=False):
    D, H, W = images_zyx.shape
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    flat = images_zyx.astype(np.float32, copy=False).reshape(D, -1)          # (D, H*W)
    flat_rs = cv2.resize(flat, (flat.shape[1], target_depth), interpolation=interpolation)
    return flat_rs.reshape(target_depth, H, W)

path = '/remote-home/share/25-jianfabai/cvpr2026/challenge1/3d/'
save_path = '/remote-home/share/25-jianfabai/medgemma-finetune/challenge1/norm/'
# types = ['train/covid','train/non-covid','valid/covid','valid/non-covid']
# types = ['train/covid1b','train/non-covid1b','valid/covid1b','valid/non-covid1b']
types = ['train/covid1b','train/non-covid1b','valid/covid1b','valid/non-covid1b']
# types = ['train/non-covid1b1','valid/covid1b1','valid/non-covid1b1']
# types = ['test']

for type in types:
    type_path = path + type + '/'
    # type_path=path
    patient_count = 0
    # for scan in sorted(os.listdir(type_path)):
    for i in range(len(sorted(os.listdir(type_path)))):
        scan = sorted(os.listdir(type_path))[i]
        img_array = np.load(os.path.join(type_path+scan))
        print(scan,img_array.shape)
        if img_array.shape[0]>150:
            begin = int(0.15*img_array.shape[0])
            end = int(0.85*img_array.shape[0])
            img_array = img_array[begin:end]
    
        if img_array.shape[0]>512:
            # resize z
            # img_array1 = rescale_z(img_array,128)
            # img_array1 = rescale_z(img_array,24)
            img_array1 = rescale_z(img_array,64)
            # resize xy
            # img_array1 = rescale_gao_xy(img_array1,256)
            img_array1 = rescale_gao_xy(img_array1,448)
        else:
            # resize xy
            # img_array1 = rescale_gao_xy(img_array,256)
            img_array1 = rescale_gao_xy(img_array,448)
            # resize z
            # img_array1 = rescale_z(img_array1,128)
            # img_array1 = rescale_z(img_array1,24)
            img_array1 = rescale_z(img_array1,64)


        # save_dir = save_path + type + '/'
        parent, last = os.path.split(type)          # parent='train', last='covid1b'
        type_area = os.path.join(parent, last + "_64_448_448")
        save_dir = os.path.join(save_path, type_area)
        # save_dir = save_path

        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        np.save(os.path.join(save_dir, scan), img_array1)
        print(save_dir,scan,img_array1.shape)
        patient_count += 1
        print(patient_count)

t1 = time.time()
elapsed = t1 - t0
print(f"elapsed={elapsed:.3f}s")