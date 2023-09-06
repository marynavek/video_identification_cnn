from importlib.resources import path
import os
import random
import shutil
from collections import namedtuple
from pathlib import Path
import time
import cv2
import numpy as np

from prnu_extract import extract_single
from sort import bubble_sort

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def get_patches(img_data, std_threshold, max_num_patches):
    patches = []

    # Default patches is returned when no patches are found with a Std.Dev. lower than the threshold
    default_patch_std = np.array([float('inf'), float('inf'), float('inf')])

    default_patch = None
    cropped_pathes = []

    # patch = namedtuple('WindowSize', ['width', 'height'])(128, 128)

    # # stride = namedtuple('Strides', ['width_step', 'height_step'])(128, 128)
    # image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    # # default_patch_std = np.std(img_data.reshape(-1, 3), axis=0)
    # default_patch_std = np.std(img_data)
    # num_pw = image.width/patch.width
    # num_ph = image.height/patch.height
    
    # width_left = image.width - patch.width*num_pw
    # height_left = image.height - patch.height*num_ph
   
    # stride_height = 128 - int(width_left/num_pw)
    # stride_width = 128 - int(height_left/num_ph)
    # stride = namedtuple('Strides', ['width_step', 'height_step'])(stride_width, stride_height)

    patch = namedtuple('WindowSize', ['width', 'height'])(256, 256)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(256, 256)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])

    std_threshold = 20
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            cropped_img = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            patch_std = np.std(cropped_img)
            patches.append(cropped_img)

#             if np.prod(np.less_equal(patch_std, default_patch_std)):
#                 patches.append(cropped_img)
#             elif np.prod(np.less_equal(patch_std, std_threshold)):
#                 std_threshold = patch_std
#                 # default_patch = cropped_img
#                 cropped_pathes.append(cropped_img)
# #sort cropped patches based on std and then append to the patches
# #consider minimum and maximum bound
#     sorted_cropped_patches = bubble_sort(cropped_pathes)

#     if len(patches) < 15:
#         for i in range(len(sorted_cropped_patches)):
#             if len(patches) > max_num_patches:
#                 break
#             patches.append(sorted_cropped_patches[i])
#     # # print(f'std_pathes = {patches} and cropped patches = {cropped_pathes}')
#     # # Filter out excess patches
    if len(patches) > 10:
        random.seed(999)
        indices = random.sample(range(len(patches)), 10)
        patches = [patches[x] for x in indices]
    
    return patches

def crop_image_into_four_sectors(img):
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]

    height_left = left1.shape[0]
    height_cutoff_left = height_left // 2
    first_sector = left1[:height_cutoff_left, :]
    second_sector = left1[height_cutoff_left:, :]
    
    # start vertical devide image
    height_right = img.shape[0]
    # Cut the image in half
    height_cutoff_right = height_right // 2
    third_sector = right1[:height_cutoff_right, :]
    forth_sector = right1[height_cutoff_right:, :]
    
    return first_sector, second_sector, third_sector, forth_sector


def save_patches(patches, source_img_path, destination_dir):
    for patch_id, patch in enumerate(patches, 1):
        img_name = source_img_path.stem + "_P-number" +'_{}'.format(str(patch_id).zfill(3)) + source_img_path.suffix
        img_path = destination_dir.joinpath(img_name)
        cv2.imwrite(str(img_path), patch * 255.0)


def main(source_data_dir, destination_patches_top_rigth, destination_patches_bottom_rigth, destination_patches_top_left, destination_patches_bottom_left):
    device_num_patches_dict_top_rigth = {}
    device_num_patches_dict_bottom_rigth = {}
    device_num_patches_dict_top_left = {}
    device_num_patches_dict_bottom_left = {}
    devices = source_data_dir.glob("*")
    if not destination_patches_top_rigth.exists():
        os.makedirs(str(destination_patches_top_rigth), exist_ok=True)
    if not destination_patches_bottom_rigth.exists():
        os.makedirs(str(destination_patches_bottom_rigth), exist_ok=True)
    if not destination_patches_top_left.exists():
        os.makedirs(str(destination_patches_top_left), exist_ok=True)
    if not destination_patches_bottom_left.exists():
        os.makedirs(str(destination_patches_bottom_left), exist_ok=True)

    t_start = time.time()
    for device in devices:
        image_paths = device.glob("*")
        destination_device_dir_pathches_top_rigth = destination_patches_top_rigth.joinpath(device.name)
        destination_device_dir_pathches_bottom_rigth = destination_patches_bottom_rigth.joinpath(device.name)
        destination_device_dir_pathches_top_left = destination_patches_top_left.joinpath(device.name)
        destination_device_dir_pathches_bottom_left = destination_patches_bottom_left.joinpath(device.name)

        # The following if-else construct makes sense on running multiple instances of this method
        if destination_device_dir_pathches_top_rigth.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_top_rigth), exist_ok=True)
        if destination_device_dir_pathches_bottom_rigth.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_bottom_rigth), exist_ok=True)
        if destination_device_dir_pathches_top_left.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_top_left), exist_ok=True)
        if destination_device_dir_pathches_bottom_left.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_bottom_left), exist_ok=True)    

        num_patches_top_right = 0
        num_patches_bottom_right = 0
        num_patches_top_left = 0
        num_patches_bottom_left = 0
        
        for image_path in image_paths:
            # For now, we only want to extract frames from original videos
            # if "WA" in image_path.stem or "YT" in image_path.stem:
            #     continue
            
            img = cv2.imread(str(image_path))
            img = np.float32(img) / 255.0

            # sector_1, sector2, sector3, sector4 = crop_image_into_four_sectors(img)
            patches = get_patches(img_data=img,std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15 )
            # pathes_1st_sector = get_patches(img_data=sector_1, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
            # pathes_2nd_sector = get_patches(img_data=sector2, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
            # pathes_3rd_sector = get_patches(img_data=sector3, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
            # pathes_4th_sector = get_patches(img_data=sector4, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)

            # num_patches_top_right += len(pathes_1st_sector)
            # num_patches_bottom_right += len(pathes_2nd_sector)
            # num_patches_top_left += len(pathes_3rd_sector)
            # num_patches_bottom_left += len(pathes_4th_sector)

            save_patches(patches, image_path, destination_device_dir_pathches_top_rigth)
            # save_patches(pathes_2nd_sector, image_path, destination_device_dir_pathches_bottom_rigth)
            # save_patches(pathes_3rd_sector, image_path, destination_device_dir_pathches_top_left)
            # save_patches(pathes_4th_sector, image_path, destination_device_dir_pathches_bottom_left)

        device_num_patches_dict_top_rigth[device.name] = num_patches_top_right
        device_num_patches_dict_bottom_rigth[device.name] = num_patches_bottom_right
        device_num_patches_dict_top_left[device.name] = num_patches_top_left
        device_num_patches_dict_bottom_left[device.name] = num_patches_bottom_left
        print(f"{device.name} | {num_patches_top_right} patches for 1st sector ({int(time.time() - t_start)} sec.)")
        print(f"{device.name} | {num_patches_bottom_right} patches for 2nd sector ({int(time.time() - t_start)} sec.)")
        print(f"{device.name} | {num_patches_top_left} patches for 3rd sector ({int(time.time() - t_start)} sec.)")
        print(f"{device.name} | {num_patches_bottom_left} patches for 4th sector ({int(time.time() - t_start)} sec.)")

    return device_num_patches_dict_top_rigth, device_num_patches_dict_bottom_rigth, device_num_patches_dict_top_left, device_num_patches_dict_bottom_left

if __name__ == "__main__":
    images_per_device = Path('/Users/marynavek/Projects/files/experiment_cortivo/train_test_frames/test/')

    patches_per_device_top_rigth = Path('/Users/marynavek/Projects/files/experiment_cortivo/patches/test/')
    patches_per_device_bottom_right = Path('/Users/marynavek/Projects/files/experiment_cortivo/sector_2_patches/test/')
    patches_per_device_top_left = Path('/Users/marynavek/Projects/files/experiment_cortivo/sector_3_patches/test/')
    patches_per_device_bottom_left = Path('/Users/marynavek/Projects/files/experiment_cortivo/sector_4_patches/test/')


    device_patch_dict_1st_sector, device_patch_dict_2nd_sector, device_patch_dict_3rd_sector, device_patch_dict_4th_sector = main(images_per_device, patches_per_device_top_rigth, patches_per_device_bottom_right, patches_per_device_top_left, patches_per_device_bottom_left)

    