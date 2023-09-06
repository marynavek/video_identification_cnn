import csv, os
import numpy as np
import cv2

from prnu_extract import extract_single

input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_1.csv"
input_sec_2 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_2.csv"
input_sec_3 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_3.csv"
input_sec_4 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_4.csv"
files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"
remove_path = "/data/home/mveksler/"
add_path = "/Users/marynavek/Projects/files/"
image_path = "/Users/marynavek/Projects/files/experiment_5/15_dev_train_test_frames/test"

DEVICE_TYPES = [
        "D01_Samsung_GalaxyS3",
        "D02_Huawei_P9",
        "D03_Apple_iPhone5c",
        "D04_Apple_iPhone6",
        "D05_Huawei_P9Lite", 
        "D06_Apple_iPhone6Plus",
        "D07_Samsung_GalaxyS5",
        "D08_Apple_iPhone5",
        "D09_Huawei_P8",
        "D10_Samsung_GalaxyS4Mini"
    ]

def determine_scene(image_name):
    scene_types = ["still", "panrot", "move"]
    for scene in scene_types:
        if scene in image_name:
            return scene

def determine_video_type(image_name):
    video_types = ["_indoor_", "_outdoor_", "_flat_", "_indoorYT_", "_outdoorYT_", "_flatYT_", "_indoorWA_", "_outdoorWA_", "_flatWA_"]
    for video_type in video_types:
        if video_type in image_name:
            return video_type

def determine_device(image_name):
    for device in DEVICE_TYPES:
        if device in image_name:
            return device

def get_video_name(image_name):
    remove_part, main_part = image_name.split("_vid_name_")
    video_name, remove = main_part.split("P-number")
    return video_name

def get_frame_number(image_name):
    remove_part, main_part = image_name.split("frame_number_")
    frame_number, remove = main_part.split("vid_name")
    return frame_number


def calculate_blur(main_path, image):
    image_path = os.path.join(main_path, image)
    image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    std = np.std(image)
    prnu_noise = extract_single(image)
    if prnu_noise is None:
        prnu_std = 0
    else:
        prnu_std = np.std(prnu_noise)
    return std, prnu_std

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]


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

def get_sector_std(main_path, image, sector):
    image_path = os.path.join(main_path, image)
    image = cv2.imread(image_path)
    sector_1, sector_2, sector_3, sector_4 = crop_image_into_four_sectors(image)
    if sector == 1:
        std = np.std(sector_1)
        prnu_noise = extract_single(sector_1)
    elif sector == 2:
        std = np.std(sector_2)
        prnu_noise = extract_single(sector_2)
    elif sector == 3:
        std = np.std(sector_3)
        prnu_noise = extract_single(sector_3)
    elif sector == 4:
        std = np.std(sector_4)
        prnu_noise = extract_single(sector_4)

    if prnu_noise is None:
            prnu_std = 0
    else:
        prnu_std = np.std(prnu_noise)
    
    return std, prnu_std


with open(input_sec_4, newline='') as file_sec_1:
    reader_1 = csv.DictReader(file_sec_1)
    newreader_1 = []
    for row in reader_1:

        newreader_1.append(row)


    sorted_newreader_1 = sorted(newreader_1,  key=lambda x: x["Device"])

    output_reader = []
    for device in DEVICE_TYPES:
        device_dict = []
        video_names = []
        for item in newreader_1:
            if item["Device"] == device:
               device_dict.append(item) 

        sorted_device_dict = sorted(device_dict, key=lambda x: x["video_name"])

        vid_name = ""

        new_image_path = os.path.join(image_path, device)
        all_images = listdir_nohidden(new_image_path)
        for n in sorted_device_dict:
            if vid_name != n["video_name"]:
               vid_name =  n["video_name"]
               video_names.append(vid_name)
        for video in video_names:
            video_dict = []
            total_patches = 0
            correct_patches = 0

            for d in sorted_device_dict:
                if d["video_name"] == video:
                    video_dict.append(d)
            frames_numbers = []
            frame = 0
            sorted_video_dict = sorted(video_dict, key=lambda x: x["frame_number"])
            for v in sorted_video_dict:
                if frame != v["frame_number"]:
                    frame =  v["frame_number"]
                    frames_numbers.append(frame)
            
            for f_number in frames_numbers:
                total_p_frame = 0
                correct_p_frame = 0
                correct = 0
                for s in sorted_video_dict:
                    if f_number == s["frame_number"]:
                        if s["true_l"] == s["predicted_l"]:
                            correct_p_frame += 1
                        total_p_frame += 1
            
                for image in all_images:
                    if f_number in image and video[:-1] in image:
                        std, prnu_std = calculate_blur(new_image_path, image)
                        quadrant_std, quadrant_prnu_std = get_sector_std(new_image_path, image, 4)
                        break
                if correct_p_frame >= 0.5*total_p_frame:
                    correct = 1
                new_item = {"Device": device, "video_name":video, "frame_number":f_number, "total_patches": total_p_frame, "true_patches": correct_p_frame, "quadrant_std": quadrant_std, "quadrant_prnu_std": quadrant_prnu_std, "std": std, "correct": correct, "prnu_std": prnu_std}
                output_reader.append(new_item)
                total_p_frame = 0
                correct_p_frame = 0

    file_name = 'stats_frames_sector_4.csv'
    csv_path = os.path.join(files_path, file_name)
    with open(csv_path, 'w') as csvfile:
            fieldnames = ["Device", "video_name","frame_number", "total_patches", "true_patches", "quadrant_std", "quadrant_prnu_std", "std", "correct", "prnu_std"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in output_reader:
                writer.writerow(data)
