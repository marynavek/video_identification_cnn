import csv, os
import numpy as np
import cv2

from prnu_extract import extract_single

input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/ensemble_results_experiment_5_1/sector_1_predictions.csv"
input_sec_2 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/ensemble_results_experiment_5_1/sector_2_predictions.csv"
input_sec_3 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/ensemble_results_experiment_5_1/sector_3_predictions.csv"
input_sec_4 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/ensemble_results_experiment_5_1/sector_4_predictions.csv"
files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"
remove_path = "/data/home/mveksler/"
add_path = "/Users/marynavek/Projects/files/"
images_path = "/Users/marynavek/Projects/files/experiment_5/15_dev_train_test_frames/test"


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

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def get_frame_number(image_name):
    remove_part, main_part = image_name.split("frame_number_")
    frame_number, remove = main_part.split("vid_name")
    return frame_number

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

def calculate_image_std(main_path, image):
    image_path = os.path.join(main_path, image)
    image = cv2.imread(image_path)
    std = np.std(image)
    prnu_noise = extract_single(image)
    if prnu_noise is None:
        prnu_std = 0
    else:
        prnu_std = np.std(prnu_noise)
    return std, prnu_std

with open(input_sec_1, newline='') as file_sec_1:
    reader_1 = csv.DictReader(file_sec_1)
    newreader_1 = []

    for row in reader_1:
        input_image_name = row["File"]
        scene = determine_scene(input_image_name)
        video_type = determine_video_type(input_image_name)
        device = determine_device(input_image_name)
        new_image_name = input_image_name.replace(remove_path, add_path)
        image = cv2.imread(new_image_name)
        std = np.std(image)

        prnu_noise = extract_single(image)
        if prnu_noise is None:
            prnu_std = 0
        else:
            prnu_std = np.std(prnu_noise)

        video_name = get_video_name(new_image_name)
        frame_number = get_frame_number(new_image_name)
        frame = "frame_number_" + str(frame_number)
        if row["True Label"] == row["Predicted Label"]:
            correct = 1
        else: correct = 0
        true_label = row["True Label"]
        predicted_label = row["Predicted Label"]


        new_image_path = os.path.join(images_path, device)
        all_images = listdir_nohidden(new_image_path)

        for image in all_images:
           
            if device in image and video_name[:-1] in image and frame in image:
                image_std, image_prnu_std = calculate_image_std(new_image_path, image)
                sector_std, sector_prnu_std = get_sector_std(new_image_path, image, 1)

        addItem = {"Device": device, "video_name":video_name, "p_std": std, "sector_std": sector_std, "image_std": image_std, "p_prnu_std": prnu_std,  "sector_prnu_std":sector_prnu_std, "image_prnu_std":image_prnu_std, "correct": correct, "true_l": true_label, "predicted_l": predicted_label}
        newreader_1.append(addItem)
    sorted_newreader_1 = sorted(newreader_1,  key=lambda x: x["Device"])

    file_name = 'stats_patch_sector_1.csv'
    csv_path = os.path.join(files_path, file_name)
    with open(csv_path, 'w') as csvfile:
            fieldnames = ["Device", "video_name", "p_std", "std", "sector_std", "image_std", "p_prnu_std",  "sector_prnu_std", "image_prnu_std", "correct", "true_l", "predicted_l"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in sorted_newreader_1:
                writer.writerow(data)
