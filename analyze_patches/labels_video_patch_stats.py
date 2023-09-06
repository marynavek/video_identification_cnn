import csv, os
import numpy as np
import cv2

from prnu_extract import extract_single

input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/sector_1_predictions.csv"
input_sec_2 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/sector_2_predictions.csv"
input_sec_3 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/sector_3_predictions.csv"
input_sec_4 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/sector_4_predictions.csv"
files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"
remove_path = "/data/home/mveksler/"
add_path = "/Users/marynavek/Projects/files/"

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

with open(input_sec_2, newline='') as file_sec_1:
    reader_1 = csv.DictReader(file_sec_1)
    newreader_1 = []
    for row in reader_1:
        input_image_name = row["File"]
        scene = determine_scene(input_image_name)
        video_type = determine_video_type(input_image_name)
        device = determine_device(input_image_name)
        new_image_name = input_image_name.replace(remove_path, add_path)
        image = cv2.imread(new_image_name)
        std = np.std(image.reshape(-1, 3), axis=0)
        prnu_noise = extract_single(image)
        prnu_std = np.std(prnu_noise)
        video_name = get_video_name(new_image_name)
        frame_number = get_frame_number(new_image_name)

        true_label = row["True Label"]
        predicted_label = row["Predicted Label"]

        addItem = {"Device": device, "video_name":video_name, "frame_number": frame_number, "video_type": video_type, "scene": scene, "true_l": true_label, "predicted_l": predicted_label}
        newreader_1.append(addItem)



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
        for n in sorted_device_dict:
            if vid_name != n["video_name"]:
               vid_name =  n["video_name"]
               video_names.append(vid_name)
    
        for video in video_names:
            total_patches = 0
            correct_patches = 0

            for d in sorted_device_dict:
                if d["video_name"] == video:
                    if n["true_l"] == n["predicted_l"]:
                        correct_patches += 1
                    total_patches += 1 
            
            new_item = {"Device": device, "video_name":video_name, "video_type": video_type, "scene": scene, "total_patches": total_patches, "true_patches": correct_patches}
            output_reader.append(new_item)
            total_patches = 0
            correct_patches = 0

    file_name = 'stats_vid_patch_sector_2.csv'
    csv_path = os.path.join(files_path, file_name)
    with open(csv_path, 'w') as csvfile:
            fieldnames = ["Device", "video_name","video_type", "scene", "total_patches", "true_patches"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in sorted_newreader_1:
                writer.writerow(data)
