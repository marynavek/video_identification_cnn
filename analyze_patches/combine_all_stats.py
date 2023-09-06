import csv, os
import numpy as np
import cv2

input_sec_1_patch = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_4.csv"
input_sec_1_frames = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_frames_sector_4.csv"
files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"


f_patch = open(input_sec_1_patch)
f_frame = open(input_sec_1_frames)

reader_patch = csv.DictReader(f_patch)
reader_frame = csv.DictReader(f_frame)

newreader = []
reader_2 = []
for i in reader_frame:
    reader_2.append(i)

for row in reader_patch:
    video_name = row["video_name"]
    f_number = row["frame_number"]
    patch_std = row["std"]
    patch_prnu_std = row["prnu_std"]
    device = row["Device"]
    correct = row["correct"]
    item = None
    for row2 in reader_2:
        video_name_1 = row2["video_name"]
        f_number_1 = row2["frame_number"]
        if video_name == video_name_1 and f_number == f_number_1:
            item = row2
            break
    quadrant_std = item["quadrant_std"]
    quadrant_prnu_std = item["quadrant_prnu_std"]
    frame_std = item["std"]
    frame_prnu_std = item["prnu_std"]
            
    addItem = {"Device": device, "video_name": video_name, "frame_number": f_number, "patch_std": patch_std,
                        "quadrant_std": quadrant_std,  "frame_std": frame_std,  "patch_prnu_std": patch_prnu_std, 
                        "quadrant_prnu_std": quadrant_prnu_std, "frame_prnu_std": frame_prnu_std, "correct": correct}
    newreader.append(addItem)


file_name = 'combined_stats_all_4.csv'
csv_path = os.path.join(files_path, file_name)          

with open(csv_path, 'w') as csvfile:
    fieldnames = ["Device", "video_name","frame_number", "patch_std", "quadrant_std", "frame_std", "patch_prnu_std", 
                "quadrant_prnu_std", "frame_prnu_std", "correct"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in newreader:
        writer.writerow(data)