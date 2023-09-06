import csv, os
import numpy as np
import cv2


input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_frames_sector_1.csv"
input_sec_2 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_frames_sector_2.csv"
input_sec_3 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_frames_sector_3.csv"
input_sec_4 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_frames_sector_4.csv"

files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"

f1 = open(input_sec_1)
f2 = open(input_sec_2)
f3 = open(input_sec_3)
f4 = open(input_sec_4)

reader1 = csv.DictReader(f1)
reader2 = csv.DictReader(f2)
reader3 = csv.DictReader(f3)
reader4 = csv.DictReader(f4)
newreader = []
for row in reader1:
    video_name = row["video_name"]
    f_number = row["frame_number"]
    item2 = None
    item3 = None
    item4 = None
    for row2 in reader2:
        if row2["frame_number"] == f_number and row2["video_name"] == video_name:
            item2 = row2
            break

    for row3 in reader3:
        if row3["frame_number"] == f_number and row3["video_name"] == video_name:
            item3 = row3
            break

    for row4 in reader4:
        if row4["frame_number"] == f_number and row4["video_name"] == video_name:
            item4 = row4
            break

    addItem = {"Device": row["Device"], "video_name": video_name, "frame_number": f_number, "total_patches": row["total_patches"], "sector1_correct": row["true_patches"],
                "sector2_correct": item2["true_patches"], "sector3_correct": item3["true_patches"], "sector4_correct": item4["true_patches"], "blur": row["blur"], "std": row["std"], "prnu_std": row["prnu_std"]}
    newreader.append(addItem)

file_name = 'combined_stats_frames.csv'
csv_path = os.path.join(files_path, file_name)
with open(csv_path, 'w') as csvfile:
    fieldnames = ["Device", "video_name","frame_number", "total_patches", "sector1_correct", "sector2_correct", "sector3_correct", "sector4_correct", "blur", "std", "correct", "prnu_std"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in newreader:
        writer.writerow(data)