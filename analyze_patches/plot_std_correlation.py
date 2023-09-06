import csv, os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd

files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"

input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/combined_stats_all.csv"

file = open(input_sec_1)
reader= csv.DictReader(file)
device = "D01_Samsung_GalaxyS3"
patch_std = []
quadrant_std = []
frame_std = []
for row in reader:
    if row["Device"] == device:
        label = row["correct"]
        if label == '1':
            if len(patch_std)  < 400:
                patch_std.append(float(row["patch_prnu_std"]))
                quadrant_std.append(float(row["quadrant_prnu_std"]))
                frame_std.append(float(row["frame_prnu_std"]))

# column_values = ["patch", "quadrant", "frame"]

# array = [patch_std, quadrant_std, frame_std]
# df = pd.DataFrame(data = array, 
                #   columns = column_values)

plt.plot(patch_std, color="blue")
plt.plot(quadrant_std, color="green")
# plt.plot(frame_std)
plt.show()