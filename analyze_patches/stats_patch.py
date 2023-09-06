import csv, os
from matplotlib import pyplot as plt
import numpy as np
import cv2


input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_frames_sector_2.csv"
input_sec_2 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_2.csv"
input_sec_3 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_3.csv"
input_sec_4 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/stats_sector_4.csv"

files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"

with open(input_sec_1, newline='') as file:
    reader = csv.DictReader(file)
    std_array = []
    for row in reader:
        label = row["correct"]
        if label == '1':
            std_array.append(float(row["std"]))
    print(len(std_array))
    plt.hist(std_array, cumulative=True, range = (np.min(std_array), np.max(std_array)))
    # density, bins, _ = plt.hist(std_array, density=True, range = (np.min(std_array), np.max(std_array)))
    
    # count, _ = np.histogram(std_array, range=(np.min(std_array), np.max(std_array)))
    # for x,y,num in zip(bins, density, count):
    #     if num != 0:
    #         plt.text(x, y+0.05, num, fontsize=10, rotation=-90) 

    plt.show()