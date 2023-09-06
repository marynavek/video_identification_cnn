import matplotlib.pyplot as plt
import numpy as np

import csv, os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd

files_path = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches"

input_sec_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_patches/combined_stats_all_3.csv"

file = open(input_sec_1)
reader= csv.DictReader(file)

total_array = []
correct_array = []
incorrect_array = []

for row in reader:
    label = row["correct"]
    total_array.append(float(row["patch_std"]))
    if label == '1':
        correct_array.append(float(row["patch_std"]))
    else:
        incorrect_array.append(float(row["patch_std"]))


men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

# density_correct, bins_correct = np.histogram(correct_array)
# density_incorrect, bins_incorrect = np.histogram(incorrect_array)
density_correct, bins_correct, _ = plt.hist(correct_array)
density_incorrect, bins_incorrect, _ = plt.hist(incorrect_array)

print(density_correct)
x = np.arange(len(bins_correct)-1)  # the label locations
width = 0.35  # the width of the bars
print(x)

fig, ax = plt.subplots()
rects1 = ax.bar(x-width/2, density_correct, width, label='Correct')
rects2 = ax.bar(x+width/2, density_incorrect, width, label='Incorrect')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Amount of items')
ax.set_title('Sector 2 - Patch std')
# ax.set_xticks(x)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
