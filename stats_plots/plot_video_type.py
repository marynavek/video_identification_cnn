import csv, os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utility import get_prediction_files

path_to_csv = '/Users/marynavek/Projects/video_identification_cnn/bayar_models/reg_patches/sector_1/all/32/predictions/all/32/statistics/most_common_label_by_video'

previous_files = get_prediction_files(path_to_csv)

output_folder = "/Users/marynavek/Projects/video_identification_cnn/bayar_models/reg_patches/sector_1/all/32/predictions/all/32/statistics"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files_path = os.path.join(output_folder, "matrix_identified_videos_per_device")
if not os.path.exists(files_path):
    os.mkdir(files_path)

DEVICE_TYPES = [
        "01V",
        "03V",
        "04V",
        "05V",
        "09V",
        "27V"
    ]

device_array = []

for i in range(len(previous_files)):
    for file in previous_files:
        create_string = "_" + str(i) + '_'
        if create_string in file:
            path = os.path.join(path_to_csv, file)
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
                main_stats_array = []
                for device_type in DEVICE_TYPES:
                    device1 = 0
                    device2 = 0
                    device3 = 0
                    device4 = 0
                    device5 = 0
                    device6 = 0
                    device_array = []
                    for row in sorted_reader:
                        if device_type in row["Video Name"]:
                            if row["Mostly predicted label"] == "0":
                                device1 += 1
                            elif row["Mostly predicted label"] == "1":
                                device2 += 1
                            elif row["Mostly predicted label"] == "2":
                                device3 += 1
                            elif row["Mostly predicted label"] == "3":
                                device4 += 1
                            elif row["Mostly predicted label"] == "4":
                                device5 += 1
                            elif row["Mostly predicted label"] == "5":
                                device6 += 1
                    device_array = [device1, device2, device3, device4, device5, device6]
                    main_stats_array.append(device_array)

            df_cm = pd.DataFrame(main_stats_array, range(6), range(6))
            plt.figure()
            # plt.figure(figsize=(10,7))
            sn.set(font_scale=1.4) # for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

            file_name = 'matrix_identified_videos_per_device_' + str(i) + '_.png'
            matrix_path = os.path.join(files_path, file_name)
            # plt.show()
            plt.savefig(matrix_path)
