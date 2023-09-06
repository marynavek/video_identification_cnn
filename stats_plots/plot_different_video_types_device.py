import csv, os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utility import get_prediction_files

path_to_csv = '/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_7/statistics/most_common_label_by_video'

previous_files = get_prediction_files(path_to_csv)

output_folder = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_7/statistics"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files_path = os.path.join(output_folder, "matrix_identified_videos_per_device_complete")
if not os.path.exists(files_path):
    os.mkdir(files_path)

DEVICE_TYPES = [
        "01SamsungGalaxyS3",
        "02HuaweiP9",
        "03AppleiPhone5c",
        "04AppleiPhone6",
        "05HuaweiP9Lite", 
        "06AppleiPhone6Plus",
        "07SamsungGalaxyS5",
        "08AppleiPhone5",
        "09HuaweiP8",
        "10SamsungGalaxyS4Mini",
        "11AppleiPhone6"
    ]

VIDEO_TYPES = [
        "flat",
        "indoor",
        "outdoor"
    ]

for i in range(len(previous_files)):
    for file in previous_files:
        create_string = "_" + str(i) + '_'
        if create_string in file:
            path = os.path.join(path_to_csv, file)
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
                main_stats_array_total = []
                main_stats_array_flat = []
                main_stats_array_outdoor = []
                main_stats_array_indoor = []
                for device_type in DEVICE_TYPES:
                    print(device_type)
                    device1 = 0
                    device2 = 0
                    device3 = 0
                    device4 = 0
                    device5 = 0
                    device6 = 0
                    device7 = 0
                    device8 = 0
                    device9 = 0
                    device10 = 0
                    device11 = 0
                    device1_flat = 0
                    device2_flat = 0
                    device3_flat = 0
                    device4_flat = 0
                    device5_flat = 0
                    device6_flat = 0
                    device7_flat = 0
                    device8_flat = 0
                    device9_flat = 0
                    device10_flat = 0
                    device11_flat = 0
                    device1_indoor = 0
                    device2_indoor = 0
                    device3_indoor = 0
                    device4_indoor = 0
                    device5_indoor = 0
                    device6_indoor = 0
                    device7_indoor = 0
                    device8_indoor = 0
                    device9_indoor = 0
                    device10_indoor = 0
                    device11_indoor = 0
                    device1_outdoor = 0
                    device2_outdoor = 0
                    device3_outdoor = 0
                    device4_outdoor = 0
                    device5_outdoor = 0
                    device6_outdoor = 0
                    device7_outdoor = 0
                    device8_outdoor = 0
                    device9_outdoor = 0
                    device10_outdoor = 0
                    device11_outdoor = 0

                    device_array_total = []
                    device_array_flat = []
                    device_array_indoor = []
                    device_array_outdoor = []

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
                            elif row["Mostly predicted label"] == "6":
                                device7 += 1
                            elif row["Mostly predicted label"] == "7":
                                device8 += 1
                            elif row["Mostly predicted label"] == "8":
                                device9 += 1
                            elif row["Mostly predicted label"] == "9":
                                device10 += 1
                            elif row["Mostly predicted label"] == "10":
                                device11 += 1
                    device_array_total = [device1, device2, device3, device4, device5, device6, device7, device8, device9, device10, device11]
                    main_stats_array_total.append(device_array_total)

                    for video_type in VIDEO_TYPES:
                        for row2 in sorted_reader:
                            if device_type in row2["Video Name"] and video_type in row2["Video Name"]:
                                if video_type == "flat":
                                    if row2["Mostly predicted label"] == "0":
                                        device1_flat += 1
                                    elif row2["Mostly predicted label"] == "1":
                                        device2_flat += 1
                                    elif row2["Mostly predicted label"] == "2":
                                        device3_flat += 1
                                    elif row2["Mostly predicted label"] == "3":
                                        device4_flat += 1
                                    elif row2["Mostly predicted label"] == "4":
                                        device5_flat += 1
                                    elif row2["Mostly predicted label"] == "5":
                                        device6_flat += 1
                                    elif row2["Mostly predicted label"] == "6":
                                        device7_flat += 1
                                    elif row2["Mostly predicted label"] == "7":
                                        device8_flat += 1
                                    elif row2["Mostly predicted label"] == "8":
                                        device9_flat += 1
                                    elif row2["Mostly predicted label"] == "9":
                                        device10_flat += 1
                                    elif row2["Mostly predicted label"] == "10":
                                        device11_flat += 1
                                elif video_type == "indoor":
                                    if row2["Mostly predicted label"] == "0":
                                        device1_indoor += 1
                                    elif row2["Mostly predicted label"] == "1":
                                        device2_indoor += 1
                                    elif row2["Mostly predicted label"] == "2":
                                        device3_indoor += 1
                                    elif row2["Mostly predicted label"] == "3":
                                        device4_indoor += 1
                                    elif row2["Mostly predicted label"] == "4":
                                        device5_indoor += 1
                                    elif row2["Mostly predicted label"] == "5":
                                        device6_indoor += 1
                                    elif row2["Mostly predicted label"] == "6":
                                        device7_indoor += 1
                                    elif row2["Mostly predicted label"] == "7":
                                        device8_indoor += 1
                                    elif row2["Mostly predicted label"] == "8":
                                        device9_indoor += 1
                                    elif row2["Mostly predicted label"] == "9":
                                        device10_indoor += 1
                                    elif row2["Mostly predicted label"] == "10":
                                        device11_indoor += 1
                                elif video_type == "outdoor":
                                    if row2["Mostly predicted label"] == "0":
                                        device1_outdoor += 1
                                    elif row2["Mostly predicted label"] == "1":
                                        device2_outdoor += 1
                                    elif row2["Mostly predicted label"] == "2":
                                        device3_outdoor += 1
                                    elif row2["Mostly predicted label"] == "3":
                                        device4_outdoor += 1
                                    elif row2["Mostly predicted label"] == "4":
                                        device5_outdoor += 1
                                    elif row2["Mostly predicted label"] == "5":
                                        device6_outdoor += 1
                                    elif row2["Mostly predicted label"] == "6":
                                        device7_outdoor += 1
                                    elif row2["Mostly predicted label"] == "7":
                                        device8_outdoor += 1
                                    elif row2["Mostly predicted label"] == "8":
                                        device9_outdoor += 1
                                    elif row2["Mostly predicted label"] == "9":
                                        device10_outdoor += 1
                                    elif row2["Mostly predicted label"] == "10":
                                        device11_outdoor += 1
                    device_array_flat = [device1_flat, device2_flat, device3_flat,
                                    device4_flat, device5_flat, device6_flat, device7_flat,
                                    device8_flat, device9_flat, device10_flat, device11_flat]
                    main_stats_array_flat.append(device_array_flat)
                    device_array_indoor = [device1_indoor, device2_indoor, device3_indoor,
                                    device4_indoor, device5_indoor, device6_indoor, device7_indoor,
                                    device8_indoor, device9_indoor, device10_indoor, device11_indoor]
                    main_stats_array_indoor.append(device_array_indoor)
                    device_array_outdoor = [device1_outdoor, device2_outdoor, device3_outdoor, 
                                    device4_outdoor, device5_outdoor, device6_outdoor, device7_outdoor, device8_outdoor, 
                                    device9_outdoor, device10_outdoor, device11_outdoor]
                    main_stats_array_outdoor.append(device_array_outdoor)

            df_cm_total = pd.DataFrame(main_stats_array_total, range(11), range(11))
            df_cm_flat = pd.DataFrame(main_stats_array_flat, range(11), range(11))
            df_cm_indoor = pd.DataFrame(main_stats_array_indoor, range(11), range(11))
            df_cm_outdoor = pd.DataFrame(main_stats_array_outdoor, range(11), range(11))
            # plt.tight_layout(pad=2.0)

            plt.figure(figsize=(12, 2.5), dpi=80)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.subplot(1,4,1)
            # plt.plot
            plt.title('Total Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_total, annot=True, annot_kws={"size": 11}, cmap="Greys", cbar=False) # font size

            plt.subplot(1,4,2)
            plt.title('Flat Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_flat, annot=True, annot_kws={"size": 11}, cmap="Greys", cbar=False, yticklabels=False) # font size

            plt.subplot(1,4,3)
            plt.title('Indoor Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_indoor, annot=True, annot_kws={"size": 11}, cmap="Greys", cbar=False, yticklabels=False) # font size

            plt.subplot(1,4,4)
            plt.title('Outdoor Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_outdoor, annot=True, annot_kws={"size": 11}, cmap="Greys", cbar=False, yticklabels=False) # font size

            file_name = 'matrix_identified_videos_per_device_' + str(i) + '_.png'
            matrix_path = os.path.join(files_path, file_name)
            # plt.show()
            plt.savefig(matrix_path)
            plt.savefig(matrix_path, format='png')
