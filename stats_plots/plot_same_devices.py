import csv, os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utility import get_prediction_files

path_to_csv = '/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_9/statistics/most_common_label_by_video'

previous_files = get_prediction_files(path_to_csv)

output_folder = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_9/statistics"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files_path = os.path.join(output_folder, "matrix_identified_videos_per_device_complete")
if not os.path.exists(files_path):
    os.mkdir(files_path)

DEVICE_TYPES = [
        "04AppleiPhone5c",
        "08AppleiPhone5c",
        "11AppleiPhone5c"
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
                    device1_flat = 0
                    device2_flat = 0
                    device3_flat = 0
                    device1_indoor = 0
                    device2_indoor = 0
                    device3_indoor = 0
                    device1_outdoor = 0
                    device2_outdoor = 0
                    device3_outdoor = 0

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
                    device_array_total = [device1, device2, device3]
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
                                elif video_type == "indoor":
                                    if row2["Mostly predicted label"] == "0":
                                        device1_indoor += 1
                                    elif row2["Mostly predicted label"] == "1":
                                        device2_indoor += 1
                                    elif row2["Mostly predicted label"] == "2":
                                        device3_indoor += 1
                                elif video_type == "outdoor":
                                    if row2["Mostly predicted label"] == "0":
                                        device1_outdoor += 1
                                    elif row2["Mostly predicted label"] == "1":
                                        device2_outdoor += 1
                                    elif row2["Mostly predicted label"] == "2":
                                        device3_outdoor += 1
                    device_array_flat = [device1_flat, device2_flat, device3_flat]
                    main_stats_array_flat.append(device_array_flat)
                    device_array_indoor = [device1_indoor, device2_indoor, device3_indoor]
                    main_stats_array_indoor.append(device_array_indoor)
                    device_array_outdoor = [device1_outdoor, device2_outdoor, device3_outdoor]
                    main_stats_array_outdoor.append(device_array_outdoor)

            df_cm_total = pd.DataFrame(main_stats_array_total, range(3), range(3))
            df_cm_flat = pd.DataFrame(main_stats_array_flat, range(3), range(3))
            df_cm_indoor = pd.DataFrame(main_stats_array_indoor, range(3), range(3))
            df_cm_outdoor = pd.DataFrame(main_stats_array_outdoor, range(3), range(3))
            # plt.tight_layout(pad=2.0)

            plt.figure(figsize=(12, 2.5), dpi=80)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.subplot(1,4,1)
            # plt.plot
            plt.title('Total Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_total, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False) # font size

            plt.subplot(1,4,2)
            plt.title('Flat Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_flat, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False, yticklabels=False) # font size

            plt.subplot(1,4,3)
            plt.title('Indoor Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_indoor, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False, yticklabels=False) # font size

            plt.subplot(1,4,4)
            plt.title('Outdoor Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_outdoor, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False, yticklabels=False) # font size

            file_name = 'matrix_identified_videos_per_device_' + str(i) + '_.png'
            matrix_path = os.path.join(files_path, file_name)
            # plt.show()
            plt.savefig(matrix_path)
            plt.savefig(matrix_path, format='png')
