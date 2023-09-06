import csv, os, argparse
from utility import get_prediction_files, find_max_mode

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--main_path', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--csvs_folder', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main_path = args.main_path
    csvs_folder = args.csvs_folder

    path_to_csv = os.path.join(main_path, csvs_folder, "initial_stats")

    previous_files = get_prediction_files(path_to_csv)

    output_folder = os.path.join(main_path, csvs_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    files_path = os.path.join(output_folder, "most_common_labels_videos_devices")

    if not os.path.exists(files_path):
        os.mkdir(files_path)

    previous_files = sorted(previous_files)


    for i in range(len(previous_files)):
        for file in previous_files:
            create_string = "_" + str(i) + '_'
            if create_string in file:
                path = os.path.join(path_to_csv, file)
                with open(path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    video_name = ""
                    new_video_name = ""
                    out_put_dictionary = []
                    number_of_frames = 0
                    predicted_labels = []
                    true_label = 0
                    label_0 = 0
                    label_1 = 0
                    label_2 = 0
                    label_3 = 0
                    label_4 = 0
                    label_5 = 0
                    label_6 = 0
                    label_7 = 0
                    label_8 = 0
                    label_9 = 0
                    label_10 = 0
                    label_11 = 0
                    label_12 = 0
                    label_13 = 0
                    label_14 = 0
                    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
                    for row in sorted_reader:

                        if not video_name == row["Video Name"]:
                            if not number_of_frames == 0:
                                most_common_label = find_max_mode(predicted_labels)
                                addValue = {"Video Name": video_name, "Total Frames": number_of_frames, "True Label": true_label, "Mostly predicted label": most_common_label, "Label 0": label_0, "Label 1": label_1, "Label 2": label_2, "Label 3": label_3, "Label 4": label_4,
                                            "Label 5": label_5, "Label 6": label_6, "Label 7": label_7, "Label 8": label_8, "Label 9": label_9,
                                             "Label 10": label_10, "Label 11": label_11, "Label 12": label_12, "Label 13": label_13, "Label 14": label_14 }
                                out_put_dictionary.append(addValue)
                            true_label = row["True Label"]
                            number_of_frames = 0
                            number_of_correct_frames = 0
                            predicted_labels = []
                            label_0 = 0
                            label_1 = 0
                            label_2 = 0
                            label_3 = 0
                            label_4 = 0
                            label_5 = 0
                            label_6 = 0
                            label_7 = 0
                            label_8 = 0
                            label_9 = 0
                            label_10 = 0
                            label_11 = 0
                            label_12 = 0
                            label_13 = 0
                            label_14 = 0
                            most_common_label = 0
                            video_name = row["Video Name"]

                        number_of_frames += 1
                        if row["Predicted Label"] == "0":
                            label_0 += 1
                        elif row["Predicted Label"] == "1":
                            label_1 += 1
                        elif row["Predicted Label"] == "2":
                            label_2 += 1
                        elif row["Predicted Label"] == "3":
                            label_3 += 1
                        elif row["Predicted Label"] == "4":
                            label_4 += 1
                        if row["Predicted Label"] == "5":
                            label_5 += 1
                        elif row["Predicted Label"] == "6":
                            label_6 += 1
                        elif row["Predicted Label"] == "7":
                            label_7 += 1
                        elif row["Predicted Label"] == "8":
                            label_8 += 1
                        elif row["Predicted Label"] == "9":
                            label_9 += 1
                        if row["Predicted Label"] == "10":
                            label_10 += 1
                        elif row["Predicted Label"] == "11":
                            label_11 += 1
                        elif row["Predicted Label"] == "12":
                            label_12 += 1
                        elif row["Predicted Label"] == "13":
                            label_13 += 1
                        elif row["Predicted Label"] == "14":
                            label_14 += 1

                        predicted_labels.append(row["Predicted Label"])


                file_name = 'output_most_common_by_videos_devices_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Video Name", "Total Frames", "True Label", "Mostly predicted label", "Label 0","Label 1", "Label 2", "Label 3", "Label 4",  "Label 5","Label 6", "Label 7", "Label 8", "Label 9",  "Label 10",  "Label 11","Label 12", "Label 13", "Label 14"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)
                        
                break


                    
