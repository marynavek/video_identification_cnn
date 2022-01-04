import os
import csv
import pandas as pd

path_to_prediction_folder = "/home/marynavek/Video_Project/video_identification_cnn/models_concat_transfer_prnu_with_stabilizer/ccnn-FC2x1024-480_800-f3-k_s5/predictions/frames"

def get_statistics(path_to_folder):
    files_list = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]

    return files_list

csv_files = get_statistics(path_to_prediction_folder)

csv_array_of_dict = []
csv_array_of_data_frames = []
for file in csv_files:
    csv_array_of_data_frames.append(pd.read_csv(os.path.join(path_to_prediction_folder, file)))

frame_name = ""
number_of_true_predicted_labels = 0
number_of_wrong_predicted_labels = 0
final_results = []
for data_frame in csv_array_of_data_frames:
    csv_array_of_dict.append(data_frame.T.to_dict().values())

for row in csv_array_of_dict[0]:
    frame_name = row["File"]
    if row["True Label"] == row["Predicted Label"]:
        number_of_true_predicted_labels = number_of_true_predicted_labels + 1
    else:
        number_of_wrong_predicted_labels = number_of_wrong_predicted_labels + 1
    
    for csv_file in csv_array_of_dict[1:]:
        for row in csv_file:
            if row["File"] == frame_name:
                if row["True Label"] == row["Predicted Label"]:
                    number_of_true_predicted_labels = number_of_true_predicted_labels + 1
                else:
                    number_of_wrong_predicted_labels = number_of_wrong_predicted_labels + 1
                break
    
    new_row = {"File": frame_name, "Amount Of Correct Predictions": number_of_true_predicted_labels, "Amount Of Wrong Predictions": number_of_wrong_predicted_labels}

    final_results.append(new_row)
    number_of_true_predicted_labels = 0
    number_of_wrong_predicted_labels = 0


with open('frames_stat_evaluation.csv', 'w') as csvfile:
        fieldnames = ["File", "Amount Of Correct Predictions", "Amount Of Wrong Predictions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in final_results:
            writer.writerow(data)
            