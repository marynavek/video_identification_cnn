import csv, os
import statistics
from statistics import mode

path_to_csv = '/Users/marynavek/Projects/Video_Project/results_4th_epoch/'
path = os.path.join(path_to_csv, "output_frames_stats.csv")
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
    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
    for row in sorted_reader:

        if not video_name == row["Video Name"]:
            if not number_of_frames == 0:
                true_label = row["True Label"]
                most_common_label = mode(predicted_labels)
                addValue = {"Video Name": video_name, "Total Frames": number_of_frames, "True Label": true_label, "Mostly predicted label": most_common_label, "Label 0": label_0, "Label 1": label_1, "Label 2": label_2, "Label 3": label_3, "Label 4": label_4 }
                out_put_dictionary.append(addValue)
            number_of_frames = 0
            number_of_correct_frames = 0
            predicted_labels = []
            label_0 = 0
            label_1 = 0
            label_2 = 0
            label_3 = 0
            label_4 = 0
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
        predicted_labels.append(row["Predicted Label"])



with open('output_most_common_by_videos_devices.csv', 'w') as csvfile:
        fieldnames = ["Video Name", "Total Frames", "True Label", "Mostly predicted label", "Label 0","Label 1", "Label 2", "Label 3", "Label 4"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in out_put_dictionary:
            writer.writerow(data)
        


        
