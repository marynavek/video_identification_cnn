import csv, os
import statistics

def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list)
    return max_mode

path_to_csv = '/Users/marynavek/Projects/Video_Project/'
path = os.path.join(path_to_csv, "output_frames_stats.csv")
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    video_name = ""
    new_video_name = ""
    out_put_dictionary = []
    
    correct_frames = 0
    frames_per_video = 0
    
    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
    for row in sorted_reader:
        if not video_name == row["Video Name"]:
            if not frames_per_video == 0:
                true_label = row["True Label"]
                addValue = {"Video Name": video_name, "Total Frames": frames_per_video, "Correct Frames": correct_frames}
                out_put_dictionary.append(addValue)
            frames_per_video = 0
            correct_frames = 0
            video_name = row["Video Name"]

        frames_per_video += 1
        if row["True Label"] == row["Predicted Label"]:
            correct_frames += 1



with open('correctly_predicted_frames_by_video.csv', 'w') as csvfile:
        fieldnames = ["Video Name", "Total Frames", "Correct Frames"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in out_put_dictionary:
            writer.writerow(data)
        