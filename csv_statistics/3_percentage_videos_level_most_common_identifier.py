import csv, os

path_to_csv = '/Users/marynavek/Projects/Video_Project/'
path = os.path.join(path_to_csv, "output_most_common_by_videos_stats.csv")
VIDEO_TYPES = [
    "_flat_",
    "_flatWA_",
    "_flatYT_",
    "_indoor_",
    "_indoorWA_",
    "_indoorYT_",
    "_outdoor_",
    "_outdoorWA_",
    "_outdoorYT_"
]
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    out_put_dictionary = []

    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 

    total_number_of_videos = 0
    correct_videos = 0
   
    for video_type in VIDEO_TYPES:
        total_number_of_videos = 0
        correct_videos = 0
        percentage = 0
        for row in sorted_reader:
            if video_type in row["Video Name"]:
                total_number_of_videos += 1
                if row["True Label"] == row["Mostly predicted label"]:
                    correct_videos += 1
        if total_number_of_videos != 0:
            percentage = correct_videos/total_number_of_videos
        addValue = {"Video Type": video_type, "Total Videos": total_number_of_videos, "Correct Videos": correct_videos, "Percentage": percentage}
        out_put_dictionary.append(addValue)        

with open('percentage_correct_videos_based_on_type_and_commonality_stats.csv', 'w') as csvfile:
        fieldnames = ["Video Type", "Total Videos", "Correct Videos", "Percentage"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in out_put_dictionary:
            writer.writerow(data)
        


        
