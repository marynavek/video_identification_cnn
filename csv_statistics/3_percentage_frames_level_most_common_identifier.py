import csv, os

path_to_csv = '/Users/marynavek/Projects/Video_Project/'
path = os.path.join(path_to_csv, "correctly_predicted_frames_by_video.csv")
VIDEO_TYPES = [
    "_flat_",
    "_flatWA_",
    "_flatYT_",
    "_indoor_",
    "_indoorWA_",
    "_indoorYT_",
    "_outdoor_",
    "_outdoorWA_",
    "_outdoorYT"
]
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    out_put_dictionary = []

    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 

    total_number_of_frames = 0
    correct_frames = 0
   
    for video_type in VIDEO_TYPES:
        total_number_of_frames = 0
        correct_frames = 0
        percentage = 0
        for row in sorted_reader:
            if video_type in row["Video Name"]:
                total_number_of_frames += int(row["Total Frames"])
                correct_frames += int(row["Correct Frames"])
                
        if total_number_of_frames != 0:
            percentage = correct_frames/total_number_of_frames
        addValue = {"Video Type": video_type, "Total Frames": total_number_of_frames, "Correct Frames": correct_frames, "Percentage": percentage}
        out_put_dictionary.append(addValue)        

with open('percentage_correct_frames_based_on_type_and_commonality_stats.csv', 'w') as csvfile:
        fieldnames = ["Video Type", "Total Frames", "Correct Frames", "Percentage"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in out_put_dictionary:
            writer.writerow(data)
        


        
