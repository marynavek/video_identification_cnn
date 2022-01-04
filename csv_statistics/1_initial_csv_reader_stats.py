import csv, os

path_to_csv = '/Users/marynavek/Projects/video_identification_cnn/models_concat_transfer_prnu/ccnn-FC2x1024-480_800-f3-k_s5/predictions/frames'
path = os.path.join(path_to_csv, "fm-e00001_F_predictions.csv")
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    
    # split frames stats by videos and devices

    # 1. get videos names
    VIDEO_DIR = "/Users/marynavek/Projects/Video_Project/Videos"
    DEVICES = [item for item in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, item))]

    FILTERED_RESULTS = []
    
    newreader = []
    for row in reader:
        file_name_row = row["File"]
        file_name_row = file_name_row.replace("['", "")
        file_name_row = file_name_row.replace("']", "")
        remove_part, video_name = file_name_row.split("D0")
        video_name = video_name.split(".")[0]
        row["Video Name"] = video_name
        newreader.append(row)
        

    # print(newreader)
    with open('output_frames_stats.csv', 'w') as csvfile:
        fieldnames = reader.fieldnames
        fieldnames.append("Video Name")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in newreader:
            writer.writerow(data)



