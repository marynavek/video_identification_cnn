import csv, os, argparse
from utility import get_prediction_files

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--main_path', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--csvs_folder', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main_path = args.main_path
    csvs_folder = args.csvs_folder


    path_to_csv = os.path.join(main_path, csvs_folder, "most_common_label_by_video")

    previous_files = get_prediction_files(path_to_csv)

    output_folder = os.path.join(main_path, csvs_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    files_path = os.path.join(output_folder, "video_percentage_device_level_and_vid_type")
    if not os.path.exists(files_path):
        os.mkdir(files_path)

    previous_files = sorted(previous_files)

    DEVICE_TYPES = [
        "03HuaweiP9",
        "04AppleiPhone5c",
        "05AppleiPhone6",
        "06AppleiPhone4",
        "10HuaweiP9",  
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

                    # getting the videos by device
                    out_put_dictionary = []

                    sorted_reader = sorted(reader, key = lambda item: item['Video Name'])

                    total_number_of_videos = 0
                    # correct_videos = 0
                    video_dict = []
                    for device_type in DEVICE_TYPES:
                        print(device_type)
                        total_number_of_videos_device = 0
                        correct_videos_device = 0
                        video_names = []
                        for row in sorted_reader:
                            if device_type in row["Video Name"]:
                                total_number_of_videos_device += 1
                                video_names.append(row["Video Name"])
                                if row["True Label"] == row["Mostly predicted label"]:
                                    
                                    correct_videos_device += 1 

                        total_flat = 0
                        total_indoor = 0
                        total_outdoor = 0
                        correct_flat = 0
                        correct_indoor = 0
                        correct_outdoor = 0

                        for video_type in VIDEO_TYPES:
                            print(video_type)
                            total_number_of_videos = 0
                            correct_videos = 0
                            for video_name in video_names:
                                if video_type in video_name:
                                    for row2 in sorted_reader:
                                        if row2["Video Name"] == video_name:
                                            total_number_of_videos += 1
                                            if row2["True Label"] == row2["Mostly predicted label"]:
                                                correct_videos += 1
                            
                            if video_type == "flat":
                                total_flat = total_number_of_videos
                                correct_flat = correct_videos
                            elif video_type == "indoor":
                                total_indoor = total_number_of_videos
                                correct_indoor = correct_videos
                            elif video_type == "outdoor":
                                total_outdoor = total_number_of_videos
                                correct_outdoor = correct_videos
                        
                        addValue = {"Device Type": device_type, "Total Videos": total_number_of_videos_device, "Correct Videos": correct_videos_device,
                            "Total Flat": total_flat, "Correct Flat": correct_flat, "Total Indoor": total_indoor, "Correct Indoor": correct_indoor,
                            "Total Outdoor": total_outdoor, "Correct Outdoor": correct_outdoor, "Videos": video_names}

                        out_put_dictionary.append(addValue)  
                    
                file_name = 'correct_videos_device' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Device Type", "Total Videos", "Correct Videos",
                        "Total Flat", "Correct Flat", "Total Indoor", "Correct Indoor",
                        "Total Outdoor", "Correct Outdoor", "Videos"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)
                        
                break
                            
                            

                    