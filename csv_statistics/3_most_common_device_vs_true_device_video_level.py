import csv, os, argparse

from utility import find_max_mode, get_prediction_files

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
        
    files_path = os.path.join(output_folder, "true_vs_predicted_device_video_level")
    if not os.path.exists(files_path):
        os.mkdir(files_path)

    DEVICE_TYPES = [
        "03HuaweiP9",
        "04AppleiPhone5c",
        "05AppleiPhone6",
        "06AppleiPhone4",
        "10HuaweiP9",  
    ]

    previous_files = sorted(previous_files)

    for i in range(len(previous_files)):
        for file in previous_files:
            create_string = "_" + str(i) + '_'
            if create_string in file:
                path = os.path.join(path_to_csv, file)

                with open(path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)

                    out_put_dictionary = []

                    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 

                    total_number_of_videos = 0
                    correct_videos = 0
                
                    for device_type in DEVICE_TYPES:
                        
                        total_number_of_videos = 0
                        mostly_predicted_labels = []
                        correct_videos = 0
                        percentage = 0
                        mostly_predicted_label = 0
                        mostly_predicted_device = ""
                        for row in sorted_reader:
                            if device_type in row["Video Name"]:
                                total_number_of_videos += 1
                                if row["True Label"] == row["Mostly predicted label"]:
                                    correct_videos += 1
                                
                                mostly_predicted_labels.append(row["Mostly predicted label"])
                                
                        mostly_predicted_label = str(int(find_max_mode(mostly_predicted_labels)) + 1)
                        for device in DEVICE_TYPES:
                            if mostly_predicted_label in device:
                                mostly_predicted_device = device  
                                break  
                        
                        addValue = {"Device Type": device_type, "Total Videos": total_number_of_videos, "Correct Videos": correct_videos, "Mostly Predicted Device": mostly_predicted_device }
                        out_put_dictionary.append(addValue)        


                file_name = 'video_identification_device_level_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Device Type", "Total Videos", "Correct Videos", "Mostly Predicted Device"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)
                        
                break


                        
