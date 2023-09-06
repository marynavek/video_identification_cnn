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

    files_path = os.path.join(output_folder, "correct_video_percentage_device_level")
    if not os.path.exists(files_path):
        os.mkdir(files_path)

    # DEVICE_TYPES = [
    #     "01SamsungGalaxyS3",
    #     "02HuaweiP9",
    #     "03AppleiPhone5c",
    #     "04AppleiPhone6",
    #     "05HuaweiP9Lite", 
    #     "06AppleiPhone6Plus",
    #     "07SamsungGalaxyS5",
    #     "08AppleiPhone5",
    #     "09HuaweiP8",
    #     "10SamsungGalaxyS4Mini",
    #     "11AppleiPhone6",
    # ]
    # DEVICE_TYPES = [
    #     "04AppleiPhone5c",
    #     "08AppleiPhone5c",
    #     "11AppleiPhone5c"
    # ]
    DEVICE_TYPES = [
        "01AppleiPhone4s",
        "02HuaweiP9",
        "03AppleiPhone6",
        "04SamsungGalaxyS3",
        "05AppleiPhone5c", 
        "06HuaweiP9Lite",
        "07AppleiPhone6Plus",
        "08SonyXperiaZ1Compact",
        "09SamsungGalaxyS5",
        "10AppleiPhone5",
        "11XiaomiRedmiNote",
        "12OnePlusA3000",
        "13HuaweiP",
        "14SamsungGalaxyS4Mini",
        "15OnePlusA3003"
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
                        correct_videos = 0
                        percentage = 0
                        for row in sorted_reader:
                            if device_type in row["Video Name"]:
                                total_number_of_videos += 1
                                if row["True Label"] == row["Mostly predicted label"]:
                                    correct_videos += 1
                        if total_number_of_videos != 0:
                            percentage = correct_videos/total_number_of_videos
                        addValue = {"Device Type": device_type, "Total Videos": total_number_of_videos, "Correct Videos": correct_videos, "Percentage": percentage}
                        out_put_dictionary.append(addValue)        

                file_name = 'percentage_correct_videos_per_device_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Device Type", "Total Videos", "Correct Videos", "Percentage"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)

                break  


                    
