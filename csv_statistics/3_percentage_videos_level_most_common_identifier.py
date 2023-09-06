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

    files_path = os.path.join(output_folder, "correct_video_percentage_types_level")
    if not os.path.exists(files_path):
        os.mkdir(files_path)

    previous_files = sorted(previous_files)

    VIDEO_TYPES = [
        "flat",
        "flatWA",
        "flatYT",
        "indoor",
        "indoorWA",
        "indoorYT",
        "outdoor",
        "outdoorWA",
        "outdoorYT"
    ]

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


                for item in out_put_dictionary:
                    if item["Video Type"] == 'flat':
                        total_flat_videos = item["Total Videos"]
                        correct_flat_videos = item ["Correct Videos"]
                    elif item["Video Type"] == 'flatWA':
                        total_flat_videos_WA = item["Total Videos"]
                        correct_flat_videos_WA = item ["Correct Videos"]
                    elif item["Video Type"] == 'flatYT':
                        total_flat_videos_YT = item["Total Videos"]
                        correct_flat_videos_YT= item["Correct Videos"]
                    elif item["Video Type"] == 'indoor':
                        total_indoor_videos = item["Total Videos"]
                        correct_indoor_videos = item["Correct Videos"]
                    elif item["Video Type"] == 'indoorWA':
                        total_indoor_videos_WA = item["Total Videos"]
                        correct_indoor_videos_WA = item["Correct Videos"]
                    elif item["Video Type"] == 'indoorYT':
                        total_indoor_videos_YT = item["Total Videos"]
                        correct_indoor_videos_YT= item["Correct Videos"]
                    elif item["Video Type"] == 'outdoor':
                        total_outdoor_videos = item["Total Videos"]
                        correct_outdoor_videos = item["Correct Videos"]
                    elif item["Video Type"] == 'outdoorWA':
                        total_outdoor_videos_WA = item["Total Videos"]
                        correct_indoor_videos_WA = item["Correct Videos"]
                    elif item["Video Type"] == 'outdoorYT':
                        total_outdoor_videos_YT = item["Total Videos"]
                        correct_outdoor_videos_YT= item["Correct Videos"]

                native_flat_total = total_flat_videos - total_flat_videos_WA - total_flat_videos_YT
                native_flat_correct = correct_flat_videos - correct_flat_videos_WA - correct_flat_videos_YT
                addValue = {"Video Type": 'flat_native', "Total Videos": native_flat_total, "Correct Videos": native_flat_correct, "Percentage": native_flat_correct/native_flat_total}
                out_put_dictionary.append(addValue)   

                native_indoor_total = total_indoor_videos - total_indoor_videos_WA - total_indoor_videos_YT
                native_indoor_correct = correct_indoor_videos - correct_indoor_videos_WA - correct_indoor_videos_YT

                addValue = {"Video Type": 'indoor_native', "Total Videos": native_indoor_total, "Correct Videos": native_indoor_correct, "Percentage": native_indoor_correct/native_indoor_total}
                out_put_dictionary.append(addValue)   

                native_outdoor_total = total_outdoor_videos - total_outdoor_videos_WA - total_outdoor_videos_YT
                native_outdoor_correct = correct_outdoor_videos - correct_indoor_videos_WA - correct_outdoor_videos_YT

                addValue = {"Video Type": 'outdoor_native', "Total Videos": native_outdoor_total, "Correct Videos": native_outdoor_correct, "Percentage": native_outdoor_correct/native_outdoor_total}
                out_put_dictionary.append(addValue)  

                file_name = 'percentage_correct_videos_based_on_type_and_commonality_stats_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Video Type", "Total Videos", "Correct Videos", "Percentage"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)
                        
                break

                        
