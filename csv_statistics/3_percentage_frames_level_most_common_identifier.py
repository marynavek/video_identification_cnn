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

    path_to_csv = os.path.join(main_path, csvs_folder, "correct_predicted_frames")

    previous_files = get_prediction_files(path_to_csv)

    output_folder = os.path.join(main_path, csvs_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    files_path = os.path.join(output_folder, "correct_frames_percetage_per_video_type")
    if not os.path.exists(files_path):
        os.mkdir(files_path)

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

                file_name = 'percentage_correct_frames_based_on_type_and_commonality_stats_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Video Type", "Total Frames", "Correct Frames", "Percentage"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)

                break   


                        
