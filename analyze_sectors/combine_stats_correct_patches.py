
import csv
import os


file_sector_1 = "/Users/marynavek/Projects/video_identification_cnn/analyze_sectors/experiment_15_devices_15_frames/sector_1_results/statistics/correct_predicted_frames/correctly_predicted_frames_by_video_0_.csv"

file_sector_2 = "/Users/marynavek/Projects/video_identification_cnn/analyze_sectors/experiment_15_devices_15_frames/sector_2_results/statistics/correct_predicted_frames/correctly_predicted_frames_by_video_0_.csv"

file_sector_3 = "/Users/marynavek/Projects/video_identification_cnn/analyze_sectors/experiment_15_devices_15_frames/sector_3_results/statistics/correct_predicted_frames/correctly_predicted_frames_by_video_0_.csv"

file_sector_4 = "/Users/marynavek/Projects/video_identification_cnn/analyze_sectors/experiment_15_devices_15_frames/sector_4_results/statistics/correct_predicted_frames/correctly_predicted_frames_by_video_0_.csv"

combined_file = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/15_phones_10_frames/statistics/correct_predicted_frames/correctly_predicted_frames_by_video_0_.csv"


f_sector_1 = open(file_sector_1)
f_sector_2 = open(file_sector_2)
f_sector_3 = open(file_sector_3)
f_sector_4 = open(file_sector_4)
f_combined = open(combined_file)

reader_1 = csv.DictReader(f_sector_1)
reader_2 = csv.DictReader(f_sector_2)
reader_3 = csv.DictReader(f_sector_3)
reader_4 = csv.DictReader(f_sector_4)
reader_combined = csv.DictReader(f_combined)

final_dict = []
for row in reader_combined:
    video_name = row["Video Name"]
    total_frames = row["Total Frames"]
    mostly_predicted_combined = row["Correct Frames"]

    for r1 in reader_1:
        if r1["Video Name"] == video_name:
            mostly_predicted_s1 = r1["Correct Frames"]
            break

    for r2 in reader_2:
        if r2["Video Name"] == video_name:
            mostly_predicted_s2 = r2["Correct Frames"]
            break
    
    for r3 in reader_3:
        if r3["Video Name"] == video_name:
            mostly_predicted_s3 = r3["Correct Frames"]
            break
    
    for r4 in reader_4:
        if r4["Video Name"] == video_name:
            mostly_predicted_s4 = r4["Correct Frames"]
            break

    addItem = {"Video Name:": video_name, "Total Frames": total_frames, "combined_correct": mostly_predicted_combined, "correct_1": mostly_predicted_s1,
                "correct_2": mostly_predicted_s2, "correct_3": mostly_predicted_s3, "correct_4": mostly_predicted_s4}

    final_dict.append(addItem)

file_name = 'combined_correct_frames.csv'
csv_path = os.path.join('/Users/marynavek/Projects/video_identification_cnn/analyze_sectors/experiment_15_devices_15_frames', file_name)
with open(csv_path, 'w') as csvfile:
    fieldnames = ["Video Name:", "Total Frames", "combined_correct", "correct_1", "correct_2", "correct_3", "correct_4"]
    fieldnames.append("Video Name")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in final_dict:
        writer.writerow(data)