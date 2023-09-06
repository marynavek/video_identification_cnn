import os
import argparse
from posix import listdir
from posixpath import join


INPUT_DIR = "/Users/marynavek/Projects/files/ten_devices_videos_ds"
OUTPUT_DIR = "/Users/marynavek/Projects/files/gop_calculate"
DEVICES = [item for item in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, item))]

print(DEVICES)



for device in DEVICES:
    print("Processing videos for " + device)
    device_folder = os.path.join(INPUT_DIR, device)

    VIDEO_TYPES = [item for item in os.listdir(device_folder) if os.path.isdir(os.path.join(device_folder, item))]

    for video_type in VIDEO_TYPES:
        print("Creating frames for videos of type " + video_type)

        video_type_folder = os.path.join(device_folder, video_type)


        VIDEO_NAMES = [item for item in os.listdir(video_type_folder) if os.path.isfile(os.path.join(video_type_folder, item))]

        
        if "flat"  == video_type:
            new_video_type = "__flat__"
        elif "flatWA" == video_type:
            new_video_type = "__flat__"
        elif "flatYT" == video_type:
            new_video_type = "__flat__"
        elif "indoor" == video_type:
            new_video_type = "__indoor__"
        elif "indoorWA" == video_type:
            new_video_type = "__indoor__"
        elif "indoorYT" == video_type:
            new_video_type = "__indoor__"
        elif "outdoor" == video_type:
            new_video_type = "__outdoor__"
        elif "outdoorWA" == video_type:
            new_video_type = "__outdoor__"
        elif "outdoorYT" == video_type:
            new_video_type = "__outdoor__"
        
        outputPath = os.path.join(OUTPUT_DIR, device, new_video_type)
        if not os.path.isdir(outputPath):
            os.makedirs(outputPath)

        
        # print(VIDEO_NAMES)
        for video in VIDEO_NAMES:

            output_video_folder_name = video.split(".")[0]
            
            
            output_video_path = os.path.join(outputPath, output_video_folder_name)
            if not os.path.isdir(output_video_path):
                os.makedirs(output_video_path)

            video_path = os.path.join(video_type_folder, video)
            print("Extracting frames for " + video)
            os.system("sudo python3 gop_detect.py -i " + video_path + " -p " + output_video_path + " -o " + video_type + " -c " + video) 

            