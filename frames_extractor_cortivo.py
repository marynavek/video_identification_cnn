import os, cv2
import argparse
from posix import listdir
import numpy as np


INPUT_DIR = "/Users/marynavek/Projects/files/ten_devices_videos_ds"
OUTPUT_DIR = "/Users/marynavek/Projects/files/experiment_cortivo/extracted_frames"
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

        # outputPath = os.path.join(OUTPUT_DIR, device, video_type)
            
        # if not os.path.isdir(outputPath):
        #     os.makedirs(outputPath)

        
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
            else: continue

            video_path = os.path.join(video_type_folder, video)
            print("Extracting frames for " + video)
            # os.system("sudo python3 iframe.py -i " + video_path + " -p " + output_video_path + " -o " + video_type + " -c " + video) 

            cap = cv2.VideoCapture(video_path)

            # Frame rate per second
            frame_rate = np.floor(cap.get(5))

            # Total number of frames
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            print(video_length)
            # Calculate modulo to save frames throughout complete video, rather than frames [1:1+FRAMES_PER_VIDEO]
            mod = 1
            if video_length >50:
                mod = video_length // 50

            number_of_frames_to_save = 50

            frames_saved = 0
            count = 0
    
            while cap.isOpened():
                # Extract the frame
                ret, frame = cap.read()

                # Frame is available
                if ret:
                    # Get current frame id
                    frame_id = cap.get(1)

                    # Determine whether we have to save this frame
                    
                    save_frame = frame_id % mod == 0

                    if save_frame:
                        # Check whether we have to resize or crop the frame
                        cv2.imwrite(output_video_path + f"/{video}-" + "%#05d.jpg" % frame_id, frame)
                        frames_saved = frames_saved + 1
                count += 1
                if (frames_saved >= number_of_frames_to_save or count >= video_length):
                # Release the feed
                    if cap.isOpened():
                        cap.release()
                    break
