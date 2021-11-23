import os
import shutil
import random
import cv2

FRAMES_PER_VIDEO = 10


FRAMES_DIR="/Users/marynavek/Projects/Video_Project/Frames"
TRAIN_SET_PATH_DIR="/Users/marynavek/Projects/Video_Project/train"
TEST_SET_PATH_DIR="/Users/marynavek/Projects/Video_Project/test"


DEVICES = [item for item in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, item))]

for device in DEVICES:
    device_folder = os.path.join(FRAMES_DIR, device)
    VIDEO_TYPES = [item for item in os.listdir(device_folder) if os.path.isdir(os.path.join(device_folder, item))]

    for video_type in VIDEO_TYPES:
        video_type_folder = os.path.join(device_folder, video_type)
        VIDEO_NAMES = [item for item in os.listdir(video_type_folder) if os.path.isdir(os.path.join(video_type_folder, item))]

        if device == "D05_Apple_iPhone5c":

            train_video_path = os.path.join(TRAIN_SET_PATH_DIR, device)
            test_video_path = os.path.join(TEST_SET_PATH_DIR, device)
        else:
            train_video_path = os.path.join(TRAIN_SET_PATH_DIR, "other")
            test_video_path = os.path.join(TEST_SET_PATH_DIR, "other") 
        if not os.path.isdir(train_video_path):
            os.makedirs(train_video_path)
        if not os.path.isdir(test_video_path):
            os.makedirs(test_video_path)
        
        train_frames = 0
        for video_name in VIDEO_NAMES:
            video_folder = os.path.join(video_type_folder, video_name)

            frames = [item for item in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, item))]

            
            # for frame in frames:
            frames_from_video = 0
            for frame in frames:

                if frames_from_video < 10:
                    # random_frame = random.choice(frames)
                    if os.path.isfile(os.path.join(train_video_path, frame)):
                        image = cv2.imread(os.path.join(video_folder, frame))
                        file_name = "im"+str(train_frames)+".png"
                        cv2.imwrite(os.path.join(train_video_path, file_name), image)
                        train_frames += 1
                    else:
                        shutil.copy2(os.path.join(video_folder, frame), train_video_path)

                    frames_from_video +=1
                else:
                    break
                # if test_frames < 11:
                #     # random_frame = random.choice(frames)
                #     if not os.path.isfile(os.path.join(test_video_path, frame)) and not os.path.isfile(os.path.join(train_video_path, frame)):
                #         shutil.copy2(os.path.join(video_folder, frame), test_video_path)
                #         test_frames +=1
                # if test_frames > 9 and train_frames > 9:
                #     break 

            
