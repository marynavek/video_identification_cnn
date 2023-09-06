import glob
import os
import random
import numpy as np
import shutil
import time
import cv2
from xgboost import cv
import librosa
import numpy as np
from numpy.random import seed, randint

from process_audio_mel import preprocess_sound

from prnu_extract import extract_single

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def get_video_compression_types(video_name):
    video_types = [video_name]
    
    for category in ['flat', 'indoor', 'outdoor']:
        if category in video_name:
            WA = video_name.replace(category, f"{category}WA")
            # if "D05" not in video_name:
            YT = video_name.replace(category, f"{category}YT")
            # if "D05" not in video_name:
            # video_types.extend([WA, YT])
            # else: 
            #     video_types.extend([WA])
            return video_types

def copy_frames(src_path, dest_path_1, original_videos, device):
    

    for video in original_videos:
        video_variations = get_video_compression_types(video)
        
        for video_variation in video_variations:
            train_frames_1 = 0
            if 'flat' in video_variation: 
                video_dest_path = os.path.join(src_path, '__flat__')
            if 'indoor' in video_variation: 
                video_dest_path = os.path.join(src_path, '__indoor__')
            if 'outdoor' in video_variation: 
                video_dest_path = os.path.join(src_path, '__outdoor__')
            vid_path = os.path.join(video_dest_path, video_variation)
            if os.path.exists(vid_path):
                video_name = os.path.basename(os.path.normpath(vid_path))
                audios = listdir_nohidden(vid_path)
                new_video_name = device + "_V_" + video_name.split("_V_")[1]

                for audio in audios:
                    audio_path = os.path.join(vid_path, audio)
                    y, sr = librosa.load(audio_path)

                    length = sr * 5
                    seg_num = 500
                    range_high = len(y) - length
                    seed(1)
                    random_start = randint(range_high, size=seg_num)
                    # data = np.zeros((seg_num * sample_num, 496, 64, 1))

                    for j in range(seg_num):
                        cur_wav = y[random_start[j]:random_start[j] + length]
                        cur_wav = cur_wav / 32768.0
                        cur_spectro = preprocess_sound(cur_wav, sr)
                        # cur_spectro = np.expand_dims(cur_spectro, 2)

                        # spectro_reshaped = cur_spectro.reshape(cur_spectro.shape[0], -1)
                        file_name = "patch_number_"+str(j)+ "_vid_name_" + new_video_name + ".txt"
                        np.savetxt(os.path.join(dest_path_1, file_name), cur_spectro)
                    
                    


input_frames_dir = "/Users/marynavek/Projects/files/15_devices_audio"
# output_frames_dir_1 = "/Users/marynavek/Projects/files/audio_csv_files"
output_frames_dir_1 = "/Users/marynavek/Projects/files/audio_experiment/train_test_audio"
# output_frames_dir_2 = "/Users/marynavek/Projects/files/Rong_devices_frames_ds_2"
# output_frames_dir_3 = "/Users/marynavek/Projects/files/Rong_devices_frames_ds_3"

if not os.path.exists(output_frames_dir_1):
        os.makedirs(output_frames_dir_1)

devices = [device for device in listdir_nohidden(input_frames_dir)]

device_train_v = {}
device_test_v = {}

train_frames_dir_1 = os.path.join(output_frames_dir_1, "train")
test_frames_dir_1 = os.path.join(output_frames_dir_1, "test")

if not os.path.exists(train_frames_dir_1):
    os.mkdir(train_frames_dir_1)
if not os.path.exists(test_frames_dir_1):
    os.mkdir(test_frames_dir_1)

for device in devices:
    # if device in DATASET_DEVICES:
        d_src_path = os.path.join(input_frames_dir, device)
        d_dest_train_path_1 = os.path.join(train_frames_dir_1, device)
        d_dest_test_path_1 = os.path.join(test_frames_dir_1, device)

        if not os.path.exists(d_dest_train_path_1):
            os.mkdir(d_dest_train_path_1)
        if not os.path.exists(d_dest_test_path_1):
            os.mkdir(d_dest_test_path_1)

        # if not os.path.exists(d_dest_path_1):
        #     os.mkdir(d_dest_path_1)

        flat_vids_dir = os.path.join(d_src_path, '__flat__')
        indoor_vids_dir = os.path.join(d_src_path, '__indoor__')
        outdoor_vids_dir = os.path.join(d_src_path, '__outdoor__')

        flat_vids = [v for v in listdir_nohidden(flat_vids_dir) if
                        os.path.isdir(os.path.join(flat_vids_dir, v)) and "_flat_" in v]
        indoor_vids = [v for v in listdir_nohidden(indoor_vids_dir) if
                        os.path.isdir(os.path.join(indoor_vids_dir, v)) and "_indoor_" in v]
        outdoor_vids = [v for v in listdir_nohidden(outdoor_vids_dir) if
                        os.path.isdir(os.path.join(outdoor_vids_dir, v)) and "_outdoor_" in v]

        num_original_vids = len(flat_vids) + len(indoor_vids) + len(outdoor_vids)

        # num_train_vids = 8
        # num_test_vids = 8

        num_flat_test_vids= int(len(flat_vids)*0.4)
        num_flat_train_vids = len(flat_vids) - num_flat_test_vids
        num_indoor_test_vids= int(len(indoor_vids)*0.4)
        num_indoor_train_vids = len(indoor_vids) - num_indoor_test_vids
        num_outdoor_test_vids= int(len(outdoor_vids)*0.4)
        num_outdoor_train_vids = len(outdoor_vids) - num_outdoor_test_vids

        num_test_vids = num_flat_test_vids + num_indoor_test_vids + num_outdoor_test_vids
        num_train_vids = num_flat_train_vids + num_indoor_train_vids + num_outdoor_train_vids

        print(f"\n{device} | Total videos: {num_original_vids}, train: {num_train_vids}, test: {num_test_vids}\n")


        random.shuffle(flat_vids)
        random.shuffle(indoor_vids)
        random.shuffle(outdoor_vids)

        train_vids = []
        test_vids = []

        train_vids.extend(flat_vids[0:1]) 
        test_vids.extend(flat_vids[1:2])

        del flat_vids[0:2]  # Remove the used flat vids
        
        train_vids.extend(indoor_vids[0:1])
        test_vids.extend(indoor_vids[1:2])

        del indoor_vids[0:2]

        train_vids.extend(outdoor_vids[0:1])
        test_vids.extend(outdoor_vids[1:2])

        del outdoor_vids[0:2]

        # unused_vids = []
        unused_flat = []
        unused_indoor = []
        unused_outdoor = []
        # unused_vids.extend(flat_vids)
        # unused_vids.extend(indoor_vids)
        # unused_vids.extend(outdoor_vids)
        unused_flat.extend(flat_vids)
        unused_indoor.extend(indoor_vids)
        unused_outdoor.extend(outdoor_vids)

        random.shuffle(unused_flat)
        random.shuffle(unused_indoor)
        random.shuffle(unused_outdoor)
    
        # num_remaining_train_vids = num_train_vids - len(train_vids)

        # num_remaining_test_vids = num_test_vids - len(test_vids)

        num_remaining_train_flat = num_flat_train_vids - int(len(train_vids)/3)
        num_remaining_train_indoor = num_indoor_train_vids - int(len(train_vids)/3)
        num_remaining_train_outdoor = num_outdoor_train_vids - int(len(train_vids)/3)

        num_remaining_test_flat = num_flat_test_vids - int(len(test_vids)/3)
        num_remaining_test_indoor = num_indoor_test_vids - int(len(test_vids)/3)
        num_remaining_test_outdoor = num_outdoor_test_vids - int(len(test_vids)/3)

        for i in range(num_remaining_train_flat):
            train_vids.append(unused_flat[i])

        del unused_flat[0:num_remaining_train_flat]
        
        for i in range(num_remaining_train_indoor):
            train_vids.append(unused_indoor[i])

        del unused_indoor[0:num_remaining_train_indoor]

        for i in range(num_remaining_train_outdoor):
            train_vids.append(unused_outdoor[i])

        del unused_outdoor[0:num_remaining_train_outdoor]

        for i in range(num_remaining_test_flat):
            test_vids.append(unused_flat[i])
        
        del unused_flat[0:num_remaining_test_flat]

        for i in range(num_remaining_test_indoor):
            test_vids.append(unused_indoor[i])
        
        del unused_indoor[0:num_remaining_test_indoor]

        for i in range(num_remaining_test_outdoor):
            test_vids.append(unused_outdoor[i])
        
        del unused_outdoor[0:num_remaining_test_outdoor]

        combined = set(train_vids).intersection(test_vids)
        if len(combined) > 0:
            print("Error! The following video(s) occur in both test and unused set:")
            for item in combined:
                print(f"{item}")

            raise ValueError("Error! Videos occur in both train and test set!")

        copy_frames(d_src_path, d_dest_train_path_1, train_vids, device)
        copy_frames(d_src_path, d_dest_test_path_1, test_vids, device)
        