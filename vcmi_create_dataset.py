import os, cv2
import random

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def listdir_all_videos(device_path):
    all_vids = []
    sub_dirs = [f for f in os.listdir(device_path) if not f.startswith('.')]
    for dir in sub_dirs:
        vids = [f for f in os.listdir(dir) if not f.startswith('.')]
        for vid in vids:
            all_vids.append(vid)
    return all_vids

DEVICES = [
    "D03_Huawei_P9",
    "D04_Apple_iPhone5c",
    "D05_Apple_iPhone6",
    "D10_Huawei_P9Lite",
    "D06_Apple_iPhone4"
]

ORIGINAL_CATEGORIES = ['flat', 'indoor', 'outdoor']
VIDEO_COMPRESSION_TYPES = ['original', 'WA', 'YT']
CATEGORIES = ['flat', 'indoor', 'outdoor',
              'flatYT', 'indoorYT', 'outdoorYT',
              'flatWA', 'indoorWA', 'outdoorWA']

TRAIN_FRAMES = 200
TEST_FRAMES = 200
VISION_DATASET_DIR = "/Users/marynavek/Projects/files/CleanUp_Videos"
VISION_FRAMES_DIR = "/Users/marynavek/Projects/files/vcmi_Frames"
OUTPUT_DIR ="/Users/marynavek/Projects/files/vcmi_temp_dataset"
SEED = 42


def init_data_dir():
    if not OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR is empty!")

    if not os.path.isdir(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of dataset directory")


def copy_frames(device, videos):
    device_src_path = os.path.join(VISION_FRAMES_DIR, device)
    device_dest_path = os.path.join(OUTPUT_DIR, device)

    # Create 'train' or 'test' directory
    if not os.path.isdir(device_dest_path):
        try:
            os.makedirs(device_dest_path)
        except OSError as e:
            print(e)
            raise ValueError(f"{device} | Error during creation of directory")
    else:
        print(f"{device} | Skipping device, path already exists in data set")
        return

    for video in videos:
        video_name = str(video).split(".")[0]
        video_types = get_video_compression_types(video_name)
        print(f"video name {video}")
        for video_type in video_types:
            print(f"{device} | Copying frames for video {video_type}")
            for category in CATEGORIES:
                new_cat = "_" + category + "_"
                if new_cat in video_type:
                    video_path = os.path.join(device_src_path, category, video_type)
                    break
            print(f"video_path name {video_path}")
            files = os.listdir(video_path)

            if len(files) == 0:
                print(f"No frames found for video {video_type}")
                break

            random.shuffle(files)
            for i in range(TRAIN_FRAMES):
                # print(video_path)
                # print(files[i])
                # file_name_p1, file_name_p2, file_name_p3 = files[i].split('.')
                # file_name = file_name_p1 + "_" + file_name_p2 + ".jpeg"
                f_src = os.path.join(video_path, files[i])
                # print(f_src)
                if not os.path.exists(os.path.join(device_dest_path,category, video_type)):
                    os.makedirs(os.path.join(device_dest_path,category, video_type))
                image = cv2.imread(f_src)
                cv2.imwrite(os.path.join(device_dest_path,category, video_type, files[i]), image)


def get_videos_by_device(device):
    #print(f"{device} | Get valid videos")
    original_valid_videos = []
    original_invalid_videos = []

    device_path = os.path.join(VISION_DATASET_DIR, device)
    video_categories = [item for item in os.listdir(device_path) if os.path.isdir(os.path.join(device_path, item))
                        and item in ORIGINAL_CATEGORIES]

    # Create list of original videos
    # We want to include at least one video per original category in train and test
    for category in video_categories:
        # Category videos
        videos = listdir_nohidden(os.path.join(device_path, category))
        print(videos)
        # Check if original video is exchanged via both WA and YT
        valid_videos, invalid_videos = check_valid_video(device, videos)
        print(valid_videos)
        # Extend list with valid videos
        original_valid_videos.extend(valid_videos)
        # Extend list with invalid videos

    return original_valid_videos, original_invalid_videos

def check_valid_video(device, original_videos, verbose=False):
    # It is considered that a video is valid if it is available for all three platforms, i.e.:
    # original, WA and YT. Otherwise, we consider it to be invalid.
    valid_videos = []
    invalid_videos = []

    device_src_path = os.path.join(VISION_FRAMES_DIR, device)
    all_vids = listdir_nohidden
    for video in original_videos:
        # print(f"video {video}")
        video_name = str(video).split(".")[0]
        print(f"video name {video_name}")
        video_types = get_video_compression_types(video_name)
        print(f"video_types {video_types}")
        
        valid = True
        for video_type in video_types:
            # if video_type not in listdir_all_videos(device_src_path):
            for coompression in VIDEO_COMPRESSION_TYPES:
                if coompression in video_type:
                    for category in ORIGINAL_CATEGORIES:
                        if category in video_type:
                            if coompression == 'WA' or coompression == "YT":
                                folder_category = category + coompression
                            else:
                                folder_category = category
                            video_path = os.path.join(device_src_path, folder_category, video_type)
                            print(video_path)
                            if not os.path.exists(video_path):
                                if verbose:
                                    print(f"Path {video_type} does not exists. Therefore, {video} is not valid.")
                                valid = False

        if valid:
            valid_videos.append(video)
        else:
            invalid_videos.append(video)

    return valid_videos, invalid_videos


def get_video_compression_types(video_name):
    video_types = [video_name]

    for category in ORIGINAL_CATEGORIES:
        if category in video_name:
            WA = video_name.replace(category, f"{category}WA")
            YT = video_name.replace(category, f"{category}YT")
            video_types.extend([WA, YT])
            return video_types


if __name__ == "__main__":
    random.seed(SEED)

    valid_device_video_dict = {}

    import time
    t_start = time.time()
    for device in DEVICES:
        print(device)
        valid_videos, invalid_videos = get_videos_by_device(device)
        print(f"{device} | Start copying frames for {len(valid_videos)} videos")
        copy_frames(device, valid_videos)
        print(f"{device} | Finished ({int(time.time() - t_start)} sec.)")