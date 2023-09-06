import numpy as np
import cv2, os
from tensorflow.keras.utils import Sequence
# from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf
from csv import writer

from prnu_extract import extract_single
# from PIL import Image

DEVICE_TYPES = [
        "D01_Samsung_GalaxyS3",
        "D02_Huawei_P9",
        "D03_Apple_iPhone5c",
        "D04_Apple_iPhone6",
        "D05_Huawei_P9Lite", 
        "D06_Apple_iPhone6Plus",
        "D07_Samsung_GalaxyS5",
        "D08_Apple_iPhone5",
        "D09_Huawei_P8",
        "D10_Samsung_GalaxyS4Mini"
    ]

class DataGeneratorBayar(Sequence):
    def __init__(self,frames_path_dict, num_classes, batch_size=32, to_fit=True, dim=(480,800,3), shuffle=True):
        self.frames_path_dict = frames_path_dict
        self.to_fit = to_fit
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dim = dim
        self.list_IDS = list(range(0, len(frames_path_dict)))
        self.shuffle = shuffle
        self.on_epoch_end()   

    def __len__(self):
        return int(np.floor(len(self.frames_path_dict)) / self.batch_size)     

    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDS[k] for k in indexes]
        
        # Generate data
        frames_batch, std_patch_batch, std_quadrant_batch, labels_batch = self.__generate_frames_ds__(list_IDs_temp)

        if self.to_fit == True:
            return [frames_batch, std_patch_batch, std_quadrant_batch], labels_batch
        else:
            with open('true_labels_for_prediction_bayar.csv', 'a', newline='') as f_object:  
                # Pass the CSV  file object to the writer() function
                writer_object = writer(f_object)
                # Result - a writer object
                # Pass the data in the list as an argument into the writerow() function
                for label in labels_batch:
                    writer_object.writerow(label)  
                # Close the file object
                f_object.close()
            return frames_batch


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDS))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_frames_ds__(self, list_IDs_temp):
        # frame_ds = np.empty((self.batch_size, 480, 800, 3), dtype=np.uint8)
        frame_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        # frame_ds = np.empty((self.batch_size, 256, 256, 3), dtype=np.uint8)
        labels_ds = np.empty((self.batch_size, self.num_classes), dtype=np.uint8)
        std_quadrant = np.empty((self.batch_size, 1), dtype=np.uint8)
        std_patch = np.empty((self.batch_size, 1), dtype=np.uint8)
        for i, id in enumerate(list_IDs_temp):
            key = "item_ID"
            val = id
            item = next((d for d in self.frames_path_dict if d.get(key) == val), None)
            
            frame = self.__get_image__(item["patch_path"])
            label = item["class_label"]
            frame_ds[i, ...] = frame
            labels_ds[i, ...] = label

            std_patch_1, std_quadrant_1 = self.__get_std__(item["patch_path"])
            std_quadrant[i, ...] = std_quadrant_1
            std_patch[i, ...] = std_patch_1
            
        return frame_ds, std_patch, std_quadrant, labels_ds


    #read image and resize it to (480,800, 3() and dt.float32 type)
    def __get_image__(self, image_path):
        # im = Image.open(image_path)
        img = cv2.imread(image_path)   
        # img = cv2.resize(img, (800,480))
        # im2arr = np.array(im) # im2arr.shape: height x width x channel
        # img = Image.fromarray(im2arr)
        # noise_patch = extract_single(img)
        # new = noise_patch.reshape(128,128,1)
        return img


    def __determine_device__(self,image_name):
        for device in DEVICE_TYPES:
            if device in image_name:
                return device
                
    def get_video_name(self, image_name):
        remove_part, main_part = image_name.split("_vid_name_")
        video_name, remove = main_part.split("P-number")
        return video_name

    def get_frame_number(self, image_name):
        remove_part, main_part = image_name.split("frame_number_")
        frame_number, remove = main_part.split("vid_name")
        return frame_number

    def listdir_nohidden(self, path):
        return [f for f in os.listdir(path) if not f.startswith('.')]

    def crop_image_into_four_sectors(self, img):
        width = img.shape[1]
        # Cut the image in half
        width_cutoff = width // 2
        left1 = img[:, :width_cutoff]
        right1 = img[:, width_cutoff:]

        height_left = left1.shape[0]
        height_cutoff_left = height_left // 2
        first_sector = left1[:height_cutoff_left, :]
        second_sector = left1[height_cutoff_left:, :]
        
        # start vertical devide image
        height_right = img.shape[0]
        # Cut the image in half
        height_cutoff_right = height_right // 2
        third_sector = right1[:height_cutoff_right, :]
        forth_sector = right1[height_cutoff_right:, :]
        
        return first_sector, second_sector, third_sector, forth_sector


    def __get_quadrant__(self, image_path):
        if self.shuffle == True:
            main_path = "/Users/marynavek/Projects/files/experiment_5/15_dev_train_test_frames/train"
        else:
            main_path = "/Users/marynavek/Projects/files/experiment_5/15_dev_train_test_frames/test"

        device = self.__determine_device__(image_path)
        video_name = self.get_video_name(image_path)
        f_number = self.get_frame_number(image_path)
        
        new_image_path = os.path.join(main_path, device)
        all_images = self.listdir_nohidden(new_image_path)
        for image in all_images:
            if device in image and video_name[:-1] in image and f_number in image:
                image_new_p = os.path.join(new_image_path, image)
                im = cv2.imread(image_new_p)
                sector_1, sector_2, sector_3, sector_4 = self.crop_image_into_four_sectors(im)

                return np.std(sector_2)

    def __get_std__(self, image_path):
        img = cv2.imread(image_path)  
        std_patch = np.std(img)
        std_quadrant = self.__get_quadrant__(image_path)
        if std_quadrant == 0:
            std_quadrant = 1000
        if std_patch == 0:
            std_patch = 1000
        f_std_patch = 1/std_patch
        f_std_quadrant = 1/std_quadrant

        return f_std_patch, f_std_quadrant
