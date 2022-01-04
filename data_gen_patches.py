import numpy as np
import cv2, os
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from csv import writer

class DataGeneratorPatches(Sequence):
    def __init__(self,frames_path_dict, to_fit=True, batch_size=32, dim=(480,800,3), shuffle=True):
        self.frames_path_dict = frames_path_dict
        self.to_fit = to_fit
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
        frames_batch, prnu_batch, labels_batch = self.__generate_frames_ds__(list_IDs_temp)

        if self.to_fit == True:
            return [prnu_batch, frames_batch], labels_batch
        else:
            with open('true_labels_for_prediction.csv', 'a', newline='') as f_object:  
                # Pass the CSV  file object to the writer() function
                writer_object = writer(f_object)
                # Result - a writer object
                # Pass the data in the list as an argument into the writerow() function
                for label in labels_batch:
                    writer_object.writerow(label)  
                # Close the file object
                f_object.close()
            return [prnu_batch, frames_batch]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDS))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_frames_ds__(self, list_IDs_temp):
        frame_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        prnu_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        labels_ds = np.empty((self.batch_size, 5), dtype=np.uint8)
        for i, id in enumerate(list_IDs_temp):
            key = "item_ID"
            val = id
            item = next((d for d in self.frames_path_dict if d.get(key) == val), None)
            
            frame = self.__get_image__(item["frame_path"], resize=False)
            prnu = self.__get_image__(item["noise_path"])
            label = item["class_label"]
            frame_ds[i, ...] = frame
            prnu_ds[i, ...] = prnu
            labels_ds[i, ...] = label
            
        return frame_ds, prnu_ds, labels_ds

    def __get_image__(self, image_path):
        img = cv2.imread(image_path)    
        return img

